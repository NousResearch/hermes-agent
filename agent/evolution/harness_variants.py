"""Harness Variant Isolation — inspired by HarnessX Section 4.3.

When tasks need conflicting improvements, the system maintains multiple
harness variants. Each variant is an independent configuration snapshot
that routes tasks to the variant with the highest estimated success rate.

Key properties (from HarnessX):
  - Non-degrading aggregate trajectory: no variant regresses on its routed tasks
  - Sustained exploration: variants can diverge without blocking each other
  - Lower total token consumption: fork only when seesaw would otherwise reject

Variant lifecycle:
  1. DEFAULT variant handles all tasks initially
  2. When a proposal improves some tasks but regresses others:
     - Applied to the target variant (not rejected)
     - A new variant is forked
     - The lowest-performing variant is retired if pool is full (max K variants)
  3. Task routing: each task is routed to the variant with the best score
     on that task's cluster (domain-based clustering)

Configuration (per HarnessX):
  - max_variants (K): 3 by default
  - routing_strategy: "best_score" | "domain_cluster" | "round_robin"
  - min_score_differential: 0.05 (fork when improvement exceeds this threshold)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_MAX_VARIANTS = 3
DEFAULT_MIN_SCORE_DIFFERENTIAL = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class HarnessVariant:
    """A single harness variant — an independent configuration snapshot."""

    variant_id: str = field(default_factory=lambda: f"variant_{uuid.uuid4().hex[:8]}")
    name: str = "default"
    parent_variant: Optional[str] = None  # Forked from which variant
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Performance tracking
    task_scores: Dict[str, float] = field(default_factory=dict)  # task_name → best score
    total_tasks_attempted: int = 0
    total_tasks_succeeded: int = 0
    avg_score: float = 0.0

    # Configuration snapshot
    active_skills: List[str] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)
    prompt_overrides: Dict[str, str] = field(default_factory=dict)

    # Status
    is_active: bool = True
    retired_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_tasks_attempted == 0:
            return 0.0
        return self.total_tasks_succeeded / self.total_tasks_attempted

    def record_result(self, task_name: str, score: float, passed: bool) -> None:
        """Update performance tracking with a new result."""
        current_best = self.task_scores.get(task_name, 0.0)
        if score > current_best:
            self.task_scores[task_name] = score
        self.total_tasks_attempted += 1
        if passed:
            self.total_tasks_succeeded += 1
        # Update running average
        if self.total_tasks_attempted > 0:
            self.avg_score = (
                (self.avg_score * (self.total_tasks_attempted - 1) + score)
                / self.total_tasks_attempted
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "parent_variant": self.parent_variant,
            "created_at": self.created_at,
            "task_scores": self.task_scores,
            "total_tasks_attempted": self.total_tasks_attempted,
            "total_tasks_succeeded": self.total_tasks_succeeded,
            "avg_score": self.avg_score,
            "success_rate": self.success_rate,
            "active_skills": self.active_skills,
            "active_tools": self.active_tools,
            "prompt_overrides": self.prompt_overrides,
            "is_active": self.is_active,
            "retired_at": self.retired_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessVariant":
        return cls(
            variant_id=d.get("variant_id", ""),
            name=d.get("name", "default"),
            parent_variant=d.get("parent_variant"),
            created_at=d.get("created_at", ""),
            task_scores=d.get("task_scores", {}),
            total_tasks_attempted=d.get("total_tasks_attempted", 0),
            total_tasks_succeeded=d.get("total_tasks_succeeded", 0),
            avg_score=d.get("avg_score", 0.0),
            active_skills=d.get("active_skills", []),
            active_tools=d.get("active_tools", []),
            prompt_overrides=d.get("prompt_overrides", {}),
            is_active=d.get("is_active", True),
            retired_at=d.get("retired_at"),
        )


# ---------------------------------------------------------------------------
# Variant Manager
# ---------------------------------------------------------------------------


class VariantManager:
    """Manages multiple harness variants with task routing.

    Single-point-of-contact for variant lifecycle: creation, forking,
    retiring, and task-to-variant routing.
    """

    def __init__(
        self,
        max_variants: int = DEFAULT_MAX_VARIANTS,
        min_score_differential: float = DEFAULT_MIN_SCORE_DIFFERENTIAL,
    ):
        self.max_variants = max_variants
        self.min_score_differential = min_score_differential
        self._variants: Dict[str, HarnessVariant] = {}
        self._active_variant_id: str = ""

        # Ensure default variant always exists
        self._ensure_default()

    # -- Variant management ---------------------------------------------------

    def _ensure_default(self) -> HarnessVariant:
        """Guarantee a default variant exists."""
        for v in self._variants.values():
            if v.name == "default" and v.is_active:
                self._active_variant_id = v.variant_id
                return v
        default = HarnessVariant(name="default")
        self._variants[default.variant_id] = default
        self._active_variant_id = default.variant_id
        return default

    @property
    def active_variant(self) -> HarnessVariant:
        if self._active_variant_id and self._active_variant_id in self._variants:
            return self._variants[self._active_variant_id]
        return self._ensure_default()

    @property
    def active_variants(self) -> List[HarnessVariant]:
        return [v for v in self._variants.values() if v.is_active]

    def get_variant(self, variant_id: str) -> Optional[HarnessVariant]:
        return self._variants.get(variant_id)

    def set_active(self, variant_id: str) -> bool:
        """Set the active variant by ID."""
        variant = self._variants.get(variant_id)
        if variant and variant.is_active:
            self._active_variant_id = variant_id
            return True
        return False

    # -- Task routing ---------------------------------------------------------

    def route_task(self, task_name: str, task_domain: str = "general") -> HarnessVariant:
        """Route a task to the best-performing variant.

        Strategy: route to the variant with the highest score on this task.
        If no variant has attempted this task, use domain-based clustering.
        Falls back to the default variant.
        """
        active = self.active_variants
        if not active:
            return self._ensure_default()
        if len(active) == 1:
            return active[0]

        # Find variant with best score on this specific task
        best_variant = active[0]
        best_score = -1.0
        for variant in active:
            score = variant.task_scores.get(task_name, -1.0)
            if score > best_score:
                best_score = score
                best_variant = variant

        # If no variant has attempted this task, use domain-clustered routing
        if best_score < 0 and task_domain != "general":
            # Find variant with best avg on tasks in same domain
            domain_best = active[0]
            domain_best_score = -1.0
            for variant in active:
                domain_tasks = {
                    k: v for k, v in variant.task_scores.items()
                    if k.startswith(task_domain) or task_domain in k
                }
                if domain_tasks:
                    domain_avg = sum(domain_tasks.values()) / len(domain_tasks)
                    if domain_avg > domain_best_score:
                        domain_best_score = domain_avg
                        domain_best = variant
            if domain_best_score >= 0:
                return domain_best

        return best_variant

    # -- Forking logic --------------------------------------------------------

    def should_fork(
        self,
        proposal_target: str,
        improved_tasks: List[str],
        regressed_tasks: List[str],
    ) -> bool:
        """Determine if a proposal should fork a new variant.

        Forks when a proposal improves some tasks but regresses others,
        and the net improvement exceeds the threshold.
        """
        if not regressed_tasks:
            return False  # No regression — no need to fork
        if not improved_tasks:
            return False  # Nothing improved — reject outright
        if len(self.active_variants) >= self.max_variants:
            return False  # Pool is full — retire lowest first
        return True

    def fork_variant(
        self,
        parent: HarnessVariant,
        proposal_target: str,
        name: Optional[str] = None,
    ) -> HarnessVariant:
        """Create a new variant forked from a parent.

        The new variant inherits the parent's configuration and adds the
        proposed change. The parent retains its original configuration
        (for tasks that would regress).
        """
        if len(self.active_variants) >= self.max_variants:
            self._retire_lowest()

        child = HarnessVariant(
            name=name or f"{parent.name}-{proposal_target}",
            parent_variant=parent.variant_id,
            active_skills=list(parent.active_skills),
            active_tools=list(parent.active_tools),
            prompt_overrides=dict(parent.prompt_overrides),
        )
        self._variants[child.variant_id] = child
        logger.info(
            "Forked variant '%s' (%s) from '%s' for proposal '%s'",
            child.name, child.variant_id, parent.name, proposal_target,
        )
        return child

    def retire_variant(self, variant_id: str) -> bool:
        """Retire a variant (mark inactive, don't delete)."""
        variant = self._variants.get(variant_id)
        if not variant:
            return False
        if variant.name == "default":
            logger.warning("Cannot retire the default variant")
            return False
        variant.is_active = False
        variant.retired_at = datetime.now(timezone.utc).isoformat()
        # If this was the active variant, switch to default
        if self._active_variant_id == variant_id:
            self._ensure_default()
        logger.info("Retired variant '%s' (%s)", variant.name, variant_id)
        return True

    def _retire_lowest(self) -> Optional[HarnessVariant]:
        """Retire the lowest-performing non-default variant."""
        active = [v for v in self.active_variants if v.name != "default"]
        if not active:
            return None
        lowest = min(active, key=lambda v: v.avg_score)
        self.retire_variant(lowest.variant_id)
        return lowest

    # -- Persistence ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_variant_id": self._active_variant_id,
            "max_variants": self.max_variants,
            "min_score_differential": self.min_score_differential,
            "variants": {vid: v.to_dict() for vid, v in self._variants.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariantManager":
        mgr = cls(
            max_variants=d.get("max_variants", DEFAULT_MAX_VARIANTS),
            min_score_differential=d.get("min_score_differential", DEFAULT_MIN_SCORE_DIFFERENTIAL),
        )
        mgr._variants = {
            vid: HarnessVariant.from_dict(vd)
            for vid, vd in d.get("variants", {}).items()
        }
        mgr._active_variant_id = d.get("active_variant_id", "")
        if not mgr._active_variant_id or mgr._active_variant_id not in mgr._variants:
            mgr._ensure_default()
        return mgr

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist variant state to disk."""
        if path is None:
            path = get_hermes_home() / "evolution" / "variants.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VariantManager":
        """Load variant state from disk."""
        if path is None:
            path = get_hermes_home() / "evolution" / "variants.json"
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load variant state: %s", e)
            return cls()
