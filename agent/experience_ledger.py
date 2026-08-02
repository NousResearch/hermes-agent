"""Experience Ledger — AIDE²-inspired skill/memory quality tracking.

Inspired by AIDE²'s public/private score split: this module records eval
outcomes for every skill and memory entry, separating agent-visible
(public) scores from hidden (private) signals that guard against reward
hacking.

Design principles (from AIDE² research):
- Public score = agent-reported completion quality (optimistic, gameable)
- Private score = objective signals (user corrections, rework rate, reuse)
- Cost tracking = USD spent per task (selection pressure)
- Lineage = which version evolved from which (audit trail)
- Stale detection = skills with poor private scores get flagged

This is the foundation for Hermes² self-improvement (P1-1 outer loop).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ========================================================================
# Data structures
# ========================================================================

@dataclass
class SkillEval:
    """Evaluation record for a skill execution."""

    skill_id: str
    eval_event_id: str
    task_family: str  # coding/research/lark/security/mlops/etc.
    public_score: float = 0.0  # Agent-reported (0-1)
    private_score: float = 0.0  # Hidden objective (0-1)
    cost_usd: float = 0.0
    outcome: str = "unknown"  # success/partial/failure/reward_hack
    tokens_in: int = 0
    tokens_out: int = 0
    duration_sec: float = 0.0
    user_corrected: bool = False
    rework_count: int = 0
    reuse_count: int = 0  # How many times this skill was reused after
    lineage: str = ""  # Parent eval_event_id if this is an evolution
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "eval_event_id": self.eval_event_id,
            "task_family": self.task_family,
            "public_score": self.public_score,
            "private_score": self.private_score,
            "cost_usd": self.cost_usd,
            "outcome": self.outcome,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_sec": self.duration_sec,
            "user_corrected": self.user_corrected,
            "rework_count": self.rework_count,
            "reuse_count": self.reuse_count,
            "lineage": self.lineage,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillEval":
        return cls(**d)


@dataclass
class SkillSummary:
    """Aggregated statistics for a skill."""

    skill_id: str
    total_evals: int = 0
    avg_public_score: float = 0.0
    avg_private_score: float = 0.0
    total_cost_usd: float = 0.0
    success_rate: float = 0.0
    avg_cost_per_success: float = 0.0
    avg_tokens_per_eval: int = 0
    user_correction_rate: float = 0.0
    last_eval_at: float = 0.0
    days_since_last_eval: float = 0.0
    staleness_score: float = 0.0  # 0=fresh, 1=very stale

    @property
    def is_stale(self) -> bool:
        """A skill is stale if: low private score OR long time since last eval."""
        return self.staleness_score > 0.8

    @property
    def needs_improvement(self) -> bool:
        """Needs improvement if private score < 0.6 with sufficient evals."""
        return self.total_evals >= 3 and self.avg_private_score < 0.6


class ExperienceLedger:
    """Records and queries skill/memory evaluation outcomes.

    Inspired by AIDE²'s experience tracking:
    - Each skill execution generates an eval record
    - Public score (agent-visible) vs private score (hidden)
    - Cost tracking for selection pressure
    - Staleness detection based on private score + recency

    Usage:
        ledger = ExperienceLedger(hermes_home=Path.home() / ".hermes")
        ledger.record_eval(SkillEval(...))
        summary = ledger.get_summary("github-pr-workflow")
        stale = ledger.get_stale_skills()
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        max_history_per_skill: int = 100,
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.ledger_path = self.hermes_home / "state" / "experience_ledger.json"
        self.max_history = max_history_per_skill
        self._evals: Dict[str, List[SkillEval]] = {}  # skill_id -> [evals]
        self._summaries: Dict[str, SkillSummary] = {}
        self._load()

    def _load(self) -> None:
        """Load existing ledger from disk."""
        if not self.ledger_path.exists():
            logger.info("Experience ledger: no existing data found")
            return

        try:
            data = json.loads(self.ledger_path.read_text())
            for skill_id, eval_list in data.get("evals", {}).items():
                self._evals[skill_id] = [
                    SkillEval.from_dict(e) for e in eval_list
                ]
            logger.info(
                "Experience ledger: loaded %d skills, %d total evals",
                len(self._evals),
                sum(len(v) for v in self._evals.values()),
            )
        except Exception as e:
            logger.warning("Experience ledger: failed to load: %s", e)

    def save(self) -> None:
        """Persist ledger to disk."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "evals": {
                skill_id: [e.to_dict() for e in evals]
                for skill_id, evals in self._evals.items()
            },
        }
        self.ledger_path.write_text(json.dumps(data, indent=2))
        logger.debug(
            "Experience ledger: saved to %s", self.ledger_path,
        )

    def record_eval(self, eval_record: SkillEval) -> None:
        """Record a new skill evaluation."""
        if eval_record.skill_id not in self._evals:
            self._evals[eval_record.skill_id] = []

        self._evals[eval_record.skill_id].append(eval_record)

        # Trim history if needed
        if len(self._evals[eval_record.skill_id]) > self.max_history:
            self._evals[eval_record.skill_id] = (
                self._evals[eval_record.skill_id][-self.max_history:]
            )

        # Update summary
        self._summaries[eval_record.skill_id] = self._compute_summary(
            eval_record.skill_id,
        )

    def get_summary(self, skill_id: str) -> Optional[SkillSummary]:
        """Get aggregated summary for a skill."""
        if skill_id not in self._summaries:
            if skill_id in self._evals:
                self._summaries[skill_id] = self._compute_summary(skill_id)
            else:
                return None
        return self._summaries[skill_id]

    def get_all_summaries(self) -> Dict[str, SkillSummary]:
        """Get summaries for all skills."""
        for skill_id in self._evals:
            if skill_id not in self._summaries:
                self._summaries[skill_id] = self._compute_summary(skill_id)
        return dict(self._summaries)

    def get_stale_skills(self) -> List[SkillSummary]:
        """Get skills that are stale (low private score or old)."""
        summaries = self.get_all_summaries()
        return [s for s in summaries.values() if s.is_stale]

    def get_skills_needing_improvement(self) -> List[SkillSummary]:
        """Get skills with low private scores that need improvement."""
        summaries = self.get_all_summaries()
        return [s for s in summaries.values() if s.needs_improvement]

    def get_top_skills(self, n: int = 5) -> List[SkillSummary]:
        """Get top N skills by private score."""
        summaries = self.get_all_summaries()
        return sorted(
            summaries.values(),
            key=lambda s: s.avg_private_score,
            reverse=True,
        )[:n]

    def get_worst_skills(self, n: int = 5) -> List[SkillSummary]:
        """Get worst N skills by private score."""
        summaries = self.get_all_summaries()
        return sorted(
            summaries.values(),
            key=lambda s: s.avg_private_score,
        )[:n]

    def get_evals_for_skill(self, skill_id: str) -> List[SkillEval]:
        """Get all evals for a skill."""
        return list(self._evals.get(skill_id, []))

    def _compute_summary(self, skill_id: str) -> SkillSummary:
        """Compute aggregated stats for a skill."""
        evals = self._evals.get(skill_id, [])
        if not evals:
            return SkillSummary(skill_id=skill_id)

        total = len(evals)
        avg_public = sum(e.public_score for e in evals) / total
        avg_private = sum(e.private_score for e in evals) / total
        total_cost = sum(e.cost_usd for e in evals)
        success_count = sum(1 for e in evals if e.outcome == "success")
        success_rate = success_count / total
        avg_cost = total_cost / max(success_count, 1)
        avg_tokens = sum(e.tokens_in + e.tokens_out for e in evals) // total
        correction_rate = sum(1 for e in evals if e.user_corrected) / total

        last_eval = max(e.created_at for e in evals)
        days_since = (time.time() - last_eval) / 86400

        # Staleness: blend of recency and private score
        # High days_since + low private_score = very stale
        recency_factor = min(days_since / 30, 1.0)  # 30 days = max
        quality_factor = 1.0 - avg_private  # Low score = high staleness
        staleness = 0.6 * recency_factor + 0.4 * quality_factor

        return SkillSummary(
            skill_id=skill_id,
            total_evals=total,
            avg_public_score=round(avg_public, 3),
            avg_private_score=round(avg_private, 3),
            total_cost_usd=round(total_cost, 4),
            success_rate=round(success_rate, 3),
            avg_cost_per_success=round(avg_cost, 4),
            avg_tokens_per_eval=avg_tokens,
            user_correction_rate=round(correction_rate, 3),
            last_eval_at=last_eval,
            days_since_last_eval=round(days_since, 1),
            staleness_score=round(staleness, 3),
        )

    @property
    def total_evals(self) -> int:
        """Total number of evals across all skills."""
        return sum(len(v) for v in self._evals.values())

    @property
    def skill_count(self) -> int:
        """Number of skills with eval data."""
        return len(self._evals)
