"""Subconscious Dream & Autonomous Memory Distillation Cycle.

Autonomous background self-reflection engine for Hermes Agent. When the system is
idle (or triggered via `hermes dream`), the Dream Cycle:
1. Harvests error episodes, tool failures, and user corrections from recent trajectories.
2. Performs counterfactual self-reflection ("what invariant rule would prevent this?").
3. Distills actionable Cognitive Reflex Rules into an immutable reflex graph.
4. Prunes stale or redundant rules to keep agent context lean and razor-sharp.
5. Injects top cognitive reflexes into subsequent sessions' system prompts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

REFLEX_FILE_NAME = "REFLEX_RULES.json"
REFLEX_MD_NAME = "REFLEXES.md"


@dataclass
class ReflexRule:
    """An invariant cognitive rule derived from past failure analysis."""
    rule_id: str
    category: str  # e.g., 'git', 'docker', 'coding', 'tools', 'system'
    trigger: str   # Situation where rule activates
    heuristic: str # The invariant action/instruction
    severity: str = "mandatory"  # 'mandatory' or 'advisory'
    confidence: float = 1.0
    times_applied: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReflexRule:
        return cls(
            rule_id=data.get("rule_id", ""),
            category=data.get("category", "general"),
            trigger=data.get("trigger", ""),
            heuristic=data.get("heuristic", ""),
            severity=data.get("severity", "mandatory"),
            confidence=float(data.get("confidence", 1.0)),
            times_applied=int(data.get("times_applied", 0)),
            created_at=int(data.get("created_at", int(time.time()))),
        )


@dataclass
class DreamEpisode:
    """A harvested trajectory episode containing an error or user correction."""
    session_id: str
    turn_index: int
    error_signal: str
    failed_tool: Optional[str] = None
    user_correction: Optional[str] = None
    context_snippet: str = ""


@dataclass
class DreamSummary:
    """Outcome report of an executed Dream Cycle."""
    episodes_harvested: int
    rules_synthesized: int
    rules_pruned: int
    total_active_rules: int
    duration_ms: float
    journal_entry: str


class DreamCycleEngine:
    """Autonomous dream, counterfactual reflection, and reflex distillation engine."""

    def __init__(self, hermes_home: Optional[Path] = None):
        self.home = hermes_home or get_hermes_home()
        self.memories_dir = self.home / "memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.reflex_json_path = self.memories_dir / REFLEX_FILE_NAME
        self.reflex_md_path = self.memories_dir / REFLEX_MD_NAME

    # --------------------------------------------------------------------------
    # 1. Reflex Storage & Persistence
    # --------------------------------------------------------------------------

    def load_reflex_rules(self) -> List[ReflexRule]:
        """Load all active reflex rules from disk."""
        if not self.reflex_json_path.exists():
            return []
        try:
            raw = json.loads(self.reflex_json_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [ReflexRule.from_dict(r) for r in raw if isinstance(r, dict)]
        except Exception as exc:
            logger.warning("Failed to load reflex rules: %s", exc)
        return []

    def save_reflex_rules(self, rules: List[ReflexRule]) -> None:
        """Atomically persist reflex rules to JSON and human-readable Markdown."""
        tmp_json = self.reflex_json_path.with_suffix(".tmp")
        data = [r.to_dict() for r in rules]
        tmp_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_json.replace(self.reflex_json_path)

        # Generate readable Markdown journal
        lines = [
            "# Cognitive Reflexes & Distilled Heuristics",
            "",
            "These behavioral invariants were autonomously synthesized during Hermes Dream Cycles.",
            "",
        ]
        if not rules:
            lines.append("*No active reflex rules recorded yet.*\n")
        else:
            by_category: Dict[str, List[ReflexRule]] = {}
            for r in rules:
                by_category.setdefault(r.category, []).append(r)

            for cat, cat_rules in sorted(by_category.items()):
                lines.append(f"## {cat.capitalize()}")
                lines.append("")
                for r in cat_rules:
                    lines.append(f"- **[{r.rule_id}]** (`{r.severity}`) **When**: {r.trigger}")
                    lines.append(f"  - **Heuristic**: {r.heuristic}")
                    lines.append("")

        tmp_md = self.reflex_md_path.with_suffix(".tmp")
        tmp_md.write_text("\n".join(lines), encoding="utf-8")
        tmp_md.replace(self.reflex_md_path)

    # --------------------------------------------------------------------------
    # 2. Episode Harvesting
    # --------------------------------------------------------------------------

    def harvest_episodes_from_messages(self, messages: List[Dict[str, Any]], session_id: str = "default") -> List[DreamEpisode]:
        """Harvest learning episodes from a list of conversation messages."""
        episodes: List[DreamEpisode] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            # Detect tool failure
            if role == "tool" or "tool_call" in msg:
                if any(err_kw in content.lower() for err_kw in ["error", "exception", "failed", "traceback", "fatal", "exit code"]):
                    episodes.append(
                        DreamEpisode(
                            session_id=session_id,
                            turn_index=idx,
                            error_signal=content[:400],
                            failed_tool=msg.get("name") or "tool",
                            context_snippet=content[:200],
                        )
                    )

            # Detect user correction
            elif role == "user":
                lower_content = content.lower()
                correction_cues = [
                    "that failed", "that's wrong", "you made a mistake",
                    "don't do that", "stop doing", "incorrect", "you broke",
                    "no, that", "you missed"
                ]
                if any(cue in lower_content for cue in correction_cues):
                    episodes.append(
                        DreamEpisode(
                            session_id=session_id,
                            turn_index=idx,
                            error_signal="user_correction",
                            user_correction=content[:400],
                            context_snippet=content[:200],
                        )
                    )

        return episodes

    # --------------------------------------------------------------------------
    # 3. Counterfactual Reflection & Distillation
    # --------------------------------------------------------------------------

    def distill_rule_from_episode(self, episode: DreamEpisode) -> Optional[ReflexRule]:
        """Synthesize a structured ReflexRule from an error episode."""
        sig = episode.error_signal.lower()
        corr = (episode.user_correction or "").lower()

        # Git branch / updater patterns
        if "branch" in sig or "git" in sig or "branch" in corr:
            return ReflexRule(
                rule_id=f"git_branch_remote_guard_{int(time.time() * 1000) % 10000}",
                category="git",
                trigger="When verifying or healing git branches with upstream/remote",
                heuristic="Verify remote-tracking ref and unmerged commit count before assuming a missing ref indicates branch deletion.",
                severity="mandatory",
                confidence=0.95,
            )

        # Docker / Windows path patterns
        if "docker" in sig or "wsl" in sig or "container" in sig:
            return ReflexRule(
                rule_id=f"docker_path_normalization_{int(time.time() * 1000) % 10000}",
                category="docker",
                trigger="When mounting files or executing Docker containers on host OS",
                heuristic="Normalize host paths to POSIX slashes and ensure daemon volume permissions are pre-checked.",
                severity="mandatory",
                confidence=0.90,
            )

        # Tool timeout / connection failure
        if "timeout" in sig or "timed out" in sig or "connection refused" in sig:
            return ReflexRule(
                rule_id=f"network_resilience_guard_{int(time.time() * 1000) % 10000}",
                category="network",
                trigger="When communicating with local daemons or external endpoints",
                heuristic="Apply exponential backoff with jitter and probe health endpoint before dispatching payload.",
                severity="advisory",
                confidence=0.85,
            )

        # General user correction
        if episode.user_correction:
            clean_hint = re.sub(r"[^a-zA-Z0-9\s]", "", episode.user_correction).strip()
            if clean_hint:
                return ReflexRule(
                    rule_id=f"user_feedback_rule_{int(time.time() * 1000) % 10000}",
                    category="user_preference",
                    trigger=f"In context of user instructions: '{clean_hint[:60]}...'",
                    heuristic=f"Follow user explicit constraint: {clean_hint[:120]}",
                    severity="mandatory",
                    confidence=0.92,
                )

        return None

    # --------------------------------------------------------------------------
    # 4. Deduplication & Pruning
    # --------------------------------------------------------------------------

    def deduplicate_and_merge(self, existing: List[ReflexRule], new_rules: List[ReflexRule]) -> List[ReflexRule]:
        """Deduplicate candidate rules against existing inventory using lexical similarity."""
        merged = list(existing)

        for candidate in new_rules:
            is_dup = False
            for rule in merged:
                # Same category and trigger overlap
                if rule.category == candidate.category:
                    overlap = len(set(rule.heuristic.lower().split()) & set(candidate.heuristic.lower().split()))
                    total = max(len(rule.heuristic.split()), len(candidate.heuristic.split()))
                    if total > 0 and (overlap / total) > 0.6:
                        rule.confidence = round(min(1.0, rule.confidence + 0.05), 4)
                        rule.times_applied += 1
                        is_dup = True
                        break
            if not is_dup:
                merged.append(candidate)

        return merged

    def prune_stale_rules(self, rules: List[ReflexRule], max_rules: int = 25, min_confidence: float = 0.5) -> Tuple[List[ReflexRule], int]:
        """Prune low-confidence or oldest overflow rules to prevent context bloat."""
        initial_count = len(rules)
        filtered = [r for r in rules if r.confidence >= min_confidence]
        filtered.sort(key=lambda r: (r.confidence, r.times_applied), reverse=True)
        retained = filtered[:max_rules]
        pruned_count = initial_count - len(retained)
        return retained, pruned_count

    # --------------------------------------------------------------------------
    # 5. Full Dream Cycle Run
    # --------------------------------------------------------------------------

    def execute_dream_cycle(self, recent_trajectories: Optional[List[List[Dict[str, Any]]]] = None) -> DreamSummary:
        """Execute a complete Dream Cycle over provided or discovered trajectories."""
        start_time = time.time()
        existing_rules = self.load_reflex_rules()

        episodes: List[DreamEpisode] = []
        if recent_trajectories:
            for s_idx, traj in enumerate(recent_trajectories):
                episodes.extend(self.harvest_episodes_from_messages(traj, session_id=f"session_{s_idx}"))

        new_rules: List[ReflexRule] = []
        for ep in episodes:
            rule = self.distill_rule_from_episode(ep)
            if rule:
                new_rules.append(rule)

        merged = self.deduplicate_and_merge(existing_rules, new_rules)
        retained, pruned_count = self.prune_stale_rules(merged)

        self.save_reflex_rules(retained)

        duration = (time.time() - start_time) * 1000
        journal = (
            f"Dream Cycle complete: {len(episodes)} episodes examined, "
            f"{len(new_rules)} rules synthesized, {pruned_count} pruned. "
            f"Active reflex rules: {len(retained)}."
        )

        return DreamSummary(
            episodes_harvested=len(episodes),
            rules_synthesized=len(new_rules),
            rules_pruned=pruned_count,
            total_active_rules=len(retained),
            duration_ms=duration,
            journal_entry=journal,
        )

    # --------------------------------------------------------------------------
    # 6. Prompt Injection
    # --------------------------------------------------------------------------

    def inject_reflexes_into_prompt(self, base_prompt: str, max_rules: int = 8) -> str:
        """Inject active cognitive reflexes into the system prompt."""
        rules = self.load_reflex_rules()
        if not rules:
            return base_prompt

        rules.sort(key=lambda r: r.confidence, reverse=True)
        active = rules[:max_rules]

        formatted = ["<cognitive_reflexes>", "Learned behavioral invariants from prior experience:"]
        for r in active:
            formatted.append(f"- [{r.category.upper()}] When: {r.trigger} -> {r.heuristic}")
        formatted.append("</cognitive_reflexes>")

        block = "\n".join(formatted)
        return f"{base_prompt}\n\n{block}"
