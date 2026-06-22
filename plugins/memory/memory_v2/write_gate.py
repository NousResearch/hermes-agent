"""Rule-based write gate classifier for Memory v2 online ingestion.

The v0 write gate is deliberately cheap and deterministic. It decides whether a
turn should create a review candidate and, if so, what kind of memory the
candidate is likely to become. Durable promotion remains a later consolidation
step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class WriteGateOutcome(str, Enum):
    """Possible v0 write-gate outcomes."""

    DISCARD = "discard"
    ARCHIVE_ONLY = "archive_only"
    EPISODIC_ONLY = "episodic_only"
    SEMANTIC_FACT = "semantic_fact"
    PROJECT_UPDATE = "project_update"
    CORE_UPDATE = "core_update"
    SKILL_CANDIDATE = "skill_candidate"
    OPEN_LOOP = "open_loop"
    SUPERSEDE_EXISTING = "supersede_existing"


@dataclass(frozen=True)
class WriteGateDecision:
    """Result of classifying a user turn for Memory v2 writes."""

    outcome: WriteGateOutcome
    claim: str = ""
    memory_type: str = "fact"
    proposed_destination: str = "inbox/candidates.jsonl"
    importance: float = 0.5
    confidence: float = 0.7
    reason: str = ""
    should_create_candidate: bool = False
    requires_review: bool = False


class RuleBasedWriteGate:
    """Low-compute v0 write gate for explicit memory candidates."""

    _REMEMBER_PATTERNS = (
        r"^\s*remember\s+that\s+(.+)$",
        r"^\s*please\s+remember\s+that\s+(.+)$",
        r"^\s*don't\s+forget\s+that\s+(.+)$",
        r"^\s*do\s+not\s+forget\s+that\s+(.+)$",
        r"^\s*remember\s+to\s+(.+)$",
        r"^\s*please\s+remember\s+to\s+(.+)$",
    )
    _EPHEMERAL_TERMS = (
        "today",
        "tomorrow",
        "tonight",
        "this morning",
        "this afternoon",
        "this evening",
        "parked",
        "parking",
        "right now",
        "for now",
        "temporary",
        "temporarily",
    )
    _PREFERENCE_TERMS = ("prefer", "prefers", "like", "likes", "want", "wants")
    _PROJECT_TERMS = ("project", "memory v2", "qwen", "research-project", "current plan", "current state", "next step")
    _ENVIRONMENT_TERMS = ("hermes is running", "host", "wsl", "macos", "linux", "windows", "environment", "installed", "path")
    _PROCEDURE_TERMS = ("when ", "workflow", "procedure", "steps", "how to", "load the", "skill", "runbook", "troubleshoot")
    _OPEN_LOOP_TERMS = ("todo", "to-do", "follow up", "remind me", "open loop", "need to")

    def classify(self, user_text: str) -> WriteGateDecision:
        text = str(user_text or "").strip()
        claim = self.extract_explicit_claim(text)
        if not claim:
            return WriteGateDecision(
                outcome=WriteGateOutcome.ARCHIVE_ONLY,
                reason="No explicit durable memory request; archive raw evidence only.",
            )

        lowered = claim.lower()
        full_lowered = text.lower()
        if self._contains(lowered, self._OPEN_LOOP_TERMS):
            return WriteGateDecision(
                outcome=WriteGateOutcome.OPEN_LOOP,
                claim=claim,
                memory_type="episode",
                proposed_destination="working/open_loops.yaml",
                importance=0.65,
                confidence=0.75,
                reason="Explicit open loop should be tracked as pending task state.",
                should_create_candidate=True,
                requires_review=True,
            )

        if self._contains(lowered, self._EPHEMERAL_TERMS):
            return WriteGateDecision(
                outcome=WriteGateOutcome.ARCHIVE_ONLY,
                claim=claim,
                reason="Claim appears ephemeral; archive only instead of creating durable memory landfill.",
            )

        if self._looks_like_conflict(full_lowered):
            return WriteGateDecision(
                outcome=WriteGateOutcome.SUPERSEDE_EXISTING,
                claim=claim,
                memory_type=self._memory_type_for_claim(lowered),
                proposed_destination="semantic/items",
                importance=0.75,
                confidence=0.65,
                reason="Explicit memory request appears to conflict with or supersede an existing claim; requires review.",
                should_create_candidate=True,
                requires_review=True,
            )

        if self._contains(lowered, self._PROCEDURE_TERMS):
            return WriteGateDecision(
                outcome=WriteGateOutcome.SKILL_CANDIDATE,
                claim=claim,
                memory_type="procedure_ref",
                proposed_destination="skills",
                importance=0.8,
                confidence=0.75,
                reason="Repeatable procedure should become a skill/procedure pointer, not a semantic fact blob.",
                should_create_candidate=True,
                requires_review=True,
            )

        if self._contains(lowered, self._ENVIRONMENT_TERMS):
            return WriteGateDecision(
                outcome=WriteGateOutcome.SEMANTIC_FACT,
                claim=claim,
                memory_type="environment",
                proposed_destination="semantic/items",
                importance=0.75,
                confidence=0.75,
                reason="Explicit environment fact; queue for review before durable promotion.",
                should_create_candidate=True,
                requires_review=True,
            )

        if self._contains(lowered, self._PREFERENCE_TERMS):
            return WriteGateDecision(
                outcome=WriteGateOutcome.CORE_UPDATE,
                claim=claim,
                memory_type="preference",
                proposed_destination="semantic/items",
                importance=0.85,
                confidence=0.85,
                reason="Explicit stable preference; candidate for core/semantic memory after gate review.",
                should_create_candidate=True,
                requires_review=False,
            )

        if self._contains(lowered, self._PROJECT_TERMS):
            project_destination = self._project_destination_for_claim(claim)
            update_kind = self._project_update_kind_for_claim(claim)
            reason = "Explicit project-state update; queue for project memory review."
            if update_kind:
                reason = f"project_update: {update_kind}"
            return WriteGateDecision(
                outcome=WriteGateOutcome.PROJECT_UPDATE,
                claim=claim,
                memory_type="project_state",
                proposed_destination=project_destination,
                importance=0.8,
                confidence=0.8,
                reason=reason,
                should_create_candidate=True,
                requires_review=True,
            )

        return WriteGateDecision(
            outcome=WriteGateOutcome.SEMANTIC_FACT,
            claim=claim,
            memory_type="fact",
            proposed_destination="semantic/items",
            importance=0.6,
            confidence=0.7,
            reason="Explicit memory request; queue as semantic fact candidate for review.",
            should_create_candidate=True,
            requires_review=True,
        )

    @classmethod
    def extract_explicit_claim(cls, user_text: str) -> str:
        for pattern in cls._REMEMBER_PATTERNS:
            match = re.match(pattern, user_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _contains(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _looks_like_conflict(text: str) -> bool:
        return " not " in text or "instead of" in text or "no longer" in text or "supersede" in text

    @staticmethod
    def _project_destination_for_claim(claim: str) -> str:
        field_pattern = r"(?:current\s+state|decision|open\s+question|next\s+action|status|goal|why\s+it\s+matters)"
        patterns = [
            rf"^\s*project\s+(.+?)\s+{field_pattern}\s*:",
            rf"^\s*for\s+project\s+(.+?)\s*,?\s*{field_pattern}\s*:",
            rf"^\s*(memory\s+v2|memory_v2|qwen|research-project)\s+{field_pattern}\s*:",
        ]
        for pattern in patterns:
            match = re.search(pattern, claim, flags=re.IGNORECASE)
            if match:
                slug = re.sub(r"[^a-z0-9]+", "-", match.group(1).lower()).strip("-")
                return f"semantic/projects/{slug or 'project'}.yaml"
        return "semantic/items"

    @staticmethod
    def _project_update_kind_for_claim(claim: str) -> str:
        lowered = claim.lower()
        if re.search(r"\bopen\s+question\s*:", lowered):
            return "open_question"
        if re.search(r"\bnext\s+action\s*:", lowered):
            return "next_action"
        if re.search(r"\bwhy\s+it\s+matters\s*:", lowered):
            return "why_it_matters"
        for kind in ("current_state", "decision", "status", "goal"):
            label = kind.replace("_", r"\s+")
            if re.search(rf"\b{label}\s*:", lowered):
                return kind
        return ""

    def _memory_type_for_claim(self, lowered_claim: str) -> str:
        if self._contains(lowered_claim, self._ENVIRONMENT_TERMS):
            return "environment"
        if self._contains(lowered_claim, self._PREFERENCE_TERMS):
            return "preference"
        if self._contains(lowered_claim, self._PROJECT_TERMS):
            return "project_state"
        return "fact"
