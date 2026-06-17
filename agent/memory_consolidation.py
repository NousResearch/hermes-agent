"""Conservative governance layer for durable memory writes.

This module is deliberately deterministic and side-effect free. It does not
save, delete, or rewrite memory; it classifies candidate entries so callers can
attach typed metadata and warnings while preserving the existing memory UX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal, Tuple

ConsolidationAction = Literal[
    "semantic_add",
    "semantic_replace",
    "procedural_skill_candidate",
    "episodic_only",
    "working_memory_only",
    "discard",
]
ConsolidationTarget = Literal["memory", "user", "skill", "none"]


@dataclass(frozen=True)
class ConsolidationDecision:
    action: ConsolidationAction
    target: ConsolidationTarget
    text: str
    reason: str
    confidence: float
    salience: float
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


_PROCEDURAL_RE = re.compile(
    r"\b(when doing|steps?:|first[, ]|then[, ]|finally[, ]|"
    r"verify by|install by|patch by|deploy by|configure by|troubleshoot by|"
    r"workflow|procedure)\b",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(
    r"(`[^`]+`|\b(?:run|execute)\s+(?:python|pytest|uv|git|npm|pnpm|curl|hermes)\b)",
    re.IGNORECASE,
)
_EPISODIC_RE = re.compile(
    r"\b(done|completed|fixed|submitted|opened pr|merged|branch|commit|sha|"
    r"phase \d+|today|this session|just now|file count|exit code|passed in \d+)\b|"
    r"\b(?:pr|issue)\s*#?\d+\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(r"\b(prefers?|likes?|wants?|does not want|hates?|values?)\b", re.IGNORECASE)
_PROJECT_FACT_RE = re.compile(r"\b(uses?|runs?|is|are|has|requires?|configured|located at)\b", re.IGNORECASE)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def classify_memory_candidate(
    target: str,
    text: str,
    *,
    explicit: bool = False,
    replace: bool = False,
) -> ConsolidationDecision:
    """Classify a candidate durable-memory entry.

    The classifier is intentionally conservative: PR3 annotates and warns, but
    does not hard-block content. Existing security scanning remains the only
    hard gate for writes.
    """
    clean = " ".join(str(text or "").strip().split())
    normalized_target: ConsolidationTarget = "user" if target == "user" else "memory"
    action: ConsolidationAction = "semantic_replace" if replace else "semantic_add"
    reason = "compact durable declarative fact"
    confidence = 0.7
    salience = 0.55
    warnings: list[str] = []

    if not clean:
        return ConsolidationDecision(
            action="discard",
            target="none",
            text="",
            reason="empty memory candidate",
            confidence=1.0,
            salience=0.0,
            warnings=("Empty memory candidates should not be saved.",),
        )

    if normalized_target == "user" and _PREFERENCE_RE.search(clean):
        reason = "durable user preference/profile fact"
        confidence = 0.86
        salience = 0.82
    elif _PROCEDURAL_RE.search(clean) or _COMMAND_RE.search(clean):
        action = "procedural_skill_candidate"
        normalized_target = "skill"
        reason = "looks like reusable procedural knowledge better suited to a skill"
        confidence = 0.78
        salience = 0.68
        warnings.append(
            "This looks procedural; keep semantic memory compact and consider promoting the workflow to a skill."
        )
    elif _EPISODIC_RE.search(clean):
        action = "episodic_only"
        normalized_target = "none"
        reason = "looks like task progress or session-local history"
        confidence = 0.74
        salience = 0.22
        warnings.append(
            "This looks episodic or task-local; session_search is usually better than durable memory."
        )
    elif _PROJECT_FACT_RE.search(clean):
        reason = "durable project/environment fact"
        confidence = 0.74
        salience = 0.62

    if explicit and warnings:
        warnings.append(
            "Explicit memory request preserved; consolidation warning is advisory only."
        )

    return ConsolidationDecision(
        action=action,
        target=normalized_target,
        text=clean,
        reason=reason,
        confidence=_clamp(confidence),
        salience=_clamp(salience),
        warnings=tuple(warnings),
    )


def semantic_record_fields_from_decision(
    decision: ConsolidationDecision,
    original_target: str,
) -> dict:
    """Map a consolidation decision onto semantic-record sidecar fields."""
    if decision.action == "procedural_skill_candidate":
        kind = "procedural_candidate"
    elif decision.action in {"episodic_only", "working_memory_only"}:
        kind = "episodic_note"
    elif original_target == "user":
        kind = "user_profile_fact"
    else:
        kind = "semantic_fact"

    fields = {
        "kind": kind,
        "salience": decision.salience,
        "confidence": decision.confidence,
        "consolidation_action": decision.action,
        "consolidation_target": decision.target,
        "consolidation_reason": decision.reason,
    }
    if decision.warnings:
        fields["consolidation_warnings"] = list(decision.warnings)
    return fields


__all__ = [
    "ConsolidationDecision",
    "classify_memory_candidate",
    "semantic_record_fields_from_decision",
]
