"""Heuristics for deciding which facts should become durable records."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

_STALE_PATTERNS = (
    re.compile(r"\bPR\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\bissue\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\bcommit\s+[0-9a-f]{6,40}\b", re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE),
)


def _is_stale(value: str) -> bool:
    return any(pattern.search(value or "") for pattern in _STALE_PATTERNS)


def _clean_list(values: Optional[Iterable[str]]) -> List[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def detect_durable_candidates(
    *,
    tool_call_count: int = 0,
    fixed_error: bool = False,
    user_corrected_procedure: bool = False,
    discovered_project_fact: bool = False,
    project_fact: str = "",
    stable_user_preference: str = "",
    produced_artifacts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Classify a completed turn's durable-state promotion candidates.

    This is intentionally conservative: it suggests where information belongs
    but does not write skills, memories, or project files by itself.
    """

    reasons: List[str] = []
    do_not_carry: List[str] = []
    artifacts: List[str] = []

    for artifact in _clean_list(produced_artifacts):
        if _is_stale(artifact):
            do_not_carry.append(artifact)
        else:
            artifacts.append(artifact)

    suggest_skill = False
    if tool_call_count >= 5:
        suggest_skill = True
        reasons.append("many-tool-calls")
    if fixed_error:
        suggest_skill = True
        reasons.append("error-recovery")
    if user_corrected_procedure:
        suggest_skill = True
        reasons.append("user-corrected-procedure")

    project_facts: List[str] = []
    if discovered_project_fact and project_fact.strip():
        if _is_stale(project_fact):
            do_not_carry.append(project_fact.strip())
        else:
            project_facts.append(project_fact.strip())
            reasons.append("stable-project-fact")

    user_preferences: List[str] = []
    if stable_user_preference.strip():
        preference = stable_user_preference.strip()
        if _is_stale(preference):
            do_not_carry.append(preference)
        else:
            user_preferences.append(preference)
            reasons.append("stable-user-preference")

    return {
        "suggest_skill": suggest_skill,
        "suggest_project_note": bool(project_facts),
        "suggest_memory": bool(user_preferences),
        "reasons": reasons,
        "project_facts": project_facts,
        "user_preferences": user_preferences,
        "artifacts": artifacts,
        "do_not_carry": do_not_carry,
    }
