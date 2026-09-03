"""Evidence and confidence thresholds that gate automatic application.

Global automation is enabled from this pipeline's first production run —
there is no queue a human must clear by hand. What keeps that safe is
conservative evidence requirements, checked here, plus the separate,
unconditional conflict gate in ``pipeline.py`` (a candidate that conflicts
with existing content is never auto-applied, regardless of how much
evidence it has accumulated).

Two independent bars, both must clear:

1. **Confidence** — the classifier's own judgment that this is a real,
   well-formed lesson. Below :data:`MIN_CONFIDENCE`, a candidate never
   qualifies no matter how many times it recurs.
2. **Evidence** — how many times, and how independently, the lesson showed
   up. The rule depends on *why* the candidate exists:

   * ``explicit_instruction`` — George said it directly and plainly. One
     confirmed occurrence is enough; this is a direct statement of intent,
     not a pattern that needs corroborating.
   * everything targeting a **skill** — needs at least three independent
     confirmed uses. A skill is a procedure Claude Code will actively run
     again, so it earns a higher bar than a passive rule.
   * everything targeting a **rule** or the **CLAUDE.md block** (i.e. every
     other case) — needs at least three independent sessions AND at least
     two distinct task occurrences. "Sessions" and "tasks" are tracked
     separately (see ``store.py``): a task is one matched occurrence, a
     session is the transcript session it came from. Requiring both stops
     one long session that repeats a phrase five times from looking like
     five independent confirmations.
"""

from __future__ import annotations

MIN_CONFIDENCE = 0.6

_MIN_SESSIONS_FOR_RULE = 3
_MIN_TASKS_FOR_RULE = 2
_MIN_USES_FOR_SKILL = 3


def is_eligible(candidate: dict) -> tuple[bool, str]:
    """Whether *candidate* (a ``store.CandidateStore`` row dict) has enough
    evidence and confidence to auto-apply. Returns ``(eligible, reason)`` —
    ``reason`` is a short, human-readable explanation, populated whether or
    not the candidate is eligible (useful for status/dry-run output)."""
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence < MIN_CONFIDENCE:
        return False, f"confidence {confidence:.2f} is below the floor of {MIN_CONFIDENCE}"

    category = candidate.get("category")
    session_count = len(candidate.get("session_ids") or [])
    task_count = int(candidate.get("occurrence_count") or 0)

    if category == "explicit_instruction":
        if task_count >= 1:
            return True, "explicit instruction, confirmed once"
        return False, "explicit instruction with no confirmed occurrence"

    if candidate.get("target_kind") == "skill":
        if task_count >= _MIN_USES_FOR_SKILL:
            return True, f"{task_count} confirmed uses (needs {_MIN_USES_FOR_SKILL})"
        return False, f"only {task_count} confirmed uses (needs {_MIN_USES_FOR_SKILL})"

    # Everything else targets the CLAUDE.md block or a rule file.
    if session_count >= _MIN_SESSIONS_FOR_RULE and task_count >= _MIN_TASKS_FOR_RULE:
        return True, (
            f"{session_count} sessions and {task_count} tasks "
            f"(needs {_MIN_SESSIONS_FOR_RULE} sessions, {_MIN_TASKS_FOR_RULE} tasks)"
        )
    return False, (
        f"only {session_count} sessions and {task_count} tasks "
        f"(needs {_MIN_SESSIONS_FOR_RULE} sessions, {_MIN_TASKS_FOR_RULE} tasks)"
    )
