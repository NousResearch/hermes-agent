"""panel trigger helpers — three V2 surfaces for hermes-agent.

This module only exposes convenience constructors. Wiring into the actual
agent code paths (skill_manage, plan-mode output, user-correction detection)
is intentionally deferred — see Agent Emitter Architecture doc.

All helpers truncate per the unit-type caps (passage/diff ≤8000, choice
text ≤2000, prompt_context ≤2000) before emitting, and use a sha1-derived
external_ref so re-firing the same trigger is idempotent.
"""

from __future__ import annotations

import hashlib

from .panel_emitter import emit

PASSAGE_CAP = 8000
DIFF_CAP = 8000
CHOICE_CAP = 2000
CONTEXT_CAP = 2000


def _trunc(s: str | None, cap: int) -> str:
    if s is None:
        return ""
    if len(s) <= cap:
        return s
    return s[:cap]


def _ref(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def emit_skill_diff(
    skill_name: str,
    diff: str,
    reason: str,
    profile: str | None = None,
) -> dict:
    """rater judges whether a skill update is an improvement."""
    unit = {
        "type": "skill_diff_review",
        "external_ref": _ref(skill_name, reason),
        "diff": _trunc(diff, DIFF_CAP),
        "prompt_context": _trunc(reason, CONTEXT_CAP),
        "binary": {"yes": "improvement", "no": "regression"},
    }
    return emit([unit], profile=profile)


def emit_process_output(
    passage: str,
    user_goal: str,
    profile: str | None = None,
) -> dict:
    """rater rates the quality of a single agent process output."""
    trimmed_passage = _trunc(passage, PASSAGE_CAP)
    unit = {
        "type": "process_output_rating",
        "external_ref": _ref(trimmed_passage[:500]),
        "passage": trimmed_passage,
        "prompt_context": _trunc(user_goal, CONTEXT_CAP),
        "choices": [
            {"label": "1", "text": _trunc("great", CHOICE_CAP)},
            {"label": "2", "text": _trunc("ok", CHOICE_CAP)},
            {"label": "3", "text": _trunc("meh", CHOICE_CAP)},
            {"label": "4", "text": _trunc("bad", CHOICE_CAP)},
        ],
    }
    return emit([unit], profile=profile)


def emit_prompt_rewrite(
    original: str,
    corrected: str,
    context: str,
    profile: str | None = None,
) -> dict:
    """rater picks the better of two phrasings."""
    a = _trunc(original, CHOICE_CAP)
    b = _trunc(corrected, CHOICE_CAP)
    unit = {
        "type": "prompt_rewrite_pair",
        "external_ref": _ref(original, corrected),
        "choices": [
            {"label": "A", "text": a},
            {"label": "B", "text": b},
        ],
        "prompt_context": _trunc(context, CONTEXT_CAP),
    }
    return emit([unit], profile=profile)
