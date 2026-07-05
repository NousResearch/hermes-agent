"""Lightweight SkillOpt-style text update primitives for Hermes skills.

This module intentionally implements only the safe, deterministic first slice of
SkillOpt: bounded atomic edits on a single skill document plus a pure validation
gate. It does not call models, read/write skill files, or decide what evidence
to collect. Higher-level `/learn`, curator, cron, or future optimizers can use
these primitives behind their own verification loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

EditOp = Literal["append", "insert_after", "replace", "delete"]
GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass(frozen=True)
class AtomicEdit:
    """A bounded edit proposed for one skill document.

    `append` ignores `target`. `insert_after`, `replace`, and `delete` require a
    unique target string. This deliberately avoids regexes and shell-like patch
    languages so model-proposed edits are easy to review and test.
    """

    op: EditOp
    content: str = ""
    target: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class GateDecision:
    """Pure validation-gate outcome for a candidate skill."""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part])


def _has_unterminated_slow_update(text: str) -> bool:
    start = text.find(SLOW_UPDATE_START)
    return start >= 0 and text.find(SLOW_UPDATE_END, start + len(SLOW_UPDATE_START)) < 0


def _protected_ranges(text: str) -> list[range]:
    """Return protected slow-update character ranges.

    Missing end marker makes the rest of the document protected. This is
    fail-closed: malformed skills should not receive step-level edits inside a
    suspected slow-update block.
    """

    ranges: list[range] = []
    start = 0
    while True:
        i = text.find(SLOW_UPDATE_START, start)
        if i < 0:
            return ranges
        j = text.find(SLOW_UPDATE_END, i + len(SLOW_UPDATE_START))
        if j < 0:
            ranges.append(range(i, len(text)))
            return ranges
        j += len(SLOW_UPDATE_END)
        ranges.append(range(i, j))
        start = j


def _overlaps_protected(text: str, start: int, end: int) -> bool:
    if start < 0 or end < start:
        return False
    for r in _protected_ranges(text):
        if start < r.stop and end > r.start:
            return True
    return False


def _find_unique(text: str, target: str) -> tuple[int, int]:
    if not target:
        raise ValueError("target is required for this edit operation")
    start = text.find(target)
    if start < 0:
        raise ValueError(f"target not found: {target!r}")
    if text.find(target, start + len(target)) >= 0:
        raise ValueError(f"target is not unique: {target!r}")
    end = start + len(target)
    if _overlaps_protected(text, start, end):
        raise ValueError("edit would modify protected slow-update block")
    return start, end


def _apply_one(skill: str, edit: AtomicEdit) -> str:
    content = edit.content.rstrip()
    if edit.op == "append":
        if _has_unterminated_slow_update(skill):
            raise ValueError("unterminated slow-update block; refusing append")
        sep = "" if skill.endswith("\n") else "\n"
        return f"{skill}{sep}{content}\n"

    start, end = _find_unique(skill, edit.target)
    if edit.op == "delete":
        return skill[:start] + skill[end:]
    if edit.op == "replace":
        return skill[:start] + content + skill[end:]
    if edit.op == "insert_after":
        sep = "" if skill[end:end + 1] == "\n" else "\n"
        return skill[:end] + sep + content + "\n" + skill[end:]
    raise ValueError(f"unknown edit op: {edit.op!r}")


def apply_bounded_edits(
    skill: str,
    edits: list[AtomicEdit],
    *,
    edit_budget: int,
    max_words: int = 2000,
) -> tuple[str, int]:
    """Apply up to `edit_budget` atomic edits and return candidate text.

    The edit budget is the SkillOpt textual learning-rate analogue. All applied
    edits are deterministic and bounded; invalid edits fail before returning a
    partially accepted result to the caller.
    """

    if edit_budget < 0:
        raise ValueError("edit_budget must be non-negative")
    if max_words <= 0:
        raise ValueError("max_words must be positive")

    candidate = skill
    applied = 0
    for edit in edits[:edit_budget]:
        candidate = _apply_one(candidate, edit)
        applied += 1

    words = _word_count(candidate)
    if words > max_words:
        raise ValueError(f"candidate skill exceeds max_words ({words} > {max_words})")
    return candidate, applied


def evaluate_skill_gate(
    *,
    candidate_skill: str,
    candidate_score: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    step: int,
    best_step: int = 0,
    epsilon: float = 0.0,
) -> GateDecision:
    """Accept a candidate only when it strictly improves validation score."""

    cand = float(candidate_score)
    cur = float(current_score)
    best = float(best_score)
    raw_epsilon = float(epsilon)
    if not all(isfinite(value) for value in (cand, cur, best, raw_epsilon)):
        raise ValueError("gate scores and epsilon must be finite numbers")
    threshold = max(0.0, raw_epsilon)

    if cand > cur + threshold:
        if cand > best + threshold:
            return GateDecision("accept_new_best", candidate_skill, cand, candidate_skill, cand, step)
        return GateDecision("accept", candidate_skill, cand, best_skill, best, best_step)
    return GateDecision("reject", current_skill, cur, best_skill, best, best_step)
