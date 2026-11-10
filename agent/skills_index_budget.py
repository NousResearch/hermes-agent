"""Budgeted skills index: when the ``<available_skills>`` block outgrows its share of the model's
context window, the least-used skills keep their name and lose their description.

Opt-in via ``skills.index_budget_percent`` (0 = off, the legacy full index). The rule:

- Nothing changes while the full index fits the budget (percent of the context window).
- A skill is only eligible to lose its description when it was neither used nor created/modified
  in the last ``RECENT_DAYS`` days (``.usage.json``: last_used_at/last_viewed_at, created_at,
  last_patched_at). A newly installed skill has no usage record yet, so it is judged by the age of
  its SKILL.md instead and keeps its description for its first ``RECENT_DAYS`` days.
- Eligible skills are ordered by ``scrub_order`` (fewest uses first) and scrubbed from the front
  until the index fits, never past ``MAX_SCRUBBED_FRACTION`` of the listed skills.
- A scrubbed skill stays listed by name and loads with ``skill_view`` exactly as before.

``scrub_order`` is a total order (the name breaks every tie), so the scrubbed set is always a prefix
of it and can be split into further tiers later. Every date is compared at whole-UTC-day resolution:
the same library renders the same bytes all day unless a skill crosses the cut, which keeps the
index reusable across sessions. A conversation never re-tiers itself: its system prompt is built
once and stored, and the per-turn skills delta compares names only.
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Mapping

RECENT_DAYS = 14
MAX_SCRUBBED_FRACTION = 0.5
_DAY = 86400


def _epoch_day(stamp: Any) -> int:
    """Whole UTC days since the epoch for an ISO timestamp; 0 when absent or unreadable."""
    if not stamp:
        return 0
    try:
        parsed = datetime.fromisoformat(str(stamp))
    except ValueError:
        return 0
    return int((parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp() // _DAY)


def _mtime_day(path: str | None) -> int:
    try:
        return int(os.stat(path).st_mtime // _DAY) if path else 0
    except OSError:
        return 0


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def scrub_order(names: Iterable[str], usage: Mapping[str, Mapping[str, Any]], skill_files: Mapping[str, str],
                today: int) -> list[str]:
    """Eligible skills, first = first to lose its description.

    Order: fewest uses, then longest since last use, then oldest, then name. A skill with no usage
    record and no file on disk (plugin-registered) has nothing to rank on and is never eligible.
    """
    cutoff = today - RECENT_DAYS
    ranked = []
    for name in names:
        record = usage.get(name) or usage.get(name.rsplit("/", 1)[-1]) or {}
        path = skill_files.get(name)
        if not record and not path:
            continue
        used = max(_epoch_day(record.get("last_used_at")), _epoch_day(record.get("last_viewed_at")))
        # The usage record is the authority once it exists: a bulk re-sync (update, git pull) rewrites
        # many SKILL.md files at once and must not shield them all. No record yet = judge by file age.
        touched = (max(_epoch_day(record.get("created_at")), _epoch_day(record.get("last_patched_at")))
                   if record else _mtime_day(path))
        if used > cutoff or touched > cutoff:
            continue
        ranked.append((_count(record.get("use_count")), used, touched, name))
    return [row[-1] for row in sorted(ranked)]


def names_to_scrub(rows: Mapping[str, str], order: list[str], excess_tokens: int,
                   estimate: Callable[[str], int]) -> frozenset[str]:
    """The prefix of *order* whose descriptions cover *excess_tokens*, capped at half of *rows*."""
    scrubbed: list[str] = []
    cap = int(len(rows) * MAX_SCRUBBED_FRACTION)
    for name in order:
        if excess_tokens <= 0 or len(scrubbed) >= cap:
            break
        if rows.get(name):
            scrubbed.append(name)
            excess_tokens -= estimate(f": {rows[name]}")
    return frozenset(scrubbed)


def select_scrubbed(full_index: str, rows: Mapping[str, str], skill_files: Mapping[str, str],
                    budget_tokens: int, usage: Mapping[str, Mapping[str, Any]], now: float | None = None) -> frozenset[str]:
    """Names that lose their description so *full_index* fits *budget_tokens*; empty when it already fits."""
    from agent.model_metadata import estimate_tokens_rough

    excess = estimate_tokens_rough(full_index) - budget_tokens
    if excess <= 0:
        return frozenset()
    today = int((time.time() if now is None else now) // _DAY)
    return names_to_scrub(rows, scrub_order(rows, usage, skill_files, today), excess, estimate_tokens_rough)
