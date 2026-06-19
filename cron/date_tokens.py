"""Deterministic date-token expansion for cron job prompts.

Cron prompts historically embedded literal date placeholders — ``$(date
+%Y-%m-%d)``, ``YYYY-MM-DD``, ``<TODAY>``, ``{TODAY}`` — and relied on the
(often weak claude-3-haiku) agent to substitute the real date at run time.
Agents frequently failed, writing the *literal placeholder* as a filename
(e.g. ``context/theo/trader-standups/$(date +%Y-%m-%d)-open.md``). This module
replaces those placeholders with the actual Pacific-Time date BEFORE the prompt
reaches the agent, so no agent substitution is required.

Applied at the single prompt-build chokepoint in ``cron/scheduler.py``
(``_build_job_prompt``). Kept as a standalone module with a fail-safe import so
it can be re-applied by the self-heal cron (``apply_cron_datefix.sh``) after a
``hermes update`` stashes local source edits.

Pacific Time is used because the BFO crons schedule and name files in PT.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo

    _PT = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - zoneinfo always present on py3.9+
    _PT = None


def _now_pt() -> datetime:
    return datetime.now(_PT) if _PT is not None else datetime.now()


def expand_cron_date_tokens(text: str, now: datetime | None = None) -> str:
    """Replace literal date placeholders in ``text`` with the resolved PT date.

    Deterministic and fail-safe: returns ``text`` unchanged if it contains no
    recognizable placeholder. Ordering is longest/most-specific first so that
    e.g. ``YYYY-MM-DD`` is consumed before ``YYYY-MM``.
    """
    if not text:
        return text
    # Cheap guard: skip the regex work when no placeholder marker is present.
    if "$(date" not in text and "YYYY" not in text and "TODAY" not in text:
        return text

    n = now or _now_pt()
    today = n.strftime("%Y-%m-%d")
    ymd_compact = n.strftime("%Y%m%d")
    month = n.strftime("%Y-%m")
    year = n.strftime("%Y")
    # Week-ending Sunday: weekday() is Mon=0..Sun=6; days until the coming Sunday.
    week_ending_sunday = (n + timedelta(days=(6 - n.weekday()) % 7)).strftime("%Y-%m-%d")

    replacements = (
        # shell command-substitution forms (most specific first)
        (r"\$\(date \+%Y-%m-%d\)", today),
        (r"\$\(date \+%F\)", today),
        (r"\$\(date \+%Y%m%d\)", ymd_compact),
        (r"\$\(date \+%Y-%m\)", month),
        (r"\$\(date \+%Y\)", year),
        # doubly-escaped variant seen in junk: $(date +\%Y-\%m-\%d)
        (r"\$\(date \+\\%Y-\\%m-\\%d\)", today),
        # angle / brace placeholders
        (r"<WEEK_ENDING_SUNDAY>", week_ending_sunday),
        (r"\{\{?\s*WEEK_ENDING_SUNDAY\s*\}?\}", week_ending_sunday),
        (r"<TODAY>", today),
        (r"\{\{?\s*TODAY\s*\}?\}", today),
        (r"<YYYY-MM-DD>", today),
        (r"\{YYYY-MM-DD\}", today),
        # bare placeholders (longest first)
        (r"\bYYYY-MM-DD\b", today),
        (r"\bYYYYMMDD\b", ymd_compact),
        (r"\bYYYY-MM\b", month),
    )
    for pattern, value in replacements:
        text = re.sub(pattern, value, text)
    return text
