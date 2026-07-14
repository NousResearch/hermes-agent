"""Curated catalog of starter cron-job suggestions.

These are the built-in automations Hermes can offer a new user out of the box —
the ``catalog`` source of the unified suggestion surface. Each entry is a
ready-to-run ``cron.jobs.create_job`` spec wrapped as a suggestion; the user
accepts via ``/suggestions``. Nothing here auto-schedules.

The "important-mail monitor" entry is where the old proactive-monitor engine
lives now: its ``classify_items.py`` (poll a source -> LLM-score urgency ->
surface only above-threshold) is ONE catalog automation, not a standalone
feature.

Adding a catalog entry: append a CatalogEntry. Keep prompts self-contained
(cron jobs run with no chat context) and schedules sensible. The ``job_spec``
is passed verbatim to ``create_job`` on accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["CatalogEntry", "CATALOG", "seed_catalog_suggestions", "classify_items_script_path"]


def classify_items_script_path() -> str:
    """Absolute path to the urgency classifier script shipped with cron/."""
    return str((Path(__file__).resolve().parent / "scripts" / "classify_items.py"))


@dataclass(frozen=True)
class CatalogEntry:
    """A curated starter automation offered as a suggestion."""

    key: str                 # stable dedup key (never re-offered once dismissed)
    title: str
    description: str
    job_spec: Dict[str, Any]  # kwargs for cron.jobs.create_job


# The curated set. Schedules use the cron/interval syntax create_job accepts.
CATALOG: List[CatalogEntry] = [
    CatalogEntry(
        key="catalog:daily-briefing",
        title="Daily briefing",
        description="Every morning at 8am, a short briefing: today's calendar, "
        "weather, and anything urgent waiting on you.",
        job_spec={
            "prompt": (
                "Produce a concise morning briefing for the user: today's "
                "calendar events, the local weather, and any urgent items "
                "(unread important email, due tasks). Keep it short and "
                "scannable. If you have no connected data sources, give a brief "
                "general good-morning with the date and offer to connect "
                "calendar/email."
            ),
            "schedule": "0 8 * * *",
            "name": "Daily briefing",
            "deliver": "origin",
        },
    ),
    CatalogEntry(
        key="catalog:important-mail-monitor",
        title="Important-mail monitor",
        description="Check your inbox periodically and ping you ONLY about mail "
        "that actually needs attention — never the newsletters.",
        job_spec={
            "prompt": (
                "Check the user's inbox for new messages since the last run. "
                "For each candidate, judge urgency against this rule: surface "
                "only mail that needs a reply today, is from a manager/family "
                "member, or mentions a deadline. Pipe candidates through the "
                "urgency classifier (run `python3 -m cron.scripts.classify_items "
                "--threshold 7 --criteria ...` from the hermes-agent install — "
                "resolve the script path at run time, do not assume a fixed "
                "location) and deliver ONLY what it returns. If nothing "
                "clears the bar, respond with [SILENT] so the user is not "
                "pinged. Requires a connected mail source; if none is "
                "configured, explain how to connect one and then stop."
            ),
            "schedule": "every 30m",
            "name": "Important-mail monitor",
            "deliver": "origin",
        },
    ),
    CatalogEntry(
        key="catalog:weekly-review",
        title="Weekly review",
        description="Every Sunday evening, a recap of the week: what got done, "
        "what's still open, and what's coming up next week.",
        job_spec={
            "prompt": (
                "Produce a weekly review for the user: summarize what was "
                "accomplished this week, list still-open items, and preview "
                "next week's calendar. Pull from whatever sources are connected "
                "(calendar, task tools, recent conversations). Keep it tight."
            ),
            "schedule": "0 18 * * 0",
            "name": "Weekly review",
            "deliver": "origin",
        },
    ),
    CatalogEntry(
        key="catalog:standup-reminder",
        title="Workday start reminder",
        description="A weekday nudge at 9am with your day's agenda and top "
        "priorities, so you start focused.",
        job_spec={
            "prompt": (
                "Give the user a brief weekday start-of-day nudge: their "
                "calendar for today and the 1-3 highest-priority things to "
                "focus on, inferred from recent context and any task tools. "
                "Encouraging, short, one message."
            ),
            "schedule": "0 9 * * 1-5",
            "name": "Workday start reminder",
            "deliver": "origin",
        },
    ),
]


def seed_catalog_suggestions(
    *,
    add_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Register catalog entries as pending suggestions.

    ``add_fn`` defaults to ``cron.suggestions.add_suggestion`` (injectable for
    tests). ``keys`` restricts to specific catalog entries; omit to seed all.
    Entries already dismissed/accepted (by dedup key) or beyond the pending cap
    are skipped by the store, so re-seeding is safe and idempotent. Returns the
    list of suggestion records actually created.
    """
    if add_fn is None:
        from cron.suggestions import add_suggestion as add_fn  # type: ignore[assignment]

    wanted = set(keys) if keys else None
    created: List[Dict[str, Any]] = []
    for entry in CATALOG:
        if wanted is not None and entry.key not in wanted:
            continue
        rec = add_fn(
            title=entry.title,
            description=entry.description,
            source="catalog",
            job_spec=dict(entry.job_spec),
            dedup_key=entry.key,
        )
        if rec is not None:
            created.append(rec)
    return created
