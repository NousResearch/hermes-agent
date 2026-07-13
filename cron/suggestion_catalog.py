"""Curated catalog of starter cron-job suggestions.

These are the built-in automations Hermes can offer a new user out of the box —
the ``catalog`` source of the unified suggestion surface. Each entry is a
ready-to-run ``cron.jobs.create_job`` spec wrapped as a suggestion; the user
accepts via ``/suggestions``. Nothing here auto-schedules.

The important-mail entry deliberately leaves semantic importance decisions to
the scheduled primary AIAgent. The legacy ``classify_items.py`` helper remains
available only for explicit backwards-compatible use; catalog suggestions do
not invoke it.

Adding a catalog entry: append a CatalogEntry. Keep prompts self-contained
(cron jobs run with no chat context) and schedules sensible. The ``job_spec``
is passed verbatim to ``create_job`` on accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "CatalogEntry",
    "CATALOG",
    "IMPORTANT_MAIL_CATALOG_KEY",
    "IMPORTANT_MAIL_CATALOG_REVISION",
    "canonical_important_mail_job_spec",
    "reconcile_pending_important_mail_suggestion",
    "seed_catalog_suggestions",
    "classify_items_script_path",
]


IMPORTANT_MAIL_CATALOG_KEY = "catalog:important-mail-monitor"
IMPORTANT_MAIL_CATALOG_REVISION = 2


def classify_items_script_path() -> str:
    """Return the deprecated compatibility helper path for existing callers."""
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
        key=IMPORTANT_MAIL_CATALOG_KEY,
        title="Important-mail monitor",
        description="Check your inbox periodically and ping you ONLY about mail "
        "that actually needs attention — never the newsletters.",
        job_spec={
            "prompt": (
                "Check the user's inbox for new messages since the last run. "
                "Using your full task context, judge each candidate against this rule: surface "
                "only mail that needs a reply today, is from a manager/family "
                "member, or mentions a deadline. Do not delegate importance or "
                "delivery decisions to an auxiliary classifier. Deliver only "
                "the messages you determine match. If nothing matches, respond "
                "with exactly [SILENT] so the user is not "
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


def _important_mail_entry() -> CatalogEntry:
    return next(entry for entry in CATALOG if entry.key == IMPORTANT_MAIL_CATALOG_KEY)


def canonical_important_mail_job_spec() -> Dict[str, Any]:
    """Return a copy of the current model-authored important-mail job spec."""
    return dict(_important_mail_entry().job_spec)


def reconcile_pending_important_mail_suggestion(*, reconcile_fn=None) -> bool:
    """Refresh only the still-pending built-in important-mail suggestion.

    Accepted/dismissed records and suggestions from non-catalog sources are
    intentionally outside this migration boundary.
    """
    if reconcile_fn is None:
        from cron.suggestions import reconcile_pending_catalog_suggestion

        reconcile_fn = reconcile_pending_catalog_suggestion
    entry = _important_mail_entry()
    return bool(
        reconcile_fn(
            dedup_key=entry.key,
            title=entry.title,
            description=entry.description,
            job_spec=dict(entry.job_spec),
            catalog_revision=IMPORTANT_MAIL_CATALOG_REVISION,
        )
    )


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
    reconcile_fn = None
    if add_fn is None:
        from cron.suggestions import add_suggestion as add_fn  # type: ignore[assignment]
        from cron.suggestions import reconcile_pending_catalog_suggestion

        reconcile_fn = reconcile_pending_catalog_suggestion

    wanted = set(keys) if keys else None
    created: List[Dict[str, Any]] = []
    for entry in CATALOG:
        if wanted is not None and entry.key not in wanted:
            continue
        if entry.key == IMPORTANT_MAIL_CATALOG_KEY and reconcile_fn is not None:
            reconcile_pending_important_mail_suggestion(
                reconcile_fn=reconcile_fn,
            )
        rec = add_fn(
            title=entry.title,
            description=entry.description,
            source="catalog",
            job_spec=dict(entry.job_spec),
            dedup_key=entry.key,
        )
        if rec is not None:
            created.append(rec)
            if entry.key == IMPORTANT_MAIL_CATALOG_KEY and reconcile_fn is not None:
                reconcile_pending_important_mail_suggestion(
                    reconcile_fn=reconcile_fn,
                )
    return created
