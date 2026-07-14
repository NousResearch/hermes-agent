"""Shared ``/suggestions`` command logic for CLI and gateway.

Both surfaces call ``handle_suggestions_command(args, origin=...)`` and present
the returned text however they present command output. Keeping the logic here
(not in cli.py / gateway/run.py) means the two surfaces can never drift.

Subcommands:
  /suggestions                 list pending suggestions (numbered)
  /suggestions accept <N|id>   create the cron job for that suggestion
  /suggestions dismiss <N|id>  dismiss it (latched, never re-offered)
  /suggestions catalog         seed the curated starter automations as pending
  /suggestions clear           drop accepted records (housekeeping)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _fmt_pending(pending: list) -> str:
    if not pending:
        return (
            "No suggested automations right now.\n"
            "Try `/suggestions catalog` to see the curated starter set, or "
            "install a blueprint skill to get one."
        )
    lines = ["Suggested automations — `/suggestions accept N` or `dismiss N`:\n"]
    for i, s in enumerate(pending, 1):
        spec = s.get("job_spec", {}) or {}
        sched = spec.get("schedule", "?")
        src = s.get("source", "?")
        lines.append(f"  {i}. {s.get('title', '(untitled)')}  [{sched}]  ({src})")
        desc = s.get("description", "").strip()
        if desc:
            lines.append(f"     {desc}")
    return "\n".join(lines)


def _resolve_origin() -> Optional[Dict[str, Any]]:
    """Best-effort current-chat origin from session env (CLI and gateway both set it).

    Mirrors cron's ``_origin_from_env`` so an accepted suggestion's job delivers
    back to the chat where it was accepted. Returns None if unavailable, in
    which case create_job falls back to a configured home channel.
    """
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": chat_id,
                "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
                "thread_id": get_session_env("HERMES_SESSION_THREAD_ID") or None,
            }
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)
    return None


def handle_suggestions_command(
    args: str,
    *,
    origin: Optional[Dict[str, Any]] = None,
    surface: str = "cli",
) -> str:
    """Dispatch a ``/suggestions`` invocation. Returns text to show the user.

    ``args`` is everything after ``/suggestions`` (already stripped of the
    command word). ``origin`` is the platform/chat dict so an accepted job's
    "origin" delivery routes back to where the user accepted; when omitted it
    is resolved from the session environment. ``surface`` (``"cli"`` |
    ``"gateway"``) picks the wording for follow-up hints — ``/cron`` only
    exists on the CLI.
    """
    if origin is None:
        origin = _resolve_origin()
    try:
        from cron import suggestions as store
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("suggestions store import failed: %s", e)
        return "Suggestions are unavailable in this build."

    parts = (args or "").strip().split()
    sub = parts[0].lower() if parts else ""
    rest = " ".join(parts[1:]).strip()

    # Bare /suggestions -> list pending.
    if not sub:
        return _fmt_pending(store.list_pending())

    if sub in ("accept", "add", "schedule"):
        if not rest:
            return "Usage: /suggestions accept <number|id>"
        job = store.accept_suggestion(rest, origin=origin)
        if job is None:
            return f"No pending suggestion matches '{rest}'. Run /suggestions to list them."
        sched = job.get("schedule_display") or (job.get("job_spec", {}) or {}).get("schedule", "")
        name = job.get("name", "automation")
        manage = (
            "Manage it with /cron."
            if surface == "cli"
            else "Ask me to list, pause, or remove it any time."
        )
        return (
            f"Scheduled '{name}'"
            + (f" ({sched})" if sched else "")
            + f". {manage}"
        )

    if sub in ("dismiss", "no", "reject"):
        if not rest:
            return "Usage: /suggestions dismiss <number|id>"
        ok = store.dismiss_suggestion(rest)
        return (
            "Dismissed. Won't suggest that again."
            if ok
            else f"No pending suggestion matches '{rest}'."
        )

    if sub == "catalog":
        try:
            from cron.suggestion_catalog import seed_catalog_suggestions

            created = seed_catalog_suggestions()
        except Exception as e:
            logger.debug("catalog seed failed: %s", e)
            return "Couldn't load the catalog."
        if not created:
            return (
                "No new catalog automations to add (already offered, dismissed, "
                "or your suggestion list is full). Run /suggestions to see pending."
            )
        added = ", ".join(c.get("title", "?") for c in created)
        return f"Added {len(created)} suggestion(s): {added}.\nRun /suggestions to review."

    if sub == "clear":
        removed = store.clear_resolved()
        return f"Cleared {removed} resolved suggestion record(s)."

    return (
        "Usage:\n"
        "  /suggestions              list pending\n"
        "  /suggestions accept N     schedule suggestion N\n"
        "  /suggestions dismiss N    dismiss suggestion N\n"
        "  /suggestions catalog      add curated starter automations\n"
        "  /suggestions clear        housekeeping"
    )
