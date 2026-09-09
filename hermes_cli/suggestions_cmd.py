"""Shared ``/suggestions`` command logic for CLI and gateway.

Both surfaces call ``handle_suggestions_command(args, origin=...)`` and present the returned text,
so the two surfaces can never drift. Subcommands are listed in ``_USAGE``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _fmt_pending(pending: list) -> str:
    if not pending:
        return ("No suggested automations right now.\n"
                "Try `/suggestions catalog` to see the curated starter set, or "
                "install a blueprint skill to get one.")
    lines = ["Suggested automations — `/suggestions accept N` or `dismiss N`:\n"]
    for i, s in enumerate(pending, 1):
        sched = (s.get("job_spec", {}) or {}).get("schedule", "?")
        lines.append(f"  {i}. {s.get('title', '(untitled)')}  [{sched}]  ({s.get('source', '?')})")
        desc = s.get("description", "").strip()
        if desc:
            lines.append(f"     {desc}")
    return "\n".join(lines)


def _resolve_origin() -> Optional[Dict[str, Any]]:
    """Best-effort current-chat origin from session env (mirrors cron's ``_origin_from_env``) so an
    accepted job delivers back to the accepting chat; None lets create_job use the home channel."""
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": chat_id,
                "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
                "thread_id": get_session_env("HERMES_SESSION_THREAD_ID") or None}
    except Exception:
        pass
    return None


def _accept(store, rest: str, origin, surface: str) -> str:
    if not rest:
        return "Usage: /suggestions accept <number|id>"
    from cron.scheduler import CronSchedulerRegistrationError
    try:
        job = store.accept_suggestion(rest, origin=origin)
    except CronSchedulerRegistrationError as e:
        return e.user_message()
    if job is None:
        return f"No pending suggestion matches '{rest}'. Run /suggestions to list them."
    sched = job.get("schedule_display") or (job.get("job_spec", {}) or {}).get("schedule", "")
    name = job.get("name", "automation")
    manage = ("Manage it with /cron." if surface == "cli"
              else "Ask me to list, pause, or remove it any time.")
    return f"Scheduled '{name}'" + (f" ({sched})" if sched else "") + f". {manage}"


def _dismiss(store, rest: str, origin, surface: str) -> str:
    if not rest:
        return "Usage: /suggestions dismiss <number|id>"
    if store.dismiss_suggestion(rest):
        return "Dismissed. Won't suggest that again."
    return f"No pending suggestion matches '{rest}'."


def _catalog(store, rest: str, origin, surface: str) -> str:
    try:
        from cron.suggestion_catalog import seed_catalog_suggestions
        created = seed_catalog_suggestions()
    except Exception as e:
        logger.debug("catalog seed failed: %s", e)
        return "Couldn't load the catalog."
    if not created:
        return ("No new catalog automations to add (already offered, dismissed, "
                "or your suggestion list is full). Run /suggestions to see pending.")
    added = ", ".join(c.get("title", "?") for c in created)
    return f"Added {len(created)} suggestion(s): {added}.\nRun /suggestions to review."


def _clear(store, rest: str, origin, surface: str) -> str:
    return f"Cleared {store.clear_resolved()} resolved suggestion record(s)."


_SUBCOMMANDS = {
    "": lambda store, rest, origin, surface: _fmt_pending(store.list_pending()),
    "accept": _accept, "add": _accept, "schedule": _accept,
    "dismiss": _dismiss, "no": _dismiss, "reject": _dismiss,
    "catalog": _catalog, "clear": _clear}

_USAGE = (
    "Usage:\n"
    "  /suggestions              list pending\n"
    "  /suggestions accept N     schedule suggestion N\n"
    "  /suggestions dismiss N    dismiss suggestion N\n"
    "  /suggestions catalog      add curated starter automations\n"
    "  /suggestions clear        housekeeping")


def handle_suggestions_command(
    args: str, *, origin: Optional[Dict[str, Any]] = None, surface: str = "cli") -> str:
    """Dispatch a ``/suggestions`` invocation (``args`` = text after the command word); returns
    text to show the user. ``origin`` defaults to the session environment's chat."""
    if origin is None:
        origin = _resolve_origin()
    try:
        from cron import suggestions as store
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("suggestions store import failed: %s", e)
        return "Suggestions are unavailable in this build."
    parts = (args or "").strip().split()
    handler = _SUBCOMMANDS.get(parts[0].lower() if parts else "")
    if handler is None:
        return _USAGE
    return handler(store, " ".join(parts[1:]).strip(), origin, surface)
