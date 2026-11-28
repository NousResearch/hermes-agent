"""The ``inbox`` delivery target: a finished cron run's own session filed into its profile's Inbox.

The run session the scheduler already created for this fire (``gateway/session_cron.py``) IS the inbox
item: nothing is copied. Filing stamps one marker on that row's ``model_config`` (``cron_inbox``: job id,
job name, delivery time and a preview of the final response) and marks the row unread through the same
``last_read_at`` watermark every session uses, so read state is backend-side and shared by every client
(Desktop, the Web app, a future mobile app). ``gateway/session_inbox.py`` serves the list and mark-read
RPCs and the one-time continuation of a finished item as an interactive chat.

Default: a job created from a Desktop/Web chat that is not the canonical Bot Chat, with ``deliver``
omitted or ``origin``, delivers to ``inbox`` (``default_deliver``; the cronjob tool skips it for a
script-only job, which writes no run session). Bot Chat origins keep bot-chat delivery; CLI/TUI
origins keep the local-chat note.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("cron.scheduler")

INBOX_PLATFORM = "inbox"
# ``model_config`` key of a filed run session; its presence is what makes the row an inbox item.
INBOX_MARKER = "cron_inbox"
PREVIEW_CHARS = 280
# Creation sources whose non-Bot-Chat jobs default to the inbox (``gateway.session_policy.SURFACES``
# labels). Data, not code: a client with its own inbox surface adds its label here.
INBOX_DEFAULT_SOURCES = frozenset({"gui"})


def is_inbox_token(part: str) -> bool:
    return (part or "").strip().lower() == INBOX_PLATFORM


def inbox_target() -> dict:
    return {"platform": INBOX_PLATFORM, "chat_id": "", "thread_id": None}


def default_deliver(deliver: Optional[str]) -> Optional[str]:
    """``inbox`` for an omitted / ``origin`` deliver created from a Desktop/Web chat that is not the
    canonical Bot Chat (the creating turn's frozen policy says which surface it is); else *deliver*."""
    if deliver not in (None, "", "origin"):
        return deliver
    from gateway.session_policy import active_policy
    policy = active_policy()
    if policy is None or policy.source not in INBOX_DEFAULT_SOURCES or _in_bot_chat():
        return deliver
    return INBOX_PLATFORM


def _in_bot_chat() -> bool:
    """Whether the creating session is the profile's canonical Bot Chat (titled exactly ``Bot Chat``)."""
    from gateway.session_context import get_session_env
    from tools.bot_mode_probe import BOT_CHAT_TITLE
    session_id = get_session_env("HERMES_SESSION_ID")
    if not session_id:
        return False
    try:
        from persistence_assembly import open_persistence, profile_scope
        store = open_persistence(profile_scope())
        lineage_root = store.lineage.lineage(session_id)
        ids = {session_id, *(lineage_root or ())}
        return any(store.sessions.get_title(sid) == BOT_CHAT_TITLE for sid in ids)
    except Exception:  # health: allow BLE001 -- an unreadable title keeps the default delivery
        logger.debug("inbox default: Bot Chat check failed for %s", session_id, exc_info=True)
        return False


def file_run(job: dict, content: str) -> Optional[str]:
    """File this fire's run session into the inbox. None once filed, else the error."""
    from persistence_assembly import open_persistence, profile_scope
    store = open_persistence(profile_scope())
    session_id = _run_session(store, job)
    if not session_id:
        return "inbox: this run has no session to file"
    preview = " ".join(str(content or "").split())[:PREVIEW_CHARS]
    store.sessions.patch_model_config(session_id, {INBOX_MARKER: {
        "job_id": job.get("id"), "job_name": job.get("name") or job.get("id"),
        "delivered_at": time.time(), "preview": preview}})
    store.sessions.set_read(session_id, False)
    return None


def _run_session(store, job: dict) -> Optional[str]:
    """This fire's run session: the job's run whose execution is this fire's (delivery runs in the
    firing scheduler's thread, outside the owner's execution context)."""
    from persistence.conversation.sessions import SessionWindow
    execution_id = job.get("execution_id")
    if not execution_id:
        return None
    runs = store.sessions.list_cron_job_runs(str(job["id"]), SessionWindow(limit=20))
    return next((str(run.entry.session.session_id) for run in runs if run.cron_execution_id == execution_id), None)
