"""
session_orchestration/feed.py — unified feed channel pusher (T007/T008).

Responsibilities
----------------
Push ONE Discord message per state transition (turn-change) to:
  1. The unified feed channel (``feed_channel_id`` from config or passed in).
  2. The task's Discord thread (``row["discord_thread_id"]``).

Heartbeat edit (T008)
---------------------
``edit_status_message`` PATCHes an existing Discord message in-place.  Used
by the watcher heartbeat hook to update a status message without firing a new
notification.

Debounce
--------
A module-level ``_last_notified`` dict (``task_id → state_value``) prevents
double-posting if somehow the same transition fires twice.  The primary
debounce is already enforced by the watcher call-site (``_on_turn_change``
is only called when ``new_state != old_state``); this is belt-and-suspenders.

Transport
---------
Reuses the Discord REST helpers from ``tools/discord_tool.py``
(``_discord_request``, ``_get_bot_token``).  The watcher runs as a
``--no-agent`` cron job so no gateway_runner is available; we post directly
to the Discord REST API using the bot token from the environment.

Called by
---------
``session_orchestration.watcher._on_turn_change`` — T007 only.
``session_orchestration.watcher._on_heartbeat_tick`` — T008.
T009 (hang notify) will add its own feed primitive here; leave the public API
open for extension.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level debounce state
# ---------------------------------------------------------------------------

# task_id → last state value we successfully posted a message for.
# Reset when the state moves back to a non-attention state (so a later
# transition to WAITING_USER is treated as a new event).
_last_notified: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _get_feed_channel_id() -> Optional[str]:
    """Return the configured feed channel id, or None if absent."""
    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import]

        cfg = load_config()
        so_cfg = cfg_get(cfg, "session_orchestration", default={}) or {}
        return so_cfg.get("feed_channel_id") or None
    except Exception as exc:
        logger.debug("feed: could not read config for feed_channel_id: %s", exc)
    # Fall back to environment (useful in tests and manual runs)
    return os.getenv("HERMES_FEED_CHANNEL_ID") or None


# ---------------------------------------------------------------------------
# Low-level Discord REST helper (reuses discord_tool pattern)
# ---------------------------------------------------------------------------


def _post_discord_message(
    channel_id: str,
    content: str,
    *,
    token: Optional[str] = None,
) -> Optional[str]:
    """POST a message to a Discord channel/thread.  Returns the message id or None.

    Uses the same REST helpers as ``tools/discord_tool``.  Best-effort:
    logs on failure and returns None rather than raising.
    """
    if not channel_id:
        return None

    try:
        from tools.discord_tool import _discord_request, _get_bot_token, DiscordAPIError  # type: ignore[import]
    except ImportError as exc:
        logger.warning("feed: cannot import discord helpers: %s", exc)
        return None

    resolved_token = token or _get_bot_token()
    if not resolved_token:
        logger.warning("feed: DISCORD_BOT_TOKEN not set; skipping post to channel=%s", channel_id)
        return None

    try:
        result = _discord_request(
            "POST",
            f"/channels/{channel_id}/messages",
            resolved_token,
            body={"content": content},
        )
        msg_id: Optional[str] = (result or {}).get("id")
        return msg_id
    except Exception as exc:
        logger.warning("feed: failed to post to channel=%s: %s", channel_id, exc)
        return None


# ---------------------------------------------------------------------------
# Public API consumed by watcher._on_turn_change
# ---------------------------------------------------------------------------


def push_turn_change(
    task_id: str,
    row: Dict[str, Any],
    new_state: str,
    old_state: str,
    *,
    feed_channel_id: Optional[str] = None,
    token: Optional[str] = None,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> bool:
    """Push a turn-change notification to the feed channel + task thread.

    Called exactly once per state transition (the watcher call-site already
    gates on ``new_state != old_state``).  Belt-and-suspenders debounce via
    ``_last_notified`` ensures idempotence inside a single cron run.

    Parameters
    ----------
    task_id:
        Registry key for the session.
    row:
        Full registry row at the time of the transition.
    new_state:
        The new ``SessionLifecycle`` value (string, e.g. ``"WAITING_USER"``).
    old_state:
        The previous state value.
    feed_channel_id:
        Override the configured feed channel id (used in tests).
    token:
        Override the Discord bot token (used in tests).
    summarize_fn:
        Optional callable ``(question: str) -> str``.  When ``new_state``
        is ``"WAITING_USER"`` and ``row["last_question"]`` is non-empty,
        this function is called with the raw question string and its return
        value is embedded in the message.  When ``None`` (the default), the
        raw question is formatted as a readable blockquote inline note.
        Never called for non-WAITING_USER transitions or when last_question
        is absent/empty.

    Returns True if at least one message was posted, False otherwise.
    """
    # Belt-and-suspenders: skip if we already notified this state for this task.
    if _last_notified.get(task_id) == new_state:
        logger.debug(
            "feed.push_turn_change: debounce suppressed for task_id=%s state=%s",
            task_id,
            new_state,
        )
        return False

    # Resolve feed channel
    resolved_feed_channel = feed_channel_id or _get_feed_channel_id()
    thread_id: Optional[str] = row.get("discord_thread_id") or None

    # Build the message text
    task_label = task_id
    agent = row.get("agent", "unknown")
    project = row.get("project") or row.get("repo") or ""
    project_part = f" | {project}" if project else ""

    if new_state == "WAITING_USER":
        icon = "🔔"
        verb = "needs your input"
    elif new_state == "PAUSED_HANDOFF":
        icon = "⏸️"
        verb = "paused — handoff detected"
    elif new_state == "DONE":
        icon = "✅"
        verb = "completed"
    elif new_state == "RUNNING":
        icon = "▶"
        verb = "running"
    else:
        icon = "ℹ️"
        verb = f"state: {new_state}"

    message = (
        f"{icon} **[{agent}] {task_label}** {verb}{project_part}\n"
        f"State: `{old_state}` → `{new_state}`"
    )
    if thread_id:
        message += f" | <#{thread_id}>"

    # Append the needs_input question when transitioning to WAITING_USER
    if new_state == "WAITING_USER":
        question = row.get("last_question") or ""
        if question:
            question_text = (
                summarize_fn(question) if summarize_fn is not None else f"Question: {question}"
            )
            message += f"\n> {question_text}"

    posted = False

    # 1. Push to the unified feed channel
    if resolved_feed_channel:
        msg_id = _post_discord_message(resolved_feed_channel, message, token=token)
        if msg_id:
            logger.info(
                "feed.push_turn_change: posted to feed channel=%s msg_id=%s task_id=%s state=%s",
                resolved_feed_channel,
                msg_id,
                task_id,
                new_state,
            )
            posted = True
    else:
        logger.debug("feed.push_turn_change: no feed_channel_id; skipping feed post")

    # 2. Push to the task's own Discord thread (if different from feed channel)
    if thread_id and thread_id != resolved_feed_channel:
        thread_msg_id = _post_discord_message(thread_id, message, token=token)
        if thread_msg_id:
            logger.info(
                "feed.push_turn_change: posted to thread=%s msg_id=%s task_id=%s",
                thread_id,
                thread_msg_id,
                task_id,
            )
            posted = True
    elif not thread_id:
        logger.debug("feed.push_turn_change: no discord_thread_id; skipping thread post")

    # 3. DM the user when transitioning to WAITING_USER (T013)
    if new_state == "WAITING_USER":
        user_id = row.get("discord_user_id")
        if user_id:
            try:
                from tools.discord_tool import _get_bot_token  # type: ignore[import]
                from session_orchestration.dm_transport import send_dm

                tok = token or _get_bot_token()
                if tok:
                    send_dm(user_id, message, tok)
            except Exception as exc:
                logger.error(
                    "feed.push_turn_change: DM failed task_id=%s: %s", task_id, exc
                )

    # Record the notified state (debounce marker)
    _last_notified[task_id] = new_state

    return posted


def push_hang_notification(
    task_id: str,
    row: Dict[str, Any],
    escalate: bool = False,
    *,
    feed_channel_id: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Push a hang (or hang-escalation) notification to the feed channel.

    Called by ``watcher._on_hang`` when a session is confirmed hung.
    This is a SEPARATE function from ``push_turn_change`` — it never
    modifies ``_last_notified`` and does not touch the turn-change debounce.

    Parameters
    ----------
    task_id:
        Registry key for the session.
    row:
        Full registry row at the time of the hang detection.
    escalate:
        If True, this is an escalation (nudge already sent, session still
        hung); message content reflects the escalation state.
    feed_channel_id:
        Override the configured feed channel id (used in tests).
    token:
        Override the Discord bot token (used in tests).

    Returns True if at least one message was posted, False otherwise.
    """
    resolved_feed_channel = feed_channel_id or _get_feed_channel_id()
    thread_id: Optional[str] = row.get("discord_thread_id") or None

    agent = row.get("agent", "unknown")
    project = row.get("project") or row.get("repo") or ""
    project_part = f" | {project}" if project else ""
    idle_ticks = row.get("idle_ticks") or 0

    if escalate:
        icon = "🚨"
        verb = "still hung after nudge — escalating to user"
    else:
        icon = "⚠️"
        verb = f"appears hung ({idle_ticks} idle ticks) — sending auto-nudge"

    message = (
        f"{icon} **[{agent}] {task_id}** {verb}{project_part}"
    )
    if thread_id:
        message += f" | <#{thread_id}>"

    posted = False

    if resolved_feed_channel:
        msg_id = _post_discord_message(resolved_feed_channel, message, token=token)
        if msg_id:
            logger.info(
                "feed.push_hang_notification: posted to feed channel=%s msg_id=%s "
                "task_id=%s escalate=%s",
                resolved_feed_channel,
                msg_id,
                task_id,
                escalate,
            )
            posted = True
    else:
        logger.debug(
            "feed.push_hang_notification: no feed_channel_id; skipping feed post"
        )

    if thread_id and thread_id != resolved_feed_channel:
        thread_msg_id = _post_discord_message(thread_id, message, token=token)
        if thread_msg_id:
            logger.info(
                "feed.push_hang_notification: posted to thread=%s msg_id=%s task_id=%s",
                thread_id,
                thread_msg_id,
                task_id,
            )
            posted = True
    elif not thread_id:
        logger.debug(
            "feed.push_hang_notification: no discord_thread_id; skipping thread post"
        )

    # DM the user on the first stale rung (escalate=False) only.
    # When escalate=True, the escalation DM is sent by watcher._on_hang (T010) — no double-DM.
    if not escalate:
        user_id = row.get("discord_user_id")
        if user_id:
            try:
                from tools.discord_tool import _get_bot_token  # type: ignore[import]
                from session_orchestration.dm_transport import send_dm

                tok = token or _get_bot_token()
                if tok:
                    send_dm(user_id, message, tok)
            except Exception as exc:
                logger.error(
                    "feed.push_hang_notification: stale DM failed: %s", exc
                )

    return posted


def clear_last_notified(task_id: str) -> None:
    """Reset the debounce marker for *task_id*.

    Called when a session transitions OUT of an attention state back to
    RUNNING — so the NEXT transition to WAITING_USER is treated as new.
    T008/T009 may also call this to re-arm after a hang resolution.

    Also useful in tests to reset state between runs.
    """
    _last_notified.pop(task_id, None)


# ---------------------------------------------------------------------------
# Heartbeat edit helper (T008) — edits a message in-place, no notification
# ---------------------------------------------------------------------------


def edit_status_message(
    channel_id: str,
    message_id: str,
    content: str,
    *,
    token: Optional[str] = None,
) -> bool:
    """PATCH an existing Discord message in-place.  Returns True on success.

    Used by the heartbeat hook to update a status message without sending a
    new notification.  Best-effort: logs on failure and returns False rather
    than raising.

    Parameters
    ----------
    channel_id:
        Discord channel (or thread) id that contains the message.
    message_id:
        The id of the message to edit.
    content:
        The new message content string.
    token:
        Override the Discord bot token (used in tests).
    """
    if not channel_id or not message_id:
        return False

    try:
        from tools.discord_tool import _discord_request, _get_bot_token, DiscordAPIError  # type: ignore[import]
    except ImportError as exc:
        logger.warning("feed: cannot import discord helpers: %s", exc)
        return False

    resolved_token = token or _get_bot_token()
    if not resolved_token:
        logger.warning(
            "feed: DISCORD_BOT_TOKEN not set; skipping edit of message=%s channel=%s",
            message_id,
            channel_id,
        )
        return False

    try:
        _discord_request(
            "PATCH",
            f"/channels/{channel_id}/messages/{message_id}",
            resolved_token,
            body={"content": content},
        )
        return True
    except Exception as exc:
        logger.warning(
            "feed: failed to edit message=%s channel=%s: %s",
            message_id,
            channel_id,
            exc,
        )
        return False
