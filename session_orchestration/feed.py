"""
session_orchestration/feed.py — Discord feed projections (T007/T008).

Responsibilities
----------------
Maintain the channel-level action digest and post optional task-thread
notices.  Attention state must have exactly one ``feed_channel_id`` projection:
the digest managed by ``reconcile_attention_digest``.

Heartbeat edit (T008)
---------------------
``edit_status_message`` PATCHes an existing Discord message in-place.  Used
by the watcher heartbeat hook to update a status message without firing a new
notification.

Debounce
--------
A module-level ``_last_notified`` dict (``task_id → state_value``) prevents
double-posting optional thread notices if somehow the same transition fires
twice.  The primary debounce is already enforced by the watcher call-site
(``_on_turn_change`` is only called when ``new_state != old_state``); this is
belt-and-suspenders.

Transport
---------
Reuses the Discord REST helpers from ``tools/discord_tool.py``
(``_discord_request``, ``_get_bot_token``).  The watcher runs as a
``--no-agent`` cron job so no gateway_runner is available; we post directly
to the Discord REST API using the bot token from the environment.

Called by
---------
``session_orchestration.watcher._on_turn_change`` — digest + thread notice.
``session_orchestration.watcher._on_heartbeat_tick`` — status heartbeat.
``session_orchestration.watcher._on_hang`` — stale/frozen digest + thread notice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_ATTENTION_DIGEST_PROJECTION_NAME = "attention_digest"
_EDIT_OK = "edited"
_EDIT_MISSING = "missing"
_EDIT_FAILED = "failed"

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
        from session_orchestration.config import load_session_orchestration_config

        cfg = load_session_orchestration_config()
        if cfg.feed_channel_id:
            return cfg.feed_channel_id
    except Exception as exc:
        logger.debug("feed: could not read typed config for feed_channel_id: %s", exc)

    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import]

        cfg = load_config()
        so_cfg = cfg_get(cfg, "session_orchestration", default={}) or {}
        if so_cfg.get("feed_channel_id"):
            return so_cfg.get("feed_channel_id")
    except Exception as exc:
        logger.debug("feed: could not read legacy config for feed_channel_id: %s", exc)

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

def _edit_discord_message(
    channel_id: str,
    message_id: str,
    content: str,
    *,
    token: Optional[str] = None,
) -> str:
    """PATCH a Discord message and distinguish missing-message recovery."""
    if not channel_id or not message_id:
        return _EDIT_FAILED

    try:
        from tools.discord_tool import (  # type: ignore[import]
            DiscordAPIError,
            _discord_request,
            _get_bot_token,
        )
    except ImportError as exc:
        logger.warning("feed: cannot import discord helpers: %s", exc)
        return _EDIT_FAILED

    resolved_token = token or _get_bot_token()
    if not resolved_token:
        logger.warning(
            "feed: DISCORD_BOT_TOKEN not set; skipping edit of message=%s channel=%s",
            message_id,
            channel_id,
        )
        return _EDIT_FAILED

    try:
        _discord_request(
            "PATCH",
            f"/channels/{channel_id}/messages/{message_id}",
            resolved_token,
            body={"content": content},
        )
        return _EDIT_OK
    except DiscordAPIError as exc:
        if exc.status == 404:
            logger.info(
                "feed: digest message missing; will recreate message=%s channel=%s",
                message_id,
                channel_id,
            )
            return _EDIT_MISSING
        logger.warning(
            "feed: failed to edit message=%s channel=%s: %s",
            message_id,
            channel_id,
            exc,
        )
        return _EDIT_FAILED
    except Exception as exc:
        logger.warning(
            "feed: failed to edit message=%s channel=%s: %s",
            message_id,
            channel_id,
            exc,
        )
        return _EDIT_FAILED


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            try:
                dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_age(value: Any, now: datetime) -> str:
    dt = _coerce_datetime(value)
    if dt is None:
        return "unknown"
    seconds = max(0, int((now - dt).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def _sort_attention_item(item: Mapping[str, Any]) -> tuple:
    return (
        -int(item.get("priority") or 0),
        str(item.get("opened_at") or ""),
        str(item.get("task_id") or ""),
        str(item.get("reason") or ""),
        int(item.get("id") or 0),
    )


def _session_identity(task_id: str, session: Mapping[str, Any]) -> str:
    tmux_session = session.get("tmux_session")
    if tmux_session:
        return f"tmux `{tmux_session}`"
    hermes_session_key = session.get("hermes_session_key")
    if hermes_session_key:
        return f"session `{hermes_session_key}`"
    run_id = session.get("run_id")
    repo = session.get("repo")
    if run_id and repo:
        return f"run `{run_id}` repo `{repo}`"
    return f"task `{task_id}`"


def _priority_label(item: Mapping[str, Any]) -> str:
    priority = int(item.get("priority") or 0)
    reason = str(item.get("reason") or "").lower()
    detail = str(item.get("detail") or "").lower()
    staleness = ""
    if "frozen" in reason or "frozen" in detail:
        staleness = "frozen/stuck"
    elif (
        "stale" in reason
        or "stuck" in reason
        or "hung" in reason
        or "idle" in reason
        or "stale" in detail
        or "stuck" in detail
        or "hung" in detail
    ):
        staleness = "stale/stuck"
    return f"P{priority}/{staleness}" if staleness else f"P{priority}"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int = 80) -> str:
    """Collapse whitespace and clip *text* to *limit* chars with an ellipsis."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def render_attention_digest(
    attention_items: Sequence[Mapping[str, Any]],
    sessions_by_task_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    now: Any = None,
) -> str:
    """Render unresolved attention items as a concise @-mention router.

    Each row leads with the clickable thread link and an @-mention of the
    owner, then the session identity, reason, optional detail, and a truncated
    question snippet. The full question + numbered options live in the thread
    (see ``push_turn_change``); the board stays a one-line-per-item router.
    """
    render_now = _coerce_datetime(now) or datetime.now(timezone.utc)
    sessions = sessions_by_task_id or {}
    ordered_items = sorted(attention_items, key=_sort_attention_item)

    if not ordered_items:
        return "**Hermes action feed**\nNo unresolved attention items."

    lines = [
        "**Hermes action feed**",
        f"{len(ordered_items)} unresolved attention item(s):",
    ]
    for item in ordered_items:
        task_id = str(item.get("task_id") or "unknown-task")
        session = sessions.get(task_id, {})
        reason = str(item.get("reason") or "attention")
        agent = session.get("agent") or item.get("agent")
        project = (
            session.get("project")
            or session.get("repo")
            or item.get("project")
            or item.get("repo")
        )
        owner_parts: List[str] = []
        if agent:
            owner_parts.append(str(agent))
        if project:
            owner_parts.append(str(project))
        owner = f" · {'/'.join(owner_parts)}" if owner_parts else ""
        thread_id = session.get("discord_thread_id") or item.get("discord_thread_id")
        detail = item.get("detail")
        detail_part = f" · {detail}" if detail else ""
        # @-mention the owner so the feed board routes/pings the right person.
        # (The digest is a single edited message; Discord does not re-ping on an
        # unchanged mention, so this is a router hint, not spam.)
        uid = session.get("discord_user_id") or item.get("discord_user_id")
        mention = f"<@{uid}> " if uid else ""
        # A short question snippet points into the thread where the full
        # question + numbered options live; the board stays concise.
        question = (session.get("last_question") or "").strip()
        question_part = f" · {_truncate(question)}" if question else ""
        if thread_id:
            # Clickable Discord thread link using native channel mention syntax.
            target = f"<#{thread_id}>"
        else:
            target = f"**{task_id}**"
        lines.append(
            "- [ ] "
            f"{target} — {mention}"
            f"({_session_identity(task_id, session)})"
            f"{owner} · reason `{reason}`{detail_part}{question_part}"
        )
    return "\n".join(lines)


def reconcile_attention_digest(
    registry: Any,
    *,
    feed_channel_id: Optional[str] = None,
    token: Optional[str] = None,
    now: Any = None,
) -> Dict[str, Any]:
    """Reconcile the feed-channel digest projection without changing attention truth."""
    channel_id = feed_channel_id or _get_feed_channel_id()
    if not channel_id:
        return {"status": "no_channel", "posted": False, "edited": False}

    attention_items = registry.list_unresolved_attention_items()
    task_ids = sorted(
        {str(item.get("task_id")) for item in attention_items if item.get("task_id")}
    )
    sessions_by_task_id = {
        task_id: (registry.get(task_id) or {})
        for task_id in task_ids
    }
    content = render_attention_digest(attention_items, sessions_by_task_id, now=now)
    digest_hash = _content_hash(content)
    payload = {
        "item_count": len(attention_items),
        "task_ids": task_ids,
    }

    projection = registry.get_projection(channel_id, _ATTENTION_DIGEST_PROJECTION_NAME)
    message_id = (projection or {}).get("message_id")
    if projection and projection.get("content_hash") == digest_hash and message_id:
        return {
            "status": "unchanged",
            "message_id": message_id,
            "content_hash": digest_hash,
            "posted": False,
            "edited": False,
        }

    if message_id:
        edit_status = _edit_discord_message(channel_id, message_id, content, token=token)
        if edit_status == _EDIT_OK:
            registry.upsert_projection(
                channel_id,
                _ATTENTION_DIGEST_PROJECTION_NAME,
                message_id=message_id,
                content_hash=digest_hash,
                payload=payload,
            )
            return {
                "status": "edited",
                "message_id": message_id,
                "content_hash": digest_hash,
                "posted": False,
                "edited": True,
            }
        if edit_status != _EDIT_MISSING:
            return {
                "status": "discord_failed",
                "message_id": message_id,
                "content_hash": digest_hash,
                "posted": False,
                "edited": False,
            }

    new_message_id = _post_discord_message(channel_id, content, token=token)
    if not new_message_id:
        return {
            "status": "discord_failed",
            "message_id": message_id,
            "content_hash": digest_hash,
            "posted": False,
            "edited": False,
        }

    registry.upsert_projection(
        channel_id,
        _ATTENTION_DIGEST_PROJECTION_NAME,
        message_id=new_message_id,
        content_hash=digest_hash,
        payload=payload,
    )
    return {
        "status": "posted" if not message_id else "recreated",
        "message_id": new_message_id,
        "content_hash": digest_hash,
        "posted": True,
        "edited": False,
    }


# ---------------------------------------------------------------------------
# Public API consumed by watcher._on_turn_change
# ---------------------------------------------------------------------------


def _parse_options(row: Dict[str, Any]) -> List[str]:
    """Decode the JSON ``last_options`` column into a list of labels."""
    raw = row.get("last_options")
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _render_needs_input_detail(row: Dict[str, Any]) -> str:
    """Render the extracted question and (for a menu) a numbered option list.

    The full question + options live in the thread; the ``#feed`` digest only
    routes to it. Returns "" when there is nothing extracted to show.
    """
    question = (row.get("last_question") or "").strip()
    options = _parse_options(row)
    lines: List[str] = []
    if question:
        lines.append(question)
    if row.get("last_input_kind") == "menu" and options:
        if lines:
            lines.append("")
        lines.extend(f"{idx}. {label}" for idx, label in enumerate(options, start=1))
        lines.append("")
        lines.append("_Reply with the option number to choose._")
    return "\n".join(lines).strip()


def _notify_key(new_state: str, row: Dict[str, Any]) -> str:
    """Debounce key = state + a digest of the current question.

    Keying on state alone suppresses a *new* question that arrives within the
    same ``WAITING_USER`` state — omp can render menu→menu faster than the
    1-minute watcher tick. Folding the question digest in lets a genuinely new
    prompt re-notify while a re-read of the same prompt stays debounced.
    """
    question = (row.get("last_question") or "").strip()
    if not question:
        return new_state
    digest = hashlib.sha1(question.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{new_state}\x00{digest}"


def push_turn_change(
    task_id: str,
    row: Dict[str, Any],
    new_state: str,
    old_state: str,
    *,
    feed_channel_id: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Post an optional task-thread notice for a user-attention transition.

    Channel-level attention state is projected exclusively by
    ``reconcile_attention_digest``.  This helper intentionally does not post
    to ``feed_channel_id``; that argument is retained only so callers can
    avoid echoing the same one-off notice into the digest channel when a task
    thread id aliases the feed channel.

    Returns True if a task-thread notice was posted, False otherwise.
    """
    # Belt-and-suspenders: skip if we already notified this state+question for
    # this task. Keying on the question digest (not state alone) lets a new
    # question within the same WAITING_USER state re-notify.
    notify_key = _notify_key(new_state, row)
    if _last_notified.get(task_id) == notify_key:
        logger.debug(
            "feed.push_turn_change: debounce suppressed for task_id=%s key=%s",
            task_id,
            notify_key,
        )
        return False

    resolved_feed_channel = feed_channel_id or _get_feed_channel_id()
    thread_id: Optional[str] = row.get("discord_thread_id") or None

    task_label = task_id
    agent = row.get("agent", "unknown")
    project = row.get("project") or row.get("repo") or ""
    project_part = f" | {project}" if project else ""

    if new_state == "WAITING_USER":
        icon = "🔔"
        verb = "needs your input"
    else:
        # PAUSED_HANDOFF
        icon = "⏸️"
        verb = "paused — handoff detected"

    uid = row.get("discord_user_id")

    # THREAD gets the DETAIL (question + numbered options for a menu), with NO
    # @-mention — the actual @-ping lives in #feed now, so we don't double-ping.
    message = (
        f"{icon} **[{agent}] {task_label}** {verb}{project_part}\n"
        f"State: `{old_state}` → `{new_state}`"
    )
    if new_state == "WAITING_USER":
        detail = _render_needs_input_detail(row)
        if detail:
            message = f"{message}\n\n{detail}"

    posted = False
    if thread_id and thread_id != resolved_feed_channel:
        thread_msg_id = _post_discord_message(thread_id, message, token=token)
        if thread_msg_id:
            logger.info(
                "feed.push_turn_change: posted detail to thread=%s msg_id=%s task_id=%s",
                thread_id,
                thread_msg_id,
                task_id,
            )
            posted = True
    elif thread_id:
        logger.debug(
            "feed.push_turn_change: thread_id matches feed_channel_id; digest owns channel projection"
        )
    else:
        logger.debug("feed.push_turn_change: no discord_thread_id; skipping thread post")

    # #feed gets a FRESH @-mention router message (a NEW post, so Discord
    # actually notifies the user — unlike the silently-edited board digest),
    # pointing at the thread where the full context lives. The board digest
    # (reconcile_attention_digest) remains the persistent state view + reap
    # target; this is the live "needs you" ping.
    if resolved_feed_channel and uid:
        thread_link = f"<#{thread_id}>" if thread_id else f"`{task_label}`"
        question = (row.get("last_question") or "").strip()
        snippet = f"\n> {_truncate(question)}" if question else ""
        router_ping = (
            f"<@{uid}> {icon} **[{agent}]** {verb}{project_part} → {thread_link}"
            f"{snippet}"
        )
        feed_msg_id = _post_discord_message(resolved_feed_channel, router_ping, token=token)
        if feed_msg_id:
            logger.info(
                "feed.push_turn_change: posted @-ping router to feed=%s msg_id=%s task_id=%s",
                resolved_feed_channel,
                feed_msg_id,
                task_id,
            )
            posted = True

    # Record the notified state+question (debounce marker)
    _last_notified[task_id] = notify_key

    return posted


def push_hang_notification(
    task_id: str,
    row: Dict[str, Any],
    escalate: bool = False,
    *,
    feed_channel_id: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Post an optional task-thread stale/frozen notice.

    Channel-level stale/frozen attention is projected exclusively by
    ``reconcile_attention_digest``.  This helper intentionally does not post
    to ``feed_channel_id``; that argument is retained only to avoid echoing a
    thread-local notice into the digest channel if ids alias.

    Returns True if a task-thread notice was posted, False otherwise.
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

    message = f"{icon} **[{agent}] {task_id}** {verb}{project_part}"

    if thread_id and thread_id != resolved_feed_channel:
        thread_msg_id = _post_discord_message(thread_id, message, token=token)
        if thread_msg_id:
            logger.info(
                "feed.push_hang_notification: posted to thread=%s msg_id=%s task_id=%s",
                thread_id,
                thread_msg_id,
                task_id,
            )
            return True
    elif thread_id:
        logger.debug(
            "feed.push_hang_notification: thread_id matches feed_channel_id; digest owns channel projection"
        )
    else:
        logger.debug(
            "feed.push_hang_notification: no discord_thread_id; skipping thread post"
        )

    return False


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
    return _edit_discord_message(
        channel_id,
        message_id,
        content,
        token=token,
    ) == _EDIT_OK
