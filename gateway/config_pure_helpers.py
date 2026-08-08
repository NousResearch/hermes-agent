"""Config/channel-override helpers extracted from gateway/run.py (#54962, slice 18).

Pure helpers for channel-override lookup, Hermes binary resolution, session-key
parsing, process-notification formatting, and gateway watch-event draining.
"""

from __future__ import annotations

import sys
from typing import Optional

from gateway.config import ChannelOverride, GatewayConfig, Platform


def _channel_override_lookup_keys(
    chat_id: str,
    *,
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> list[str]:
    """Ordered, de-duplicated keys for ``channel_overrides`` lookup.

    Matches ``resolve_channel_prompt`` semantics: exact thread/channel id first,
    then parent channel/forum id (Discord threads inherit parent overrides).
    """
    keys: list[str] = []
    seen: set[str] = set()
    for key in (chat_id, thread_id, parent_id):
        if not key:
            continue
        sk = str(key)
        if sk in seen:
            continue
        seen.add(sk)
        keys.append(sk)
    return keys


def _get_channel_override(
    config: GatewayConfig,
    platform: Platform,
    chat_id: str,
    *,
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Optional[ChannelOverride]:
    """Return per-channel override for this platform/chat_id, or None.

    Looks up ``channel_overrides`` by ``chat_id``, then ``thread_id``, then
    ``parent_id`` (forum threads / child channels inherit the parent entry).
    """
    platforms = getattr(config, "platforms", None)
    if not platforms:
        return None
    platform_config = platforms.get(platform)
    if not platform_config or not platform_config.channel_overrides:
        return None
    overrides = platform_config.channel_overrides
    for key in _channel_override_lookup_keys(
        chat_id, thread_id=thread_id, parent_id=parent_id
    ):
        ov = overrides.get(key)
        if ov is not None:
            return ov
    return None


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key into its component parts.

    Session keys follow the format
    ``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``.
    Returns a dict with ``platform``, ``chat_type``, ``chat_id``, and
    optionally ``thread_id`` keys, or None if the key doesn't match.

    The 6th element is only returned as ``thread_id`` for chat types where
    it is unambiguous (``dm`` and ``thread``).  For group/channel sessions
    the suffix may be a user_id (per-user isolation) rather than a
    thread_id, so we leave ``thread_id`` out to avoid mis-routing.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {
            "platform": parts[2],
            "chat_type": parts[3],
            "chat_id": parts[4],
        }
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None


def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        # Reuse the shared rich formatter (self-contained task-source block).
        from tools.process_registry import format_process_notification
        return format_process_notification(evt)

    return None


def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.

    Watch events are handled by the post-turn gateway drain. Process
    completions are owned by their per-process watcher task, and async
    delegation completions are owned by ``_async_delegation_watcher``.
    Requeueing async events inside ``while not queue.empty()`` would make the
    loop non-terminating, so detach the current batch first, then requeue any
    events this drain does not own after the queue is empty.
    """
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {"watch_match", "watch_disabled"}:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events
