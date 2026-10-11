"""Telegram presence for bounded work after its foreground turn returns.

Reuse gateway supervision, registry liveness and canonical completion routes.
Daemons without completion notification are deliberately excluded.
"""
import asyncio
import logging

from gateway.config import Platform
from gateway.platforms.typing_scope import typing_scope

logger = logging.getLogger(__name__)


def _live_work_routes():
    from tools.process_registry import process_registry
    from tools.async_delegation import list_async_delegations

    routes = []
    for item in process_registry.list_sessions():
        if item["status"] != "running" or not item.get("notify_on_complete"):
            continue
        session = process_registry.get(item["session_id"])
        if session is None or session.exited:
            continue
        routes.append({
            "session_key": session.session_key,
            "parent_session_id": session.parent_session_id,
            **{key: getattr(session, "watcher_" + key) for key in
               ("platform", "chat_id", "user_id", "user_name", "thread_id", "message_id")},
        })
    routes.extend(r for r in list_async_delegations() if r.get("status") in {"running", "stalling"})
    # Work without a known owning conversation cannot safely light up a topic.
    return {(r["session_key"], r["parent_session_id"]): r for r in routes
            if r.get("session_key") and r.get("parent_session_id")}


def _running(runner):
    return runner._running and not getattr(runner, "_draining", False) and not getattr(runner, "_restart_requested", False)


async def background_typing_loop(runner, interval=2.0):
    while runner._running:
        try:
            await refresh_background_typing(runner)
        except Exception:
            logger.debug("Background typing snapshot failed", exc_info=True)
        await asyncio.sleep(interval)


async def refresh_background_typing(runner):
    if not _running(runner):
        return
    routes = await asyncio.to_thread(_live_work_routes)
    seen = set()
    # One slow Telegram request must not starve every other topic's five-second TTL.
    await asyncio.gather(*(_refresh_route(runner, key, event, seen) for key, event in routes.items()))


async def _refresh_route(runner, key, event, seen):
    session_key, parent_id = key
    try:
        async with runner._completion_event_scope(event):
            # /new and closed sessions must not borrow an old task's activity.
            if await runner._classify_completion_target(parent_id) != "deliver":
                return
            source = await asyncio.to_thread(runner._build_process_event_source, event)
            if source is None or source.platform != Platform.TELEGRAM:
                return
            adapter = runner._delivery_adapter_for(source)
            if adapter is None or not adapter.config.typing_indicator:
                return
            # The foreground already owns a refresh loop (including prompt pauses).
            if session_key in adapter._active_sessions:
                return
            metadata = runner._thread_metadata_for_source(source)
            scope = typing_scope(source.platform, source.chat_id, metadata)
            target = (id(adapter), scope)
            if target in seen or scope in adapter._typing_paused or source.chat_id in adapter._typing_paused:
                return
            if not _running(runner):
                return
            seen.add(target)
            await asyncio.wait_for(adapter.send_typing(source.chat_id, metadata=metadata), 1.5)
    except Exception:
        # Presence must never prevent a completion from being delivered.
        logger.debug("Background typing refresh failed", exc_info=True)
