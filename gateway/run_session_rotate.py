"""Control-socket verb ``rotate-session``: rotate one chat/thread's live session (#125590).

The session rotation lives in the RUNNING gateway's in-memory store (``SessionStore.reset_session``
swaps the routing entry under ``_lock`` and only then ends the old ``state.db`` row), so an external
process that ends the row directly rotates nothing: the gateway never re-reads routing from the DB
per message. Event-driven consumers (an orchestrator that opens work for a channel and wants that
conversation to start clean) had exactly one supported entry point: a human typing ``/new``.

This verb reuses the same rotation funnel ``/new`` runs — generation bump, running-agent slot
release, agent teardown + cache eviction, conversation-scope clear, in-flight delegation interrupt,
``SessionStore.reset_session``, the ``on_session_reset`` plugin hook, Telegram topic-lane rebind —
so an external rotation cannot leave a stale cached agent or a zombie topic binding pointing at the
ended session.
One deliberate divergence, not an omission: ``/new`` falls back to
``get_or_create_session(force_new=True)`` when nothing was routed, while this verb answers
``rotated: false`` — a caller polling before the channel's first message must not special-case a
miss, and must not fabricate a session either.
``/new`` additionally calls ``_reset_process_scoped_tool_state()`` (env-passthrough allowlist +
credential-file registry). Both stores are ContextVar-backed; a control-socket verb executes in the
socket thread's fresh context (``run_coroutine_threadsafe`` does not carry the loop lineage's
context), where the clear would auto-vivify a fresh empty store — a no-op cosplaying as parity.
Until the clear has a context-correct mechanism (or the stores are shown to hold nothing across
turns), the verb deliberately does NOT mirror that call; see the PR discussion on #125605.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.run")


def rotate_session_verb(runner):
    """Build the ``rotate-session`` handler; socket handlers run on executor threads."""
    loop = asyncio.get_running_loop()

    async def apply(params: dict[str, Any]) -> dict[str, Any]:
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"error": "session store is not available"}

        source = _source_from_params(params)
        if isinstance(source, str):
            return {"error": source}
        session_key = runner._session_key_for_source(source)

        old_sid: Optional[str] = None
        with store._lock:
            entry = store._entry_locked(session_key)
            old_sid = entry.session_id if entry is not None else None
        if old_sid is None:
            # Present-but-unknown key is a miss, not an error: a caller polling before the
            # channel's first message (nothing routed yet) gets a truthful no-op.
            return {"rotated": False, "session_key": session_key}

        # Same generation/slot/cache/scope funnel as /new, so an external rotation cannot
        # strand an in-flight run's zombie agent slot or conversation-scoped state.
        runner._invalidate_session_run_generation(session_key, reason="session_reset")
        runner._release_running_agent_state(session_key)
        await runner._cleanup_old_agent_for_reset(session_key)
        runner._evict_cached_agent(session_key)
        runner._clear_conversation_scope(session_key, reason="session_reset")
        # In-flight async delegations end WITH the conversation: once the id rotates their
        # completions have no live owner (mirrors the /new path).
        with contextlib.suppress(Exception):
            from tools.async_delegation import interrupt_for_session

            interrupt_for_session(session_key=session_key, reason="session_reset",
                                  parent_session_id=str(old_sid))

        new_entry = await runner.async_session_store.reset_session(session_key)
        await runner._fire_session_reset_hooks(
            source, session_key, old_sid, new_entry.session_id if new_entry is not None else None)
        # Plugin on_session_reset hook after the new session exists (mirrors /new); best-effort.
        if new_entry is not None:
            with contextlib.suppress(Exception):
                from hermes_cli.lifecycle import invoke_hook
                invoke_hook("on_session_reset", session_id=new_entry.session_id,
                            reason="session_rotate",
                            platform=source.platform.value if source.platform else "",
                            old_session_id=old_sid, new_session_id=new_entry.session_id)
        # Telegram private-chat topic lanes bind (chat_id, thread_id) -> session_id durably;
        # without the rebind the binding-heal walk switches the next message back onto the
        # ended session (same invariant the /new path enforces after compression resets).
        if new_entry is not None and runner._is_telegram_topic_lane(source):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(runner._record_telegram_topic_binding, source, new_entry)

        return {
            "rotated": new_entry is not None,
            "session_key": session_key,
            "old_session_id": old_sid,
            "new_session_id": new_entry.session_id if new_entry is not None else None,
            "end_reason": "session_reset",
        }

    def handler(params):
        future = asyncio.run_coroutine_threadsafe(apply(params if isinstance(params, dict) else {}), loop)
        try:
            return future.result(timeout=5.0)
        except TimeoutError:
            return {"pending": True}
        except Exception as exc:
            logger.warning("rotate-session request failed", exc_info=True)
            return {"error": f"{type(exc).__name__}: {exc}"}

    return handler


def _source_from_params(params: dict[str, Any]):
    """Build the canonical :class:`SessionSource` for a request, or an error string.

    ``session_key`` may be supplied verbatim (power callers that already know the canonical
    identity); otherwise ``platform`` + ``chat_id`` (+ ``thread_id``) are rebuilt through the
    same ``SessionSource`` every ingress path uses, so the verb and message routing can never
    disagree about which conversation a request targets.
    """
    from gateway.config import Platform
    from gateway.session import SessionSource

    platform_value = str(params.get("platform") or "").strip().lower()
    if not platform_value:
        return "platform is required"
    try:
        platform = Platform(platform_value)
    except ValueError:
        return f"unknown platform: {platform_value!r}"
    chat_id = params.get("chat_id")
    if chat_id is None or not str(chat_id).strip():
        return "chat_id is required"
    thread_id = params.get("thread_id")
    return SessionSource(
        platform=platform,
        chat_id=str(chat_id).strip(),
        chat_type=str(params.get("chat_type") or "group").strip() or "group",
        thread_id=str(thread_id).strip() if thread_id is not None and str(thread_id).strip() else None,
    )
