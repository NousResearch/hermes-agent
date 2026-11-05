"""Model-free API delegation delivery probe; run in an isolated HERMES_HOME.

Uses real request-context binding, dispatch policy and SQLite persistence. This
is local mechanism evidence, not provider inference or native Windows evidence.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path


async def probe():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import clear_session_vars
    from gateway.wake import persist_delegation_delivery
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence, profile_scope
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid

    decisions = {}
    for name, capable in (("headerless", ""), ("explicit", "1")):
        kw = dict(chat_id="api-parent", session_id="api-parent")
        if "session_history_delivery" in inspect.signature(APIServerAdapter._bind_api_server_session).parameters:
            kw["session_history_delivery"] = capable
        tokens = APIServerAdapter._bind_api_server_session(**kw)
        try:
            args = ["api-parent"]
            if len(inspect.signature(_resolve_async_wake_sid).parameters) > 1:
                args.append(capable == "1")
            decisions[name] = _resolve_async_wake_sid(*args)
        finally:
            clear_session_vars(tokens)
    store = open_persistence(profile_scope(home=Path(os.environ["HERMES_HOME"])), eager=True)
    store.sessions.create(NewSession(session_id="api-parent", source="api_server"))
    store.sessions.create(NewSession(session_id="stranger", source="api_server"))
    # The adapter opens this home's store itself.
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-api-key"}))
    evt = {"type": "async_delegation", "delegation_id": "probe-unit-1"}
    await asyncio.gather(*(persist_delegation_delivery(adapter, text="OWNER_RESULT", session_id="api-parent", evt=evt) for _ in range(2)))
    decisions["replayed_rows"] = len(store.messages.read_transcript("api-parent"))
    store.sessions.end("api-parent", "compression")
    store.sessions.create(NewSession(session_id="api-child", source="api_server", parent_session_id="api-parent"))
    try:
        await persist_delegation_delivery(adapter, text="LATE_RESULT", session_id="api-parent", evt={**evt, "delegation_id": "probe-unit-2"})
        decisions["rotation"] = "persisted"
    except Exception as exc:
        decisions["rotation"] = type(exc).__name__
    decisions["child_rows"] = len(store.messages.read_transcript("api-child"))
    store.coordination.try_acquire_turn_lease(session_id="api-child", holder="fixture-client-turn")
    busy_evt = {**evt, "delegation_id": "busy-unit"}
    try:
        await persist_delegation_delivery(adapter, text="BUSY_RESULT", session_id="api-child", evt=busy_evt)
        decisions["busy_delivery"] = "inserted"
    except Exception as exc:
        decisions["busy_delivery"] = type(exc).__name__
    finally:
        store.coordination.release_turn_lease(session_id="api-child", holder="fixture-client-turn")
    await persist_delegation_delivery(adapter, text="BUSY_RESULT", session_id="api-child", evt=busy_evt)
    decisions["after_release_rows"] = len(store.messages.read_transcript("api-child"))
    decisions["stranger_rows"] = len(store.messages.read_transcript("stranger"))
    adapter._close_cached_session_dbs()
    store.close()
    return decisions


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), indent=2))
