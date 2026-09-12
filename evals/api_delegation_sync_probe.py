"""Model-free API delegation delivery probe; run in an isolated HERMES_HOME.

Uses real request-context binding, dispatch policy and SQLite persistence. This
is local mechanism evidence, not provider inference or native Windows evidence.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace


async def probe():
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import clear_session_vars
    from gateway.wake import persist_delegation_delivery
    from hermes_state import SessionDB
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
    db = SessionDB(db_path=Path(os.environ["HERMES_HOME"]) / "state.db")
    db.create_session("api-parent", source="api_server")
    db.create_session("stranger", source="api_server")
    adapter = SimpleNamespace(_ensure_session_db=lambda: db)
    evt = {"type": "async_delegation", "delegation_id": "probe-unit-1"}
    await asyncio.gather(*(persist_delegation_delivery(adapter, text="OWNER_RESULT", session_id="api-parent", evt=evt) for _ in range(2)))
    decisions["replayed_rows"] = len(db.get_messages("api-parent"))
    db.end_session("api-parent", "compression")
    db.create_session("api-child", source="api_server", parent_session_id="api-parent")
    try:
        await persist_delegation_delivery(adapter, text="LATE_RESULT", session_id="api-parent", evt={**evt, "delegation_id": "probe-unit-2"})
        decisions["rotation"] = "persisted"
    except Exception as exc:
        decisions["rotation"] = type(exc).__name__
    decisions["child_rows"] = len(db.get_messages("api-child"))
    db.acquire_session_turn_lease("api-child", "fixture-client-turn", wait_seconds=0)
    busy_evt = {**evt, "delegation_id": "busy-unit"}
    try:
        await persist_delegation_delivery(adapter, text="BUSY_RESULT", session_id="api-child", evt=busy_evt)
        decisions["busy_delivery"] = "inserted"
    except Exception as exc:
        decisions["busy_delivery"] = type(exc).__name__
    finally:
        db.release_session_turn_lease("api-child", "fixture-client-turn")
    await persist_delegation_delivery(adapter, text="BUSY_RESULT", session_id="api-child", evt=busy_evt)
    decisions["after_release_rows"] = len(db.get_messages("api-child"))
    decisions["stranger_rows"] = len(db.get_messages("stranger"))
    db.close()
    return decisions


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), indent=2))
