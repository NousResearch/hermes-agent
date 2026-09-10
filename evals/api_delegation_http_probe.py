"""Loopback HTTP + real API executor/session bindings, with inference replaced.

No paid model calls. Captures what the runtime hands the model, not provider behavior.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def probe():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import get_session_env
    from gateway.wake import persist_delegation_delivery
    from hermes_state import SessionDB
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid
    import gateway.session_context as sc

    db = SessionDB(db_path=Path(os.environ["HERMES_HOME"]) / "state.db")
    db.create_session("parent", source="api_server")
    db.append_message("parent", "user", "request")
    db.append_message("parent", "assistant", "acknowledged")
    db.end_session("parent", "compression")
    db.create_session("child", source="api_server", parent_session_id="parent")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-api-key"}))
    adapter._session_db = db
    captured = []

    def create_agent(**kwargs):
        agent = MagicMock()
        agent.session_id = kwargs.get("session_id")
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        def run(**turn):
            sid = get_session_env("HERMES_SESSION_CHAT_ID", "")
            args = [sid]
            if len(inspect.signature(_resolve_async_wake_sid).parameters) > 1:
                args.append(sc.session_history_delivery_supported())
            captured.append({"session_id": sid, "target": _resolve_async_wake_sid(*args), "history": turn.get("conversation_history")})
            callback = kwargs.get("stream_delta_callback")
            if callback:
                callback("fixture reply")
            return {"final_response": "fixture reply", "session_id": sid, "messages": [], "api_calls": 1}
        agent.run_conversation.side_effect = run
        return agent
    adapter._create_agent = create_agent
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    records = []
    runs = []
    async with TestClient(TestServer(app)) as client:
        for stream in (False, True):
            for explicit in (False, True):
                headers = {"Authorization": "Bearer fixture-api-key"}
                if explicit:
                    headers["X-Hermes-Session-Id"] = "parent"
                response = await client.post("/v1/chat/completions", headers=headers, json={"messages": [{"role": "user", "content": "continue"}], "stream": stream})
                body = await response.text()
                records.append({"stream": stream, "explicit": explicit, "status": response.status, "header": response.headers.get("X-Hermes-Session-Id"), "runtime": captured[-1] if captured else None, "body": body[:120]})
        calls_before = len(captured)
        evt = {"type": "async_delegation", "delegation_id": "unit-http"}
        delivery_error = None
        try:
            await asyncio.gather(*(persist_delegation_delivery(adapter, text="DELIVERY_RESULT", session_id="parent", evt=evt) for _ in range(2)))
        except Exception as exc:
            delivery_error = type(exc).__name__
        calls_after = len(captured)
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer fixture-api-key", "X-Hermes-Session-Id": "parent"}, json={"messages": [{"role": "user", "content": "read result"}]})
        await response.read()
        resumed_history = captured[-1]["history"]
        snapshot = [{"role": "user", "content": "caller snapshot"}]
        adapter._response_store.put("resp-seed", {"conversation_history": snapshot, "session_id": "parent"})
        db.create_session("declared", source="api_server", session_key="fixture-key")
        db.append_message("declared", "assistant", "DECLARED_HISTORY")
        for name, body, extra_headers in (
            ("session", {"session_id": "parent"}, {}),
            ("caller_history", {"session_id": "parent", "conversation_history": snapshot}, {}),
            ("response_chain", {"previous_response_id": "resp-seed"}, {}),
            ("declared_key", {}, {"X-Hermes-Session-Key": "fixture-key"}),
        ):
            count = len(captured)
            response = await client.post("/v1/runs", headers={"Authorization": "Bearer fixture-api-key", **extra_headers}, json={"input": "continue", **body})
            await response.read()
            async def wait_for_run():
                while len(captured) == count:
                    await asyncio.sleep(0.01)
            if response.status == 202:
                await asyncio.wait_for(wait_for_run(), 10)
            runs.append({"name": name, "status": response.status, "runtime": captured[-1] if len(captured) > count else None})
    result = {"requests": records, "runs": runs, "delivery_error": delivery_error, "unsolicited_calls": calls_after-calls_before, "resumed_history": resumed_history, "durable_child_rows": len(db.get_messages("child"))}
    db.close()
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), indent=2, default=str))
