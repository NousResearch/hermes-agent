"""A real stdio turn must not let delivered progress hide its terminal diagnostic."""

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest

from agent.codex_runtime import make_codex_app_server_event_bridge, run_codex_app_server_turn
from agent.transports.codex_app_server_session import CodexAppServerSession
from gateway.run_turn import GatewayTurnMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_state import SessionDB
from run_agent import AIAgent


_WIRE_SERVER = '''
import json, sys
status = sys.argv[2]
def send(payload):
    print(json.dumps(payload), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    result = {}
    if method == "thread/start":
        result = {"thread": {"id": "thread-test"}}
    elif method == "turn/start":
        result = {"turn": {"id": "turn-test"}}
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
    if method == "turn/start":
        send({"method": "item/completed", "params": {
            "threadId": "thread-test", "turnId": "turn-test",
            "item": {"type": "agentMessage", "id": "message-test",
                     "text": "Checking notifications next.", "phase": "commentary"}}})
        if status != "deadline":
            send({"method": "turn/completed", "params": {
                "threadId": "thread-test", "turn": {"id": "turn-test", "status": status}}})
'''


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["deadline", "completed", "interrupted", "failed"])
async def test_terminal_status_survives_progress_delivery(tmp_path, monkeypatch, status):
    """Real client/session/projector/DB/runtime and gateway suppression, no provider or live chat."""
    executable = tmp_path / "codex-fixture"
    executable.write_text(f"#!{sys.executable}\n" + _WIRE_SERVER)
    executable.chmod(0o700)
    # The real client supplies `app-server` as argv[1]; the fixture receives its outcome as argv[2].
    from agent.transports.codex_app_server import CodexAppServerClient
    session = CodexAppServerSession(
        cwd=str(tmp_path), codex_bin=str(executable),
        client_factory=lambda **kw: CodexAppServerClient(**kw, extra_args=[status]),
    )
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config", lambda: {"agent": {"codex_turn_timeout": 2.0, "codex_idle_timeout": 20}})
    agent = AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                    api_mode="codex_app_server", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._codex_session = session
    agent._codex_session_prompt = None
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="terminal-test", source="feishu", model="stub")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "terminal-test"
    progress = "Checking notifications next."
    adapter = SimpleNamespace(
        MAX_MESSAGE_LENGTH=39000, draft_stream_is_message=False,
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="progress-message")),
    )
    consumer = GatewayStreamConsumer(adapter, "test-chat", StreamConsumerConfig(cursor=""))
    loop = asyncio.get_running_loop()

    def deliver_interim(message):
        future = asyncio.run_coroutine_threadsafe(consumer._send_commentary(message["content"]), loop)
        assert future.result(timeout=5) is True

    monkeypatch.setattr(agent, "_emit_interim_assistant_message", deliver_interim)
    session._on_event = make_codex_app_server_event_bridge(agent)
    try:
        result = await asyncio.to_thread(
            run_codex_app_server_turn, agent, user_message="check", original_user_message="check",
            messages=[{"role": "user", "content": "check"}], effective_task_id="terminal-test",
        )
        assert result["completed"] is (status == "completed")
        assert result["partial"] is (status != "completed")
        assert any(msg.get("content") == progress for msg in result["messages"])
        assert any(msg.get("content") == progress for msg in db.get_messages("terminal-test"))
        if status == "completed":
            assert result["final_response"] == progress
        else:
            assert result["final_response"] != progress
            assert "incomplete" in result["final_response"] or "interrupted" in result["final_response"]
            assert progress in result["final_response"]
            if status == "deadline":
                assert "timed out" in result["error"]
                assert agent._codex_session is None
        ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(platform="feishu"),
                              session_key="terminal-test")
        await GatewayTurnMixin()._run_agent_mark_streamed_delivery(result, ctx)
        assert bool(result.get("already_sent")) is (status == "completed")
        assert adapter.send.await_args_list[0].kwargs["content"] == progress
        if not result.get("already_sent"):
            await adapter.send(chat_id="test-chat", content=result["final_response"])
            assert adapter.send.await_args_list[-1].kwargs["content"] == result["final_response"]
        assert adapter.send.await_count == (1 if status == "completed" else 2)
    finally:
        session.close()
        db.close()
