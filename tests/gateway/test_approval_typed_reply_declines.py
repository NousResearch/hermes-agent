"""``approvals.typed_reply_declines``: a message sent while an approval is pending declines it and
reaches the agent as the user's next instruction (chat busy path and ``POST /v1/runs/{id}/steer``)."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import tools.approval as approval_module
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.waiting import wait_until

REPLY = "no, run git clean -n first so I can see what goes"


def _set_typed_reply_declines(enabled: bool) -> None:
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["approvals"].update(mode="manual", typed_reply_declines=enabled)
    save_config(config)


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")


def _runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter._send_with_retry = AsyncMock(return_value=SimpleNamespace(success=True, message_id="r1"))
    adapter._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents, runner._pending_messages = {}, {}
    runner._draining, runner.session_store = False, None
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = runner._busy_text_mode = "interrupt"
    return runner, adapter


@pytest.fixture
def gate(monkeypatch):
    """Run the real dangerous-command gate on a thread, blocked on a gateway approval for the chat."""
    for key in ("HERMES_YOLO_MODE", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    approval_module._gateway_queues.clear()
    started = []

    def start(session_key):
        out = {}

        def run():
            from tools.approval_context import reset_current_session_key, set_current_session_key
            monkeypatch.setenv("HERMES_EXEC_ASK", "1")
            token = set_current_session_key(session_key)
            approval_module.register_gateway_notify(session_key, lambda data: None)
            try:
                out["result"] = approval_module.check_all_command_guards("rm -rf /srv/build-cache", "local")
            finally:
                approval_module.unregister_gateway_notify(session_key)
                reset_current_session_key(token)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        started.append((session_key, thread))
        wait_until(lambda: approval_module.has_blocking_approval(session_key), what="approval pending")
        return out, thread

    yield start
    for session_key, thread in started:
        approval_module.resolve_gateway_approval(session_key, "deny", resolve_all=True)
        thread.join(timeout=5)
    approval_module._gateway_queues.clear()


def test_chat_message_declines_pending_approval_as_next_instruction(gate):
    _set_typed_reply_declines(True)
    runner, adapter = _runner()
    session_key = runner._session_key_for_source(_source())
    out, thread = gate(session_key)

    event = MessageEvent(text=REPLY, message_type=MessageType.TEXT, source=_source(), message_id="m1")
    assert asyncio.run(runner._handle_active_session_busy_message(event, session_key)) is True
    thread.join(timeout=5)

    result = out["result"]
    assert result["approved"] is False
    assert REPLY in result["message"] and "next instruction" in result["message"]
    assert not runner._pending_messages  # the reply went into the blocked turn, not the queue
    sent = adapter._send_with_retry.await_args.kwargs["content"]
    assert "next instruction" in sent


def test_chat_message_leaves_approval_pending_when_off(gate):
    _set_typed_reply_declines(False)
    runner, _adapter = _runner()
    session_key = runner._session_key_for_source(_source())
    gate(session_key)

    event = MessageEvent(text=REPLY, message_type=MessageType.TEXT, source=_source(), message_id="m1")
    asyncio.run(runner._route_plaintext_approval_while_busy(event, session_key))
    assert approval_module.has_blocking_approval(session_key)


@pytest.mark.asyncio
async def test_api_steer_declines_the_runs_pending_approval(gate):
    from tests.gateway.test_api_server_runs import _claim_run, _create_runs_app, _make_adapter
    _set_typed_reply_declines(True)
    adapter = _make_adapter()
    run_id = "run_wait"
    agent = MagicMock()
    adapter._active_run_agents[run_id] = agent
    adapter._run_approval_sessions[run_id] = run_id
    adapter._set_run_status(run_id, "waiting_for_approval")
    _claim_run(adapter, run_id)
    out, thread = gate(run_id)

    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        resp = await cli.post(f"/v1/runs/{run_id}/steer", json={"input": REPLY})
        payload = await resp.json()
    await asyncio.to_thread(thread.join, 5)

    assert resp.status == 200 and payload["declined_approvals"] == 1
    agent.steer.assert_not_called()  # delivered once, as the declined approval's reply
    assert REPLY in out["result"]["message"]
