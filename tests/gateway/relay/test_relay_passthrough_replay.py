"""Regression for #140260: a forwarded interaction is one delivery, even after reconnect/restart."""
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult, _reply_anchor_for_event
from gateway.relay.adapter import RelayAdapter
from gateway.relay.ws_transport import WebSocketRelayTransport
from tests.gateway.relay.test_relay_passthrough import _desc
from tests.waiting import async_wait_until


def _frame(kind=5, *, interaction_id="interaction-1", buffer_id="buffer-1"):
    data = {2: {"name": "status"}, 3: {"custom_id": "button"},
            5: {"custom_id": "feedback", "components": [
                {"type": 1, "components": [{"type": 4, "custom_id": "title", "value": "marker"}]}]}}[kind]
    payload = {"type": kind, "channel_id": "chat-1", "guild_id": "guild-1", "data": data,
               "member": {"user": {"id": "user-1", "username": "ben"}}}
    if interaction_id is not None:
        payload["id"] = interaction_id
    if kind == 3:
        payload["message"] = {"id": "shared-message"}
    return {"type": "passthrough_forward", "bufferId": buffer_id, "forward": {
        "platform": "discord", "botId": "app", "method": "POST", "path": "/interactions",
        "headers": [], "bodyB64": base64.b64encode(json.dumps(payload).encode()).decode()}}


def _adapter():
    desc = _desc()
    transport = WebSocketRelayTransport("ws://127.0.0.1:1/relay", "discord", "app",
                                        gateway_id="t", upgrade_secret="test-only")
    transport._ws = MagicMock()
    transport._descriptor = desc
    transport._descriptors_by_platform = {"discord": desc}
    adapter = RelayAdapter(PlatformConfig(enabled=True), desc, transport=transport)
    transport.set_passthrough_handler(adapter._on_passthrough)
    return adapter, transport


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [2, 3, 5])
@pytest.mark.parametrize("identity", ["buffer", "interaction", "buffer-only", "live"])
async def test_passthrough_replay_dispatches_once(kind, identity):
    adapter, transport = _adapter()
    adapter.handle_message = AsyncMock()
    first = _frame(kind, interaction_id=None if identity == "buffer-only" else "interaction-1",
                   buffer_id=None if identity == "live" else "buffer-1")
    second = json.loads(json.dumps(first))
    if identity == "interaction":
        second["bufferId"] = "buffer-2"
    await transport._handle_frame(json.dumps(first))
    await transport._handle_frame(json.dumps(second))
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.message_id is None
    assert _reply_anchor_for_event(event) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["in-flight", "terminal", "restart", "reset", "rewrite-reset"])
@pytest.mark.parametrize("kind", [3, 5])
@pytest.mark.parametrize("identity", ["interaction", "buffer-only"])
async def test_native_interaction_replay_has_one_durable_turn(tmp_path, monkeypatch, retry, kind, identity):
    import gateway.run as gr
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    hook_calls = []
    if retry == "rewrite-reset":
        from hermes_cli import lifecycle

        async def hook(name, **kwargs):
            if name == "pre_gateway_dispatch":
                hook_calls.append(kwargs["event"].delivery_id)
                return [{"action": "rewrite", "text": "rewritten: " + kwargs["event"].text}]
            return []

        monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_gateway_dispatch")
        monkeypatch.setattr(lifecycle, "ainvoke_hook", hook)
    calls, entered, release = [], asyncio.Event(), asyncio.Event()
    if retry != "in-flight":
        release.set()

    async def run(**kwargs):
        calls.append(kwargs["message"])
        entered.set()
        await release.wait()
        return {"final_response": "ok", "messages": [], "tools": [], "history_offset": 0,
                "last_prompt_tokens": 0}

    async def boot():
        runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        adapter, transport = _adapter()
        runner.adapters = {Platform.RELAY: adapter}
        adapter.gateway_runner = runner
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="t")
        runner._wire_adapter_handlers(adapter)
        runner._run_agent = run
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
        return adapter, transport, authority

    adapter, transport, authority = await boot()
    first = _frame(kind, interaction_id=None if identity == "buffer-only" else "interaction-1")
    replay = json.loads(json.dumps(first))
    if identity == "interaction":
        replay["bufferId"] = "buffer-2"

    def settled():
        return (not adapter._background_tasks and not adapter._session_tasks
                and all(live.task is None or live.task.done() for live in authority.sessions.values()))

    try:
        await transport._handle_frame(json.dumps(first))
        await async_wait_until(entered.is_set, timeout=30)
        if retry != "in-flight":
            await async_wait_until(settled, timeout=30)
        if retry == "restart":
            authority.db.close()
            adapter, transport, authority = await boot()
        if retry in ("reset", "rewrite-reset"):
            route = next(iter(authority.sessions.values())).route
            authority.runner.session_store.reset_session(route)
        # A cold receiver cannot rely on the process-local seen set. The provider id survives a
        # changed connector buffer id; the admission ledger must recognize the original turn.
        adapter._seen_inbound.clear()
        await transport._handle_frame(json.dumps(replay))
        release.set()
        await async_wait_until(settled, timeout=30)
        rows = authority.db._read_all("SELECT request_id FROM session_admissions")
        assert len(rows) == 1
        text = "button" if kind == 3 else "feedback\ntitle: marker"
        if retry == "rewrite-reset":
            text = "rewritten: " + text
            assert hook_calls == [rows[0]["request_id"]]
        assert calls == [text]
        # A second user action in the same channel is a new turn.
        await transport._handle_frame(json.dumps(_frame(kind, interaction_id="interaction-2", buffer_id="buffer-3")))
        await async_wait_until(settled, timeout=30)
        assert len(authority.db._read_all("SELECT request_id FROM session_admissions")) == 2
        assert calls == [text] * 2
    finally:
        release.set()
        await async_wait_until(settled, timeout=30)
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("handoff failed"), asyncio.CancelledError()])
async def test_failed_passthrough_handoff_remains_replayable(error):
    adapter, transport = _adapter()
    adapter.handle_message = AsyncMock(side_effect=[error, None])
    frame = json.dumps(_frame(2))
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await transport._handle_frame(frame)
    else:
        await transport._handle_frame(frame)
    await transport._handle_frame(frame)
    assert adapter.handle_message.await_count == 2
