"""A relayed Discord interaction reaches the default native session authority only through the
authenticated WS reader: real frame dispatcher -> relay adapter -> GatewayRunner/SQLite authority."""
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

_MODAL = {"id": "i1", "type": 5, "channel_id": "c1", "guild_id": "g1",
          "member": {"user": {"id": "u1", "username": "ben"}},
          "data": {"custom_id": "feedback", "components": [
              {"type": 1, "components": [{"type": 4, "custom_id": "title", "value": "MODAL_MARKER"}]}]}}
_FRAME = {"type": "passthrough_forward", "bufferId": "b1", "forward": {
    "platform": "discord", "botId": "app", "method": "POST", "path": "/interactions", "headers": [],
    "bodyB64": base64.b64encode(json.dumps(_MODAL).encode()).decode()}}


@pytest.mark.asyncio
async def test_relayed_modal_admitted_only_with_reader_proof(tmp_path, monkeypatch):
    import gateway.run as gr
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.ws_transport import WebSocketRelayTransport, _passthrough_from_wire
    from gateway.session_authority import initialize_session_authority
    from tests.gateway.relay.test_relay_passthrough import _desc
    from tests.waiting import async_wait_until

    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    transport = WebSocketRelayTransport("ws://127.0.0.1:1/relay", "discord", "app",
                                        gateway_id="t", upgrade_secret="test-only")
    desc = _desc()
    transport._ws = MagicMock()
    transport._descriptor = desc
    transport._descriptors_by_platform = {"discord": desc}
    adapter = RelayAdapter(PlatformConfig(enabled=True), desc, transport=transport)
    runner.adapters = {Platform.RELAY: adapter}
    adapter.gateway_runner = runner
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="t")
    runner._wire_adapter_handlers(adapter)
    transport.set_passthrough_handler(adapter._on_passthrough)
    calls = []

    async def run(**kwargs):
        calls.append(kwargs["message"])
        return {"final_response": "ok", "messages": [], "tools": [], "history_offset": 0,
                "last_prompt_tokens": 0}

    runner._run_agent = run
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent"))

    def settled():
        return not adapter._background_tasks and not adapter._session_tasks

    try:
        # Forged: the same forward handed to the adapter without the reader's proof is refused.
        await adapter._on_passthrough(_passthrough_from_wire(_FRAME["forward"]))
        await async_wait_until(settled, timeout=30)
        assert calls == []
        # Authentic: the real WS reader dispatches the frame; the modal text is admitted.
        await transport._handle_frame(json.dumps(_FRAME))
        await async_wait_until(lambda: calls and settled(), timeout=30)
        assert calls == ["feedback\ntitle: MODAL_MARKER"]
    finally:
        authority.db.close()
