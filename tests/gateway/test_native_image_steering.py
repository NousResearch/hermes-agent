"""Native busy image corrections traverse ingress, the active loop, and durable replay (#146835)."""

import asyncio
import base64
from copy import deepcopy
from itertools import pairwise
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from run_agent import AIAgent
from tests.agent.test_tool_call_incremental_persistence import (
    _make_tool_defs,
)


class CaptureAdapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="ack")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  supports_vision: true\nagent:\n  image_input_mode: native\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="synthetic-key", base_url="https://example.invalid/v1", model="gpt-4o",
            quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=8,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "Inspect the supplied diagrams."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *a, **k: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._busy_input_mode = "steer"
    images = [tmp_path / "circle.png", tmp_path / "square.png"]
    for path, color in zip(images, ["blue", "red"]):
        Image.new("RGB", (8, 8), color).save(path)
    return runner, agent, images


def install_route(runner, agent, images, platform):
    adapter = CaptureAdapter(PlatformConfig(enabled=True, extra={
        "allow_from": ["sender"], "group_allow_from": ["sender"],
        "group_sessions_per_user": False, "thread_sessions_per_user": False,
    }), platform)
    runner.adapters[platform] = adapter
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="1234", user_id="sender", chat_type="dm", thread_id="7")
    event = MessageEvent(
        text="Compare these replacement diagrams", message_type=MessageType.PHOTO,
        source=source, message_id="image-event", media_urls=[str(p) for p in images],
        media_types=["image/png"] * len(images), reply_to_message_id="initial",
    )
    key = runner._session_key_for_source(source)
    runner._session_state(key).turn.agent = agent
    return adapter, key, event


def pixel_bytes(content):
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            for part in content if part["type"] == "image_url"]


def assert_provider_projections(messages, images):
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    original = deepcopy(messages)
    _, anthropic = convert_messages_to_anthropic(messages)
    assert all(a["role"] != b["role"] for a, b in pairwise(anthropic))
    parts = anthropic[-1]["content"]
    assert [p["tool_use_id"] for p in parts if p["type"] == "tool_result"] == ["first", "second"]
    assert [base64.b64decode(p["source"]["data"]) for p in parts if p["type"] == "image"] == images
    responses = _chat_messages_to_responses_input(messages, current_issuer_kind="codex_backend")
    assert [p["call_id"] for p in responses if p.get("type") == "function_call"] == ["first", "second"]
    assert [p["call_id"] for p in responses if p.get("type") == "function_call_output"] == ["first", "second"]
    assert responses[-1]["role"] == "user"
    types = [p["type"] for p in responses[-1]["content"]]
    assert types[-2:] == ["input_image", "input_image"] and all(t == "input_text" for t in types[:-2])
    assert [base64.b64decode(p["image_url"].split(",", 1)[1]) for p in responses[-1]["content"]
            if p["type"] == "input_image"] == images
    assert messages == original
    assert "fallback" not in str(responses) and "content_steer_ids" not in str(responses)




async def deliver_relay(runner, event):
    """A real WS reader mints provenance; only the remote connector is synthetic."""
    import json
    import websockets
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    descriptor = CapabilityDescriptor(1, 'slack', 'Fixture', 4096, False, True, True, 'plain', 'chars')
    ready, acknowledged = asyncio.Queue(), asyncio.Event()
    async def connector(ws):
        await ws.recv()
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        await ready.put(ws)
        async for data in ws:
            frame = json.loads(data)
            if frame['type'] == 'inbound_ack':
                acknowledged.set()
            elif frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'ack'}}) + '\n')
    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay', 'slack', 'fixture-bot',
            gateway_id='fixture-gateway', upgrade_secret='synthetic-secret', reconnect=False)
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters.clear()
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            raw = {'text': event.text, 'message_id': event.message_id, 'message_type': event.message_type.value,
                   'source': event.source.to_dict(), 'media_urls': event.media_urls, 'media_types': event.media_types}
            await ws.send(json.dumps({'type': 'inbound', 'event': raw, 'bufferId': 'image-frame'}) + '\n')
            await asyncio.wait_for(acknowledged.wait(), 10)
        finally:
            await adapter.disconnect()
