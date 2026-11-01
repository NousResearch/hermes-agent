"""Reasoning is part of the declared stream final, exactly once (#5892)."""
import asyncio
import json
import sys
from itertools import pairwise
from types import ModuleType, SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext
from tests.waiting import async_wait_until
from tests.gateway.test_stale_finalize_suppression import FinalizeCaptureAdapter, _make_runner
from tests.gateway.test_stream_final_contract import _make_draft_adapter
from tests.gateway.test_slack_native_streaming import _make_adapter, META


class ReasoningAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, **kwargs):
        if self.stream_delta_callback:
            self.stream_delta_callback("Final answer")
        return {"final_response": "Final answer", "last_reasoning": "Consider the evidence",
                "messages": [], "api_calls": 1, "completed": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("platform, display, shown", [
    (Platform.TELEGRAM, {"show_reasoning": True}, True),
    (Platform.DISCORD, {"show_reasoning": True, "reasoning_style": "subtext"}, True),
    (Platform.TELEGRAM, {"show_reasoning": False}, False),
    (Platform.MATTERMOST, {"show_reasoning": True}, False),
    (Platform.MATTERMOST, {"show_reasoning": True,
                          "platforms": {"mattermost": {"show_reasoning": True}}}, True),
])
async def test_streamed_reasoning_obeys_display_without_duplicate_final(
    monkeypatch, tmp_path, platform, display, shown,
):
    config = {"display": {"tool_progress": "off", "interim_assistant_messages": False, **display}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    fake_agent = ModuleType("run_agent")
    fake_agent.AIAgent = ReasoningAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_agent)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "fixture"})
    adapter = FinalizeCaptureAdapter(platform)
    runner = _make_runner(adapter)
    runner._show_reasoning = True
    source = SessionSource(platform=platform, chat_id="room", chat_type="group", thread_id="thread")
    result = await runner._run_agent(
        message="Explain", context_prompt="", history=[], source=source,
        session_id="reasoning", session_key=f"agent:main:{platform.value}:group:room",
    )
    response = runner._hmwa_prepend_reasoning(result, result["final_response"], source, False)
    assert ("Consider the evidence" in response) is shown
    assert response.count("Consider the evidence") == int(shown)
    assert result.get("already_sent") is True
    payloads = [entry["content"] for entry in adapter.sent + adapter.edits]
    assert response in payloads, "suppressed final must contain the configured reasoning"
    assert len(adapter.sent) == 1
    if shown and platform == Platform.DISCORD:
        assert "-# Consider the evidence" in response
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    event = MessageEvent(text="Explain", message_type=MessageType.TEXT, source=source, message_id="inbound")
    delivered = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="reasoning"), "reasoning", None,
        result, [], response, "", False,
    )
    assert delivered is None
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native_slack", [False, True])
@pytest.mark.parametrize("state, shown", [
    ({"completed": True}, True),
    ({"failed": True}, False),
    ({"interrupted": True, "completed": False}, False),
    ({"completed": False}, False),
    ({"completed": True, "final_response": "[SILENT]"}, False),
])
async def test_native_stream_seals_reasoning_final_with_prefix_stable_drafts(monkeypatch, state, shown, native_slack):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"display": {"show_reasoning": True}})
    adapter = _make_draft_adapter()
    if native_slack:
        adapter, client = _make_adapter()
    source = SessionSource(platform=Platform.SLACK, chat_id="room", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner._show_reasoning = True
    ctx = TurnContext(source=source)
    consumer = GatewayStreamConsumer(adapter, "room", StreamConsumerConfig(
        transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor="",
    ), metadata=META)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Final answer")
    await async_wait_until(
        lambda: bool(client.chat_startStream.await_count if native_slack else adapter.draft_calls), timeout=2,
    )
    result = {"final_response": "Final answer", "last_reasoning": "Consider the evidence", **state}
    TurnRunner(runner, ctx)._finish_stream_consumer(result, [], consumer)
    await task
    if native_slack:
        client.chat_postMessage.assert_not_awaited()
        client.chat_startStream.assert_awaited_once()
        if state.get("final_response") != "[SILENT]":
            client.chat_stopStream.assert_awaited_once()
        payloads = [call.kwargs.get("text", "") for call in client.chat_update.await_args_list]
        payloads += [call.kwargs.get("markdown_text", "") for call in client.chat_stopStream.await_args_list]
        assert any("Consider the evidence" in payload for payload in payloads) is shown
        if shown:
            assert consumer.delivered_final_matches(result["final_response"]) is True
        return
    frames = [call["content"] for call in adapter.draft_calls]
    assert all(right.startswith(left) for left, right in pairwise(frames))
    if state.get("final_response") == "[SILENT]":
        assert not adapter.send_calls
        assert not result.get("_gateway_reasoning_rendered")
        return
    assert len(adapter.send_calls) == 1
    final = adapter.send_calls[0]["content"]
    assert ("Consider the evidence" in final) is shown
    assert final.count(result["final_response"]) == 1
    assert consumer.delivered_final_matches(result["final_response"]) is True
