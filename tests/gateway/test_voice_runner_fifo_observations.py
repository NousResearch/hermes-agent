"""Runner FIFO drain, continuation and finalization observations.

The recursive agent result, input preparation, pin/cache persistence, speech
provider and transport are controlled. FIFO drain and continuation/delivery
methods are real. This does not exercise a live model or native streaming.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_voice_attachment_observations import CaptureDelivery, make_reply
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "voice_only", "all"])
@pytest.mark.parametrize("opening_type", [MessageType.TEXT, MessageType.VOICE])
@pytest.mark.parametrize("terminal_type", [MessageType.TEXT, MessageType.VOICE])
@pytest.mark.parametrize("attachment", ["none", "media_pdf"])
async def test_each_runner_queued_reply_obeys_its_own_voice_policy(
    mode, opening_type, terminal_type, attachment, tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = CaptureDelivery()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {"telegram:fixture-chat": mode}
    runner._draining = False

    adapter.gateway_runner = runner
    runner._sync_voice_mode_state_to_adapter(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", chat_type="dm", user_id="fixture-user")
    opening = MessageEvent(text="First request", source=source, message_type=opening_type, message_id="opening")
    terminal = MessageEvent(text="Next request", source=source, message_type=terminal_type, message_id="terminal")
    key = adapter._source_session_key(source)
    first_response = make_reply(attachment, tmp_path).replace("The requested answer.", "First reply.")
    terminal_response = make_reply(attachment, tmp_path).replace("The requested answer.", "Terminal reply.")
    generations, recursive_inputs, completed = [], [], []

    def provider(*, text, output_path):
        path = tmp_path / "auto.ogg"
        path.write_bytes(b"controlled speech fixture")
        generations.append(text)
        return json.dumps({"success": True, "file_path": str(path)})

    async def recursive_model(**kwargs):
        recursive_inputs.append(kwargs["message_type"])
        return {"final_response": terminal_response, "messages": [{"role": "user", "content": "Next request"}],
                "already_sent": False}

    runner._run_agent = recursive_model
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="Next request")
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()

    async def handler(ev):
        ctx = TurnContext(source=source, voice_turn=opening_type == MessageType.VOICE,
                          session_id="fixture-session", session_key=key, run_generation=None,
                          history=[], context_prompt=None, event_message_id="opening", inbound_message_id="opening")
        ctx.reply_event = ev
        result = {"final_response": first_response, "messages": [{"role": "user", "content": "First request"}]}
        runner._enqueue_fifo(key, terminal, adapter)
        pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
        assert pending_event is terminal
        merged = await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event, first_response, result, None)
        response = await runner._hmwa_deliver_turn_response(
            ev, source, SimpleNamespace(session_id="fixture-session", session_key=key), key, None,
            merged, merged["messages"], merged["final_response"], "", False,
        )
        completed.append(True)
        return response

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", provider)
    adapter.set_message_handler(handler)
    await adapter.handle_message(opening)
    await async_wait_until(lambda: completed and key not in adapter._active_sessions, timeout=10)
    counts = {"first": sum("First reply" in text for text in generations),
              "terminal": sum("Terminal reply" in text for text in generations)}
    expected = {"first": int(mode == "all" or (mode == "voice_only" and opening_type == MessageType.VOICE)),
                "terminal": int(mode == "all" or (mode == "voice_only" and terminal_type == MessageType.VOICE))}
    print("VOICE_OBSERVATION " + json.dumps({"delivery": "runner_fifo_continuation", "mode": mode,
          "opening_input": opening_type.value, "terminal_input": terminal_type.value, "attachment": attachment,
          "speech_counts": counts, "expected_counts": expected, "recursive_input": [t.value for t in recursive_inputs],
          "wire": adapter.wire}, sort_keys=True))
    assert recursive_inputs == [terminal_type]
    assert runner._queue_depth(key, adapter=adapter) == 0
    for body in ["First reply.", "Terminal reply."]:
        assert sum(body in w.get("text", "") + w.get("caption", "") for w in adapter.wire) == 1
    assert sum(w["kind"] == "document" for w in adapter.wire) == 2 * int(attachment != "none")
    assert counts == expected


@pytest.mark.asyncio
async def test_terminal_reply_retains_its_actual_voice_channel_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = CaptureDelivery()
    adapter.platform = Platform.DISCORD
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {"discord:fixture-chat": "all"}
    runner._draining = False
    adapter.gateway_runner = runner
    source = SessionSource(platform=Platform.DISCORD, chat_id="fixture-chat", chat_type="group", user_id="fixture-user")
    opening = MessageEvent(text="Opening", source=source, message_type=MessageType.TEXT, message_id="opening", internal=True)
    terminal = MessageEvent(text="Next", source=source, message_type=MessageType.TEXT, message_id="terminal",
                            raw_message=SimpleNamespace(guild_id=42))
    key = adapter._source_session_key(source)
    played, generated = [], []

    def provider(*, text, output_path):
        generated.append(text)
        audio = tmp_path / "auto.ogg"
        audio.write_bytes(b"controlled speech fixture")
        return json.dumps({"success": True, "file_path": str(audio)})

    async def playback(guild, audio_path):
        played.append(guild)
        return True

    adapter.is_in_voice_channel = lambda guild: guild == 42
    adapter.play_in_voice_channel = playback
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="Next")
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._run_agent = AsyncMock(return_value={
        "final_response": "Terminal completed answer.", "messages": [{"role": "user", "content": "Next"}],
    })
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", provider)
    ctx = TurnContext(source=source, session_key=key, session_id="fixture-session", history=[])
    ctx.reply_event = opening
    result = {"final_response": "Opening completed answer.", "messages": [{"role": "user", "content": "Opening"}]}
    runner._enqueue_fifo(key, terminal, adapter)
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
    merged = await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event, result["final_response"], result, None)
    assert merged["_queued_terminal_reply_event"] is terminal
    await runner._hmwa_deliver_turn_response(
        opening, source, SimpleNamespace(session_id="fixture-session"), key, None,
        merged, merged["messages"], merged["final_response"], "", False,
    )
    assert played == [42]
    assert len(generated) == 2
    assert opening.raw_message is None
    assert opening.message_id == "opening"
    assert "_queued_terminal_reply_event" not in merged
