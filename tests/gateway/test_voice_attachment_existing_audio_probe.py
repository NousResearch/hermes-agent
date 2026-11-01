"""Independent guard checks for already-produced speech and existing audio."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.test_voice_attachment_observations import CaptureDelivery
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", [MessageType.TEXT, MessageType.VOICE])
@pytest.mark.parametrize("existing_audio", [
    "stream_confirmed", "tool_output", "failed_tool", "tool_output_omitted", "unrelated_audio", "prebuilt_voice",
])
async def test_existing_audio_is_not_synthesized_again(existing_audio, input_type, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = CaptureDelivery()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {"telegram:fixture-chat": "all"}
    adapter.gateway_runner = runner
    runner._sync_voice_mode_state_to_adapter(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="Request", source=source, message_type=input_type, message_id="request")
    key = adapter._source_session_key(source)
    audio = tmp_path / ("auto.ogg" if existing_audio == "stream_confirmed" else "tool.ogg")
    audio.write_bytes(b"controlled previously-produced audio fixture")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n")
    calls = []
    handler_done = []

    def forbidden_provider(**kwargs):
        calls.append(kwargs)
        generated = tmp_path / "auto.ogg"
        generated.write_bytes(b"controlled duplicate-speech fixture")
        return json.dumps({"success": True, "file_path": str(generated)})

    async def handler(ev):
        messages = [{"role": "user", "content": "Request"}]
        streamed = existing_audio == "stream_confirmed"
        if streamed:
            adapter._active_sessions[key]._hermes_run_generation = 17
            await adapter.send(source.chat_id, "The requested answer.")
            await adapter.send_voice(source.chat_id, str(audio))
            adapter._mark_streaming_tts_completed_turn(key, 17, event=ev)
            response = f"The requested answer.\nMEDIA:{pdf}"
        elif existing_audio not in {"unrelated_audio", "prebuilt_voice"}:
            messages.append({"role": "assistant", "tool_calls": [
                {"id": "tts-current", "function": {"name": "text_to_speech"}}]})
            messages.append({"role": "tool", "tool_call_id": "tts-current", "content": json.dumps(
                {"success": True, "file_path": str(audio), "voice_compatible": True}
                if existing_audio in {"tool_output", "tool_output_omitted"}
                else {"success": False, "error": "controlled generation failure"})})
            response = "The requested answer."
            if existing_audio == "tool_output":
                response += f"\n[[audio_as_voice]]\nMEDIA:{audio}"
            elif existing_audio == "tool_output_omitted":
                response += f"\nMEDIA:{pdf}"
        else:
            directive = "[[audio_as_voice]]\n" if existing_audio == "prebuilt_voice" else ""
            response = f"The requested answer.\n{directive}MEDIA:{audio}"
        result = await runner._hmwa_deliver_turn_response(
            ev, source, SimpleNamespace(session_id="fixture-session", session_key=key), key,
            17 if streamed else None, {"already_sent": streamed}, messages, response, "", False,
        )
        handler_done.append(True)
        return result

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", forbidden_provider)
    adapter.set_message_handler(handler)
    await adapter.handle_message(event)
    await async_wait_until(lambda: handler_done and key not in adapter._active_sessions, timeout=10)
    print("VOICE_OBSERVATION " + json.dumps({"boundary": existing_audio, "input": input_type.value,
          "provider_calls": len(calls), "wire": adapter.wire}, sort_keys=True))
    assert len(calls) == int(existing_audio in {"failed_tool", "unrelated_audio"})
    assert sum(w["kind"] in {"spoken", "attached_audio"} for w in adapter.wire) == 1
    assert sum(w["kind"] == "document" for w in adapter.wire) == int(
        existing_audio in {"stream_confirmed", "tool_output_omitted", "unrelated_audio"})
    if existing_audio == "unrelated_audio":
        assert sum(w["kind"] == "spoken" for w in adapter.wire) == 1
