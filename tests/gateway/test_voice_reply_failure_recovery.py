"""Speech failure must not discard the written answer or its document."""
import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.test_voice_attachment_observations import CaptureDelivery
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "exception"])
async def test_failed_voice_preserves_text_and_document(failure, tmp_path, monkeypatch):
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
    event = MessageEvent(text="Request", source=source, message_type=MessageType.VOICE, message_id="request")
    key = adapter._source_session_key(source)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n")
    speech_calls, send_attempts, completed = [], [], []

    def provider(*, text, output_path):
        speech_calls.append(text)
        path = tmp_path / "auto.ogg"
        path.write_bytes(b"controlled speech fixture")
        return json.dumps({"success": True, "file_path": str(path)})

    async def failed_send(*args, **kwargs):
        send_attempts.append(True)
        if failure == "exception":
            raise RuntimeError("controlled voice transport failure")
        return SendResult(success=False, error="controlled refusal")

    async def handler(ev):
        try:
            return await runner._hmwa_deliver_turn_response(
                ev, source, SimpleNamespace(session_id="fixture-session", session_key=key), key, None,
                {"already_sent": False}, [{"role": "user", "content": "Request"}],
                f"The requested answer.\nMEDIA:{pdf}", "", False,
            )
        finally:
            completed.append(True)

    monkeypatch.setattr(adapter, "send_voice", failed_send)
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", provider)
    adapter.set_message_handler(handler)
    await adapter.handle_message(event)
    await async_wait_until(lambda: completed and key not in adapter._active_sessions, timeout=10)
    assert len(speech_calls) == len(send_attempts) == 1
    assert sum(w["kind"] == "document" for w in adapter.wire) == 1
    assert sum(w.get("text") == "The requested answer." for w in adapter.wire) == 1
    assert not (tmp_path / "auto.ogg").exists()
