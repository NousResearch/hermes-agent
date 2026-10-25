"""Regression coverage for spoken replies with files across delivery routes.

Runner and adapter delivery methods are real. Model results, speech providers
and platform transport are controlled. No external calls are made.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.waiting import async_wait_until


class CaptureDelivery(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fixture"), Platform.TELEGRAM)
        self.wire = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.wire.append({"kind": "text", "text": content})
        return SendResult(success=True, message_id="text")

    async def send_voice(self, chat_id, audio_path, caption=None, **kwargs):
        assert Path(audio_path).is_file()
        kind = "spoken" if Path(audio_path).name == "auto.ogg" else "attached_audio"
        self.wire.append({"kind": kind, "caption": caption or ""})
        return SendResult(success=True, message_id="voice")

    async def send_document(self, chat_id, file_path, **kwargs):
        assert Path(file_path).is_file()
        self.wire.append({"kind": "document"})
        return SendResult(success=True, message_id="document")

    async def send_multiple_images(self, chat_id, images, **kwargs):
        self.wire.append({"kind": "image", "count": len(images)})
        return SendResult(success=True, message_id="image")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def make_reply(kind, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n")
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    suffixes = {
        "none": "", "media_pdf": f"\nMEDIA:{pdf}", "media_image": f"\nMEDIA:{image}",
        "bare_pdf": f"\n{pdf}", "markdown_image": "\n![figure](https://example.invalid/figure.png)",
    }
    return "The requested answer." + suffixes[kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["ordinary", "queued", "post_stream"])
@pytest.mark.parametrize("mode", ["off", "voice_only", "all"])
@pytest.mark.parametrize("input_type", [MessageType.TEXT, MessageType.VOICE])
@pytest.mark.parametrize("attachment", ["none", "media_pdf", "media_image", "bare_pdf", "markdown_image"])
async def test_outgoing_spoken_reply_matrix(delivery, mode, input_type, attachment, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = CaptureDelivery()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {"telegram:fixture-chat": mode}
    adapter.gateway_runner = runner
    runner._sync_voice_mode_state_to_adapter(adapter)
    adapter._busy_text_mode = "queue"
    response = make_reply(attachment, tmp_path)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="Request", source=source, message_type=input_type, message_id="request")
    key = adapter._source_session_key(source)
    generation_calls = []
    handler_calls = []
    blocker_started, release_blocker = asyncio.Event(), asyncio.Event()

    def fake_provider(*, text, output_path):
        path = tmp_path / "auto.ogg"
        path.write_bytes(b"controlled speech fixture, not encoded provider output")
        generation_calls.append({"contains_media_directive": "MEDIA:" in text, "text_length": len(text)})
        return json.dumps({"success": True, "file_path": str(path)})

    async def model_boundary(ev):
        if ev.message_id == "blocker":
            blocker_started.set()
            await release_blocker.wait()
            return None
        handler_calls.append(ev.message_id)
        already_sent = delivery == "post_stream"
        if already_sent:
            await adapter.send(ev.source.chat_id, response)
        return await runner._hmwa_deliver_turn_response(
            ev, ev.source, SimpleNamespace(session_id="fixture-session", session_key=key), key, None,
            {"already_sent": already_sent}, [{"role": "user", "content": "Request"}], response, "", False,
        )

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_provider)
    adapter.set_message_handler(model_boundary)
    first = None
    try:
        if delivery == "queued":
            blocker = MessageEvent(text="Busy", source=source, message_type=MessageType.TEXT, message_id="blocker")
            first = asyncio.create_task(adapter.handle_message(blocker))
            await asyncio.wait_for(blocker_started.wait(), 10)
            await adapter.handle_message(event)
            release_blocker.set()
            await asyncio.wait_for(first, 10)
            await async_wait_until(lambda: handler_calls == ["request"] and key not in adapter._active_sessions, timeout=10)
        else:
            await adapter.handle_message(event)
            await async_wait_until(lambda: handler_calls == ["request"] and key not in adapter._active_sessions, timeout=10)
        expected_voice = mode == "all" or (mode == "voice_only" and input_type == MessageType.VOICE)
        expected_attachment = (
            attachment in {"media_pdf", "media_image", "markdown_image"} if delivery == "post_stream" else attachment != "none"
        )
        observation = {
            "delivery": delivery, "mode": mode, "input": input_type.value, "attachment": attachment,
            "provider_calls": len(generation_calls), "wire": adapter.wire,
            "handler_calls": handler_calls, "expected_spoken": expected_voice,
            "expected_attachment": expected_attachment,
            "generation_details": generation_calls,
        }
        print("VOICE_OBSERVATION " + json.dumps(observation, sort_keys=True))
        assert handler_calls == ["request"]
        assert any("The requested answer." in w.get("text", "") + w.get("caption", "") for w in adapter.wire)
        assert sum(w["kind"] in {"document", "image"} for w in adapter.wire) == int(expected_attachment)
        assert sum(w["kind"] == "spoken" for w in adapter.wire) == int(expected_voice)
        if expected_voice and input_type == MessageType.VOICE and delivery != "post_stream":
            assert any(w.get("caption") == "The requested answer." for w in adapter.wire)
            assert not any(w["kind"] == "text" for w in adapter.wire)
    finally:
        release_blocker.set()
        adapter._discard_text_debounce(key)
        if first is not None and not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
