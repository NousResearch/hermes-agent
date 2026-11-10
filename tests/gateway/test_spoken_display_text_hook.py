"""``transform_spoken_display_text``: the chat text sent with a spoken reply (issue #152238).

The speech keeps its delivery cues while the chat bubble / Telegram caption gets the display
text. Fires only when the gateway actually spoke the reply (auto-TTS audio, or the runner's voice
reply); a plain text turn is never touched. The plugin dispatch is faked at
``hermes_cli.plugins``; the TTS tool is a fake that writes a real file.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

CUED = "[sighs] Fine, it passed."


def _fake_hooks(monkeypatch, display):
    """``transform_spoken_text`` leaves speech alone; the display hook answers with *display*."""
    calls = []

    def _invoke(hook_name, **kwargs):
        if hook_name == "transform_spoken_display_text":
            calls.append(kwargs)
            return [display(kwargs)]
        return []

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name.startswith("transform_spoken"))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return calls


def _strip_cue(kwargs):
    return kwargs["text"].replace("[sighs] ", "")


def _fake_tts_tool(monkeypatch):
    spoken = []

    def _tts(*, text, output_path, **_kwargs):
        spoken.append(text)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"\x00" * 32)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _tts)
    return spoken


class _Adapter(BasePlatformAdapter):
    def __init__(self, platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []
        self._should_auto_tts_for_chat = lambda _chat_id: True
        self.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
        self._session_store = SimpleNamespace(peek_session_id=lambda key: "sess-1")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


async def _run_turn(platform, message_type, handler):
    adapter = _Adapter(platform)

    async def _hold_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="hello", message_type=message_type, message_id="m-1",
        source=SessionSource(platform=platform, chat_id="-1001", chat_type="group"))
    with patch("tools.tts_tool.check_tts_requirements", return_value=True):
        await adapter._process_message_background(event, build_session_key(event.source))
    return adapter


def _reply(text, *, voice_reply_sent=False):
    async def _handler(event):
        if voice_reply_sent:
            event._voice_reply_sent = True
        return text
    return _handler


@pytest.mark.asyncio
async def test_auto_tts_speaks_the_cue_and_sends_the_display_text(monkeypatch):
    spoken = _fake_tts_tool(monkeypatch)
    calls = _fake_hooks(monkeypatch, _strip_cue)

    adapter = await _run_turn(Platform.DISCORD, MessageType.VOICE, _reply(CUED))

    assert spoken == [CUED]
    assert adapter.sent == ["Fine, it passed."]
    assert [(c["surface"], c["platform"], c["session_id"]) for c in calls] == [("auto_tts", "discord", "sess-1")]


@pytest.mark.asyncio
async def test_telegram_caption_uses_the_display_text(monkeypatch):
    _fake_tts_tool(monkeypatch)
    _fake_hooks(monkeypatch, _strip_cue)

    adapter = await _run_turn(Platform.TELEGRAM, MessageType.VOICE, _reply(CUED))

    assert adapter.play_tts.await_args.kwargs["caption"] == "Fine, it passed."
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_empty_display_text_sends_the_voice_note_alone(monkeypatch):
    _fake_tts_tool(monkeypatch)
    _fake_hooks(monkeypatch, lambda kw: "")

    adapter = await _run_turn(Platform.DISCORD, MessageType.VOICE, _reply(CUED))

    adapter.play_tts.assert_awaited_once()
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_runner_voice_reply_turn_sends_the_display_text(monkeypatch):
    calls = _fake_hooks(monkeypatch, _strip_cue)

    adapter = await _run_turn(Platform.DISCORD, MessageType.TEXT, _reply(CUED, voice_reply_sent=True))

    assert adapter.sent == ["Fine, it passed."]
    assert [c["surface"] for c in calls] == ["voice_reply"]


@pytest.mark.asyncio
async def test_unspoken_text_turn_is_never_touched(monkeypatch):
    calls = _fake_hooks(monkeypatch, _strip_cue)

    adapter = await _run_turn(Platform.DISCORD, MessageType.TEXT, _reply(CUED))

    assert adapter.sent == [CUED]
    assert calls == []


@pytest.mark.asyncio
async def test_failed_auto_tts_leaves_the_text_unchanged(monkeypatch):
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool", lambda **_kw: json.dumps({"success": False, "error": "x"}))
    calls = _fake_hooks(monkeypatch, _strip_cue)

    adapter = await _run_turn(Platform.DISCORD, MessageType.VOICE, _reply(CUED))

    assert adapter.sent == [CUED]
    assert calls == []


@pytest.mark.asyncio
async def test_runner_marks_the_event_only_after_delivering_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    _fake_tts_tool(monkeypatch)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(
        send_voice=AsyncMock(), is_in_voice_channel=lambda *_a, **_k: False)}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    spoken = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="m1")
    silent = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="m2")

    await runner._send_voice_reply(spoken, CUED)
    await runner._send_voice_reply(silent, "```\ncode only\n```")

    assert getattr(spoken, "_voice_reply_sent", False) is True
    assert getattr(silent, "_voice_reply_sent", False) is False
