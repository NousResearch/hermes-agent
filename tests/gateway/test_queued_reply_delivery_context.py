"""Queued audio/text reply targets through shared and native Telegram send methods.

Provider output and Bot API calls are controlled. No live delivery or encoding
claim. Admission identity remains separate from final reply routing.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.waiting import async_wait_until


class CaptureTargets(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fixture"), Platform.TELEGRAM)
        self._reply_to_mode = "all"
        self.wire = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.wire.append({"kind": "text", "body": content, "reply_to": reply_to,
                          "metadata": metadata})
        return SendResult(success=True, message_id="sent-text")

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None, **kwargs):
        self.wire.append({"kind": "audio", "body": Path(audio_path).read_text(encoding="utf-8"),
                          "reply_to": reply_to, "metadata": metadata})
        return SendResult(success=True, message_id="sent-audio")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None


def _native_adapter(monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", reply_to_mode="all"))
    adapter.wire = []

    async def send_message(**kwargs):
        anchor = kwargs.get("reply_to_message_id")
        adapter.wire.append({"kind": "text", "body": kwargs["text"].replace("\\", ""),
                             "reply_to": str(anchor) if anchor is not None else None,
                             "thread": kwargs.get("message_thread_id")})
        return SimpleNamespace(message_id=800)

    async def send_voice(**kwargs):
        anchor = kwargs.get("reply_to_message_id")
        adapter.wire.append({"kind": "audio", "body": kwargs["voice"].read().decode("utf-8"),
                             "reply_to": str(anchor) if anchor is not None else None,
                             "thread": kwargs.get("message_thread_id")})
        return SimpleNamespace(message_id=801)

    adapter._bot = SimpleNamespace(send_message=send_message, send_voice=send_voice,
                                   send_chat_action=AsyncMock())
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter._probe_voice_duration_seconds", lambda path: None)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["ordinary", "runner_fifo"])
@pytest.mark.parametrize("quoted_id", [None, "900"])
@pytest.mark.parametrize("transport", ["capture", "native"])
@pytest.mark.parametrize("chat", ["dm", "dm_topic", "forum"])
async def test_audio_and_text_target_the_answered_request(lane, quoted_id, transport, chat, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from gateway.platforms import base
    monkeypatch.setattr(base, "_HERMES_ROOT", tmp_path / "isolated-root")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", [str(tmp_path)])
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = CaptureTargets() if transport == "capture" else _native_adapter(monkeypatch)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {"telegram:12345": "all"}
    runner._draining = False
    adapter.gateway_runner = runner
    runner._sync_voice_mode_state_to_adapter(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345",
                           chat_type="group" if chat == "forum" else "dm", user_id="fixture-user",
                           thread_id="7" if chat != "dm" else None)
    opening = MessageEvent(text="First request", source=source, message_type=MessageType.TEXT,
                           message_id="101", reply_to_message_id=quoted_id)
    terminal = MessageEvent(text="Next request", source=source, message_type=MessageType.TEXT,
                            message_id="202", reply_to_message_id=quoted_id)
    key = adapter._source_session_key(source)
    generated, completed = [], []

    def provider(*, text, output_path):
        generated.append(text)
        audio = tmp_path / f"speech-{len(generated)}.ogg"
        audio.write_text(text, encoding="utf-8")
        return json.dumps({"success": True, "file_path": str(audio)})

    runner._run_agent = AsyncMock(return_value={
        "final_response": "Terminal completed answer.",
        "messages": [{"role": "user", "content": "Next request"}],
        "already_sent": False,
    })
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="Next request")
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()

    async def handler(event):
        result = {"final_response": "First completed answer.",
                  "messages": [{"role": "user", "content": "First request"}], "already_sent": False}
        if lane == "runner_fifo":
            ctx = TurnContext(source=source, voice_turn=False, session_id="fixture-session", session_key=key,
                              history=[], event_message_id="101", inbound_message_id="101")
            ctx.reply_event = event
            runner._enqueue_fifo(key, terminal, adapter)
            pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
            assert pending_event is terminal
            result = await runner._run_agent_queued_followup(
                ctx, adapter, pending, pending_event, result["final_response"], result, None)
        response = await runner._hmwa_deliver_turn_response(
            event, source, SimpleNamespace(session_id="fixture-session", session_key=key), key, None,
            result, result["messages"], result["final_response"], "", False)
        completed.append(True)
        return response

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", provider)
    adapter.set_message_handler(handler)
    await adapter.handle_message(opening)
    await async_wait_until(lambda: completed and key not in adapter._active_sessions, timeout=10)
    print("ANCHOR_OBSERVATION " + json.dumps({"lane": lane, "quoted_id": quoted_id,
          "transport": transport, "chat": chat, "wire": adapter.wire}, sort_keys=True))
    target = ("202" if lane == "runner_fifo" else "101") if chat != "forum" else None
    expected = {"Terminal completed answer.": target} if lane == "runner_fifo" else {"First completed answer.": target}
    # Isolate anchor correctness from the baseline's separate missing-first-speech defect.
    for body, target in expected.items():
        sends = [w for w in adapter.wire if w["body"] == body]
        assert sorted(w["kind"] for w in sends) == ["audio", "text"]
        assert {w["kind"]: w["reply_to"] for w in sends} == {"audio": target, "text": target}


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", [False, True])
async def test_actual_outer_handler_preserves_admission_identity_and_routes_both_replies(topic, tmp_path, monkeypatch):
    from tests.gateway.test_gateway_silence_tokens import _runner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from gateway.platforms import base
    monkeypatch.setattr(base, "_HERMES_ROOT", tmp_path / "isolated-root")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", [str(tmp_path)])
    (tmp_path / "config.yaml").write_text(json.dumps({"voice": {"auto_tts": True}}), encoding="utf-8")
    adapter = _native_adapter(monkeypatch)
    runner = _runner(monkeypatch, tmp_path)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {"telegram:12345": "all"}
    runner._should_send_voice_reply = GatewayRunner._should_send_voice_reply.__get__(runner)
    runner._reply_anchor_for_event = base._reply_anchor_for_event
    adapter.gateway_runner = runner
    runner._sync_voice_mode_state_to_adapter(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm",
                           user_id="fixture-user", thread_id="7" if topic else None)
    opening = MessageEvent(text="First request", source=source, message_id="101")
    terminal = MessageEvent(text="Next request", source=source, message_id="202")
    key = adapter._source_session_key(source)
    runner.session_store.get_or_create_session.return_value.session_key = key
    result = {"final_response": "Terminal completed answer.", "messages": [
        {"role": "user", "content": "Next request"},
        {"role": "assistant", "content": "Terminal completed answer."}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1,
        "failed": False, "queued_terminal_inbound_id": "202",
        "_queued_terminal_reply_event": terminal}
    runner._run_agent = AsyncMock(return_value=result)

    def provider(*, text, output_path):
        path = tmp_path / "outer.ogg"
        path.write_text(text, encoding="utf-8")
        return json.dumps({"success": True, "file_path": str(path)})

    completed = []

    async def handler(event):
        response = await runner._handle_message_with_agent(event, source, key, 1)
        completed.append(True)
        return response

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", provider)
    adapter.set_message_handler(handler)
    await adapter.handle_message(opening)
    await async_wait_until(lambda: completed and key not in adapter._active_sessions, timeout=10)
    assert runner._run_agent.await_count == 1
    assert opening.message_id == "101"
    assert opening.ledger_message_id == "202"
    assert "_queued_terminal_reply_event" not in result
    assert {w["kind"]: w["reply_to"] for w in adapter.wire} == {"audio": "202", "text": "202"}
