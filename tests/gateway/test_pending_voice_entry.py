"""Pending voice delivery through real Telegram handlers, with external I/O replaced."""
import asyncio
import json
import sys
import threading
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# gateway/conftest installs optional-library stubs before collection. This file
# exercises real PTB dispatch; the canonical runner isolates it in its own process.
if not isinstance(sys.modules.get("telegram"), ModuleType):
    for name in list(sys.modules):
        if name == "telegram" or name.startswith("telegram."):
            del sys.modules[name]
pytest.importorskip("telegram")

from telegram import Update
from telegram.ext import Application
from telegram.request import BaseRequest, RequestData

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


class LocalTelegramRequest(BaseRequest):
    def __init__(self):
        super().__init__()
        self.echoes = []
        self.echo_entered = asyncio.Event()
        self.echo_release = asyncio.Event()
        self.echo_release.set()

    @property
    def read_timeout(self):
        return 5

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(
        self, url: str, method: str, request_data: RequestData | None = None,
        read_timeout=BaseRequest.DEFAULT_NONE, write_timeout=BaseRequest.DEFAULT_NONE,
        connect_timeout=BaseRequest.DEFAULT_NONE, pool_timeout=BaseRequest.DEFAULT_NONE,
    ) -> tuple[int, bytes]:
        method_name = url.rsplit("/", 1)[-1]
        if "/file/" in url:
            return 200, b"OggSfixtureA" if method_name == "voice-a.ogg" else b"OggSfixtureB"
        if method_name == "getMe":
            result = {"id": 999, "is_bot": True, "first_name": "Fixture", "username": "fixture_bot"}
        elif method_name == "getFile":
            assert request_data is not None
            file_id = request_data.parameters["file_id"]
            assert isinstance(file_id, str)
            result = {"file_id": file_id, "file_unique_id": "unique-" + file_id,
                      "file_path": "voice/" + file_id + ".ogg", "file_size": 12}
        elif method_name in {"sendMessage", "editMessageText"}:
            assert request_data is not None
            content = request_data.parameters["text"]
            assert isinstance(content, str)
            if content.startswith("🎙️"):
                self.echoes.append(content)
                self.echo_entered.set()
                await asyncio.wait_for(self.echo_release.wait(), 10)
            result = {"message_id": 500, "date": 1, "chat": {"id": 12345, "type": "private"}, "text": "fixture"}
        else:
            result = True
        return 200, json.dumps({"ok": True, "result": result}).encode()


class WaitingModel(InterruptControlMixin):
    """Fake external model with the production redirect decision logic."""
    instances = []
    messages = []
    started = threading.Event()
    release = threading.Event()
    redirects = []

    def __init__(self, **kwargs):
        type(self).instances.append(self)
        self.tools = []
        self.model = "fixture-model"
        self.provider = "fixture-provider"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._model_request_active = threading.Event()
        self._supports_active_turn_redirect = True
        self._execution_thread_id = None
        self._pending_redirect = None

    @property
    def is_interrupted(self):
        return self._interrupt_requested

    def interrupt(
        self, message: str | None = None, *, hard_cancel: bool = False,
        tool_reason: str | None = None, require_generation: int | None = None,
    ) -> bool:
        self._interrupt_requested = True
        self._interrupt_message = message
        return True

    def redirect(self, text: str) -> bool:
        accepted = super().redirect(text)
        if accepted:
            type(self).redirects.append(text)
        return accepted

    def run_conversation(self, message, conversation_history=None, **kwargs):
        type(self).messages.append(message)
        if len(type(self).messages) == 1:
            self._model_request_active.set()
            type(self).started.set()
            assert type(self).release.wait(15), "test did not release model"
            self._model_request_active.clear()
        else:
            self._interrupt_requested = False
            self._interrupt_message = None
        return {
            "final_response": "fixture complete", "messages": [], "api_calls": 1,
            "interrupted": self._interrupt_requested, "interrupt_message": self._interrupt_message,
        }


def update(bot, number, *, text=None, voice=False):
    message = {"message_id": number, "date": 1,
               "chat": {"id": 12345, "type": "private", "first_name": "User"},
               "from": {"id": 12345, "is_bot": False, "first_name": "User"}}
    if text is not None:
        message["caption" if voice else "text"] = text
    if voice:
        file_id = "voice-a" if voice is True else str(voice)
        message["voice"] = {"file_id": file_id, "file_unique_id": "unique-" + file_id,
                            "duration": 1, "mime_type": "audio/ogg", "file_size": 11}
    return Update.de_json({"update_id": number, "message": message}, bot)


async def until(predicate, timeout=8):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


@asynccontextmanager
async def telegram_gateway(monkeypatch, tmp_path, busy_text_mode, echo):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: fixture-model\n  provider: openai\n"
        f"display:\n  busy_input_mode: interrupt\n  busy_text_mode: {busy_text_mode}\n"
        "  tool_progress: off\n  busy_ack_enabled: false\nsession_reset:\n  mode: none\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key, value in {
        "TELEGRAM_ALLOWED_USERS": "12345",
        "HERMES_GATEWAY_BUSY_INPUT_MODE": "interrupt",
        "HERMES_GATEWAY_BUSY_TEXT_MODE": busy_text_mode,
        "HERMES_GATEWAY_BUSY_ACK_ENABLED": "false",
        "HERMES_TOOL_PROGRESS_MODE": "off",
        "HERMES_GATEWAY_NOTIFY_INTERVAL": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda **kwargs: {"api_key": "fixture"})
    monkeypatch.setattr("run_agent.AIAgent", WaitingModel)
    monkeypatch.setattr("tools.tirith_security._install_tirith", lambda **kwargs: (None, "offline test"))
    WaitingModel.instances, WaitingModel.messages, WaitingModel.redirects = [], [], []
    WaitingModel.started, WaitingModel.release = threading.Event(), threading.Event()
    platform_config = PlatformConfig(enabled=True, token="999:fixture")
    runner = GatewayRunner(GatewayConfig(
        platforms={Platform.TELEGRAM: platform_config}, sessions_dir=hermes_home / "sessions",
        stt_echo_transcripts=echo,
    ))
    adapter = TelegramAdapter(platform_config)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._wire_adapter_handlers(adapter)
    transport = LocalTelegramRequest()
    app = (Application.builder().token("999:fixture").request(transport)
           .get_updates_request(LocalTelegramRequest()).build())
    adapter._app, adapter._bot = app, app.bot
    adapter._mark_connected()
    adapter._register_handlers(app)
    await app.initialize()
    assert app.concurrent_updates == 1
    errors = []

    async def record_error(_update, context):
        errors.append(context.error)

    app.add_error_handler(record_error)
    try:
        yield runner, adapter, app, transport, errors
    finally:
        WaitingModel.release.set()
        transport.echo_release.set()
        tasks = list(adapter._background_tasks) + list(adapter._pending_text_batch_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await app.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_text_mode", ["interrupt", "queue"])
@pytest.mark.parametrize("case,echo", [
    (case, echo)
    for case in ["active_a", "pending_a", "inflight_b", "buffered_text_stt", "buffered_text_echo"]
    for echo in [False, True] if echo or case != "buffered_text_echo"
])
async def test_telegram_voice_followups_reach_model_once(case, echo, busy_text_mode, monkeypatch, tmp_path):
    calls = []
    stt_entered, stt_release = threading.Event(), threading.Event()
    blocked_clip = "B" if case == "inflight_b" else "A" if case == "buffered_text_stt" else None
    if blocked_clip is None:
        stt_release.set()

    def transcribe(path, *args):
        clip = {b"OggSfixtureA": "A", b"OggSfixtureB": "B"}[Path(path).read_bytes()]
        calls.append(clip)
        if clip == blocked_clip:
            stt_entered.set()
            assert stt_release.wait(10), "test did not release STT"
        return {"success": True, "transcript": f"voice {clip} request", "provider": "fixture"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    async with telegram_gateway(monkeypatch, tmp_path, busy_text_mode, echo) as context:
        runner, adapter, app, transport, errors = context
        voice_task = None
        try:
            first = update(app.bot, 1, voice="voice-a") if case == "active_a" else update(app.bot, 1, text="older task X")
            await app.process_update(first)
            await until(WaitingModel.started.is_set)
            await until(lambda: any(isinstance(agent, WaitingModel) for _, agent in runner._running_agent_items()))
            assert not adapter._pending_messages
            assert not errors, errors
            if case.startswith("buffered_text"):
                added_text = "stop sending the pictures"
                await app.process_update(update(app.bot, 2, text=added_text))
                assert adapter._pending_text_batches
                if case == "buffered_text_echo":
                    transport.echo_release.clear()
                voice_task = asyncio.create_task(app.process_update(update(app.bot, 3, voice="voice-a")))
                await until(stt_entered.is_set if case.endswith("stt") else transport.echo_entered.is_set)
                await until(lambda: not adapter._pending_text_batch_tasks and not adapter._text_debounce_store())
                pending = next(iter(adapter._pending_messages.values()))
                if busy_text_mode == "queue":
                    assert added_text in pending.text
                else:
                    assert any(added_text in redirect for redirect in WaitingModel.redirects)
                stt_release.set()
                transport.echo_release.set()
                await asyncio.wait_for(voice_task, 10)
            else:
                if case != "active_a":
                    await app.process_update(update(app.bot, 2, voice="voice-a"))
                    assert len(adapter._pending_messages) == 1
                assert calls == ["A"]
                voice_task = asyncio.create_task(app.process_update(update(app.bot, 3, voice="voice-b")))
                if case == "inflight_b":
                    await until(stt_entered.is_set)
                    # The update is still awaiting B while the existing model turn drains A+B.
                    WaitingModel.release.set()
                    await until(lambda: not adapter._pending_messages)
                else:
                    await asyncio.wait_for(voice_task, 10)
                stt_release.set()
                await asyncio.wait_for(voice_task, 10)
            WaitingModel.release.set()
            await until(lambda: not adapter._background_tasks, timeout=10)
            assert not errors, errors
            assert len(WaitingModel.messages) == 2, WaitingModel.messages
            delivered = "\n".join(WaitingModel.messages + WaitingModel.redirects)
            expected_clips = ["A"] if case.startswith("buffered_text") else ["A", "B"]
            assert Counter(calls) == Counter(expected_clips), (calls, WaitingModel.messages)
            for clip in expected_clips:
                model_input = WaitingModel.messages[0 if case == "active_a" and clip == "A" else 1]
                assert model_input.count(f"voice {clip} request") == 1
            if case.startswith("buffered_text"):
                assert added_text in delivered
            elif case != "active_a":
                assert WaitingModel.messages[1].index("voice A request") < WaitingModel.messages[1].index("voice B request")
            assert Counter(transport.echoes) == Counter(
                [f'🎙️ "voice {clip} request"' for clip in expected_clips] if echo else []
            )
        finally:
            stt_release.set()
            transport.echo_release.set()
            if voice_task is not None:
                await asyncio.gather(voice_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", [None, "", "call-specific caption"])
@pytest.mark.parametrize("case", ["cold", "warm", "inflight", "cancelled", "error_retry", "failed_audio", "same_words", "during_echo", "legacy_photo"])
async def test_merged_voice_cache_preserves_clip_and_caption_ownership(case, user_text, monkeypatch, tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True, stt_echo_transcripts=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    adapter = SimpleNamespace(send=AsyncMock())
    paths = [str(tmp_path / f"{clip}.ogg") for clip in ("A", "B")]
    head, incoming = [MessageEvent(
        text=f"caption {clip}", message_type=MessageType.VOICE, source=source,
        media_urls=[path], media_types=["audio/ogg"],
    ) for clip, path in zip(("A", "B"), paths)]
    pending = {"conversation": head}
    late_text = MessageEvent(text="later text", message_type=MessageType.TEXT, source=source)
    entered, release = threading.Event(), threading.Event()
    calls = []
    fallback_calls = []
    tasks = []
    b_results = []
    merged_results = []
    blocking = case in {"inflight", "cancelled"}
    if not blocking:
        release.set()

    def transcribe(path, *args):
        calls.append(path)
        if path == paths[1] and blocking:
            entered.set()
            assert release.wait(10), "test did not release STT"
        if path == paths[1] and case == "failed_audio":
            return {"success": False, "error": "fixture failure"}
        words = "same words" if case == "same_words" else f"voice {Path(path).stem}"
        return {"success": True, "transcript": words, "provider": "fixture"}

    def fallback(path):
        fallback_calls.append(path)
        return {"success": False, "error": "fixture fallback failure"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio_local_fallback", fallback)
    try:
        if case in {"during_echo", "legacy_photo"}:
            # The echo await can admit another event before the pending input is returned.
            if case == "legacy_photo":
                head.media_types = [""]
                incoming.message_type = MessageType.PHOTO
                incoming.media_urls = [str(tmp_path / "photo.jpg")]
                incoming.media_types = ["image/jpeg"]

            async def merge_on_first_echo(*args, **kwargs):
                if adapter.send.await_count == 1:
                    merge_pending_message_event(pending, "conversation", incoming)
                    merge_pending_message_event(pending, "conversation", late_text)

            adapter.send.side_effect = merge_on_first_echo
            text, transcripts = await runner._transcribe_and_echo_pending_voice(
                head, adapter, source, head.text, metadata=None, log_context="test",
            )
            assert incoming.text in text
            assert late_text.text in text
            if case == "legacy_photo":
                assert calls == [paths[0]]
                assert transcripts == []
                return
            merged_results.append((text, transcripts))
        else:
            _, a_transcripts = await runner._transcribe_pending_audio_event_once(head)
            await runner._echo_pending_stt_transcripts_once(head, adapter, source, a_transcripts)
            if case == "warm":
                b_text, b_transcripts = await runner._transcribe_pending_audio_event_once(incoming)
                b_results.append((b_text, b_transcripts))
                await runner._echo_pending_stt_transcripts_once(incoming, adapter, source, b_transcripts)
            elif blocking:
                first = asyncio.create_task(runner._transcribe_pending_audio_event_once(incoming))
                tasks.append(first)
                await until(entered.is_set)
            merge_pending_message_event(pending, "conversation", incoming)
            if case == "error_retry":
                with monkeypatch.context() as transient:
                    transient.setattr(runner, "_enrich_message_with_transcription", AsyncMock(side_effect=RuntimeError("fixture task failure")))
                    with pytest.raises(RuntimeError, match="fixture task failure"):
                        await runner._transcribe_pending_audio_event_once(incoming)
            if blocking:
                draining = asyncio.create_task(runner._transcribe_pending_audio_event_once(head))
                tasks.append(draining)
                await asyncio.sleep(0)
                merge_pending_message_event(pending, "conversation", late_text)
                if case == "cancelled":
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(first, 2)
                    assert first.cancelled()
                    assert not release.is_set() and not draining.done()
                release.set()
                for task in tasks:
                    if case != "cancelled" or task is not first:
                        result = await task
                        (b_results if task is first else merged_results).append(result)

        override, b_transcripts = await runner._transcribe_pending_audio_event_once(incoming, user_text)
        b_results.append((override, b_transcripts))
        caption = incoming.text if user_text is None else user_text
        if caption:
            assert override.endswith(caption)
        else:
            assert incoming.text not in override
        expected_a = "same words" if case == "same_words" else "voice A"
        expected_b = [] if case == "failed_audio" else ["same words" if case == "same_words" else "voice B"]
        for b_text, transcripts in b_results:
            assert b_text is not None
            assert transcripts == expected_b
            assert "caption A" not in b_text and late_text.text not in b_text
            if case != "same_words":
                assert '"voice A"' not in b_text
            for transcript in expected_b:
                assert b_text.count(f'"{transcript}"') == 1
            if case == "failed_audio":
                assert "could not be transcribed" in b_text
        await runner._echo_pending_stt_transcripts_once(incoming, adapter, source, b_transcripts)
        canonical, all_transcripts = await runner._transcribe_pending_audio_event_once(head)
        await runner._echo_pending_stt_transcripts_once(head, adapter, source, all_transcripts)
        merged_results.append((canonical, all_transcripts))
        expected_all = [expected_a, *expected_b]
        for merged_text, transcripts in merged_results:
            assert merged_text is not None
            assert transcripts == expected_all
            assert "caption A" in merged_text and "caption B" in merged_text
            assert "call-specific caption" not in merged_text
            for transcript, count in Counter(expected_all).items():
                assert merged_text.count(f'"{transcript}"') == count
            if expected_b and case != "same_words":
                assert merged_text.index(f'"{expected_a}"') < merged_text.index(f'"{expected_b[0]}"')
            if blocking or case == "during_echo":
                assert late_text.text in merged_text
        assert Counter(calls) == Counter(paths)
        if case == "failed_audio":
            assert fallback_calls == [paths[1]]
            assert "could not be transcribed" in canonical
        assert adapter.send.await_count == len(all_transcripts)

        # Equal file paths do not join unrelated conversations or multiplexed profiles.
        for index, other_source in enumerate([
            SessionSource(platform=Platform.TELEGRAM, chat_id="67890", chat_type="dm"),
            SessionSource(platform=Platform.TELEGRAM, chat_id=source.chat_id, chat_type="dm", profile="other"),
        ], start=2):
            separate = MessageEvent(text="separate caption", message_type=MessageType.VOICE,
                                    source=other_source, media_urls=[paths[0]], media_types=["audio/ogg"])
            other_text, other_transcripts = await runner._transcribe_pending_audio_event_once(separate)
            await runner._echo_pending_stt_transcripts_once(separate, adapter, other_source, other_transcripts)
            assert "separate caption" in other_text and "caption B" not in other_text
            assert calls.count(paths[0]) == index
            assert adapter.send.call_args.args[0] == other_source.chat_id
            assert adapter.send.await_count == len(all_transcripts) + index - 1

    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
