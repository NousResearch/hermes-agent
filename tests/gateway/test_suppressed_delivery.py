"""Generic per-turn delivery suppression contract."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _event(*, delivery_mode=None):
    return MessageEvent(
        text="model input",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="dm"),
        delivery_mode=delivery_mode,
    )


def _adapter():
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=True),
        Platform.SLACK,
    )
    adapter.send_typing = AsyncMock(return_value=None)
    adapter.stop_typing = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=None)
    adapter.send = AsyncMock(return_value=None)
    adapter._message_handler = AsyncMock(return_value="public response")
    return adapter


def _session_key():
    return build_session_key(_event().source)


@pytest.mark.asyncio
async def test_suppress_runs_handler_without_typing_or_final_delivery():
    adapter = _adapter()
    key = _session_key()
    event = _event(delivery_mode="suppress")
    adapter._active_sessions[key] = asyncio.Event()

    await adapter._process_message_background(event, key)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter.send_typing.assert_not_awaited()
    adapter.stop_typing.assert_not_awaited()
    adapter._send_with_retry.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_event_after_suppressed_event_delivers_normally():
    adapter = _adapter()
    key = _session_key()
    adapter._active_sessions[key] = asyncio.Event()
    await adapter._process_message_background(_event(delivery_mode="suppress"), key)

    adapter._active_sessions[key] = asyncio.Event()
    await adapter._process_message_background(_event(), key)

    assert adapter._message_handler.await_count == 2
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_suppress_drops_errors_and_post_delivery_callbacks():
    adapter = _adapter()
    key = _session_key()
    callback = AsyncMock()
    adapter._post_delivery_callbacks[key] = callback
    adapter._message_handler.side_effect = RuntimeError("private failure")
    adapter._active_sessions[key] = asyncio.Event()

    await adapter._process_message_background(_event(delivery_mode="suppress"), key)

    adapter.send.assert_not_awaited()
    adapter._send_with_retry.assert_not_awaited()
    callback.assert_not_awaited()


class _SurfaceCaptureAdapter(_StubAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="t", typing_indicator=True),
            Platform.SLACK,
        )
        self.sent = []
        self.edits = []

    async def send(self, chat_id, content, **kwargs):
        self.sent.append(content)
        return SendResult(success=True, message_id="sent-1")

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append(content)
        return SendResult(success=True, message_id=message_id)


class _CallbackAgent:
    init_kwargs = None
    run_kwargs = None
    tool_executed = False
    last_instance = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        type(self).last_instance = self
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self):
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).run_kwargs = {"message": message, **kwargs}
        type(self).tool_executed = True
        progress = type(self).init_kwargs.get("tool_progress_callback")
        if progress:
            progress("tool.started", "terminal", "secret progress", {"command": "true"})
        interim = type(self).init_kwargs.get("interim_assistant_callback")
        if interim:
            interim("secret interim")
        stream = type(self).init_kwargs.get("stream_delta_callback")
        if stream:
            stream("secret stream")
        time.sleep(0.35)
        return {
            "final_response": "secret final",
            "messages": [{"role": "user", "content": message}],
            "api_calls": 1,
        }


def _runner_for_surfaces(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=True,
        streaming=SimpleNamespace(
            enabled=True,
            transport="edit",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0,
        ),
    )
    return runner


@pytest.mark.asyncio
async def test_suppress_runs_agent_and_tool_without_progress_interim_stream_or_edits(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CallbackAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"interim_assistant_messages": True, "thinking_progress": True}},
    )

    adapter = _SurfaceCaptureAdapter()
    runner = _runner_for_surfaces(adapter)
    source = _event(delivery_mode="suppress").source

    result = await runner._run_agent(
        message="full private model input",
        context_prompt="",
        history=[],
        source=source,
        session_id="suppressed-session",
        session_key="agent:main:slack:dm:C123",
        persist_user_message="[private inbound event]",
        delivery_suppressed=True,
    )

    assert result["final_response"] == "secret final"
    assert _CallbackAgent.tool_executed is True
    assert _CallbackAgent.run_kwargs["message"] == "full private model input"
    assert _CallbackAgent.run_kwargs["persist_user_message"] == "[private inbound event]"
    assert adapter.sent == []
    assert adapter.edits == []
    assert _CallbackAgent.last_instance.status_callback is None
    assert _CallbackAgent.last_instance.notice_callback is None
    assert _CallbackAgent.last_instance.clarify_callback is None


@pytest.mark.asyncio
async def test_suppress_transcribes_voice_without_echo(monkeypatch):
    adapter = _SurfaceCaptureAdapter()
    runner = _runner_for_surfaces(adapter)
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=("private transcript", ["private transcript"])
    )
    runner._should_echo_stt_transcripts = lambda: True
    runner._thread_metadata_for_source = lambda *_args: None
    runner._reply_anchor_for_event = lambda _event: None
    monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: "session")

    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_event().source,
        media_urls=["/tmp/private.ogg"],
        media_types=["audio/ogg"],
        delivery_mode="suppress",
    )

    text = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
        session_key="session",
    )

    assert text == "private transcript"
    assert adapter.sent == []