"""Gateway integration tests for Telegram run-scoped status updates."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _StatusUpdateAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.status_callback("lifecycle", "Preparing context")
        self.tool_progress_callback("tool.started", "todo", "planning", {"todos": []})
        time.sleep(0.35)
        self.status_callback("lifecycle", "Still working")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class _LongProgressAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        for idx in range(20):
            self.tool_progress_callback(
                "tool.started",
                "web_search",
                f"old progress line {idx:02d}",
                {},
            )
        self.tool_progress_callback(
            "tool.started",
            "web_search",
            "current work: final query",
            {},
        )
        time.sleep(0.45)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class _InterimAssistantAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.status_callback = kwargs.get("status_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.interim_assistant_callback("You're welcome.", already_streamed=False)
        self.tool_progress_callback("tool.started", "todo", "planning", {"todos": []})
        self.status_callback("lifecycle", "Still working")
        time.sleep(0.35)
        return {
            "final_response": "You're welcome.",
            "response_previewed": True,
            "messages": [],
            "api_calls": 1,
        }


class _BaseCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.typing: list[dict] = []
        self._next_id = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._next_id += 1
        message_id = f"send-{self._next_id}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                "message_id": message_id,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, *, metadata=None) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _TelegramStatusAdapter(_BaseCaptureAdapter):
    def __init__(self):
        super().__init__(Platform.TELEGRAM)
        self.status_updates: list[dict] = []
        self.deleted: list[dict] = []

    async def send_or_update_status(
        self,
        chat_id,
        thread_id,
        status_key,
        text,
        metadata=None,
    ) -> SendResult:
        self.status_updates.append(
            {
                "chat_id": chat_id,
                "thread_id": thread_id,
                "status_key": status_key,
                "text": text,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="status-1")

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": message_id})
        return True


class _SlackEditableAdapter(_BaseCaptureAdapter):
    def __init__(self):
        super().__init__(Platform.SLACK)


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._session_run_generation = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=SimpleNamespace(enabled=False, transport="off"),
    )
    return runner


async def _run_gateway_agent(
    monkeypatch,
    tmp_path,
    adapter,
    platform: Platform,
    *,
    cleanup_progress: bool = False,
    agent_cls=_StatusUpdateAgent,
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    display_config = (
        {"platforms": {platform.value: {"cleanup_progress": True}}}
        if cleanup_progress
        else {}
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": display_config},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake", "model": "fake-model"},
    )

    runner = _make_runner(adapter)
    session_key = f"agent:main:{platform.value}:group:chat-1:thread-1"
    runner._session_run_generation[session_key] = 7
    source = SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type="group",
        thread_id="thread-1",
    )
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=session_key,
        run_generation=7,
        event_message_id="msg-1",
    )
    return result


@pytest.mark.asyncio
async def test_telegram_progress_and_status_share_one_status_key_without_normal_sends(monkeypatch, tmp_path):
    adapter = _TelegramStatusAdapter()

    result = await _run_gateway_agent(monkeypatch, tmp_path, adapter, Platform.TELEGRAM)

    assert result["final_response"] == "done"
    assert len(adapter.status_updates) >= 2
    assert {call["status_key"] for call in adapter.status_updates} == {
        "gateway-run:agent:main:telegram:group:chat-1:thread-1:7"
    }
    assert {call["thread_id"] for call in adapter.status_updates} == {"thread-1"}
    assert all(call["metadata"]["thread_id"] == "thread-1" for call in adapter.status_updates)
    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_telegram_interim_assistant_uses_normal_send_not_status_update(monkeypatch, tmp_path):
    adapter = _TelegramStatusAdapter()

    result = await _run_gateway_agent(
        monkeypatch,
        tmp_path,
        adapter,
        Platform.TELEGRAM,
        agent_cls=_InterimAssistantAgent,
    )

    assert result["final_response"] == "You're welcome."
    assert result.get("already_sent") is True
    assert [call["content"] for call in adapter.sent] == ["You're welcome."]
    assert all(call["text"] != "You're welcome." for call in adapter.status_updates)
    assert adapter.status_updates
    assert {call["status_key"] for call in adapter.status_updates} == {
        "gateway-run:agent:main:telegram:group:chat-1:thread-1:7"
    }


@pytest.mark.asyncio
async def test_telegram_status_cleanup_deletes_repeated_status_message_once(monkeypatch, tmp_path):
    adapter = _TelegramStatusAdapter()
    session_key = "agent:main:telegram:group:chat-1:thread-1"

    result = await _run_gateway_agent(
        monkeypatch,
        tmp_path,
        adapter,
        Platform.TELEGRAM,
        cleanup_progress=True,
    )

    assert result["final_response"] == "done"
    assert len(adapter.status_updates) >= 2
    cb = adapter.pop_post_delivery_callback(session_key)
    assert callable(cb)
    cb()
    for _ in range(20):
        await asyncio.sleep(0.01)
        if adapter.deleted:
            break

    assert adapter.deleted == [{"chat_id": "chat-1", "message_id": "status-1"}]


@pytest.mark.asyncio
async def test_telegram_status_update_progress_preserves_latest_line(monkeypatch, tmp_path):
    adapter = _TelegramStatusAdapter()
    adapter.STATUS_MESSAGE_LENGTH = 120

    result = await _run_gateway_agent(
        monkeypatch,
        tmp_path,
        adapter,
        Platform.TELEGRAM,
        agent_cls=_LongProgressAgent,
    )

    assert result["final_response"] == "done"
    assert adapter.status_updates
    latest_render = adapter.status_updates[-1]["text"]
    assert "current work: final query" in latest_render
    assert "old progress line 00" not in latest_render
    assert len(latest_render) <= adapter.STATUS_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_non_telegram_progress_and_status_still_use_normal_send(monkeypatch, tmp_path):
    adapter = _SlackEditableAdapter()

    result = await _run_gateway_agent(monkeypatch, tmp_path, adapter, Platform.SLACK)

    assert result["final_response"] == "done"
    rendered = "\n".join(call["content"] for call in adapter.sent)
    assert "Preparing context" in rendered
    assert "Still working" in rendered
    assert any("todo" in call["content"] for call in adapter.sent + adapter.edits)


@pytest.mark.asyncio
async def test_final_response_delivery_remains_normal_send_new_behavior():
    adapter = _TelegramStatusAdapter()

    async def handler(_event):
        return "Final response"

    adapter.set_message_handler(handler)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
        thread_id="thread-1",
    )
    event = MessageEvent(text="hi", source=source, message_id="msg-1")

    await adapter._process_message_background(event, "session-key")

    assert adapter.status_updates == []
    assert [call["content"] for call in adapter.sent] == ["Final response"]
    assert adapter.sent[0]["metadata"]["thread_id"] == "thread-1"
    assert adapter.sent[0]["metadata"]["notify"] is True
