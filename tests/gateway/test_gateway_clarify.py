"""Tests for gateway Clarify handling on messaging platforms."""

import asyncio
import sys
import types
from threading import Event, Lock
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


class _DummyAdapter:
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self):
        self.sent = []
        self._pending_messages = {}
        self.cleared_clarify = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return MagicMock(success=True, message_id="1")

    async def edit_message(self, chat_id, message_id, content, metadata=None):
        return MagicMock(success=True, message_id=message_id)

    def clear_clarify(self, session_key):
        self.cleared_clarify.append(session_key)
        return 1

    def get_pending_message(self, session_key):
        return None

    def has_pending_interrupt(self, session_key):
        return False

    def clear_interrupt(self, session_key):
        return None


def _make_source(user_id="user-1", chat_id="12345", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_name="Telegram",
        chat_type="dm",
        user_id=user_id,
        thread_id=thread_id,
    )


def _make_event(text: str, user_id="user-1", chat_id="12345", thread_id=None):
    return MessageEvent(text=text, source=_make_source(user_id=user_id, chat_id=chat_id, thread_id=thread_id))


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _DummyAdapter()}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_clarify = {}
    runner._pending_clarify_lock = Lock()
    runner._session_db = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._session_key_for_source = lambda source: f"telegram:{source.chat_id}:{source.thread_id or 'root'}"
    return runner


@pytest.mark.asyncio
async def test_run_agent_assigns_gateway_clarify_callback(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("", encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    result = await runner._run_agent(
        message="ping",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="telegram:12345:root",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert callable(_CapturingAgent.last_init["clarify_callback"])


def test_resolve_gateway_clarify_action_other_keeps_prompt_pending():
    runner = _make_runner()
    event = Event()
    runner._pending_clarify["telegram:12345:root"] = {
        "event": event,
        "user_id": "user-1",
        "choices": ["Red", "Blue"],
        "awaiting_text": False,
        "response": None,
    }

    status = runner._resolve_gateway_clarify_action(
        "telegram:12345:root",
        action="other",
        user_id="user-1",
        user_name="Rahim",
    )

    assert status == "awaiting_text"
    assert runner._pending_clarify["telegram:12345:root"]["awaiting_text"] is True
    assert event.is_set() is False


def test_handle_pending_clarify_response_maps_numeric_choice():
    runner = _make_runner()
    event = Event()
    runner._pending_clarify["telegram:12345:root"] = {
        "event": event,
        "user_id": "user-1",
        "choices": ["Red", "Blue"],
        "awaiting_text": False,
        "response": None,
    }

    reply = runner._handle_pending_clarify_response(_make_event("2"))

    assert "Clarification received" in reply
    assert event.is_set() is True
    assert runner._pending_clarify["telegram:12345:root"]["response"] == "Blue"
    assert runner.adapters[Platform.TELEGRAM].cleared_clarify == ["telegram:12345:root"]


def test_handle_pending_clarify_response_rejects_wrong_user():
    runner = _make_runner()
    event = Event()
    runner._pending_clarify["telegram:12345:root"] = {
        "event": event,
        "user_id": "user-1",
        "choices": ["Red", "Blue"],
        "awaiting_text": False,
        "response": None,
    }

    reply = runner._handle_pending_clarify_response(_make_event("Blue", user_id="user-2"))

    assert "original requester" in reply
    assert event.is_set() is False
    assert runner._pending_clarify["telegram:12345:root"]["response"] is None


def test_handle_pending_clarify_response_ignores_already_resolved_prompt():
    runner = _make_runner()
    event = Event()
    runner._pending_clarify["telegram:12345:root"] = {
        "event": event,
        "user_id": "user-1",
        "choices": ["Red", "Blue"],
        "awaiting_text": False,
        "response": "Red",
        "resolved": True,
    }

    reply = runner._handle_pending_clarify_response(_make_event("Blue"))

    assert reply is None
    assert runner._pending_clarify["telegram:12345:root"]["response"] == "Red"
    assert runner.adapters[Platform.TELEGRAM].cleared_clarify == []


@pytest.mark.asyncio
async def test_gateway_clarify_timeout_clears_runner_and_adapter_state():
    runner = _make_runner()
    session_key = "telegram:12345:root"
    callback = runner._build_gateway_clarify_callback(
        adapter=runner.adapters[Platform.TELEGRAM],
        source=_make_source(),
        session_key=session_key,
        loop=asyncio.get_running_loop(),
        metadata=None,
        timeout_seconds=0.05,
    )

    result = await asyncio.to_thread(callback, "Need more detail", None)

    assert "timed out" in result
    assert session_key not in runner._pending_clarify
    assert runner.adapters[Platform.TELEGRAM].cleared_clarify == [session_key]
