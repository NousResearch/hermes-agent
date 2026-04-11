"""Regression tests for failed gateway turns and transcript persistence."""

import sys
import threading
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


class _FakeAgent:
    next_result = None

    def __init__(self, *args, **kwargs):
        self.tools = []
        self.model = kwargs.get("model")
        self.session_id = kwargs.get("session_id")
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
    ):
        result = dict(type(self).next_result or {})
        result.setdefault("messages", [])
        result.setdefault("api_calls", 1)
        return result

    def interrupt(self, _pending_text=None):
        return None


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner(*, history):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.config.streaming = SimpleNamespace(enabled=False, transport="off")
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.has_pending_interrupt = lambda _session_key: False
    adapter.get_pending_message = lambda _session_key: None
    adapter.queue_message = lambda _session_key, _message: None
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = None
    runner._smart_model_routing = None
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_model_notes = {}
    runner._effective_model = None
    runner._effective_provider = None
    runner._model = None
    runner._base_url = None
    runner._service_tier = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = lambda _tokens: None
    runner._get_or_create_gateway_honcho = lambda _session_key: (None, None)
    runner._load_reasoning_config = lambda: None
    runner._load_service_tier = lambda: None
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "fake-model",
        {
            "api_key": "***",
            "base_url": "https://example.invalid/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )
    runner._resolve_turn_agent_config = (
        lambda _message, model, runtime: {
            "model": model,
            "runtime": runtime,
            "request_overrides": None,
        }
    )
    runner._agent_config_signature = lambda *_args: ("sig",)
    runner._is_intentional_model_switch = lambda *_args, **_kwargs: False
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    runner._enrich_message_with_vision = AsyncMock(side_effect=lambda text, *_a, **_kw: text)
    runner._enrich_message_with_transcription = AsyncMock(
        side_effect=lambda text, *_a, **_kw: text
    )
    runner._evict_cached_agent = lambda *_a, **_kw: None
    return runner


def _install_fake_agent(monkeypatch, result):
    import gateway.run as gateway_run

    _FakeAgent.next_result = dict(result)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"tool_progress": "off"}},
    )


def _persisted_roles(runner):
    return [
        call.args[1]["role"]
        for call in runner.session_store.append_to_transcript.call_args_list
    ]


@pytest.mark.asyncio
async def test_run_agent_wrapper_preserves_no_response_failure_metadata(monkeypatch):
    runner = _make_runner(history=[{"role": "user", "content": "earlier"}])
    _install_fake_agent(
        monkeypatch,
        {
            "failed": True,
            "final_response": None,
            "messages": [],
            "api_calls": 1,
            "error": "429 rate limit exceeded",
            "error_reason": "rate_limit",
        },
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=runner.session_store.load_transcript.return_value,
        source=_make_source(),
        session_id="sess-1",
        session_key=build_session_key(_make_source()),
    )

    assert "429 rate limit exceeded" in result["final_response"]
    assert result["has_direct_response"] is False
    assert result["failed"] is True
    assert result["error"] == "429 rate limit exceeded"
    assert result["error_reason"] == "rate_limit"


@pytest.mark.asyncio
async def test_auth_failure_persists_user_turn_and_surfaces_error(monkeypatch):
    runner = _make_runner(history=[{"role": "user", "content": "earlier"}])
    _install_fake_agent(
        monkeypatch,
        {
            "failed": True,
            "final_response": None,
            "messages": [],
            "api_calls": 1,
            "error": "401 invalid api key",
            "error_reason": "auth",
        },
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result == (
        "The request failed: 401 invalid api key\n"
        "Try again or use /reset to start a fresh session."
    )
    assert _persisted_roles(runner) == ["user", "assistant"]


@pytest.mark.asyncio
async def test_rate_limit_failure_persists_user_turn_and_does_not_show_compact_guidance(monkeypatch):
    runner = _make_runner(history=[{"role": "user", "content": "earlier"}] * 60)
    _install_fake_agent(
        monkeypatch,
        {
            "failed": True,
            "final_response": None,
            "messages": [],
            "api_calls": 1,
            "error": "429 rate limit exceeded",
            "error_reason": "rate_limit",
        },
    )

    result = await runner._handle_message(_make_event("hello"))

    assert "Session too large for the model's context window." not in result
    assert "429 rate limit exceeded" in result
    assert _persisted_roles(runner) == ["user", "assistant"]


@pytest.mark.asyncio
async def test_context_overflow_skips_persistence_and_shows_compact_guidance(monkeypatch):
    runner = _make_runner(history=[{"role": "user", "content": "earlier"}] * 60)
    _install_fake_agent(
        monkeypatch,
        {
            "final_response": None,
            "messages": [],
            "api_calls": 1,
            "error": "Context length exceeded: max compression attempts (3) reached.",
            "partial": True,
            "error_reason": "context_overflow",
        },
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result == (
        "⚠️ Session too large for the model's context window.\n"
        "Use /compact to compress the conversation, or /reset to start fresh."
    )
    assert runner.session_store.append_to_transcript.call_count == 0
