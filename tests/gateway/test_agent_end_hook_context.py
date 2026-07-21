"""Tests for #45721 — agent:end hook should expose full_response and full_message.

The ``agent:end`` hook context previously only provided ``response`` truncated
to 500 chars and ``message`` truncated to 500 chars.  This test verifies that
``full_response`` and ``full_message`` fields are now emitted alongside the
truncated versions.
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup for hook context testing."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-hook",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event(text="hello world"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-42",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _agent_result(response_text: str):
    """Build a successful agent result dict."""
    return {
        "final_response": response_text,
        "messages": [],
        "api_calls": 1,
        "last_prompt_tokens": 100,
        "history_offset": 0,
    }


def _get_agent_end_context(runner) -> dict:
    """Extract the context dict from the agent:end hook emit call."""
    for call in runner.hooks.emit.call_args_list:
        args, kwargs = call.args, call.kwargs
        if args and args[0] == "agent:end":
            return args[1] if len(args) > 1 else kwargs
    return {}


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_end_includes_full_response(monkeypatch, tmp_path):
    """agent:end context must include untruncated full_response."""
    runner = _bootstrap(monkeypatch, tmp_path)
    long_response = "A" * 600  # >500 chars
    runner._run_agent = AsyncMock(return_value=_agent_result(long_response))

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    ctx = _get_agent_end_context(runner)
    assert ctx, "agent:end hook was not emitted"
    assert "full_response" in ctx, "full_response missing from agent:end context"
    assert ctx["full_response"] == long_response
    assert ctx["response"] == "A" * 500, "response should still be truncated"


@pytest.mark.asyncio
async def test_agent_end_includes_full_message(monkeypatch, tmp_path):
    """agent:end context must include untruncated full_message."""
    runner = _bootstrap(monkeypatch, tmp_path)
    long_message = "B" * 600  # >500 chars
    runner._run_agent = AsyncMock(return_value=_agent_result("ok"))

    await runner._handle_message_with_agent(
        _event(text=long_message), _source(),
        "agent:main:telegram:group:-1001:12345", 1,
    )

    ctx = _get_agent_end_context(runner)
    assert ctx, "agent:end hook was not emitted"
    assert "full_message" in ctx, "full_message missing from agent:end context"
    assert ctx["full_message"] == long_message
    assert ctx["message"] == "B" * 500, "message should still be truncated"


@pytest.mark.asyncio
async def test_agent_end_truncated_fields_backward_compat(monkeypatch, tmp_path):
    """Existing truncated response/message fields remain unchanged."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_agent_result("short reply"))

    await runner._handle_message_with_agent(
        _event(text="hi"), _source(),
        "agent:main:telegram:group:-1001:12345", 1,
    )

    ctx = _get_agent_end_context(runner)
    assert ctx["response"] == "short reply"
    assert ctx["message"] == "hi"
    assert ctx["full_response"] == "short reply"
    assert ctx["full_message"] == "hi"


@pytest.mark.asyncio
async def test_agent_end_full_response_matches_normalized_error(monkeypatch, tmp_path):
    """full_response must capture the normalized error message, not None."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "",
            "messages": [],
            "api_calls": 0,
            "last_prompt_tokens": 0,
            "history_offset": 0,
            "failed": True,
            "error": "test error",
        }
    )

    await runner._handle_message_with_agent(
        _event(text="hi"), _source(),
        "agent:main:telegram:group:-1001:12345", 1,
    )

    ctx = _get_agent_end_context(runner)
    # The normalized response replaces "" with an error message.
    # full_response should capture that normalized text.
    assert ctx.get("full_response") is not None
    assert ctx.get("full_response") == ctx.get("response")
