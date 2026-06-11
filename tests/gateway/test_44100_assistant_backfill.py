"""Gateway-side tests for issue #44100 — delivered responses must reach state.db.

``state.db`` is the canonical transcript store (spec 002), so
``append_to_transcript(..., skip_db=True)`` is a complete no-op. When an
agent recovery path delivers a final response without appending an
assistant message to its message list, the agent's own DB flush writes
only the user turn — and the gateway's fallback writes were all skipped
via ``skip_db=agent_persisted``. The delivered response was silently
dropped, and the model re-answered every "unanswered" user message on
the next turn.

Pins the contract:

1. When the turn's new messages contain no assistant text, the gateway
   backfills the delivered response with ``skip_db=False``.
2. When the turn DID produce an assistant message, no backfill happens
   (the #860 / #42039 duplicate-write protection stays intact).
3. The ``not new_messages`` fallback writes the assistant response with
   ``skip_db=False`` (a response generated this turn cannot already be
   in the loaded history).
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
    """Minimal GatewayRunner setup (mirrors test_42039_duplicate_user_message)."""
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
        session_id="sess-44100",
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
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event():
    return MessageEvent(
        text="hello world",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-44100",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _assistant_calls(calls):
    out = []
    for call in calls:
        args = call.args
        if len(args) >= 2 and isinstance(args[1], dict):
            if args[1].get("role") == "assistant":
                out.append(call)
    return out


@pytest.mark.asyncio
async def test_backfill_when_turn_has_no_assistant_message(monkeypatch, tmp_path):
    """Recovery turn: agent delivered a response but its message list ends
    at the user message. The gateway must persist the response itself."""
    runner = _bootstrap(monkeypatch, tmp_path)

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Recovered streamed answer.",
            "messages": [{"role": "user", "content": "hello world"}],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    calls = _assistant_calls(
        runner.session_store.append_to_transcript.call_args_list
    )
    backfills = [
        c for c in calls
        if c.kwargs.get("skip_db", False) is False
        and c.args[1].get("content") == "Recovered streamed answer."
    ]
    assert len(backfills) == 1, (
        "Delivered response missing from the agent transcript must be "
        "backfilled with skip_db=False — skip_db=True writes are no-ops "
        "now that state.db is the canonical store (#44100). "
        f"assistant calls: {[(c.args[1].get('content'), c.kwargs) for c in calls]}"
    )


@pytest.mark.asyncio
async def test_no_backfill_when_assistant_message_present(monkeypatch, tmp_path):
    """Normal turn: the agent persisted the assistant message itself.
    No skip_db=False assistant write may happen (#860 / #42039)."""
    runner = _bootstrap(monkeypatch, tmp_path)

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    calls = _assistant_calls(
        runner.session_store.append_to_transcript.call_args_list
    )
    assert calls, "expected the assistant message to be written to the transcript"
    for call in calls:
        assert call.kwargs.get("skip_db", False) is True, (
            "When the agent already persisted the assistant message, the "
            "gateway must not double-write it (#860 / #42039). "
            f"kwargs: {call.kwargs}"
        )


@pytest.mark.asyncio
async def test_backfill_in_tool_turn_without_final_assistant_text(
    monkeypatch, tmp_path
):
    """Recovery in a tool-calling turn: assistant(tool_calls) + tool result
    but no final assistant text → response must still be backfilled."""
    runner = _bootstrap(monkeypatch, tmp_path)

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Recovered after tools.",
            "messages": [
                {"role": "user", "content": "hello world"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}],
                },
                {"role": "tool", "content": "result", "tool_call_id": "c1"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    calls = _assistant_calls(
        runner.session_store.append_to_transcript.call_args_list
    )
    backfills = [
        c for c in calls
        if c.kwargs.get("skip_db", False) is False
        and c.args[1].get("content") == "Recovered after tools."
    ]
    assert len(backfills) == 1, (
        "A tool-call assistant message with no text does not carry the "
        "delivered response — the gateway must still backfill it. "
        f"assistant calls: {[(c.args[1].get('content'), c.kwargs) for c in calls]}"
    )


@pytest.mark.asyncio
async def test_not_new_messages_fallback_persists_assistant(monkeypatch, tmp_path):
    """``not new_messages`` edge case: the assistant response was generated
    this turn and cannot be in loaded history — it must be written with
    skip_db=False (the user entry keeps its #42039 protection)."""
    runner = _bootstrap(monkeypatch, tmp_path)

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "history_offset": 1,  # equals len(messages) → new_messages=[]
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    calls = _assistant_calls(
        runner.session_store.append_to_transcript.call_args_list
    )
    assert calls, "expected an assistant fallback write"
    for call in calls:
        assert call.kwargs.get("skip_db", True) is False, (
            "The not-new-messages assistant fallback must not skip the DB "
            "write — the response was never persisted by the agent (#44100). "
            f"kwargs: {call.kwargs}"
        )
