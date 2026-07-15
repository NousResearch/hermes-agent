"""Tests for #42039 — user messages stored twice in state.db.

When the agent has its own SessionDB reference (``_session_db is not None``),
``_flush_messages_to_session_db()`` persists messages to SQLite during the
agent run.  The gateway's ``append_to_transcript()`` must then use
``skip_db=True`` on all fallback paths to prevent writing a second copy
to the same SQLite file.

This test covers the two fallback paths that previously lacked
``skip_db=agent_persisted``:

1. ``agent_failed_early`` path — transient 429/timeout failures
2. ``not new_messages`` path — edge case where ``history_offset`` exceeds
   the actual message count
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
    """Minimal GatewayRunner setup shared by all tests in this module."""
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
        session_id="sess-dedup",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    # Mock has_platform_message_id to return False so the dedupe guard
    # (#47237) in gateway/run.py does not skip the append_to_transcript call.
    runner.session_store.has_platform_message_id.return_value = False
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
        message_id="msg-42",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _assert_user_call_has_skip_db(calls, expected_skip_db: bool):
    """Find append_to_transcript calls with role='user' and check skip_db."""
    user_calls = []
    for call in calls:
        args = call.args
        if len(args) >= 2 and isinstance(args[1], dict):
            if args[1].get("role") == "user":
                user_calls.append(call)
    assert len(user_calls) >= 1, (
        f"Expected at least one user-role append_to_transcript call, "
        f"got calls: {[c.args for c in calls if len(c.args)>=2]}"
    )
    for call in user_calls:
        actual = call.kwargs.get("skip_db", False)
        assert actual == expected_skip_db, (
            f"Expected skip_db={expected_skip_db} for user-role call, "
            f"got skip_db={actual}. kwargs={call.kwargs}"
        )


# ── Test 1: agent_failed_early path uses skip_db=True ─────────────────


@pytest.mark.asyncio
async def test_agent_failed_early_skip_db_when_agent_has_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    # Agent fails with transient 429
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": None,
            "error": "429 Too Many Requests — rate limit exceeded",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, True
    )


# ── Test 2: agent_failed_early with no _session_db → skip_db not True ─


@pytest.mark.asyncio
async def test_agent_failed_early_no_skip_db_when_no_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._session_db = None  # No agent DB → agent_persisted=False

    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "final_response": None,
            "error": "ReadTimeout: timed out",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, False
    )


# ── Test 3: not-new-messages path uses skip_db=True ───────────────────


@pytest.mark.asyncio
async def test_not_new_messages_skip_db_when_agent_has_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    # Agent succeeds but history_offset equals messages length → no new messages
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

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, True
    )


# ── Test 4: normal path (new_messages found) uses skip_db=True ────────


@pytest.mark.asyncio
async def test_normal_path_skip_db_when_agent_has_session_db(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    # Agent succeeds with new messages
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [
                {"role": "user", "content": "hi"},
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

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, True
    )


@pytest.mark.asyncio
async def test_nonempty_rate_limit_error_is_not_persisted_as_assistant_message(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)

    error_response = "API call failed after 3 retries: 429 Too Many Requests"
    runner._run_agent = AsyncMock(
        return_value={
            "failed": True,
            "failure_reason": "rate_limit",
            "completed": False,
            "final_response": error_response,
            "error": "429 Too Many Requests",
            "messages": [
                {"role": "user", "content": "hello world"},
            ],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    persisted_entries = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) >= 2 and isinstance(call.args[1], dict)
    ]

    assert any(entry.get("role") == "user" for entry in persisted_entries)
    assert not any(
        entry.get("role") == "assistant"
        and error_response in str(entry.get("content", ""))
        for entry in persisted_entries
    )
