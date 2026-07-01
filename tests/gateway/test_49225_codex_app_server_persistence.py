"""Tests for #49225 — codex app-server gateway turns persisted nowhere.

The ``codex_app_server`` runtime path (``run_codex_app_server_turn`` in
``agent/codex_runtime.py``) is an early-return that bypasses
``conversation_loop`` and therefore never calls
``_flush_messages_to_session_db()``.  The gateway, meanwhile, hard-coded

    agent_persisted = self._session_db is not None   # always True

and passed ``skip_db=agent_persisted`` to every ``append_to_transcript``
call, assuming the agent self-persisted.  For codex that assumption is
false, so codex turn messages were written to neither location and
``session_search`` (FTS over ``state.db``) returned nothing.

The fix is a three-part signal:

1. ``run_codex_app_server_turn`` returns ``agent_persisted=False``.
2. The gateway reads ``agent_result.get("agent_persisted", ...)`` instead
   of hard-coding ``True``.
3. ``_run_agent``'s rebuilt result dict passes the flag through.

These tests lock in all three, plus the standard-runtime regression
(no duplicate write, #860 / #42039):

- Codex layer: ``run_codex_app_server_turn`` sets ``agent_persisted=False``.
- Gateway: an ``agent_result`` with ``agent_persisted=False`` and a
  non-None ``_session_db`` writes user messages with ``skip_db=False``
  (gateway persists → session_search sees them).
- Regression: an ``agent_result`` WITHOUT the key (standard runtime) keeps
  ``skip_db=True`` when ``_session_db`` is set — existing behaviour, no
  duplicate write.
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


# ── Codex-layer test: run_codex_app_server_turn returns agent_persisted=False


def test_codex_app_server_turn_signals_not_self_persisted():
    """The codex success return must carry agent_persisted=False so the
    gateway knows it has to write the turn to state.db itself."""
    from agent.codex_runtime import run_codex_app_server_turn
    from agent.transports.codex_app_server_session import TurnResult

    turn = TurnResult(
        final_text="hi there",
        projected_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        tool_iterations=0,
        interrupted=False,
        error=None,
        turn_id="turn-1",
        thread_id="thread-1",
    )

    agent = MagicMock()
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = turn
    # Numeric session counters so _record_codex_app_server_usage's
    # in-place accumulation doesn't blow up on MagicMock arithmetic.
    for attr in (
        "session_prompt_tokens", "session_completion_tokens",
        "session_total_tokens", "session_input_tokens",
        "session_output_tokens", "session_cache_read_tokens",
        "session_cache_write_tokens", "session_reasoning_tokens",
        "session_estimated_cost_usd", "_iters_since_skill",
    ):
        setattr(agent, attr, 0)
    agent.context_compressor = None
    agent._session_db = None          # skip token-persistence DB branch
    agent.session_id = None
    agent._skill_nudge_interval = 0   # no skill review
    agent.valid_tool_names = set()

    messages = [{"role": "user", "content": "hello"}]

    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=messages,
        effective_task_id="task-1",
        should_review_memory=False,
    )

    assert result["agent_persisted"] is False, (
        "codex app-server turn must signal it did NOT self-persist so the "
        "gateway writes the turn to state.db (fixes #49225 starved recall)"
    )


# ── Gateway harness (mirrors tests/gateway/test_42039_duplicate_user_message)


def _bootstrap(monkeypatch, tmp_path):
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
        session_id="sess-codex",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
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
        message_id="msg-49225",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _assert_user_call_has_skip_db(calls, expected_skip_db: bool):
    user_calls = []
    for call in calls:
        args = call.args
        if len(args) >= 2 and isinstance(args[1], dict):
            if args[1].get("role") == "user":
                user_calls.append(call)
    assert len(user_calls) >= 1, (
        f"Expected at least one user-role append_to_transcript call, "
        f"got calls: {[c.args for c in calls if len(c.args) >= 2]}"
    )
    for call in user_calls:
        actual = call.kwargs.get("skip_db", False)
        assert actual == expected_skip_db, (
            f"Expected skip_db={expected_skip_db} for user-role call, "
            f"got skip_db={actual}. kwargs={call.kwargs}"
        )


# ── Gateway test: codex (agent_persisted=False) → gateway writes to DB ──


@pytest.mark.asyncio
async def test_codex_agent_persisted_false_writes_to_db(monkeypatch, tmp_path):
    """When the agent result carries agent_persisted=False (codex path),
    the gateway must NOT skip the DB write even though _session_db is set —
    otherwise codex turns land nowhere and session_search is blind."""
    runner = _bootstrap(monkeypatch, tmp_path)
    assert runner._session_db is not None  # standard hard-coded value would be True

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
            "agent_persisted": False,  # ← codex signal
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, False
    )


# ── Regression: standard runtime (no key) keeps skip_db=True ──────────────


@pytest.mark.asyncio
async def test_standard_runtime_missing_key_keeps_skip_db(monkeypatch, tmp_path):
    """Standard runtime results do NOT set agent_persisted; the gateway must
    fall back to `self._session_db is not None` (True) and skip the DB write,
    preserving the no-duplicate-write behaviour of #860 / #42039."""
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
            # no agent_persisted key — standard runtime
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    _assert_user_call_has_skip_db(
        runner.session_store.append_to_transcript.call_args_list, True
    )
