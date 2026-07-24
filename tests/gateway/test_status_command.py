from hermes_state import AsyncSessionDB
"""Tests for gateway /status behavior and token persistence."""

from datetime import datetime
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, *, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry, *, platform: Platform = Platform.TELEGRAM):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = None
    # Default: no DB row → /status reports 0 tokens.  Tests that exercise
    # the populated path override this.
    runner._session_db._db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_status_command_reports_running_agent_without_interrupt(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    # Token total comes from the SQLite SessionDB, not SessionEntry.
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 200,
        "output_tokens": 121,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "**Session ID:** `sess-1`" in result
    assert "**Cumulative API tokens (re-sent each call):** 321" in result
    assert "**Agent Running:** Yes ⚡" in result
    assert "**Title:**" not in result
    running_agent.interrupt.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_status_command_includes_session_title_when_present():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session_title.return_value = "My titled session"

    result = await runner._handle_message(_make_event("/status"))

    assert "**Session ID:** `sess-1`" in result
    assert "**Title:** My titled session" in result


@pytest.mark.asyncio
async def test_status_command_reads_token_totals_from_session_db():
    """Regression test for #17158: /status must source token totals from the
    SQLite SessionDB (where run_agent.py persists them) and sum all component
    counts, not from SessionEntry (which the agent never writes)."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,  # SessionEntry never gets written to — always 0.
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 500,
        "cache_write_tokens": 100,
        "reasoning_tokens": 50,
    }

    result = await runner._handle_message(_make_event("/status"))

    # 1000 + 250 + 500 + 100 + 50 = 1,900
    assert "**Cumulative API tokens (re-sent each call):** 1,900" in result


@pytest.mark.asyncio
async def test_status_command_tokens_zero_when_session_db_row_missing():
    """When the SessionDB has no row for the current session yet (fresh
    session, no agent calls), /status reports 0 without raising."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=999,  # This should be ignored.
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = None

    result = await runner._handle_message(_make_event("/status"))

    assert "**Cumulative API tokens (re-sent each call):** 0" in result


@pytest.mark.asyncio
async def test_status_command_includes_live_agent_model_and_context():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "openai/gpt-test",
    }
    running_agent = SimpleNamespace(
        model="openai/gpt-test",
        provider="openai",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=12_345,
            context_length=100_000,
        ),
        interrupt=MagicMock(),
    )
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `openai/gpt-test` (openai)" in result
    assert "**Context:** 12,345 / 100,000 (12%)" in result
    assert "**Cumulative API tokens (re-sent each call):** 1,250" in result
    assert "1,250 (cumulative)" not in result


@pytest.mark.asyncio
async def test_status_command_includes_persisted_model_and_context_when_agent_not_running(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        last_prompt_tokens=24_000,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 2000,
        "output_tokens": 500,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "openai/gpt-persisted",
        "billing_provider": "openai-codex",
        "billing_base_url": "https://example.invalid/v1",
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"model": {"context_length": 272_000}})

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `openai/gpt-persisted` (openai-codex)" in result
    assert "**Context:** 24,000 / 272,000 (9%)" in result
    assert "**Cumulative API tokens (re-sent each call):** 2,500" in result


@pytest.mark.asyncio
async def test_status_command_includes_cached_agent_model_and_context():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    cached_agent = SimpleNamespace(
        model="anthropic/claude-sonnet-test",
        provider="openrouter",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=10_000,
            context_length=200_000,
        ),
    )
    runner._agent_cache = {session_entry.session_key: (cached_agent, time.time())}

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `anthropic/claude-sonnet-test` (openrouter)" in result
    assert "**Context:** 10,000 / 200,000 (5%)" in result


@pytest.mark.asyncio
async def test_agents_command_reports_active_agents_and_processes(monkeypatch):
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    running_agent = SimpleNamespace(
        session_id="sess-running",
        model="openrouter/test-model",
        interrupt=MagicMock(),
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts = {session_key: time.time() - 8}
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            procs = [
                {
                    "session_id": "proc-1",
                    "status": "running",
                    "uptime_seconds": 17,
                    "command": "sleep 30",
                    "session_key": session_key or build_session_key(_make_source()),
                }
            ]
            if session_key is not None:
                return [p for p in procs if p.get("session_key") == session_key]
            return procs

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active agents:** 1" in result
    assert "**Running background processes:** 1" in result
    assert "proc-1" in result
    # Non-admin scoped view omits process-global async job counts.
    assert "**Gateway async jobs:**" not in result
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_agents_hides_other_sessions_agents_and_processes(monkeypatch):
    """Non-admin /agents must not enumerate other gateway sessions (CWE-639)."""
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-a",
        chat_id="chat-a",
        user_name="Alice",
        chat_type="dm",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-b",
        chat_id="chat-b",
        user_name="Bob",
        chat_type="dm",
    )
    key_a = build_session_key(source_a)
    key_b = build_session_key(source_b)
    entry_a = SessionEntry(
        session_key=key_a,
        session_id="sess-a",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(entry_a)
    runner._resume_caller_is_admin = lambda _src: False
    runner._running_agents = {
        key_a: SimpleNamespace(session_id="sid-a", model="model-a"),
        key_b: SimpleNamespace(session_id="sid-b-secret", model="model-b-secret"),
    }
    runner._running_agents_ts = {
        key_a: time.time() - 5,
        key_b: time.time() - 3,
    }
    runner._background_tasks = [SimpleNamespace(done=lambda: False)]

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            procs = [
                {
                    "session_id": "proc-a",
                    "status": "running",
                    "uptime_seconds": 10,
                    "command": "echo mine",
                    "session_key": key_a,
                },
                {
                    "session_id": "proc-b-secret",
                    "status": "running",
                    "uptime_seconds": 20,
                    "command": "curl http://internal/secret",
                    "session_key": key_b,
                },
            ]
            if session_key is not None:
                return [p for p in procs if p.get("session_key") == session_key]
            return procs

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    event = MessageEvent(text="/agents", source=source_a, message_id="m-a")
    result = await runner._handle_agents_command(event)

    assert "sid-a" in result
    assert "model-a" in result
    assert "proc-a" in result
    assert "sid-b-secret" not in result
    assert "model-b-secret" not in result
    assert key_b not in result
    assert "proc-b-secret" not in result
    assert "curl http://internal/secret" not in result
    assert "**Active agents:** 1" in result
    assert "**Running background processes:** 1" in result
    # Non-empty global _background_tasks must stay invisible without admin --all.
    assert "**Gateway async jobs:**" not in result


@pytest.mark.asyncio
async def test_agents_admin_all_shows_other_sessions(monkeypatch):
    """Configured slash admin with --all may see cross-session agents."""
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="admin",
        chat_id="chat-a",
        user_name="Admin",
        chat_type="dm",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-b",
        chat_id="chat-b",
        user_name="Bob",
        chat_type="dm",
    )
    key_a = build_session_key(source_a)
    key_b = build_session_key(source_b)
    entry_a = SessionEntry(
        session_key=key_a,
        session_id="sess-a",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(entry_a)
    runner._resume_caller_is_admin = lambda _src: True
    runner._running_agents = {
        key_a: SimpleNamespace(session_id="sid-a", model="model-a"),
        key_b: SimpleNamespace(session_id="sid-b", model="model-b"),
    }
    runner._running_agents_ts = {
        key_a: time.time() - 5,
        key_b: time.time() - 3,
    }
    runner._background_tasks = [SimpleNamespace(done=lambda: False)]

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            return [
                {
                    "session_id": "proc-b",
                    "status": "running",
                    "uptime_seconds": 20,
                    "command": "sleep 99",
                    "session_key": key_b,
                }
            ]

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    event = MessageEvent(text="/agents --all", source=source_a, message_id="m-a")
    result = await runner._handle_agents_command(event)

    assert "**Active agents:** 2" in result
    assert "sid-b" in result
    assert "model-b" in result
    assert key_b in result
    assert "proc-b" in result
    assert "**Gateway async jobs:** 1" in result


@pytest.mark.asyncio
async def test_agents_non_admin_all_still_scoped(monkeypatch):
    """--all without a configured slash admin must not widen the listing."""
    source_a = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-a",
        chat_id="chat-a",
        user_name="Alice",
        chat_type="dm",
    )
    source_b = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-b",
        chat_id="chat-b",
        user_name="Bob",
        chat_type="dm",
    )
    key_a = build_session_key(source_a)
    key_b = build_session_key(source_b)
    entry_a = SessionEntry(
        session_key=key_a,
        session_id="sess-a",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(entry_a)
    runner._resume_caller_is_admin = lambda _src: False
    runner._running_agents = {
        key_a: SimpleNamespace(session_id="sid-a", model="model-a"),
        key_b: SimpleNamespace(session_id="sid-b-secret", model="model-b-secret"),
    }
    runner._running_agents_ts = {
        key_a: time.time() - 5,
        key_b: time.time() - 3,
    }
    runner._background_tasks = [SimpleNamespace(done=lambda: False)]

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    event = MessageEvent(text="/agents --all", source=source_a, message_id="m-a")
    result = await runner._handle_agents_command(event)

    assert "sid-b-secret" not in result
    assert "model-b-secret" not in result
    assert key_b not in result
    assert "**Active agents:** 1" in result
    assert "**Gateway async jobs:**" not in result


@pytest.mark.asyncio
async def test_agents_matrix_redacts_session_key(monkeypatch):
    """Matrix /agents must fingerprint session keys like /status."""
    source = SessionSource(
        platform=Platform.MATRIX,
        user_id="@alice:example.org",
        chat_id="!room:example.org",
        user_name="Alice",
        chat_type="room",
    )
    session_key = build_session_key(source)
    entry = SessionEntry(
        session_key=session_key,
        session_id="sess-m",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.MATRIX,
        chat_type="room",
        total_tokens=0,
    )
    runner = _make_runner(entry, platform=Platform.MATRIX)
    runner._resume_caller_is_admin = lambda _src: False
    runner._running_agents = {
        session_key: SimpleNamespace(session_id="sid-m", model="model-m"),
    }
    runner._running_agents_ts = {session_key: time.time() - 2}
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    event = MessageEvent(text="/agents", source=source, message_id="m1")
    result = await runner._handle_agents_command(event)

    assert session_key not in result
    assert session_key[:8] not in result
    assert "sha256:" in result
    assert "sid-m" in result


@pytest.mark.asyncio
async def test_agents_admin_all_redacts_matrix_keys_from_other_platform(monkeypatch):
    """Admin --all from Telegram must still fingerprint Matrix session keys."""
    tg_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="admin",
        chat_id="chat-admin",
        user_name="Admin",
        chat_type="dm",
    )
    mx_source = SessionSource(
        platform=Platform.MATRIX,
        user_id="@bob:example.org",
        chat_id="!secret:example.org",
        user_name="Bob",
        chat_type="room",
    )
    key_tg = build_session_key(tg_source)
    key_mx = build_session_key(mx_source)
    entry = SessionEntry(
        session_key=key_tg,
        session_id="sess-tg",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(entry)
    runner._resume_caller_is_admin = lambda _src: True
    runner._running_agents = {
        key_tg: SimpleNamespace(session_id="sid-tg", model="model-tg"),
        key_mx: SimpleNamespace(session_id="sid-mx", model="model-mx"),
    }
    runner._running_agents_ts = {
        key_tg: time.time() - 5,
        key_mx: time.time() - 3,
    }
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    event = MessageEvent(text="/agents --all", source=tg_source, message_id="m1")
    result = await runner._handle_agents_command(event)

    assert key_mx not in result
    assert "!secret:example.org" not in result
    assert "sha256:" in result
    assert "sid-mx" in result
    assert key_tg in result


@pytest.mark.asyncio
async def test_tasks_alias_routes_to_agents_command(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._background_tasks = set()

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None):
            return []

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    result = await runner._handle_message(_make_event("/tasks"))

    assert "Active Agents & Tasks" in result


@pytest.mark.asyncio
async def test_handle_message_persists_agent_token_counts(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }
    )

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result == "ok"
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=80,
    )


@pytest.mark.asyncio
async def test_first_run_slack_home_channel_onboarding_uses_parent_command(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(Platform.SLACK)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="dm",
    )
    runner = _make_runner(session_entry, platform=Platform.SLACK)
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "openai/test-model",
        }
    )

    monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello", platform=Platform.SLACK))

    assert result == "ok"
    runner.adapters[Platform.SLACK].send.assert_awaited_once()
    onboarding = runner.adapters[Platform.SLACK].send.await_args.args[1]
    assert "/hermes sethome" in onboarding
    assert "Type /sethome" not in onboarding


@pytest.mark.asyncio
async def test_first_run_non_slack_home_channel_onboarding_keeps_direct_command(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(Platform.TELEGRAM)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry, platform=Platform.TELEGRAM)
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "openai/test-model",
        }
    )

    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello", platform=Platform.TELEGRAM))

    assert result == "ok"
    runner.adapters[Platform.TELEGRAM].send.assert_awaited_once()
    onboarding = runner.adapters[Platform.TELEGRAM].send.await_args.args[1]
    assert "Type /sethome" in onboarding


@pytest.mark.asyncio
async def test_handle_message_discards_stale_result_after_session_invalidation(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    session_key = session_entry.session_key
    runner.adapters[Platform.TELEGRAM]._post_delivery_callbacks = {session_key: object()}

    async def _stale_result(**kwargs):
        runner._invalidate_session_run_generation(kwargs["session_key"], reason="test_stale_result")
        return {
            "final_response": "late reply",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }

    runner._run_agent = AsyncMock(side_effect=_stale_result)

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result is None
    runner.session_store.append_to_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()
    assert session_key not in runner.adapters[Platform.TELEGRAM]._post_delivery_callbacks


@pytest.mark.asyncio
async def test_handle_message_stale_result_keeps_newer_generation_callback(monkeypatch):
    import gateway.run as gateway_run

    class _Adapter:
        def __init__(self):
            self._post_delivery_callbacks = {}

        async def send(self, *args, **kwargs):
            return None

        def pop_post_delivery_callback(self, session_key, *, generation=None):
            entry = self._post_delivery_callbacks.get(session_key)
            if entry is None:
                return None
            if isinstance(entry, tuple):
                entry_generation, callback = entry
                if generation is not None and entry_generation != generation:
                    return None
                self._post_delivery_callbacks.pop(session_key, None)
                return callback
            if generation is not None:
                return None
            return self._post_delivery_callbacks.pop(session_key, None)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    session_key = session_entry.session_key
    adapter = _Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter

    async def _stale_result(**kwargs):
        # Simulate a newer run claiming the callback slot before the stale run unwinds.
        runner._session_run_generation[session_key] = 2
        adapter._post_delivery_callbacks[session_key] = (2, lambda: None)
        return {
            "final_response": "late reply",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }

    runner._run_agent = AsyncMock(side_effect=_stale_result)

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result is None
    assert session_key in adapter._post_delivery_callbacks
    assert adapter._post_delivery_callbacks[session_key][0] == 2



@pytest.mark.asyncio
async def test_status_command_bypasses_active_session_guard():
    """When an agent is running, /status must be dispatched immediately via
    base.handle_message — not queued or treated as an interrupt (#5046)."""
    import asyncio
    from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
    from gateway.session import build_session_key
    from gateway.config import Platform, PlatformConfig

    source = _make_source()
    session_key = build_session_key(source)

    handler_called_with = []

    async def fake_handler(event):
        handler_called_with.append(event)
        return "📊 **Hermes Gateway Status**\n**Agent Running:** Yes ⚡"

    # Concrete subclass to avoid abstract method errors
    class _ConcreteAdapter(BasePlatformAdapter):
        platform = Platform.TELEGRAM

        async def connect(self, *, is_reconnect: bool = False): pass
        async def disconnect(self): pass
        async def send(self, chat_id, content, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}

    platform_config = PlatformConfig(enabled=True, token="***")
    adapter = _ConcreteAdapter(platform_config, Platform.TELEGRAM)
    adapter.set_message_handler(fake_handler)

    sent = []

    async def fake_send_with_retry(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)

    adapter._send_with_retry = fake_send_with_retry

    # Simulate an active session
    interrupt_event = asyncio.Event()
    adapter._active_sessions[session_key] = interrupt_event

    event = MessageEvent(
        text="/status",
        source=source,
        message_id="m1",
        message_type=MessageType.COMMAND,
    )
    await adapter.handle_message(event)

    assert handler_called_with, "/status handler was never called (event was queued or dropped)"
    assert sent, "/status response was never sent"
    assert "Agent Running" in sent[0]
    assert not interrupt_event.is_set(), "/status incorrectly triggered an agent interrupt"
    assert session_key not in adapter._pending_messages, "/status was incorrectly queued"


@pytest.mark.asyncio
async def test_profile_command_reports_custom_root_profile(monkeypatch, tmp_path):
    """Gateway /profile detects custom-root profiles (not under ~/.hermes)."""
    from pathlib import Path

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    profile_home = tmp_path / "profiles" / "coder"

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "unrelated-home")

    result = await runner._handle_profile_command(_make_event("/profile"))

    assert "**Profile:** `coder`" in result
    assert f"**Home:** `{profile_home}`" in result


@pytest.mark.asyncio
async def test_profile_command_reports_source_stamped_profile(monkeypatch, tmp_path):
    """On a multiplexed gateway, /profile reports the profile SERVING the
    source (source.profile — URL prefix / per-credential adapter / room map),
    not the multiplexer's active profile, which is always the default and
    made /profile answer "default" in every persona chat."""
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "milo"
    profile_home.mkdir(parents=True)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.config.multiplex_profiles = True
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    event = _make_event("/profile")
    event.source.profile = "milo"

    result = await runner._handle_profile_command(event)

    assert "**Profile:** `milo`" in result
    assert f"**Home:** `{profile_home}`" in result


@pytest.mark.asyncio
async def test_profile_command_ignores_stamp_when_multiplexing_off(monkeypatch, tmp_path):
    """Without ``gateway.multiplex_profiles`` a stamped source is ignored:
    /profile keeps reporting the active profile and the default home,
    mirroring the multiplex gating in ``_run_agent`` and
    ``_reset_notice_session_info``."""
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "milo"
    profile_home.mkdir(parents=True)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    assert runner.config.multiplex_profiles is False
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    event = _make_event("/profile")
    event.source.profile = "milo"

    result = await runner._handle_profile_command(event)

    assert "**Profile:** `default`" in result
    assert f"**Home:** `{hermes_home}`" in result


@pytest.mark.asyncio
async def test_profile_command_unstamped_source_unchanged(monkeypatch, tmp_path):
    """Single-profile behavior is untouched: an unstamped source reports the
    active profile and the default home."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = await runner._handle_profile_command(_make_event("/profile"))

    assert "**Profile:** `default`" in result
    assert f"**Home:** `{hermes_home}`" in result


@pytest.mark.asyncio
async def test_post_delivery_callback_generation_snapshot_happens_after_bind():
    """Regression: the callback_generation snapshot in _process_message_background
    must happen AFTER the handler runs, not before.

    _hermes_run_generation is set on the interrupt event by
    GatewayRunner._bind_adapter_run_generation during _handle_message_with_agent.
    The earlier snapshot-at-task-start always captured None, which bypassed the
    generation-ownership check in pop_post_delivery_callback and let stale runs
    fire a fresher run's callbacks.
    """
    import asyncio
    from gateway.platforms.base import BasePlatformAdapter

    source = _make_source()
    session_key = build_session_key(source)
    fired = []

    class _ConcreteAdapter(BasePlatformAdapter):
        platform = Platform.TELEGRAM

        async def connect(self, *, is_reconnect: bool = False): pass
        async def disconnect(self): pass
        async def send(self, chat_id, content, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}

    adapter = _ConcreteAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )

    async def fake_handler(event):
        # Simulate what _bind_adapter_run_generation does mid-run.
        interrupt_event = adapter._active_sessions.get(session_key)
        setattr(interrupt_event, "_hermes_run_generation", 1)
        # Stale run registers its callback at generation=1.
        adapter.register_post_delivery_callback(
            session_key,
            lambda: fired.append("older"),
            generation=1,
        )
        # A fresher run overwrites with generation=2 (different dict entry).
        adapter.register_post_delivery_callback(
            session_key,
            lambda: fired.append("newer"),
            generation=2,
        )
        return None

    adapter.set_message_handler(fake_handler)
    event = MessageEvent(text="hello", source=source, message_id="m1")

    await adapter.handle_message(event)
    tasks = list(adapter._background_tasks)
    assert tasks, "expected background task to be created"
    await asyncio.gather(*tasks)

    # The stale run (generation=1) must NOT fire the fresher run's callback
    # (generation=2). With the pre-fix code, callback_generation was snapshotted
    # as None before the handler ran, bypassing the ownership check and firing
    # "newer" anyway.
    assert fired == []
    assert session_key in adapter._post_delivery_callbacks
    assert adapter._post_delivery_callbacks[session_key][0] == 2
