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
    assert "**Lifetime tokens billed:** 1,900" in result


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
    assert "**Lifetime tokens billed:** 1,250" in result


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


# ── /context command tests ────────────────────────────────────────────────

def _stub_agent(**overrides) -> SimpleNamespace:
    """Build a stub agent with the attributes _handle_context_command reads."""
    props = dict(
        model="openai/gpt-test",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=47_231,
            context_length=200_000,
            threshold_tokens=100_000,
            threshold_percent=0.5,
            compression_count=2,
            _last_compression_savings_pct=63.0,
        ),
        session_api_calls=47,
        session_input_tokens=410_000,
        session_output_tokens=38_000,
        session_reasoning_tokens=12_000,
        session_total_tokens=3_158_641,
        session_cache_read_tokens=2_900_000,
        session_cache_write_tokens=48_000,
    )
    props.update(overrides)
    return SimpleNamespace(**props)


@pytest.mark.asyncio
async def test_context_all_appends_expanded_listings():
    """/context all appends per-toolset and per-skill cost listings."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-6",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    agent = _stub_agent()
    runner._running_agents[session_entry.session_key] = agent

    fake_payload = {
        "categories": [
            {"id": "skills", "label": "Skills", "tokens": 2_000},
        ],
        "context_max": 200_000,
        "context_percent": 24,
        "context_used": 47_231,
        "estimated_total": 2_000,
        "model": "openai/gpt-test",
    }
    fake_details = {
        "skills": [
            {"name": "hermes-agent", "index_tokens": 30, "skill_md_tokens": 2_500},
        ],
        "toolsets": [
            {"toolset": "terminal", "tool_count": 4, "schema_tokens": 5_100},
        ],
    }
    from unittest.mock import patch as _patch
    with _patch(
        "agent.context_breakdown.compute_session_context_breakdown",
        return_value=fake_payload,
    ), _patch(
        "agent.context_breakdown.compute_context_details",
        return_value=fake_details,
    ):
        result = await runner._handle_context_command(_make_event("/context all"))

    assert "Toolsets by schema cost" in result
    assert "terminal" in result and "5,100 tokens" in result
    assert "Skills by cost" in result
    assert "hermes-agent" in result
    # Expanded view drops the hint
    assert "Use /context all" not in result


