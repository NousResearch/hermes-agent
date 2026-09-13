from hermes_state import AsyncSessionDB, SessionDB
"""Tests for gateway /status behavior and token persistence."""

from datetime import datetime
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture(autouse=True)
def status_metadata_lookup(monkeypatch):
    """Only the metadata I/O boundary is stubbed; /status dispatch stays real."""
    from agent.model_metadata import get_model_context_length_async

    lookup = AsyncMock(return_value=None)
    lookup.real_lookup = get_model_context_length_async
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", lookup)
    return lookup


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
@pytest.mark.parametrize("resident", ["live", "cached"])
async def test_status_command_includes_live_agent_model_and_context(resident, status_metadata_lookup):
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
    if resident == "live":
        runner._running_agents[session_entry.session_key] = running_agent
    else:
        runner._agent_cache[session_entry.session_key] = (running_agent, "signature")

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `openai/gpt-test` (openai)" in result
    assert "**Context:** 12,345 / 100,000 (12%)" in result
    assert "**Lifetime tokens billed:** 1,250" in result
    status_metadata_lookup.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_uses_dominant_persisted_model_route(tmp_path, monkeypatch, status_metadata_lookup):
    """Persisted status must not combine a model and provider from different calls."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    db = SessionDB(db_path=tmp_path / "state.db")
    session_entry.last_prompt_tokens = 12_345
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "model": {"default": "config-model", "provider": "custom-config",
                  "base_url": "https://config.invalid/v1"},
    })
    status_metadata_lookup.return_value = 100_000
    runner._session_db = AsyncSessionDB(db)
    try:
        db.create_session("sess-1", "telegram", model="z-ai/glm-5.2")
        db.update_token_counts(
            "sess-1",
            model="z-ai/glm-5.2",
            billing_provider="nvidia",
            billing_base_url="https://integrate.api.nvidia.com/v1/",
            input_tokens=480,
            api_call_count=48,
        )
        db.update_token_counts(
            "sess-1",
            model="upstage/solar-pro4:free",
            billing_provider="nous",
            billing_base_url="https://inference-api.nousresearch.com/v1/",
            input_tokens=60,
            api_call_count=6,
        )
        # Reproduce the inconsistent legacy summary observed in #87227.
        db.update_session_model("sess-1", "z-ai/glm-5.2")
        db.update_session_billing_route(
            "sess-1",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1/",
        )

        result = await runner._handle_message(_make_event("/status"))

        assert "**Model:** `z-ai/glm-5.2` (nvidia)" in result
        assert "**Model:** `z-ai/glm-5.2` (nous)" not in result
        assert "**Context:** 12,345 / 100,000 (12%)" in result
        assert "**Lifetime tokens billed:** 540" in result
        status_metadata_lookup.assert_awaited_once_with(
            "z-ai/glm-5.2", provider="nvidia",
            base_url="https://integrate.api.nvidia.com/v1/", api_key="",
            config_context_length=None, custom_providers=None,
        )
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route_source", ["live", "cached", "row", "pending", "config"])
@pytest.mark.parametrize("route_url", ["https://selected.invalid/v1", ""])
@pytest.mark.parametrize("configured_context", [None, 80_000])
async def test_status_metadata_uses_selected_route(
    monkeypatch, status_metadata_lookup, route_source, route_url, configured_context,
):
    """Carry the selected route together even when its endpoint is deliberately blank."""
    from gateway.run import _AGENT_PENDING_SENTINEL
    from hermes_cli.config import get_compatible_custom_providers

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-metadata",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", last_prompt_tokens=12_345,
    )
    runner = _make_runner(session_entry)
    config = {
        "model": {"default": "config-model", "provider": "custom-config",
                  "base_url": "https://unrelated.invalid/v1",
                  "context_length": configured_context, "api_key": "unused-test-key"},
        "custom_providers": [
            {"name": "custom-selected", "base_url": "https://selected.invalid/v1",
             "models": [{"id": "selected-model", "context_length": 100_000}]},
            {"name": "custom-config", "base_url": "https://unrelated.invalid/v1"},
        ],
    }
    row = {"model": "row-model", "billing_provider": "custom-row",
           "billing_base_url": "https://row.invalid/v1", "input_tokens": 1_000,
           "output_tokens": 250, "reasoning_tokens": 50}
    runner._session_db._db.get_session.return_value = row
    runner._session_db._db.get_dominant_session_model_route.return_value = {}
    expected_model, expected_provider = "selected-model", "custom-selected"
    if route_source in {"live", "cached"}:
        resident = SimpleNamespace(
            model=expected_model, provider=expected_provider, base_url=route_url,
            context_compressor=SimpleNamespace(last_prompt_tokens=12_345, context_length=0),
            interrupt=MagicMock(),
        )
        runner._session_db._db.get_dominant_session_model_route.return_value = {
            "model": "persisted-model", "billing_provider": "custom-persisted",
            "billing_base_url": "https://persisted.invalid/v1",
        }
        if route_source == "live":
            runner._running_agents[session_entry.session_key] = resident
        else:
            runner._agent_cache[session_entry.session_key] = (resident, "signature")
    elif route_source in {"row", "pending"}:
        row.update(model=expected_model, billing_provider=expected_provider,
                   billing_base_url=route_url)
        if route_source == "pending":
            runner._running_agents[session_entry.session_key] = _AGENT_PENDING_SENTINEL
    else:
        row.update(model="", billing_provider="", billing_base_url="")
        config["model"]["base_url"] = route_url
        expected_model, expected_provider = "config-model", "custom-config"
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    status_metadata_lookup.side_effect = lambda *args, **kwargs: kwargs["config_context_length"] or 100_000

    result = await runner._handle_message(_make_event("/status"))

    assert isinstance(result, str)
    assert f"**Model:** `{expected_model}` ({expected_provider})" in result
    expected_total = configured_context or 100_000
    expected_pct = round(12_345 / expected_total * 100)
    assert f"**Context:** 12,345 / {expected_total:,} ({expected_pct}%)" in result
    assert "**Lifetime tokens billed:** 1,300" in result
    status_metadata_lookup.assert_awaited_once_with(
        expected_model, provider=expected_provider, base_url=route_url, api_key="",
        config_context_length=configured_context, custom_providers=get_compatible_custom_providers(config),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", ["legacy", "providers"])
async def test_status_context_uses_provider_config_without_network(
    tmp_path, monkeypatch, status_metadata_lookup, schema,
):
    """Read real config and resolve the selected capacity before any network probe."""
    from functools import partial
    import json
    from gateway.run import _load_gateway_config

    provider_entry = {
        "name": "custom-selected", "base_url": "https://selected.invalid/v1",
        "models": {"selected-model": {"context_length": 100_000}},
    }
    config: dict = {"model": {"default": "selected-model", "provider": "custom-selected"}}
    if schema == "legacy":
        config["custom_providers"] = [provider_entry]
    else:
        config["providers"] = {"custom-selected": provider_entry}
    config_path = tmp_path / "config.yaml"
    config_bytes = json.dumps(config)
    config_path.write_text(config_bytes)
    monkeypatch.setattr("gateway.run._load_gateway_config", partial(_load_gateway_config, config_path))
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", status_metadata_lookup.real_lookup)
    network = MagicMock(side_effect=AssertionError("configured capacity must not probe the network"))
    monkeypatch.setattr("requests.sessions.Session.request", network)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-provider-config",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", last_prompt_tokens=12_345,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_dominant_session_model_route.return_value = {}
    runner._session_db._db.get_session.return_value = {
        "model": "selected-model", "billing_provider": "custom-selected",
        "billing_base_url": "https://selected.invalid/v1", "input_tokens": 1_000,
    }

    result = await runner._handle_message(_make_event("/status"))

    assert isinstance(result, str)
    assert "**Context:** 12,345 / 100,000 (12%)" in result
    assert "**Lifetime tokens billed:** 1,000" in result
    network.assert_not_called()
    assert config_path.read_text() == config_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, True, False, 0, -1, "100000", 100_000.0, "exception", "timeout"])
async def test_status_metadata_failure_keeps_used_tokens(monkeypatch, status_metadata_lookup, outcome):
    """Invalid metadata or a timed-out lookup must leave status and billing readable."""
    import asyncio

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()), session_id="sess-metadata-failure",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm", last_prompt_tokens=12_345,
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "model": "selected-model", "billing_provider": "custom-selected",
        "input_tokens": 1_000, "reasoning_tokens": 50,
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"model": {}})
    cancelled = asyncio.Event()

    async def hanging_lookup(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    if outcome == "timeout":
        status_metadata_lookup.side_effect = hanging_lookup
    elif outcome == "exception":
        status_metadata_lookup.side_effect = RuntimeError("metadata unavailable")
    else:
        status_metadata_lookup.return_value = outcome
    real_wait_for = asyncio.wait_for
    with patch("gateway.slash_commands_status.asyncio.wait_for", wraps=real_wait_for) as bounded:
        result = await real_wait_for(runner._handle_message(_make_event("/status")), timeout=6.0)

    assert isinstance(result, str)
    assert "**Model:** `selected-model` (custom-selected)" in result
    assert "**Context:** ~12,345 tokens" in result
    assert "**Lifetime tokens billed:** 1,050" in result
    status_metadata_lookup.assert_awaited_once()
    assert any(call.kwargs.get("timeout") == 3.0 for call in bounded.call_args_list)
    if outcome == "timeout":
        assert cancelled.is_set()


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
        def list_sessions(self):
            return [
                {
                    "session_id": "proc-1",
                    "status": "running",
                    "uptime_seconds": 17,
                    "command": "sleep 30",
                }
            ]

    monkeypatch.setattr("tools.process_registry.process_registry", _FakeRegistry())

    result = await runner._handle_message(_make_event("/agents"))

    assert "**Active agents:** 1" in result
    assert "**Running background processes:** 1" in result
    assert "proc-1" in result
    running_agent.interrupt.assert_not_called()


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
        def list_sessions(self):
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
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent, MessageType
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


@pytest.mark.asyncio
async def test_context_command_keeps_configured_window_without_resident_agent():
    """The no-agent fallback must not replace a custom-provider context pin."""
    model = "unsloth/Qwen3.8-27B-GGUF:Q8_0"
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-context-pin",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    session_entry.last_prompt_tokens = 66_570
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {"model": model}

    config = {
        "model": {
            "default": model,
            "provider": "custom-local-qwen",
            "context_length": 262_144,
        },
        "custom_providers": [
            {
                "name": "custom-local-qwen",
                "base_url": "http://127.0.0.1:8080/v1",
                "models": {},
            }
        ],
    }
    runtime = {
        "provider": "custom-local-qwen",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
    }

    with patch("gateway.run._load_gateway_config", return_value=config), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value=runtime
    ), patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=config["custom_providers"],
    ), patch(
        "agent.model_metadata.get_model_context_length",
        side_effect=lambda *args, **kwargs: kwargs.get("config_context_length") or 131_072,
    ) as context_lookup:
        result = await runner._handle_context_command(_make_event("/context"))

    assert "Window: 262,144 tokens" in result
    assert "In use: 66,570 / 262,144 (25%)" in result
    assert "131,072" not in result
    assert context_lookup.call_count == 1
    assert context_lookup.call_args.kwargs["config_context_length"] == 262_144


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
