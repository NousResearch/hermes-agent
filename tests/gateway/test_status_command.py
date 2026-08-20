from hermes_state import AsyncSessionDB, SessionDB
"""Tests for gateway /status behavior and token persistence."""

from dataclasses import replace
from datetime import datetime
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
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
    runner.session_store.get_model_override.return_value = None
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
async def test_status_command_uses_dominant_persisted_model_route(tmp_path):
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
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_command_falls_back_to_config_for_incomplete_live_agent(monkeypatch):
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "stale-model",
        "billing_provider": "stale-provider",
    }
    runner._running_agents[session_entry.session_key] = SimpleNamespace(
        model="",
        provider="",
        base_url="",
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
    )

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )
    # /status must not call the resolver; keep a tripwire that fails if it does.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `configured-model` (configured-provider)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_preserves_live_model_when_provider_is_missing(monkeypatch):
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "stale-model",
        "billing_provider": "stale-provider",
    }
    runner._running_agents[session_entry.session_key] = SimpleNamespace(
        model="live-model",
        provider="",
        context_compressor=SimpleNamespace(last_prompt_tokens=12, context_length=100),
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )
    # /status must not call the resolver; keep a tripwire that fails if it does.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    result = await runner._handle_message(_make_event("/status"))

    # The live model is preserved; the missing provider now comes from the
    # current config tier (no resolver call).
    assert "**Model:** `live-model` (configured-provider)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_prefers_cached_agent_over_current_config(monkeypatch):
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    runner._agent_cache[session_entry.session_key] = (
        SimpleNamespace(
            model="cached-model",
            provider="cached-provider",
            base_url="https://cached.example/v1",
            context_compressor=SimpleNamespace(
                last_prompt_tokens=12,
                context_length=100,
            ),
        ),
        "signature",
        0,
        session_entry.session_id,
    )
    runner.config.platforms[Platform.TELEGRAM].channel_overrides["c1"] = ChannelOverride(
        model="channel-model",
        provider="channel-provider",
    )
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("wrong-model", {"provider": "wrong-provider"})
    )

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `cached-model` (cached-provider)" in result
    assert "**Context:** 12 / 100 (12%)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_prefers_runtime_session_override_over_config(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        model_override={
            "model": "persisted-model",
            "provider": "persisted-provider",
        },
    )
    runner = _make_runner(session_entry)
    runner._session_state(session_entry.session_key).conversation.model_override = {
        "model": "once-model",
        "provider": "once-provider",
    }
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "persisted-model",
        "billing_provider": "persisted-provider",
    }
    # The in-memory session override drives the render purely; the resolver is
    # never consulted.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `once-model` (once-provider)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_prefers_channel_override_over_global_config(monkeypatch):
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
    runner.config.platforms[Platform.TELEGRAM].channel_overrides["c1"] = ChannelOverride(
        model="channel-model",
        provider="channel-provider",
    )
    # The channel override drives the render purely; the resolver is never
    # consulted.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `channel-model` (channel-provider)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_resolves_config_for_pending_agent(monkeypatch):
    from gateway.run import _AGENT_PENDING_SENTINEL

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
    runner._running_agents[session_entry.session_key] = _AGENT_PENDING_SENTINEL
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "stale-model",
        "billing_provider": "stale-provider",
    }

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )
    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `configured-model` (configured-provider)" in result


@pytest.mark.asyncio
async def test_status_command_uses_normalized_session_key_for_override():
    raw_source = _make_source()
    normalized_source = replace(raw_source, thread_id="topic-1")
    normalized_key = build_session_key(normalized_source)
    session_entry = SessionEntry(
        session_key=normalized_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    runner._normalize_source_for_session_key = MagicMock(return_value=normalized_source)
    runner._session_state(normalized_key).conversation.model_override = {
        "model": "topic-model",
        "provider": "topic-provider",
    }
    # Tripwire: the normalized-key override is read directly; the resolver must
    # not run for /status.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    event = MessageEvent(text="/status", source=raw_source, message_id="m1")
    result = await runner._handle_status_command(event)

    assert "**Model:** `topic-model` (topic-provider)" in result
    runner._normalize_source_for_session_key.assert_called_once_with(raw_source)
    runner.session_store.get_or_create_session.assert_called_once_with(normalized_source)


@pytest.mark.asyncio
async def test_status_command_prefers_current_config_over_stale_billing_route(monkeypatch):
    """A persisted billing route must not mask the gateway's current config."""
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "stepfun-ai/step-3.7-flash",
        "billing_provider": "openrouter",
        "billing_base_url": "https://openrouter.ai/api/v1",
    }
    runner._session_db._db.get_dominant_session_model_route.return_value = {
        "model": "stepfun-ai/step-3.7-flash",
        "billing_provider": "openrouter",
        "billing_base_url": "https://openrouter.ai/api/v1",
    }

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "stepfun-ai/step-3.7-flash",
                "provider": "custom",
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `stepfun-ai/step-3.7-flash` (custom)" in result


@pytest.mark.asyncio
async def test_status_command_prefers_persisted_runtime_over_billing_and_config(monkeypatch):
    """The last actual gateway route must beat accounting and changed defaults."""
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "persisted-model",
        "billing_provider": "openrouter",
        "model_config": json.dumps(
            {
                "gateway_runtime": {
                    "provider": "custom",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                }
            }
        ),
    }
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `persisted-model` (custom)" in result


@pytest.mark.asyncio
async def test_status_command_prefers_persisted_runtime_over_config_for_incomplete_live_agent(
    monkeypatch,
):
    """#77521 (precedence): for an INCOMPLETE live route with no session/channel
    override, the persisted ``gateway_runtime`` (the last route that actually
    answered) must beat the current global-config default the resolver falls
    back to.  The resolver here returns the global-config values (reproducing
    the pre-fill defect); the render must still show the persisted route."""
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "billing-model",
        "billing_provider": "billing-provider",
        "model_config": json.dumps(
            {
                "gateway_runtime": {
                    "model": "persisted-gw-model",
                    "provider": "persisted-gw-provider",
                }
            }
        ),
    }
    # Incomplete live route: both model and provider missing.
    runner._running_agents[session_entry.session_key] = SimpleNamespace(
        model="",
        provider="",
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )
    # /status resolves the route purely: persisted runtime must win over the
    # current global config for the no-override incomplete-live path, without
    # ever calling the resolver.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `persisted-gw-model` (persisted-gw-provider)" in result
    assert "configured-model" not in result
    runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_does_not_mutate_session_state_on_display(monkeypatch):
    """#77521 (read-without-mutation): /status is display-only.  For an
    incomplete live route whose /model override is persisted on the
    ``SessionEntry`` (but not yet rehydrated into conversation state), /status
    must RENDER that override while leaving no trace in session/process state —
    no rehydrated ``model_override`` and no primed ``last_resolved_model``
    recovery cache."""
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        # Override persisted on the SessionEntry only — nothing in memory yet.
        model_override={
            "model": "override-model",
            "provider": "override-provider",
        },
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "billing-model",
        "billing_provider": "billing-provider",
    }
    # Incomplete live route — the former code path that triggered the resolver.
    runner._running_agents[session_key] = SimpleNamespace(
        model="",
        provider="",
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "configured-model", "provider": "configured-provider"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    # Sanity: the real resolver is in play (not a MagicMock).
    assert not isinstance(runner._resolve_session_agent_runtime, MagicMock)

    result = await runner._handle_message(_make_event("/status"))

    # The persisted override is rendered — read directly from the SessionEntry.
    assert "**Model:** `override-model` (override-provider)" in result

    # The display-only render left no trace in session/process state.
    peeked = runner._peek_session_state(session_key)
    assert peeked is None or peeked.conversation.model_override is None
    assert runner._session_state(session_key).conversation.last_resolved_model == ""
    assert runner._session_state("*").conversation.last_resolved_model == ""


@pytest.mark.asyncio
async def test_status_command_never_invokes_runtime_resolver(monkeypatch):
    """#77521: /status must NEVER call the state-mutating runtime resolver,
    across BOTH former gate paths — an idle session with a persisted override,
    and a live agent with an incomplete route."""
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "configured-model", "provider": "configured-provider"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    # Path A: idle session (no live agent) with a persisted /model override.
    idle_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-idle",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        model_override={"model": "idle-model", "provider": "idle-provider"},
    )
    idle_runner = _make_runner(idle_entry)
    idle_runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )
    idle_result = await idle_runner._handle_message(_make_event("/status"))
    assert "**Model:** `idle-model` (idle-provider)" in idle_result
    idle_runner._resolve_session_agent_runtime.assert_not_called()

    # Path B: live agent with an incomplete route (model + provider missing).
    live_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-live",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    live_runner = _make_runner(live_entry)
    live_runner._running_agents[live_entry.session_key] = SimpleNamespace(
        model="",
        provider="",
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
    )
    live_runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )
    live_result = await live_runner._handle_message(_make_event("/status"))
    assert "**Model:** `configured-model` (configured-provider)" in live_result
    live_runner._resolve_session_agent_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_does_not_touch_credential_resolution(monkeypatch):
    """#77521: /status must never reach any credential helper — OAuth
    refresh/persist and round-robin rotation live behind these — across BOTH
    former gate paths."""
    kwargs_mock = MagicMock(return_value={})
    kwargs_provider_mock = MagicMock(
        return_value={
            "api_key": None,
            "api_mode": None,
            "credential_pool": None,
            "base_url": None,
        }
    )
    pool_mock = MagicMock(return_value=None)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", kwargs_mock)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider", kwargs_provider_mock
    )
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", pool_mock)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "configured-model", "provider": "configured-provider"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    # Path A: idle session with a persisted /model override.
    idle_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-idle",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        model_override={"model": "idle-model", "provider": "idle-provider"},
    )
    await _make_runner(idle_entry)._handle_message(_make_event("/status"))

    # Path B: live agent with an incomplete route.
    live_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-live",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    live_runner = _make_runner(live_entry)
    live_runner._running_agents[live_entry.session_key] = SimpleNamespace(
        model="",
        provider="",
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
    )
    await live_runner._handle_message(_make_event("/status"))

    kwargs_mock.assert_not_called()
    kwargs_provider_mock.assert_not_called()
    pool_mock.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_displays_persisted_override_for_idle_session_without_rehydration(
    monkeypatch,
):
    """#77521: after a gateway restart, an idle session's persisted /model
    override (on the SessionEntry, no live agent, no in-memory SessionState)
    must RENDER without the store creating or rehydrating any session state."""
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        model_override={"model": "restart-model", "provider": "restart-provider"},
    )
    runner = _make_runner(session_entry)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "configured-model", "provider": "configured-provider"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `restart-model` (restart-provider)" in result
    # No in-memory session state was created or rehydrated for the display.
    assert runner._peek_session_state(session_key) is None


@pytest.mark.asyncio
async def test_status_command_ignores_malformed_persisted_runtime(monkeypatch):
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
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "stale-model",
        "billing_provider": "stale-provider",
        "model_config": "{not-json",
    }
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "configured-model",
                "provider": "configured-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `configured-model` (configured-provider)" in result


@pytest.mark.asyncio
async def test_status_command_preserves_explicit_session_model_override(monkeypatch):
    """A session-scoped /model choice must remain authoritative over global config."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
        model_override={
            "model": "session-model",
            "provider": "session-provider",
            "base_url": "https://session.example/v1",
        },
    )
    runner = _make_runner(session_entry)
    runner._session_db._db.get_session.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "model": "old-model",
        "billing_provider": "old-provider",
    }

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "global-model",
                "provider": "global-provider",
            }
        },
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda config: config["model"]["default"],
    )
    # The persisted session override (session_entry.model_override) drives the
    # render directly; the resolver must not run for /status.
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=AssertionError("resolver must not run for /status")
    )

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `session-model` (session-provider)" in result
    runner._resolve_session_agent_runtime.assert_not_called()


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
