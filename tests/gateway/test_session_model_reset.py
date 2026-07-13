"""Tests that manual /new and /reset preserve deliberate route preferences."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


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


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store.entry_for.side_effect = runner.session_store._entries.get
    runner.session_store.get_model_override.return_value = None

    def _reset(key, *, preserve_route_preferences=False, **_kwargs):
        old = runner.session_store._entries[key]
        fresh = SessionEntry(
            session_key=key,
            session_id="sess-2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=old.platform,
            chat_type=old.chat_type,
            model_override_identity=(
                dict(old.model_override_identity)
                if preserve_route_preferences and old.model_override_identity
                else None
            ),
            reasoning_override=(
                dict(old.reasoning_override)
                if preserve_route_preferences and old.reasoning_override
                else None
            ),
        )
        runner.session_store._entries[key] = fresh
        return fresh

    runner.session_store.reset_session.side_effect = _reset
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None  # disables _evict_cached_agent lock path
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    runner._reresolve_model_override_credentials = lambda identity: {
        **identity,
        "api_key": "fresh-key",
    }

    return runner


def test_session_runtime_unavailable_persisted_route_raises():
    from gateway.run import SessionRouteUnavailableError
    from gateway.session import PersistedSessionRouteLookup

    runner = _make_runner()

    with pytest.raises(SessionRouteUnavailableError):
        runner._resolve_session_agent_runtime(
            session_key=build_session_key(_make_source()),
            user_config={"model": {"default": "configured-model"}},
            persisted_route_lookup=PersistedSessionRouteLookup("unavailable"),
        )


def test_session_runtime_absent_route_returns_configured_model(monkeypatch):
    import gateway.run as gateway_run
    from gateway.session import PersistedSessionRouteLookup

    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "custom-local", "api_key": "secret"},
    )

    model, _runtime = runner._resolve_session_agent_runtime(
        session_key=build_session_key(_make_source()),
        user_config={"model": {"default": "configured-model"}},
        persisted_route_lookup=PersistedSessionRouteLookup("absent"),
    )

    assert model == "configured-model"


@pytest.mark.asyncio
async def test_new_command_preserves_session_route_preferences():
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    # Simulate a prior /model switch stored as a session override
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "***",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    entry = runner.session_store._entries[session_key]
    entry.model_override_identity = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_mode": "codex_responses",
    }
    entry.reasoning_override = {"enabled": True, "effort": "high"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert runner._session_model_overrides[session_key]["model"] == "gpt-4o"
    assert runner._session_reasoning_overrides[session_key] == {
        "enabled": True,
        "effort": "high",
    }
    assert session_key not in runner._pending_model_notes


@pytest.mark.parametrize("command", ["/new", "/reset"])
@pytest.mark.asyncio
async def test_manual_reset_banner_uses_preserved_effective_session_route(
    monkeypatch, command
):
    """The reset notice must describe the route the next turn will use."""
    import gateway.run as gateway_run
    import agent.model_metadata as model_metadata

    runner = _make_runner()
    del runner._format_session_info
    session_key = build_session_key(_make_source())
    entry = runner.session_store._entries[session_key]
    entry.model_override_identity = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    entry.reasoning_override = {"enabled": True, "effort": "high"}
    runner._reresolve_model_override_credentials = lambda identity: {
        **identity,
        "api_key": "session-secret",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }

    global_config = {
        "model": {
            "default": "claude-apr/claude-opus-4-8",
            "provider": "claude-apr",
            "base_url": "http://localhost:9999/claude",
        },
        "agent": {"reasoning_effort": "low"},
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: global_config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda _config=None: "claude-apr/claude-opus-4-8",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "claude-apr",
            "api_key": "global-secret",
            "base_url": "http://localhost:9999/claude",
        },
    )
    monkeypatch.setattr(
        model_metadata,
        "get_model_context_length",
        lambda model, **_kwargs: 400_000 if model == "gpt-5.6-sol" else 1_000_000,
    )

    notice = await runner._handle_reset_command(_make_event(command))

    assert "◆ Model: `gpt-5.6-sol`" in notice
    assert "◆ Provider: openai-codex" in notice
    assert "◆ Reasoning: high" in notice
    assert "◆ Context: 400K tokens" in notice
    assert "claude-apr/claude-opus-4-8" not in notice
    assert "localhost:9999/claude" not in notice
    assert "session-secret" not in notice
    assert "global-secret" not in notice


@pytest.mark.asyncio
async def test_new_banner_without_session_override_uses_configured_route(monkeypatch):
    """The common no-override reset keeps the configured banner behavior."""
    import agent.model_metadata as model_metadata
    import gateway.run as gateway_run

    runner = _make_runner()
    del runner._format_session_info
    global_config = {
        "model": {
            "default": "local/llama-4",
            "provider": "custom-local",
            "base_url": "http://localhost:11434/v1",
            "context_length": 262_144,
        },
        "agent": {"reasoning_effort": "medium"},
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: global_config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda _config=None: "local/llama-4",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom-local",
            "api_key": "global-secret",
            "base_url": "http://localhost:11434/v1",
        },
    )
    monkeypatch.setattr(
        model_metadata,
        "get_model_context_length",
        lambda _model, **kwargs: kwargs["config_context_length"],
    )

    notice = await runner._handle_reset_command(_make_event("/new"))

    assert "◆ Model: `local/llama-4`" in notice
    assert "◆ Provider: custom-local" in notice
    assert "◆ Reasoning: medium" in notice
    assert "◆ Context: 262K tokens (config)" in notice
    assert "◆ Endpoint: http://localhost:11434" in notice
    assert "global-secret" not in notice


@pytest.mark.asyncio
async def test_reset_banner_redacts_session_endpoint_credentials(monkeypatch):
    """Session endpoint display cannot expose URL credentials."""
    import agent.model_metadata as model_metadata
    import gateway.run as gateway_run

    runner = _make_runner()
    del runner._format_session_info
    entry = runner.session_store._entries[build_session_key(_make_source())]
    entry.model_override_identity = {
        "model": "local-model",
        "provider": "custom-local",
        "api_mode": "openai",
    }
    runner._reresolve_model_override_credentials = lambda identity: {
        **identity,
        "api_key": "route-api-key",
        "base_url": (
            "http://endpoint-user:endpoint-password@localhost:8787/v1/models"
            "?region=dev&api_key=query-secret&token=second-secret"
        ),
    }
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "global-model", "provider": "openrouter"}},
    )
    probe_kwargs = {}

    def capture_context_probe(_model, **kwargs):
        probe_kwargs.update(kwargs)
        return 32_000

    monkeypatch.setattr(
        model_metadata, "get_model_context_length", capture_context_probe
    )

    notice = await runner._handle_reset_command(_make_event("/reset"))

    assert "◆ Endpoint: http://localhost:8787" in notice
    for secret in (
        "endpoint-user",
        "endpoint-password",
        "query-secret",
        "second-secret",
        "route-api-key",
    ):
        assert secret not in notice
    assert probe_kwargs["base_url"].endswith(
        "?region=dev&api_key=query-secret&token=second-secret"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://localhost@api.example.com/v1",
        "https://api.example.com/localhost/v1",
        "https://api.example.com/v1?next=localhost",
    ],
)
@pytest.mark.asyncio
async def test_reset_banner_does_not_display_external_endpoint_spoofed_as_local(
    monkeypatch, base_url
):
    import agent.model_metadata as model_metadata
    import gateway.run as gateway_run

    runner = _make_runner()
    del runner._format_session_info
    entry = runner.session_store._entries[build_session_key(_make_source())]
    entry.model_override_identity = {
        "model": "external-model",
        "provider": "custom",
        "api_mode": "openai",
    }
    runner._reresolve_model_override_credentials = lambda identity: {
        **identity,
        "api_key": "route-secret",
        "base_url": base_url,
    }
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "global-model", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        model_metadata, "get_model_context_length", lambda _model, **_kwargs: 32_000
    )

    notice = await runner._handle_reset_command(_make_event("/reset"))

    assert "◆ Endpoint:" not in notice
    assert "route-secret" not in notice


@pytest.mark.asyncio
async def test_reset_banner_omits_local_endpoint_path_secret(monkeypatch):
    import agent.model_metadata as model_metadata
    import gateway.run as gateway_run

    runner = _make_runner()
    del runner._format_session_info
    entry = runner.session_store._entries[build_session_key(_make_source())]
    entry.model_override_identity = {
        "model": "local-model",
        "provider": "custom-local",
        "api_mode": "openai",
    }
    runner._reresolve_model_override_credentials = lambda identity: {
        **identity,
        "api_key": "route-secret",
        "base_url": "http://[::1]:8787/v1/path-secret-token?query-secret=yes#fragment-secret",
    }
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "global-model", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        model_metadata, "get_model_context_length", lambda _model, **_kwargs: 32_000
    )

    notice = await runner._handle_reset_command(_make_event("/reset"))

    assert "◆ Endpoint: http://[::1]:8787" in notice
    for secret in ("path-secret-token", "query-secret", "fragment-secret", "route-secret"):
        assert secret not in notice


@pytest.mark.asyncio
async def test_new_command_no_override_is_noop():
    """/new with no prior model override must not raise."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_new_command_only_rotates_own_session_preferences():
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    other_key = "other_session_key"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_model_overrides[other_key] = {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_key": "***",
        "base_url": "",
        "api_mode": "anthropic",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides[other_key] = {"enabled": True, "effort": "low"}
    entry = runner.session_store._entries[session_key]
    entry.model_override_identity = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_mode": "codex_responses",
    }
    entry.reasoning_override = {"enabled": True, "effort": "high"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"
    runner._pending_model_notes[other_key] = "[Note: switched to claude-sonnet-4-6.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert runner._session_model_overrides[session_key]["model"] == "gpt-4o"
    assert other_key in runner._session_model_overrides
    assert runner._session_reasoning_overrides[session_key]["effort"] == "high"
    assert other_key in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes
    assert other_key in runner._pending_model_notes
