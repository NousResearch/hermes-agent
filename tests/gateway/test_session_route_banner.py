"""Real session-store/provider-resolution regressions for the reset banner (#65463)."""

import socket

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


@pytest.fixture
def route_session(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("Network disabled in route-banner integration tests")

    monkeypatch.setattr(socket, "getaddrinfo", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-global-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-routed-key")
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: gpt-4o\n  provider: custom\n"
        "  base_url: https://api.openai.com/v1\n  context_length: 8192\n",
        encoding="utf-8",
    )
    config = GatewayConfig()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="room", user_id="user")
    stores = []

    def restart():
        store = SessionStore(tmp_path / "sessions", config)
        stores.append(store)
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.session_store = store
        return runner

    runner = restart()
    entry = runner.session_store.get_or_create_session(source)
    yield runner, source, entry.session_key, restart
    for store in stores:
        if store._db is not None:
            store._db.close()


def test_persisted_route_without_credentials_is_unavailable(route_session, monkeypatch):
    runner, source, key, restart = route_session
    runner.session_store.set_model_override(key, {"model": "deepseek-chat", "provider": "deepseek"})
    # First restart proves that the durable override resolves with real provider credentials.
    runner = restart()
    info = runner._reset_notice_session_info(source)
    assert "deepseek-chat" in info
    assert "Route: session override" in info
    assert "Context: 8K" not in info  # the global context pin does not belong to this route

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    runner = restart()
    info = runner._reset_notice_session_info(source)
    assert "Route unavailable" in info
    assert "session override" in info
    assert "deepseek" in info
    assert "credentials" in info
    assert "hermes auth" in info
    assert "gpt-4o" not in info or "Fallback" in info
    assert runner.session_store.get_model_override(key)["provider"] == "deepseek"


def test_channel_credentials_removed_reports_unavailable(route_session, monkeypatch):
    runner, source, _, restart = route_session
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={
        "room": ChannelOverride(model="deepseek-chat", provider="deepseek"),
    })
    assert "deepseek-chat" in runner._reset_notice_session_info(source)
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    info = restart()._reset_notice_session_info(source)
    assert "Route unavailable: channel override" in info
    assert "deepseek" in info and "credentials" in info
    assert "hermes auth" in info
    assert "gpt-4o" not in info


@pytest.mark.parametrize("lookup", ["chat", "thread", "parent"])
def test_reset_restores_channel_route_then_global(route_session, lookup):
    runner, source, key, restart = route_session
    source.thread_id, source.parent_chat_id = "topic", "parent"
    channel_key = {"chat": "room", "thread": "topic", "parent": "parent"}[lookup]
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={
        channel_key: ChannelOverride(model="deepseek-reasoner", provider="deepseek"),
    })
    # Source identity changed when adding a thread; persist under that session's actual key.
    key = runner.session_store.get_or_create_session(source).session_key
    runner.session_store.set_model_override(key, {"model": "deepseek-chat", "provider": "deepseek"})
    runner = restart()
    info = runner._reset_notice_session_info(source)
    assert "deepseek-chat" in info and "Route: session override" in info
    assert "deepseek-reasoner" not in info

    runner.session_store.reset_session(key)
    runner = restart()
    info = runner._reset_notice_session_info(source)
    assert "deepseek-reasoner" in info and "Route: channel override" in info
    assert "Context: 8K" not in info

    runner.config.platforms.clear()
    info = runner._reset_notice_session_info(source)
    assert "gpt-4o" in info and "Context: 8K" in info and "(config)" in info
    assert "Route:" not in info


def test_prompt_only_channel_override_keeps_global_label(route_session):
    runner, source, _, _ = route_session
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={
        "room": ChannelOverride(system_prompt="Be concise."),
    })
    info = runner._reset_notice_session_info(source)
    assert "gpt-4o" in info and "Context: 8K" in info
    assert "Route:" not in info


@pytest.mark.parametrize("override", [
    ChannelOverride(model="gpt-4o-mini"),
    ChannelOverride(provider="deepseek"),
    ChannelOverride(model="gpt-4o"),
])
def test_partial_channel_route_matches_live_resolution(route_session, override):
    runner, source, _, _ = route_session
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={"room": override})
    model, runtime = runner._resolve_session_agent_runtime(source=source)
    info = runner._reset_notice_session_info(source)
    assert f"Model: `{model}`" in info
    assert f"Provider: {runtime['provider']}" in info
    assert "Route: channel override" in info
    # An explicitly selected route that still equals the global route owns its context pin.
    assert ("Context: 8K" in info) == (override.model == "gpt-4o")


def test_empty_runtime_snapshot_does_not_borrow_global_context_or_endpoint(route_session):
    from gateway.run_model_context import _resolve_gateway_model_context

    resolved = _resolve_gateway_model_context("gpt-4o", runtime_snapshot={})
    assert resolved.provider == ""
    assert resolved.base_url == ""
    assert resolved.context_source != "config"
    assert resolved.context_length != 8192


def test_unknown_channel_provider_does_not_crash_banner(route_session):
    runner, source, _, _ = route_session
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={
        "room": ChannelOverride(model="missing-model", provider="not-installed-provider"),
    })
    info = runner._reset_notice_session_info(source)
    assert "Route unavailable: channel override" in info
    assert "not-installed-provider" in info and "hermes auth" in info
    assert "gpt-4o" not in info


@pytest.mark.asyncio
async def test_new_command_banner_uses_channel_after_clearing_session(route_session):
    from gateway.hooks import HookRegistry
    from gateway.platforms.event import MessageEvent

    runner, source, key, restart = route_session
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(channel_overrides={
        "room": ChannelOverride(model="deepseek-reasoner", provider="deepseek"),
    })
    runner.session_store.set_model_override(key, {"model": "deepseek-chat", "provider": "deepseek"})
    runner = restart()
    assert "deepseek-chat" in runner._reset_notice_session_info(source)
    runner.hooks = HookRegistry()
    runner.adapters = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._background_tasks = set()

    info = await runner._handle_reset_command(MessageEvent(text="/new", source=source, message_id="m"))
    assert "deepseek-reasoner" in info and "Route: channel override" in info
    assert "deepseek-chat" not in info
    assert restart().session_store.get_model_override(key) is None


def test_without_source_keeps_global_banner(route_session):
    runner, _, key, _ = route_session
    runner.session_store.set_model_override(key, {"model": "deepseek-chat", "provider": "deepseek"})
    info = runner._format_session_info()
    assert "gpt-4o" in info and "Context: 8K" in info
    assert "Route:" not in info
