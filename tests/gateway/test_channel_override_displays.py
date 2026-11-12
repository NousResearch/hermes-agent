"""/model and /footer report the chat's channel_overrides model, the one its next turn runs."""

import pytest
import hermes_agent.hermes_yaml as yaml

from hermes_agent.gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, token="***",
        channel_overrides={"777": ChannelOverride(model="pinned/lane-model")})})
    return runner


def _event(text, thread_id):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", thread_id=thread_id))


@pytest.fixture()
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "global/default-model", "provider": "openrouter"}}), encoding="utf-8")
    import hermes_agent.gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id, expected, absent", [
    ("777", "pinned/lane-model", "global/default-model"),
    ("888", "global/default-model", "pinned/lane-model"),
])
async def test_model_header_reports_channel_override(home, monkeypatch, thread_id, expected, absent):
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [])
    reply = await _runner()._handle_model_command(_event("/model", thread_id))
    header = reply.splitlines()[0]
    assert expected in header and absent not in header


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id, expected, absent", [
    ("777", "lane-model", "default-model"),
    ("888", "default-model", "lane-model"),
])
async def test_footer_preview_reports_channel_override(home, thread_id, expected, absent):
    reply = await _runner()._handle_footer_command(_event("/footer on", thread_id))
    assert expected in reply and absent not in reply


@pytest.mark.asyncio
async def test_footer_preview_prefers_session_model_override_over_channel_pin(home):
    """A session /model override outranks the channel pin for the turn, so the preview shows it."""
    runner = _runner()
    event = _event("/footer on", "777")
    runner._session_model_overrides[runner._session_key_for_source(event.source)] = {
        "model": "session/picked-model", "provider": "openrouter", "api_key": "sk-test",
        "base_url": "https://openrouter.ai/api/v1"}
    reply = await runner._handle_footer_command(event)
    assert "picked-model" in reply and "lane-model" not in reply


@pytest.mark.asyncio
async def test_model_switch_keeps_config_backed_custom_endpoint(home, monkeypatch):
    """A channel pinned to ``provider: custom`` hands switch_model the endpoint config.yaml defines."""
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {
        "default": "local-model", "provider": "custom", "base_url": "http://127.0.0.1:8999/v1"}}), encoding="utf-8")
    runner = _runner()
    runner.config.platforms[Platform.TELEGRAM].channel_overrides = {
        "777": ChannelOverride(model="pinned/lane-model", provider="custom")}
    seen = {}

    async def _capture(ctx, raw_input, explicit_provider, source):
        seen.update(provider=ctx.current_provider, base_url=ctx.current_base_url, model=ctx.current_model)
        return None, "stop"
    monkeypatch.setattr(runner, "_perform_model_switch", _capture)
    await runner._handle_model_command(_event("/model other-model --provider custom", "777"))
    assert seen["model"] == "pinned/lane-model"
    assert seen["base_url"].rstrip("/") == "http://127.0.0.1:8999/v1"


@pytest.mark.asyncio
async def test_model_switch_keeps_named_custom_identity_of_channel_pin(home):
    """Two named endpoints declare ``shared-model``; the channel pins one. /model reports that named
    provider and a bare-name switch resolves to it, not an ambiguous bare ``custom`` transport."""
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "shared-model", "provider": "custom:global-local"},
        "custom_providers": [
            {"name": "global-local", "base_url": "http://127.0.0.1:9/v1", "api_key": "k-global",
             "models": ["shared-model"]},
            {"name": "channel-local", "base_url": "http://127.0.0.1:7/v1", "api_key": "k-channel",
             "models": ["shared-model"]}]}), encoding="utf-8")
    runner = _runner()
    runner.config.platforms[Platform.TELEGRAM].channel_overrides = {
        "777": ChannelOverride(model="shared-model", provider="custom:channel-local")}
    event = _event("/model shared-model --session", "777")
    assert "custom:channel-local" in runner._format_session_info(event.source)
    reply = await runner._handle_model_command(event)
    assert "multiple configured providers" not in reply
    override = runner._session_model_overrides[runner._session_key_for_source(event.source)]
    assert (override["provider"], override["base_url"].rstrip("/")) == ("custom:channel-local", "http://127.0.0.1:7/v1")
