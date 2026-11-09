"""/model and /footer report the chat's channel_overrides model, the one its next turn runs."""

import pytest
import hermes_yaml as yaml

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


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
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id, expected, absent", [
    ("777", "pinned/lane-model", "global/default-model"),
    ("888", "global/default-model", "pinned/lane-model"),
])
async def test_model_header_reports_channel_override(home, monkeypatch, thread_id, expected, absent):
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [])
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
