"""Tests for /sethome gateway slash command."""

import os

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(
    text="/sethome",
    platform=Platform.TELEGRAM,
    user_id="12345",
    chat_id="67890",
    chat_name="Home Chat",
):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        chat_name=chat_name,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


@pytest.mark.asyncio
async def test_sethome_persists_home_channel_to_env_file_not_config(monkeypatch, tmp_path):
    """The gateway /sethome command should persist HOME_CHANNEL in .env."""
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text("display:\n  compact: true\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    runner = object.__new__(GatewayRunner)
    event = _make_event()

    result = await runner._handle_set_home_command(event)

    env_path = hermes_home / ".env"
    assert env_path.exists()
    assert "TELEGRAM_HOME_CHANNEL=67890" in env_path.read_text(encoding="utf-8")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assert "TELEGRAM_HOME_CHANNEL" not in config

    assert os.environ["TELEGRAM_HOME_CHANNEL"] == "67890"
    assert "Home Chat" in result
