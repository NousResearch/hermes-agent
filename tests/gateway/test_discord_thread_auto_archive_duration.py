"""``thread_auto_archive_duration`` sets the auto-archive window of threads Hermes creates itself."""

import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.delenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", raising=False)
    return DiscordAdapter(PlatformConfig(enabled=True, token="fake"))


def test_thread_auto_archive_duration_defaults_to_one_day(adapter):
    assert adapter._discord_thread_auto_archive_duration() == 1440


def test_thread_auto_archive_duration_env_override(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", "10080")
    assert adapter._discord_thread_auto_archive_duration() == 10080


def test_thread_auto_archive_duration_invalid_falls_back_with_warning(adapter, caplog):
    adapter.config.extra["thread_auto_archive_duration"] = 90
    with caplog.at_level(logging.WARNING):
        assert adapter._discord_thread_auto_archive_duration() == 1440
    assert "Invalid thread_auto_archive_duration" in caplog.text


def test_thread_auto_archive_duration_authored_extra_survives_unrelated_root_block(tmp_path, monkeypatch):
    """A root ``discord:`` block without the key seeds nothing over ``platforms.discord.extra``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", raising=False)
    (tmp_path / "config.yaml").write_text(
        "discord:\n  require_mention: false\n"
        "platforms:\n  discord:\n    enabled: false\n    extra:\n      thread_auto_archive_duration: 10080\n",
        encoding="utf-8",
    )
    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    assert adapter._discord_thread_auto_archive_duration() == 10080
    assert "DISCORD_THREAD_AUTO_ARCHIVE_DURATION" not in os.environ


@pytest.mark.asyncio
async def test_auto_thread_uses_configured_archive_duration(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", "10080")
    thread = SimpleNamespace(id=800)
    message = SimpleNamespace(
        content="deploy the fix", author=SimpleNamespace(display_name="Alice"),
        create_thread=AsyncMock(return_value=thread),
    )

    assert await adapter._auto_create_thread(message) is thread

    assert message.create_thread.await_args.kwargs["auto_archive_duration"] == 10080
