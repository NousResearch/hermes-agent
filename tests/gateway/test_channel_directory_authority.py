"""Adapter enumeration errors must not grant less-scoped session discovery authority."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import channel_directory
from gateway.config import Platform


@pytest.fixture
def discovery_fallbacks(tmp_path, monkeypatch):
    ambient = [{"id": "unowned", "name": "Another receiver's target", "type": "group"}]
    sessions = Mock(return_value=ambient)
    discord = Mock(return_value=ambient)
    slack = AsyncMock(return_value=ambient)
    monkeypatch.setattr(channel_directory, "DIRECTORY_PATH", tmp_path / "directory.json")
    monkeypatch.setattr(channel_directory, "CHANNEL_ALIASES_PATH", tmp_path / "no-aliases.json")
    monkeypatch.setattr(channel_directory, "_build_from_sessions", sessions)
    monkeypatch.setattr(channel_directory, "_build_discord", discord)
    monkeypatch.setattr(channel_directory, "_build_slack", slack)
    return ambient, sessions, discord, slack


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.DISCORD])
@pytest.mark.parametrize("raises", [False, True], ids=["empty-authoritative", "guard-error"])
async def test_authoritative_enumerator_never_grants_fallback(platform, raises, discovery_fallbacks, caplog):
    _, sessions, discord, slack = discovery_fallbacks
    hook = AsyncMock(return_value=[])
    if raises:
        hook.side_effect = RuntimeError("synthetic enumeration guard")
    with caplog.at_level(logging.WARNING, logger="gateway.channel_directory"):
        directory = await channel_directory.build_channel_directory({platform: SimpleNamespace(list_channels=hook)})
    hook.assert_awaited_once_with()
    assert directory["platforms"][platform.value] == []
    assert channel_directory.load_directory() == directory
    sessions.assert_not_called()
    discord.assert_not_called()
    slack.assert_not_called()
    if raises:
        assert "failed to build" in caplog.text
        assert platform.value in caplog.text
        assert "synthetic enumeration guard" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.DISCORD])
@pytest.mark.parametrize("explicit_none", [False, True], ids=["no-enumerator", "explicit-opt-out"])
async def test_only_opt_out_or_absent_enumerator_keeps_existing_fallback(platform, explicit_none, discovery_fallbacks):
    ambient, sessions, discord, slack = discovery_fallbacks
    adapter = SimpleNamespace()
    if explicit_none:
        adapter.list_channels = AsyncMock(return_value=None)
    directory = await channel_directory.build_channel_directory({platform: adapter})
    assert directory["platforms"][platform.value] == ambient
    assert channel_directory.load_directory() == directory
    if explicit_none:
        adapter.list_channels.assert_awaited_once_with()
    if platform == Platform.DISCORD:
        discord.assert_called_once_with(adapter)
        sessions.assert_not_called()
    else:
        sessions.assert_called_once_with(platform.value)
        discord.assert_not_called()
    slack.assert_not_called()
