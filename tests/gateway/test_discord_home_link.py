"""Discord channel links are accepted wherever a channel id is (#115216).

Copy Link sits next to Copy Channel ID in Discord's context menu; a pasted link used to reach
``int()`` inside the adapter and every home-channel / cron delivery died as a generic send failure.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import HomeChannel, Platform, load_gateway_config
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.parametrize("source", ["env", "yaml"])
def test_discord_home_link_loads_as_channel_id(tmp_path, monkeypatch, source):
    channel_id = "202523857364451329"
    link = f"https://discord.com/channels/302523857364451329/{channel_id}"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    config_text = "platforms:\n  discord:\n    enabled: false\n"
    if source == "env":
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", link)
    else:
        config_text += f"    home_channel:\n      platform: discord\n      chat_id: '{link}'\n"
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    home = load_gateway_config().platforms[Platform.DISCORD].home_channel
    assert home.chat_id == channel_id
    # Anything that is not a channel link keeps its own error path (message links included).
    for target in ("123456789", "https://discord.com/channels/123/456/789", "123/456"):
        assert HomeChannel(Platform.DISCORD, target, "Home").chat_id == target
    assert HomeChannel(Platform.SLACK, link, "Home").chat_id == link


def test_resolve_channel_accepts_a_pasted_link_for_explicit_targets():
    """Cron ``deliver: discord:<link>`` bypasses HomeChannel, so the adapter's resolver — the
    chokepoint every outbound target passes through — must accept the link too."""
    adapter = object.__new__(DiscordAdapter)
    channel = SimpleNamespace(id=456)
    adapter._client = SimpleNamespace(get_channel=Mock(return_value=channel), fetch_channel=AsyncMock())

    resolved = asyncio.run(adapter._resolve_channel(" https://ptb.discord.com/channels/123/456 "))

    assert resolved is channel
    adapter._client.get_channel.assert_called_once_with(456)
