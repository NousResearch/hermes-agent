"""Discord home links must be normalized before any delivery consumer sees them."""
import pytest
from gateway.config import HomeChannel, Platform, load_gateway_config


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


@pytest.mark.parametrize("host", ["discord.com", "ptb.discord.com", "canary.discord.com", "discordapp.com"])
def test_home_link_preserves_metadata_and_roundtrips(host):
    home = HomeChannel(Platform.DISCORD, f" https://{host}/channels/@me/123456789/ ",
                       "Home", thread_id="987", user_id="u", scope_id="s")
    assert home.chat_id == "123456789"
    assert HomeChannel.from_dict(home.to_dict()) == home
    assert (home.thread_id, home.user_id, home.scope_id) == ("987", "u", "s")


@pytest.mark.parametrize("target", [
    "123456789", "https://example.com/channels/123/456",
    "https://discord.com.evil.test/channels/123/456",
    "https://discord.com/channels/123/456/789",  # message link, not channel
    "https://discord.com/channels/123/not-a-channel", "123/456",
])
def test_unrecognized_targets_are_not_reinterpreted(target):
    assert HomeChannel(Platform.DISCORD, target, "Home").chat_id == target


def test_other_platform_home_is_unchanged():
    target = "https://discord.com/channels/123/456"
    assert HomeChannel(Platform.SLACK, target, "Home").chat_id == target
