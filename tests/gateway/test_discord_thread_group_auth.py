"""Discord channel participation and DM grants remain separate in threads."""
import pytest
from gateway.config import load_gateway_config, Platform
from gateway.session import SessionSource


def _runner(tmp_path, monkeypatch, policy):
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("discord:\n  allowed_channels: ['123']\n" + policy)
    config = load_gateway_config()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.DISCORD: DiscordAdapter(config.platforms[Platform.DISCORD])}
    runner.pairing_store = None
    return runner


@pytest.mark.parametrize("grant", ["*", "456"])
def test_group_grant_applies_to_discord_threads_but_not_dms(tmp_path, monkeypatch, grant):
    runner = _runner(tmp_path, monkeypatch, f"  group_allow_from: ['{grant}']\n")
    for chat_type in ("group", "thread"):
        source = SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="456", chat_type=chat_type)
        assert runner._is_user_authorized(source)
    dm = SessionSource(platform=Platform.DISCORD, chat_id="789", user_id="456", chat_type="dm")
    assert not runner._is_user_authorized(dm)
    adapter = runner.adapters[Platform.DISCORD]
    assert adapter._is_allowed_user("456", is_dm=False, channel_ids={"123"})
    assert not adapter._is_allowed_user("456", is_dm=False, channel_ids={"999"})
    assert not adapter._is_allowed_user("456", is_dm=True)


def test_dm_config_grant_does_not_become_a_thread_grant(tmp_path, monkeypatch):
    # Use the generic config fallback without Discord's platform-wide env bridge.
    runner = _runner(tmp_path, monkeypatch, "  group_allow_from: ['someone-else']\n")
    runner.adapters[Platform.DISCORD].config.extra['allow_from'] = ['456']
    dm = SessionSource(platform=Platform.DISCORD, chat_id="789", user_id="456", chat_type="dm")
    thread = SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="456", chat_type="thread")
    assert runner._is_user_authorized(dm)
    assert not runner._is_user_authorized(thread)
