import logging

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, interactive_setup


def _make_adapter() -> DiscordAdapter:
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def test_discord_fail_closed_default_logs_once(monkeypatch, caplog):
    adapter = _make_adapter()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    for var in (
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)

    with caplog.at_level(logging.WARNING):
        adapter._warn_if_fail_closed_default()
        adapter._warn_if_fail_closed_default()

    messages = [record.message for record in caplog.records]
    matches = [
        msg for msg in messages
        if "Discord messages are being denied because no allowlist is configured" in msg
    ]
    assert len(matches) == 1
    assert "DISCORD_ALLOWED_USERS" in matches[0]
    assert "DISCORD_ALLOW_ALL_USERS=true" in matches[0]


