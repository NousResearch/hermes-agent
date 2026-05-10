"""Tests for the config.yaml bridge of discord.thread_require_mention."""

import os

from gateway.config import Platform, load_gateway_config


def test_discord_thread_require_mention_yaml_bridges_to_platform_extra_and_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  require_mention: true\n"
        "  thread_require_mention: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_THREAD_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    discord_cfg = config.platforms[Platform.DISCORD]
    assert discord_cfg.extra["require_mention"] is True
    assert discord_cfg.extra["thread_require_mention"] is True
    assert os.getenv("DISCORD_REQUIRE_MENTION") == "true"
    assert os.getenv("DISCORD_THREAD_REQUIRE_MENTION") == "true"
