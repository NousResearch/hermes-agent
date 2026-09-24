"""The CLI must persist reply modes as strings, not YAML booleans (#121430)."""

import argparse

import pytest
import yaml

from gateway.config import Platform, load_gateway_config
from hermes_cli.config import config_command, set_config_value


@pytest.mark.parametrize("platform", ["telegram", "discord", "slack"])
@pytest.mark.parametrize("prefix", ["platforms", "gateway.platforms"])
@pytest.mark.parametrize("mode", ["off", "first", "all"])
def test_config_command_reply_mode_roundtrip(tmp_path, monkeypatch, platform, prefix, mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The gateway's YAML-to-env bridge may have run in a preceding case.
    monkeypatch.delenv("TELEGRAM_REPLY_TO_MODE", raising=False)
    monkeypatch.delenv("DISCORD_REPLY_TO_MODE", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"# Keep this comment\nplatforms:\n  {platform}:\n    enabled: true\n",
        encoding="utf-8",
    )
    config_command(argparse.Namespace(
        config_command="set", key=f"{prefix}.{platform}.reply_to_mode",
        value=mode, force=False,
    ))
    written = config_path.read_text(encoding="utf-8")
    stored = yaml.safe_load(written)
    assert stored["platforms"][platform]["reply_to_mode"] == mode
    assert stored["platforms"][platform]["enabled"] is True
    assert "# Keep this comment" in written
    assert "gateway" not in stored
    runtime = load_gateway_config().platforms[Platform(platform)]
    assert runtime.reply_to_mode == mode
    assert (runtime.reply_to_mode or "first") == mode


def test_other_platform_values_keep_existing_coercion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key, value in [
        ("platforms.telegram.enabled", "off"),
        ("platforms.telegram.extra.retry_count", "3"),
        ("platforms.telegram.extra.reply_to_mode", "off"),
        ("platforms.telegram.extra.channels", '["one", "two"]'),
        ("human_delay.mode", "off"),
    ]:
        set_config_value(key, value)
    stored = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    platform = stored["platforms"]["telegram"]
    assert platform["enabled"] is False
    assert platform["extra"] == {
        "retry_count": 3, "reply_to_mode": False, "channels": ["one", "two"],
    }
    assert stored["human_delay"]["mode"] == "off"


def test_invalid_container_still_refused_without_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    set_config_value("platforms.telegram.extra", '{"keep": true}')
    path = tmp_path / "config.yaml"
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        set_config_value("platforms.telegram.extra", "[broken")
    assert path.read_bytes() == before
