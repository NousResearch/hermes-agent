from __future__ import annotations

from gateway.config import Platform, load_gateway_config


def test_config_yaml_activates_only_credential_free_relay_consumer(
    tmp_path, monkeypatch
) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
gateway:
  platforms:
    discord:
      enabled: false
canonical_brain:
  audit_bridge:
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv(
        "GATEWAY_RELAY_URL",
        "unix:///run/muncho-discord-connector/connector.sock",
    )
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)

    config = load_gateway_config()

    assert config.platforms[Platform.RELAY].enabled is True
    assert config.platforms[Platform.RELAY].extra["relay_url"] == (
        "unix:///run/muncho-discord-connector/connector.sock"
    )
    assert config.platforms[Platform.DISCORD].enabled is False
    assert config.platforms[Platform.DISCORD].token is None
    assert Platform.RELAY in config.get_connected_platforms()
    assert Platform.DISCORD not in config.get_connected_platforms()
