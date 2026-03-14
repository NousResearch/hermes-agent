"""Tests for gateway configuration management."""

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
    load_gateway_config,
)


class TestHomeChannelRoundtrip:
    def test_to_dict_from_dict(self):
        hc = HomeChannel(platform=Platform.DISCORD, chat_id="999", name="general")
        d = hc.to_dict()
        restored = HomeChannel.from_dict(d)

        assert restored.platform == Platform.DISCORD
        assert restored.chat_id == "999"
        assert restored.name == "general"


class TestPlatformConfigRoundtrip:
    def test_to_dict_from_dict(self):
        pc = PlatformConfig(
            enabled=True,
            token="tok_123",
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="555",
                name="Home",
            ),
            extra={"foo": "bar"},
        )
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)

        assert restored.enabled is True
        assert restored.token == "tok_123"
        assert restored.home_channel.chat_id == "555"
        assert restored.extra == {"foo": "bar"}

    def test_disabled_no_token(self):
        pc = PlatformConfig()
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)
        assert restored.enabled is False
        assert restored.token is None


class TestGetConnectedPlatforms:
    def test_returns_enabled_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
                Platform.DISCORD: PlatformConfig(enabled=False, token="d"),
                Platform.SLACK: PlatformConfig(enabled=True),  # no token
            },
        )
        connected = config.get_connected_platforms()
        assert Platform.TELEGRAM in connected
        assert Platform.DISCORD not in connected
        assert Platform.SLACK not in connected

    def test_empty_platforms(self):
        config = GatewayConfig()
        assert config.get_connected_platforms() == []


class TestSessionResetPolicy:
    def test_roundtrip(self):
        policy = SessionResetPolicy(mode="idle", at_hour=6, idle_minutes=120)
        d = policy.to_dict()
        restored = SessionResetPolicy.from_dict(d)
        assert restored.mode == "idle"
        assert restored.at_hour == 6
        assert restored.idle_minutes == 120

    def test_defaults(self):
        policy = SessionResetPolicy()
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440


class TestGatewayConfigRoundtrip:
    def test_full_roundtrip(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="tok_123",
                    home_channel=HomeChannel(Platform.TELEGRAM, "123", "Home"),
                ),
            },
            reset_triggers=["/new"],
            quick_commands={"limits": {"type": "exec", "command": "echo ok"}},
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok_123"
        assert restored.reset_triggers == ["/new"]
        assert restored.quick_commands == {"limits": {"type": "exec", "command": "echo ok"}}


class TestLoadGatewayConfig:
    def test_bridges_quick_commands_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "quick_commands:\n"
            "  limits:\n"
            "    type: exec\n"
            "    command: echo ok\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.quick_commands == {"limits": {"type": "exec", "command": "echo ok"}}

    def test_invalid_quick_commands_in_config_yaml_are_ignored(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("quick_commands: not-a-mapping\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.quick_commands == {}
