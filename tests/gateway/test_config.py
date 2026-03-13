"""Tests for gateway configuration management."""

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
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

    def test_explicit_null_values_use_defaults(self):
        """
        Regression test for issue #1119: explicit null values in config.yaml
        should use defaults, not None.
        
        Python's dict.get() returns None when key exists with null value,
        which differs from when the key is missing entirely.
        """
        # Simulate YAML with explicit null values
        data = {
            "mode": None,
            "at_hour": None,
            "idle_minutes": None,
        }
        policy = SessionResetPolicy.from_dict(data)
        
        # Should use defaults, not None
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440

    def test_partial_null_values(self):
        """Partial null values should default while preserving set values."""
        data = {
            "mode": "idle",
            "at_hour": None,  # null
            "idle_minutes": 60,  # set
        }
        policy = SessionResetPolicy.from_dict(data)
        
        assert policy.mode == "idle"
        assert policy.at_hour == 4  # defaulted
        assert policy.idle_minutes == 60  # preserved

    def test_empty_dict_uses_defaults(self):
        """Empty dict (missing keys) should use all defaults."""
        policy = SessionResetPolicy.from_dict({})
        
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440


class TestGatewayConfigRoundtrip:
    def test_full_roundtrip(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="tok",
                    home_channel=HomeChannel(Platform.TELEGRAM, "123", "Home"),
                ),
            },
            reset_triggers=["/new"],
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok"
        assert restored.reset_triggers == ["/new"]
