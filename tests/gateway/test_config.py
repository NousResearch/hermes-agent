"""Tests for gateway configuration management."""

import os
from unittest.mock import patch

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
    _apply_env_overrides,
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

    def test_from_dict_treats_null_values_as_defaults(self):
        restored = SessionResetPolicy.from_dict(
            {"mode": None, "at_hour": None, "idle_minutes": None}
        )
        assert restored.mode == "both"
        assert restored.at_hour == 4
        assert restored.idle_minutes == 1440


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
            group_sessions_per_user=False,
            thread_sessions_per_user=True,
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok_123"
        assert restored.reset_triggers == ["/new"]
        assert restored.quick_commands == {"limits": {"type": "exec", "command": "echo ok"}}
        assert restored.group_sessions_per_user is False
        assert restored.thread_sessions_per_user is True

    def test_roundtrip_preserves_unauthorized_dm_behavior(self):
        config = GatewayConfig(
            unauthorized_dm_behavior="ignore",
            platforms={
                Platform.WHATSAPP: PlatformConfig(
                    enabled=True,
                    extra={"unauthorized_dm_behavior": "pair"},
                ),
            },
        )

        restored = GatewayConfig.from_dict(config.to_dict())

        assert restored.unauthorized_dm_behavior == "ignore"
        assert restored.platforms[Platform.WHATSAPP].extra["unauthorized_dm_behavior"] == "pair"


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

    def test_bridges_group_sessions_per_user_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("group_sessions_per_user: false\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.group_sessions_per_user is False

    def test_bridges_thread_sessions_per_user_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("thread_sessions_per_user: true\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.thread_sessions_per_user is True

    def test_thread_sessions_per_user_defaults_to_false(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.thread_sessions_per_user is False

    def test_invalid_quick_commands_in_config_yaml_are_ignored(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("quick_commands: not-a-mapping\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.quick_commands == {}

    def test_bridges_unauthorized_dm_behavior_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "unauthorized_dm_behavior: ignore\n"
            "whatsapp:\n"
            "  unauthorized_dm_behavior: pair\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.unauthorized_dm_behavior == "ignore"
        assert config.platforms[Platform.WHATSAPP].extra["unauthorized_dm_behavior"] == "pair"

    def test_bridges_qq_napcat_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "qq_napcat:\n"
            "  ws_url: ws://127.0.0.1:3001\n"
            "  access_token: napcat-token\n"
            "  admin_users:\n"
            "    - '179033731'\n"
            "  allow_all_users: true\n"
            "  require_mention: true\n"
            "  mention_patterns:\n"
            "    - '^\\s*马噶\\b'\n"
            "  allowed_groups:\n"
            "    - '987654'\n"
            "  home_channel: group:987654\n"
            "  home_channel_name: QQ Home\n"
            "  project_group_mode: true\n"
            "  group_batch_debounce_seconds: 1.5\n"
            "  group_min_model_interval_seconds: 8\n"
            "  group_batch_max_messages: 40\n"
            "  reconnect_interval: 9\n"
            "  system_prompt: Return [[NO_REPLY]] when the group message does not need a reply.\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        platform = getattr(Platform, "QQ_NAPCAT")
        assert platform in config.platforms
        assert config.platforms[platform].enabled is True
        assert config.platforms[platform].home_channel.chat_id == "group:987654"
        assert config.platforms[platform].home_channel.name == "QQ Home"
        assert config.platforms[platform].extra["ws_url"] == "ws://127.0.0.1:3001"
        assert config.platforms[platform].extra["access_token"] == "napcat-token"
        assert config.platforms[platform].extra["admin_users"] == ["179033731"]
        assert config.platforms[platform].extra["allow_all_users"] is True
        assert config.platforms[platform].extra["require_mention"] is True
        assert config.platforms[platform].extra["mention_patterns"] == [r"^\s*马噶\b"]
        assert config.platforms[platform].extra["allowed_groups"] == ["987654"]
        assert config.platforms[platform].extra["project_group_mode"] is True
        assert config.platforms[platform].extra["group_batch_debounce_seconds"] == 1.5
        assert config.platforms[platform].extra["group_min_model_interval_seconds"] == 8.0
        assert config.platforms[platform].extra["group_batch_max_messages"] == 40
        assert config.platforms[platform].extra["reconnect_interval"] == 9
        assert "[[NO_REPLY]]" in config.platforms[platform].extra["system_prompt"]


class TestHomeChannelEnvOverrides:
    """Home channel env vars should apply even when the platform was already
    configured via config.yaml (not just when credential env vars create it)."""

    def test_existing_platform_configs_accept_home_channel_env_overrides(self):
        cases = [
            (
                Platform.SLACK,
                PlatformConfig(enabled=True, token="xoxb-from-config"),
                {"SLACK_HOME_CHANNEL": "C123", "SLACK_HOME_CHANNEL_NAME": "Ops"},
                ("C123", "Ops"),
            ),
            (
                Platform.SIGNAL,
                PlatformConfig(
                    enabled=True,
                    extra={"http_url": "http://localhost:9090", "account": "+15551234567"},
                ),
                {"SIGNAL_HOME_CHANNEL": "+1555000", "SIGNAL_HOME_CHANNEL_NAME": "Phone"},
                ("+1555000", "Phone"),
            ),
            (
                Platform.MATTERMOST,
                PlatformConfig(
                    enabled=True,
                    token="mm-token",
                    extra={"url": "https://mm.example.com"},
                ),
                {"MATTERMOST_HOME_CHANNEL": "ch_abc123", "MATTERMOST_HOME_CHANNEL_NAME": "General"},
                ("ch_abc123", "General"),
            ),
            (
                Platform.MATRIX,
                PlatformConfig(
                    enabled=True,
                    token="syt_abc123",
                    extra={"homeserver": "https://matrix.example.org"},
                ),
                {"MATRIX_HOME_ROOM": "!room123:example.org", "MATRIX_HOME_ROOM_NAME": "Bot Room"},
                ("!room123:example.org", "Bot Room"),
            ),
            (
                Platform.EMAIL,
                PlatformConfig(
                    enabled=True,
                    extra={
                        "address": "hermes@test.com",
                        "imap_host": "imap.test.com",
                        "smtp_host": "smtp.test.com",
                    },
                ),
                {"EMAIL_HOME_ADDRESS": "user@test.com", "EMAIL_HOME_ADDRESS_NAME": "Inbox"},
                ("user@test.com", "Inbox"),
            ),
            (
                Platform.SMS,
                PlatformConfig(enabled=True, api_key="token_abc"),
                {"SMS_HOME_CHANNEL": "+15559876543", "SMS_HOME_CHANNEL_NAME": "My Phone"},
                ("+15559876543", "My Phone"),
            ),
        ]

        for platform, platform_config, env, expected in cases:
            config = GatewayConfig(platforms={platform: platform_config})
            with patch.dict(os.environ, env, clear=True):
                _apply_env_overrides(config)

            home = config.platforms[platform].home_channel
            assert home is not None, f"{platform.value}: home_channel should not be None"
            assert (home.chat_id, home.name) == expected, platform.value


class TestQqNapCatEnvOverrides:
    def test_apply_env_overrides_sets_qq_napcat_fields(self):
        config = GatewayConfig()

        with patch.dict(
            os.environ,
            {
                "QQ_NAPCAT_WS_URL": "ws://127.0.0.1:3001",
                "QQ_NAPCAT_ACCESS_TOKEN": "napcat-token",
                "QQ_NAPCAT_ADMIN_USERS": "179033731",
                "QQ_NAPCAT_REQUIRE_MENTION": "true",
                "QQ_NAPCAT_MENTION_PATTERNS": "[\"^\\\\s*马噶\\\\b\"]",
                "QQ_NAPCAT_ALLOWED_GROUPS": "12345, 67890",
                "QQ_NAPCAT_ALLOW_ALL_GROUPS": "true",
                "QQ_NAPCAT_PROJECT_GROUP_MODE": "true",
                "QQ_NAPCAT_HOME_CHANNEL": "group:12345",
                "QQ_NAPCAT_HOME_CHANNEL_NAME": "QQ Home",
                "QQ_NAPCAT_SYSTEM_PROMPT": "Use [[NO_REPLY]] for low-signal group chatter.",
                "QQ_NAPCAT_GROUP_BATCH_DEBOUNCE_SECONDS": "1.25",
                "QQ_NAPCAT_GROUP_MIN_MODEL_INTERVAL_SECONDS": "8",
                "QQ_NAPCAT_GROUP_BATCH_MAX_MESSAGES": "50",
                "QQ_NAPCAT_RECONNECT_INTERVAL": "12",
            },
            clear=False,
        ):
            _apply_env_overrides(config)

        platform = getattr(Platform, "QQ_NAPCAT")
        assert config.platforms[platform].enabled is True
        assert config.platforms[platform].home_channel.chat_id == "group:12345"
        assert config.platforms[platform].home_channel.name == "QQ Home"
        assert config.platforms[platform].extra["ws_url"] == "ws://127.0.0.1:3001"
        assert config.platforms[platform].extra["access_token"] == "napcat-token"
        assert config.platforms[platform].extra["admin_users"] == ["179033731"]
        assert config.platforms[platform].extra["require_mention"] is True
        assert config.platforms[platform].extra["mention_patterns"] == [r"^\s*马噶\b"]
        assert config.platforms[platform].extra["allowed_groups"] == ["12345", "67890"]
        assert config.platforms[platform].extra["allow_all_groups"] is True
        assert config.platforms[platform].extra["project_group_mode"] is True
        assert config.platforms[platform].extra["group_batch_debounce_seconds"] == 1.25
        assert config.platforms[platform].extra["group_min_model_interval_seconds"] == 8.0
        assert config.platforms[platform].extra["group_batch_max_messages"] == 50
        assert config.platforms[platform].extra["reconnect_interval"] == 12
        assert "[[NO_REPLY]]" in config.platforms[platform].extra["system_prompt"]

    def test_qq_platform_can_override_group_session_isolation(self):
        platform = getattr(Platform, "QQ_NAPCAT")
        config = GatewayConfig(
            group_sessions_per_user=True,
            thread_sessions_per_user=True,
            platforms={
                platform: PlatformConfig(
                    enabled=True,
                    extra={
                        "group_sessions_per_user": False,
                        "thread_sessions_per_user": False,
                    },
                ),
            },
        )

        group_per_user, thread_per_user = config.get_session_isolation(platform)

        assert group_per_user is False
        assert thread_per_user is False

    def test_project_group_mode_defaults_qq_group_sessions_to_shared(self):
        platform = getattr(Platform, "QQ_NAPCAT")
        config = GatewayConfig(
            group_sessions_per_user=True,
            platforms={
                platform: PlatformConfig(
                    enabled=True,
                    extra={"project_group_mode": True},
                ),
            },
        )

        group_per_user, thread_per_user = config.get_session_isolation(platform)

        assert group_per_user is False
        assert thread_per_user is False
