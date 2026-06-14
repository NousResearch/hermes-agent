"""Tests for gateway configuration management."""

import logging
import os
from unittest.mock import patch

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
    StreamingConfig,
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

    def test_from_dict_coerces_quoted_false_enabled(self):
        restored = PlatformConfig.from_dict({"enabled": "false"})
        assert restored.enabled is False

    def test_gateway_restart_notification_defaults_true(self):
        assert PlatformConfig().gateway_restart_notification is True
        assert PlatformConfig.from_dict({}).gateway_restart_notification is True

    def test_gateway_restart_notification_roundtrip_false(self):
        pc = PlatformConfig(enabled=True, gateway_restart_notification=False)
        restored = PlatformConfig.from_dict(pc.to_dict())
        assert restored.gateway_restart_notification is False

    def test_gateway_restart_notification_coerces_quoted_false(self):
        restored = PlatformConfig.from_dict({"gateway_restart_notification": "false"})
        assert restored.gateway_restart_notification is False


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

    def test_dingtalk_recognised_via_extras(self):
        config = GatewayConfig(
            platforms={
                Platform.DINGTALK: PlatformConfig(
                    enabled=True,
                    extra={"client_id": "cid", "client_secret": "sec"},
                ),
            },
        )
        assert Platform.DINGTALK in config.get_connected_platforms()

    def test_dingtalk_recognised_via_env_vars(self, monkeypatch):
        """DingTalk configured via env vars (no extras) should still be
        recognised as connected — covers the case where _apply_env_overrides
        hasn't populated extras yet."""
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "env_cid")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "env_sec")
        config = GatewayConfig(
            platforms={
                Platform.DINGTALK: PlatformConfig(enabled=True, extra={}),
            },
        )
        assert Platform.DINGTALK in config.get_connected_platforms()

    def test_dingtalk_missing_creds_not_connected(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        config = GatewayConfig(
            platforms={
                Platform.DINGTALK: PlatformConfig(enabled=True, extra={}),
            },
        )
        assert Platform.DINGTALK not in config.get_connected_platforms()

    def test_dingtalk_disabled_not_connected(self):
        config = GatewayConfig(
            platforms={
                Platform.DINGTALK: PlatformConfig(
                    enabled=False,
                    extra={"client_id": "cid", "client_secret": "sec"},
                ),
            },
        )
        assert Platform.DINGTALK not in config.get_connected_platforms()


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

    def test_from_dict_coerces_quoted_false_notify(self):
        restored = SessionResetPolicy.from_dict({"notify": "false"})
        assert restored.notify is False


class TestStreamingConfig:
    def test_defaults_to_auto_transport(self):
        # "auto" prefers native draft streaming where the platform supports
        # it (Telegram DMs) and falls back to edit-based everywhere else, so
        # it is safe as the global out-of-the-box default.
        restored = StreamingConfig.from_dict({"enabled": "true"})
        assert restored.transport == "auto"

    def test_from_dict_coerces_quoted_false_enabled(self):
        restored = StreamingConfig.from_dict({"enabled": "false"})
        assert restored.enabled is False

    def test_from_dict_malformed_numeric_values_fall_back_to_defaults(self):
        restored = StreamingConfig.from_dict(
            {
                "edit_interval": "oops",
                "buffer_threshold": "oops",
                "fresh_final_after_seconds": "oops",
            }
        )
        assert restored.edit_interval == 0.8
        assert restored.buffer_threshold == 24
        assert restored.fresh_final_after_seconds == 60.0


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

    def test_max_concurrent_sessions_from_dict_normalizes_disabled_values(self):
        assert GatewayConfig.from_dict({}).max_concurrent_sessions is None
        assert GatewayConfig.from_dict({"max_concurrent_sessions": None}).max_concurrent_sessions is None
        assert GatewayConfig.from_dict({"max_concurrent_sessions": 0}).max_concurrent_sessions is None
        assert GatewayConfig.from_dict({"max_concurrent_sessions": -1}).max_concurrent_sessions is None

    def test_max_concurrent_sessions_from_dict_accepts_positive_integer(self):
        config = GatewayConfig.from_dict({"max_concurrent_sessions": "3"})

        assert config.max_concurrent_sessions == 3

    def test_max_concurrent_sessions_from_dict_ignores_invalid_values(self, caplog):
        caplog.set_level(logging.WARNING, logger="gateway.config")

        config = GatewayConfig.from_dict({"max_concurrent_sessions": "many"})

        assert config.max_concurrent_sessions is None
        assert any(
            "Ignoring invalid max_concurrent_sessions='many'" in record.message
            for record in caplog.records
        )

    def test_max_concurrent_sessions_from_dict_accepts_nested_fallback(self):
        config = GatewayConfig.from_dict({"gateway": {"max_concurrent_sessions": 4}})

        assert config.max_concurrent_sessions == 4

    def test_max_concurrent_sessions_top_level_overrides_nested(self):
        config = GatewayConfig.from_dict(
            {
                "gateway": {"max_concurrent_sessions": 4},
                "max_concurrent_sessions": 2,
            }
        )

        assert config.max_concurrent_sessions == 2

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

    def test_from_dict_coerces_quoted_false_always_log_local(self):
        restored = GatewayConfig.from_dict({"always_log_local": "false"})
        assert restored.always_log_local is False

    def test_get_notice_delivery_defaults_to_public(self):
        config = GatewayConfig(
            platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")}
        )

        assert config.get_notice_delivery(Platform.SLACK) == "public"

    def test_get_notice_delivery_honors_platform_override(self):
        config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={"notice_delivery": "private"},
                ),
            }
        )

        assert config.get_notice_delivery(Platform.SLACK) == "private"


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

    def test_bridges_top_level_max_concurrent_sessions_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("max_concurrent_sessions: 2\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.max_concurrent_sessions == 2

    def test_bridges_nested_max_concurrent_sessions_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "gateway:\n"
            "  max_concurrent_sessions: 3\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.max_concurrent_sessions == 3

    def test_top_level_max_concurrent_sessions_overrides_nested_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "max_concurrent_sessions: 2\n"
            "gateway:\n"
            "  max_concurrent_sessions: 3\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.max_concurrent_sessions == 2

    def test_bridges_discord_thread_require_mention_from_config_yaml(self, tmp_path, monkeypatch):
        """discord.thread_require_mention in config.yaml should reach the runtime env var."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  thread_require_mention: true\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_THREAD_REQUIRE_MENTION", raising=False)

        load_gateway_config()

        assert os.environ.get("DISCORD_THREAD_REQUIRE_MENTION") == "true"

    def test_thread_require_mention_yaml_does_not_overwrite_env(self, tmp_path, monkeypatch):
        """Explicit env var should win over config.yaml (env > yaml precedence)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  thread_require_mention: false\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "true")  # user override

        load_gateway_config()

        # Env value preserved, not clobbered by yaml.
        assert os.environ.get("DISCORD_THREAD_REQUIRE_MENTION") == "true"

    def test_bridges_discord_allow_from_from_config_yaml(self, tmp_path, monkeypatch):
        """discord.allow_from should populate DISCORD_ALLOWED_USERS for auth."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  allow_from:\n"
            "    - \"123456789012345678\"\n"
            "    - \"999888777666555444\"\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)

        config = load_gateway_config()

        assert config.platforms[Platform.DISCORD].extra["allow_from"] == [
            "123456789012345678",
            "999888777666555444",
        ]
        assert os.environ.get("DISCORD_ALLOWED_USERS") == (
            "123456789012345678,999888777666555444"
        )

    def test_bridges_discord_platform_extra_allow_from_to_env(self, tmp_path, monkeypatch):
        """platforms.discord.extra.allow_from should reach DISCORD_ALLOWED_USERS too."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "platforms:\n"
            "  discord:\n"
            "    extra:\n"
            "      allow_from:\n"
            "        - \"123456789012345678\"\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)

        config = load_gateway_config()

        assert config.platforms[Platform.DISCORD].extra["allow_from"] == [
            "123456789012345678",
        ]
        assert os.environ.get("DISCORD_ALLOWED_USERS") == "123456789012345678"

    def test_bridges_quoted_false_platform_enabled_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "platforms:\n"
            "  api_server:\n"
            "    enabled: \"false\"\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.API_SERVER].enabled is False
        assert Platform.API_SERVER not in config.get_connected_platforms()

    def test_bridges_nested_gateway_platforms_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      enabled: true\n"
            "      token: nested-token\n"
            "      home_channel:\n"
            "        platform: telegram\n"
            "        chat_id: \"123\"\n"
            "        name: Nested Home\n"
            "      extra:\n"
            "        reply_prefix: nested\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        telegram = config.platforms[Platform.TELEGRAM]
        assert telegram.enabled is True
        assert telegram.token == "nested-token"
        assert telegram.home_channel == HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="123",
            name="Nested Home",
        )
        assert telegram.extra["reply_prefix"] == "nested"

    def test_top_level_platforms_override_nested_gateway_platforms(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      enabled: false\n"
            "      token: nested-token\n"
            "      extra:\n"
            "        reply_prefix: nested\n"
            "platforms:\n"
            "  telegram:\n"
            "    enabled: true\n"
            "    token: top-token\n"
            "    extra:\n"
            "      reply_prefix: top\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        telegram = config.platforms[Platform.TELEGRAM]
        assert telegram.enabled is True
        assert telegram.token == "top-token"
        assert telegram.extra["reply_prefix"] == "top"

    def test_shared_key_loop_bridges_allow_from_from_nested_platforms(self, tmp_path, monkeypatch):
        """Regression: shared-key loop must bridge allow_from / require_mention
        into PlatformConfig.extra even when the platform is configured only
        under ``platforms:`` (no top-level ``telegram:`` block).

        Before the fix, ``platform_cfg = yaml_cfg.get('telegram')`` returned
        None for nested-only configs, so the loop skipped the platform entirely
        and allow_from was silently ignored.  The apply_yaml_config_fn dispatch
        received the same fix in #44f3e51; the shared-key loop now mirrors it.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "platforms:\n"
            "  telegram:\n"
            "    allow_from:\n"
            "      - \"111222333\"\n"
            "      - \"444555666\"\n"
            "    require_mention: true\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        telegram = config.platforms[Platform.TELEGRAM]
        assert telegram.extra.get("allow_from") == ["111222333", "444555666"], (
            "allow_from configured under platforms.telegram must be bridged "
            "into PlatformConfig.extra by the shared-key loop"
        )
        assert telegram.extra.get("require_mention") is True, (
            "require_mention configured under platforms.telegram must be "
            "bridged into PlatformConfig.extra by the shared-key loop"
        )

    def test_shared_key_loop_bridges_allow_from_from_nested_gateway_platforms(self, tmp_path, monkeypatch):
        """Same regression check for ``gateway.platforms:`` path."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      allow_from:\n"
            "        - \"777888999\"\n"
            "      require_mention: false\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        telegram = config.platforms[Platform.TELEGRAM]
        assert telegram.extra.get("allow_from") == ["777888999"], (
            "allow_from configured under gateway.platforms.telegram must be "
            "bridged into PlatformConfig.extra by the shared-key loop"
        )
        assert telegram.extra.get("require_mention") is False

    def test_bridges_quoted_false_session_notify_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "session_reset:\n"
            "  notify: \"false\"\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.default_reset_policy.notify is False

    def test_bridges_quoted_false_always_log_local_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "always_log_local: \"false\"\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.always_log_local is False

    def test_bridges_discord_channel_prompts_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  channel_prompts:\n"
            "    \"123\": Research mode\n"
            "    456: Therapist mode\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.DISCORD].extra["channel_prompts"] == {
            "123": "Research mode",
            "456": "Therapist mode",
        }

    def test_bridges_discord_history_backfill_settings_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  history_backfill: true\n"
            "  history_backfill_limit: 17\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_HISTORY_BACKFILL", raising=False)
        monkeypatch.delenv("DISCORD_HISTORY_BACKFILL_LIMIT", raising=False)

        load_gateway_config()

        assert os.getenv("DISCORD_HISTORY_BACKFILL") == "true"
        assert os.getenv("DISCORD_HISTORY_BACKFILL_LIMIT") == "17"

    def test_bridges_telegram_channel_prompts_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "telegram:\n"
            "  channel_prompts:\n"
            '    "-1001234567": Research assistant\n'
            "    789: Creative writing\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.TELEGRAM].extra["channel_prompts"] == {
            "-1001234567": "Research assistant",
            "789": "Creative writing",
        }

    def test_bridges_slack_channel_prompts_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "slack:\n"
            "  channel_prompts:\n"
            '    "C01ABC": Code review mode\n',
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.SLACK].extra["channel_prompts"] == {
            "C01ABC": "Code review mode",
        }

    def test_bridges_feishu_allow_bots_from_config_yaml_to_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "feishu:\n  allow_bots: mentions\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("FEISHU_ALLOW_BOTS", raising=False)

        load_gateway_config()

        assert os.environ.get("FEISHU_ALLOW_BOTS") == "mentions"

    def test_feishu_allow_bots_env_takes_precedence_over_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "feishu:\n  allow_bots: all\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("FEISHU_ALLOW_BOTS", "none")

        load_gateway_config()

        assert os.environ.get("FEISHU_ALLOW_BOTS") == "none"

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

    def test_bridges_telegram_disable_link_previews_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "telegram:\n"
            "  disable_link_previews: true\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.TELEGRAM].extra["disable_link_previews"] is True

    def test_loads_telegram_rich_messages_from_gateway_platform_extra(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      extra:\n"
            "        rich_messages: false\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.TELEGRAM].extra["rich_messages"] is False

    def test_load_config_default_includes_telegram_rich_messages(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from hermes_cli.config import load_config

        config = load_config()

        assert config["telegram"]["extra"]["rich_messages"] is True

    def test_bridges_telegram_extra_base_url_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "telegram:\n"
            "  extra:\n"
            "    base_url: https://custom-proxy.example.com/bot\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert (
            config.platforms[Platform.TELEGRAM].extra["base_url"]
            == "https://custom-proxy.example.com/bot"
        )

    def test_bridges_notice_delivery_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "slack:\n"
            "  notice_delivery: private\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.get_notice_delivery(Platform.SLACK) == "private"

    def test_bridges_telegram_proxy_url_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "telegram:\n"
            "  proxy_url: socks5://127.0.0.1:1080\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TELEGRAM_PROXY", raising=False)

        load_gateway_config()

        import os
        assert os.environ.get("TELEGRAM_PROXY") == "socks5://127.0.0.1:1080"

    def test_telegram_proxy_env_takes_precedence_over_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "telegram:\n"
            "  proxy_url: http://from-config:8080\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TELEGRAM_PROXY", "socks5://from-env:1080")

        load_gateway_config()

        import os
        assert os.environ.get("TELEGRAM_PROXY") == "socks5://from-env:1080"


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
                Platform.WHATSAPP,
                PlatformConfig(enabled=True),
                {
                    "WHATSAPP_HOME_CHANNEL": "1234567890@lid",
                    "WHATSAPP_HOME_CHANNEL_NAME": "Owner DM",
                },
                ("1234567890@lid", "Owner DM"),
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
