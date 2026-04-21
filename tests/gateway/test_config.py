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

    def test_from_dict_coerces_quoted_false_always_log_local(self):
        restored = GatewayConfig.from_dict({"always_log_local": "false"})
        assert restored.always_log_local is False


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


# ---------------------------------------------------------------------------
# Regression coverage for #13582 — warn when non-secret gateway settings
# are set in BOTH config.yaml and the environment.
#
# Environment variables win by design, but stale .env entries silently
# override config.yaml edits (reporter's case: DISCORD_REPLY_TO_MODE=off
# in .env overriding discord.reply_to_mode: first in config.yaml).  These
# tests pin that ``_warn_yaml_env_conflicts`` surfaces the conflict as a
# WARNING-level log record so users can clean up the duplicated value.
# ---------------------------------------------------------------------------


class TestYamlEnvConflictWarnings:

    def _load(self, tmp_path, monkeypatch, yaml_text: str) -> None:
        """Write config.yaml under HERMES_HOME and call load_gateway_config.
        ``exist_ok=True`` so tests that invoke ``_load`` multiple times
        in a single scenario don't trip on directory reuse."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "config.yaml").write_text(yaml_text, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        load_gateway_config()

    def _conflict_records(self, caplog) -> list:
        """Return only the conflict-warning records ``_warn_yaml_env_conflicts``
        emitted (filter out other gateway.config warnings)."""
        return [
            r for r in caplog.records
            if r.name == "gateway.config"
            and r.levelno == logging.WARNING
            and "Gateway config conflict" in r.getMessage()
        ]

    def test_reporter_repro_discord_reply_to_mode_warns(
        self, tmp_path, monkeypatch, caplog
    ):
        """Reporter's exact scenario: discord.reply_to_mode=first in
        config.yaml, DISCORD_REPLY_TO_MODE=off in env. Both set, values
        differ → warning fires (#13582)."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  reply_to_mode: first\n")
        recs = self._conflict_records(caplog)
        assert len(recs) == 1, (
            f"Expected exactly 1 conflict warning for discord.reply_to_mode; "
            f"got {len(recs)}: {[r.getMessage() for r in recs]}"
        )
        msg = recs[0].getMessage()
        assert "discord.reply_to_mode" in msg
        assert "DISCORD_REPLY_TO_MODE" in msg
        # Both values surface so the user knows which to change.
        assert "first" in msg
        assert "off" in msg
        # Actionable guidance mentions the env file.
        assert ".env" in msg

    def test_no_warning_when_yaml_only(self, tmp_path, monkeypatch, caplog):
        """YAML set, env unset → no conflict to warn about (YAML will
        bridge to env normally)."""
        monkeypatch.delenv("DISCORD_REPLY_TO_MODE", raising=False)
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  reply_to_mode: first\n")
        assert self._conflict_records(caplog) == []

    def test_no_warning_when_env_only(self, tmp_path, monkeypatch, caplog):
        """Env set, YAML unset → no conflict (no surprise)."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch, "{}\n")
        assert self._conflict_records(caplog) == []

    def test_no_warning_when_values_match(self, tmp_path, monkeypatch, caplog):
        """Both set to the same effective value → silent (no-op for user)."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "first")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  reply_to_mode: first\n")
        assert self._conflict_records(caplog) == []

    def test_value_match_is_case_insensitive(self, tmp_path, monkeypatch, caplog):
        """YAML "First" + env "FIRST" = same effective value, no warning."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "FIRST")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  reply_to_mode: First\n")
        assert self._conflict_records(caplog) == []

    def test_boolean_yaml_compared_as_lowercased_string(
        self, tmp_path, monkeypatch, caplog
    ):
        """YAML ``require_mention: true`` vs env ``DISCORD_REQUIRE_MENTION=false``
        → conflict (bool must normalise to the same ``true``/``false`` string
        that the YAML→env bridges later emit)."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  require_mention: true\n")
        recs = self._conflict_records(caplog)
        assert len(recs) == 1
        assert "discord.require_mention" in recs[0].getMessage()

    def test_list_yaml_compared_as_comma_joined(
        self, tmp_path, monkeypatch, caplog
    ):
        """YAML list → env var bridges serialise as comma-joined strings
        (``DISCORD_ALLOWED_CHANNELS=chan1,chan2``).  Conflict detection
        must use the same serialisation so list-vs-string env comparison
        works correctly."""
        # Matching list + env value: no warning.
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "c1,c2")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  allowed_channels:\n    - c1\n    - c2\n")
        assert self._conflict_records(caplog) == []

        caplog.clear()
        # Mismatched → warning fires.
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "different")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  allowed_channels:\n    - c1\n    - c2\n")
        recs = self._conflict_records(caplog)
        assert len(recs) == 1
        assert "discord.allowed_channels" in recs[0].getMessage()

    def test_multiple_conflicts_each_warn_once(
        self, tmp_path, monkeypatch, caplog
    ):
        """Multiple conflicting settings → one warning each (no suppression,
        no duplication)."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "false")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(
                tmp_path, monkeypatch,
                "discord:\n  reply_to_mode: first\n"
                "telegram:\n  require_mention: true\n",
            )
        recs = self._conflict_records(caplog)
        assert len(recs) == 2
        msgs = [r.getMessage() for r in recs]
        assert any("discord.reply_to_mode" in m for m in msgs)
        assert any("telegram.require_mention" in m for m in msgs)

    def test_slack_mappings_covered(self, tmp_path, monkeypatch, caplog):
        """Slack keys are in _YAML_ENV_NON_SECRET_MAPPINGS too — pin that
        the warning coverage isn't Discord-only."""
        monkeypatch.setenv("SLACK_REQUIRE_MENTION", "true")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "slack:\n  require_mention: false\n")
        recs = self._conflict_records(caplog)
        assert len(recs) == 1
        assert "slack.require_mention" in recs[0].getMessage()

    def test_tokens_not_covered(self, tmp_path, monkeypatch, caplog):
        """Narrow-scope canary: tokens/credentials are DELIBERATELY absent
        from ``_YAML_ENV_NON_SECRET_MAPPINGS`` because env is the correct
        authoritative location for secrets.  Setting a token in both places
        must NOT produce a warning."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-token")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(
                tmp_path, monkeypatch,
                "platforms:\n  discord:\n    token: yaml-token\n",
            )
        assert self._conflict_records(caplog) == []

    def test_nested_missing_path_no_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """If the YAML doesn't have the parent key (``discord:``), the
        conflict walker must not raise even when env var is set."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch, "slack:\n  token: xyz\n")
        assert self._conflict_records(caplog) == []

    def test_empty_env_value_treated_as_unset(
        self, tmp_path, monkeypatch, caplog
    ):
        """``DISCORD_REPLY_TO_MODE=""`` (empty string) is treated as unset
        — consistent with how ``os.getenv`` + downstream code uses it.
        No warning even though the env var is technically present."""
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "")
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            self._load(tmp_path, monkeypatch,
                       "discord:\n  reply_to_mode: first\n")
        assert self._conflict_records(caplog) == []
