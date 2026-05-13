"""Tests for gateway.display_config — per-platform display/verbosity resolver."""
import pytest


# ---------------------------------------------------------------------------
# Resolver: resolution order
# ---------------------------------------------------------------------------

class TestResolveDisplaySetting:
    """resolve_display_setting() resolves with correct priority."""

    def test_explicit_platform_override_wins(self):
        """display.platforms.<plat>.<key> takes top priority."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "platforms": {
                    "telegram": {"tool_progress": "verbose"},
                },
            }
        }
        assert resolve_display_setting(config, "telegram", "tool_progress") == "verbose"

    def test_global_setting_when_no_platform_override(self):
        """Falls back to display.<key> when no platform override exists."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "new",
                "platforms": {},
            }
        }
        assert resolve_display_setting(config, "telegram", "tool_progress") == "new"

    def test_platform_default_when_no_user_config(self):
        """Falls back to built-in platform default."""
        from gateway.display_config import resolve_display_setting

        # Empty config — should get built-in defaults
        config = {}
        # Telegram tier_high override: "new" (not "all") to reduce edit
        # pressure during streaming on Telegram's ~1 edit/s flood envelope.
        assert resolve_display_setting(config, "telegram", "tool_progress") == "new"
        # Email defaults to tier_minimal → "off"
        assert resolve_display_setting(config, "email", "tool_progress") == "off"

    def test_global_default_for_unknown_platform(self):
        """Unknown platforms get the global defaults."""
        from gateway.display_config import resolve_display_setting

        config = {}
        # Unknown platform, no config → global default "all"
        assert resolve_display_setting(config, "unknown_platform", "tool_progress") == "all"

    def test_fallback_parameter_used_last(self):
        """Explicit fallback is used when nothing else matches."""
        from gateway.display_config import resolve_display_setting

        config = {}
        # "nonexistent_key" isn't in any defaults
        result = resolve_display_setting(config, "telegram", "nonexistent_key", "my_fallback")
        assert result == "my_fallback"

    def test_platform_override_only_affects_that_platform(self):
        """Other platforms are unaffected by a specific platform override."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "platforms": {
                    "slack": {"tool_progress": "off"},
                },
            }
        }
        assert resolve_display_setting(config, "slack", "tool_progress") == "off"
        assert resolve_display_setting(config, "telegram", "tool_progress") == "all"


# ---------------------------------------------------------------------------
# Backward compatibility: tool_progress_overrides
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Legacy tool_progress_overrides is still respected as a fallback."""

    def test_legacy_overrides_read(self):
        """tool_progress_overrides is read when no platforms entry exists."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "tool_progress_overrides": {
                    "signal": "off",
                    "telegram": "verbose",
                },
            }
        }
        assert resolve_display_setting(config, "signal", "tool_progress") == "off"
        assert resolve_display_setting(config, "telegram", "tool_progress") == "verbose"

    def test_new_platforms_takes_precedence_over_legacy(self):
        """display.platforms beats tool_progress_overrides."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "tool_progress_overrides": {"telegram": "verbose"},
                "platforms": {"telegram": {"tool_progress": "new"}},
            }
        }
        assert resolve_display_setting(config, "telegram", "tool_progress") == "new"

    def test_legacy_overrides_only_for_tool_progress(self):
        """Legacy overrides don't affect other settings."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress_overrides": {"telegram": "verbose"},
            }
        }
        # show_reasoning should NOT read from tool_progress_overrides
        assert resolve_display_setting(config, "telegram", "show_reasoning") is False


# ---------------------------------------------------------------------------
# YAML normalisation
# ---------------------------------------------------------------------------

class TestYAMLNormalisation:
    """YAML 1.1 quirks (bare off → False, on → True) are handled."""

    def test_tool_progress_false_normalised_to_off(self):
        """YAML's bare `off` parses as False — normalised to 'off' string."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress": False}}
        assert resolve_display_setting(config, "telegram", "tool_progress") == "off"

    def test_tool_progress_true_normalised_to_all(self):
        """YAML's bare `on` parses as True — normalised to 'all'."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress": True}}
        assert resolve_display_setting(config, "telegram", "tool_progress") == "all"

    def test_show_reasoning_string_true(self):
        """String 'true' is normalised to bool True."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"telegram": {"show_reasoning": "true"}}}}
        assert resolve_display_setting(config, "telegram", "show_reasoning") is True

    def test_tool_preview_length_string(self):
        """String numbers are normalised to int."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"slack": {"tool_preview_length": "80"}}}}
        assert resolve_display_setting(config, "slack", "tool_preview_length") == 80

    def test_platform_override_false_tool_progress(self):
        """Per-platform bare off → normalised."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"slack": {"tool_progress": False}}}}
        assert resolve_display_setting(config, "slack", "tool_progress") == "off"


# ---------------------------------------------------------------------------
# Built-in platform defaults (tier system)
# ---------------------------------------------------------------------------

class TestPlatformDefaults:
    """Built-in defaults reflect platform capability tiers."""

    def test_high_tier_platforms(self):
        """Discord defaults to 'all' tool progress; Telegram is in tier_high
        but overrides tool_progress to 'new' (less edit pressure)."""
        from gateway.display_config import resolve_display_setting

        # Telegram: tier_high member with tool_progress="new" override.
        assert resolve_display_setting({}, "telegram", "tool_progress") == "new"
        # Discord: pure tier_high.
        assert resolve_display_setting({}, "discord", "tool_progress") == "all"

    def test_medium_tier_platforms(self):
        """Mattermost, Matrix, Feishu, WhatsApp default to 'new' tool progress."""
        from gateway.display_config import resolve_display_setting

        for plat in ("mattermost", "matrix", "feishu", "whatsapp"):
            assert resolve_display_setting({}, plat, "tool_progress") == "new", plat

    def test_slack_defaults_tool_progress_off(self):
        """Slack defaults to quiet tool progress (permanent chat noise otherwise)."""
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({}, "slack", "tool_progress") == "off"

    def test_low_tier_platforms(self):
        """Signal, BlueBubbles, etc. default to 'off' tool progress."""
        from gateway.display_config import resolve_display_setting

        for plat in ("signal", "bluebubbles", "weixin", "wecom", "dingtalk"):
            assert resolve_display_setting({}, plat, "tool_progress") == "off", plat

    def test_minimal_tier_platforms(self):
        """Email, SMS, webhook default to 'off' tool progress."""
        from gateway.display_config import resolve_display_setting

        for plat in ("email", "sms", "webhook", "homeassistant"):
            assert resolve_display_setting({}, plat, "tool_progress") == "off", plat

    def test_low_tier_streaming_defaults_to_false(self):
        """Low-tier platforms default streaming to False."""
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({}, "signal", "streaming") is False
        assert resolve_display_setting({}, "email", "streaming") is False

    def test_high_tier_streaming_defaults_to_none(self):
        """High-tier platforms default streaming to None (follow global)."""
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({}, "telegram", "streaming") is None


# ---------------------------------------------------------------------------
# Config migration: tool_progress_overrides → display.platforms
# ---------------------------------------------------------------------------

class TestConfigMigration:
    """Version 16 migration moves tool_progress_overrides into display.platforms."""

    def test_migration_creates_platforms_entries(self, tmp_path, monkeypatch):
        """Old overrides are migrated into display.platforms.<plat>.tool_progress."""
        import yaml

        config_path = tmp_path / "config.yaml"
        config = {
            "_config_version": 15,
            "display": {
                "tool_progress_overrides": {
                    "signal": "off",
                    "telegram": "all",
                },
            },
        }
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Re-import to pick up the new HERMES_HOME
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        result = cfg_mod.migrate_config(interactive=False, quiet=True)
        # Re-read config
        updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        platforms = updated.get("display", {}).get("platforms", {})
        assert platforms.get("signal", {}).get("tool_progress") == "off"
        assert platforms.get("telegram", {}).get("tool_progress") == "all"

    def test_migration_preserves_existing_platforms_entries(self, tmp_path, monkeypatch):
        """Existing display.platforms entries are NOT overwritten by migration."""
        import yaml

        config_path = tmp_path / "config.yaml"
        config = {
            "_config_version": 15,
            "display": {
                "tool_progress_overrides": {"telegram": "off"},
                "platforms": {"telegram": {"tool_progress": "verbose"}},
            },
        }
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg_mod.migrate_config(interactive=False, quiet=True)
        updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Existing "verbose" should NOT be overwritten by legacy "off"
        assert updated["display"]["platforms"]["telegram"]["tool_progress"] == "verbose"


# ---------------------------------------------------------------------------
# Streaming per-platform (None = follow global)
# ---------------------------------------------------------------------------

class TestStreamingPerPlatform:
    """Streaming per-platform override semantics."""

    def test_none_means_follow_global(self):
        """When streaming is None, the caller should use global config."""
        from gateway.display_config import resolve_display_setting

        config = {}
        # Telegram has no streaming override in defaults → None
        result = resolve_display_setting(config, "telegram", "streaming")
        assert result is None  # caller should check global StreamingConfig

    def test_global_display_streaming_is_cli_only(self):
        """display.streaming must not act as a gateway streaming override."""
        from gateway.display_config import resolve_display_setting

        for value in (True, False):
            config = {"display": {"streaming": value}}
            assert resolve_display_setting(config, "telegram", "streaming") is None
            assert resolve_display_setting(config, "discord", "streaming") is None

    def test_explicit_false_disables(self):
        """Explicit False disables streaming for that platform."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {"telegram": {"streaming": False}},
            }
        }
        assert resolve_display_setting(config, "telegram", "streaming") is False

    def test_explicit_true_enables(self):
        """Explicit True enables streaming for that platform."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {"email": {"streaming": True}},
            }
        }
        assert resolve_display_setting(config, "email", "streaming") is True


# ---------------------------------------------------------------------------
# cleanup_progress — opt-in deletion of temporary progress bubbles
# ---------------------------------------------------------------------------

class TestCleanupProgress:
    """``cleanup_progress`` is off by default and resolvable per-platform."""

    def test_default_off_for_all_platforms(self):
        """No config set → cleanup_progress resolves to False everywhere."""
        from gateway.display_config import resolve_display_setting

        for plat in ("telegram", "discord", "slack", "email"):
            assert resolve_display_setting({}, plat, "cleanup_progress") is False

    def test_global_true_applies_to_all_platforms(self):
        """display.cleanup_progress=true opts in globally."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"cleanup_progress": True}}
        assert resolve_display_setting(config, "telegram", "cleanup_progress") is True
        assert resolve_display_setting(config, "discord", "cleanup_progress") is True

    def test_per_platform_override_wins(self):
        """display.platforms.<plat>.cleanup_progress beats the global value."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "cleanup_progress": False,
                "platforms": {
                    "telegram": {"cleanup_progress": True},
                },
            }
        }
        assert resolve_display_setting(config, "telegram", "cleanup_progress") is True
        assert resolve_display_setting(config, "discord", "cleanup_progress") is False

    def test_yaml_off_string_normalises_to_false(self):
        """YAML 1.1 bare ``off`` becomes string 'off' — treat as False."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {"telegram": {"cleanup_progress": "off"}},
            }
        }
        assert resolve_display_setting(config, "telegram", "cleanup_progress") is False

    def test_yaml_true_string_normalises_to_true(self):
        """String 'true'/'yes'/'on' all resolve to True."""
        from gateway.display_config import resolve_display_setting

        for val in ("true", "yes", "on", "1"):
            config = {
                "display": {
                    "platforms": {"telegram": {"cleanup_progress": val}},
                }
            }
            assert resolve_display_setting(config, "telegram", "cleanup_progress") is True, val
