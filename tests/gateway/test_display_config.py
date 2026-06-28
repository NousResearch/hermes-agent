"""Tests for gateway.display_config — per-platform display/verbosity resolver."""


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
        # Telegram is a mobile inbox by default — final-answer-first unless
        # explicitly configured otherwise.
        assert resolve_display_setting(config, "telegram", "tool_progress") == "off"
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

    def test_tool_progress_generic_is_not_a_mode(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"whatsapp": {"tool_progress": "generic"}}}}
        assert resolve_display_setting(config, "whatsapp", "tool_progress") == "all"

    def test_only_long_running_visibility_accepts_generic_mode(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "thinking_progress": "generic",
                        "interim_assistant_messages": "generic",
                        "long_running_notifications": "generic",
                    }
                }
            }
        }
        assert resolve_display_setting(config, "whatsapp", "thinking_progress") is False
        assert resolve_display_setting(config, "whatsapp", "interim_assistant_messages") is False
        assert resolve_display_setting(config, "whatsapp", "long_running_notifications") == "generic"

    def test_thinking_progress_string_false_normalised_to_false(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"whatsapp": {"thinking_progress": "false"}}}}
        assert resolve_display_setting(config, "whatsapp", "thinking_progress") is False

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
        """Discord defaults to 'all'; Telegram defaults quiet for mobile."""
        from gateway.display_config import resolve_display_setting

        # Telegram: tier_high transport, but quiet mobile default.
        assert resolve_display_setting({}, "telegram", "tool_progress") == "off"
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

        for plat in ("signal", "bluebubbles", "weixin", "wecom", "dingtalk", "whatsapp_cloud"):
            assert resolve_display_setting({}, plat, "tool_progress") == "off", plat

    def test_whatsapp_cloud_locked_to_low_tier_until_edit_message_lands(self):
        """Regression guard: ``whatsapp_cloud`` must stay TIER_LOW until the
        adapter implements edit_message. Without an edit endpoint, raising
        the tier to MEDIUM would spam separate WhatsApp messages for every
        tool-progress update, which is the exact failure mode this entry
        exists to avoid.

        When/if Cloud's edit_message lands, update _PLATFORM_DEFAULTS to
        TIER_MEDIUM and update this test to assert ``"new"`` accordingly.
        """
        from gateway.display_config import resolve_display_setting
        assert resolve_display_setting({}, "whatsapp_cloud", "tool_progress") == "off"
        assert resolve_display_setting({}, "whatsapp_cloud", "streaming") is False

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

    def test_telegram_mobile_chatter_defaults(self):
        """Telegram keeps real mid-turn signal (interim commentary + heartbeats)
        but skips the verbose busy-ack iteration counter by default."""
        from gateway.display_config import resolve_display_setting

        # Real model voice — keep on. Without this, Telegram users see
        # "typing..." for the entire turn duration with no feedback.
        assert resolve_display_setting({}, "telegram", "interim_assistant_messages") is True
        # Periodic "Working — N min" heartbeat — keep on. Otherwise long
        # turns appear completely silent.
        assert resolve_display_setting({}, "telegram", "long_running_notifications") is True
        # Verbose iteration counter in busy-ack and heartbeat — off by
        # default on Telegram (mobile chat is cramped enough without
        # "iteration 21/60" debug detail).
        assert resolve_display_setting({}, "telegram", "busy_ack_detail") is False
        # Discord keeps all of these on (desktop-first, more vertical space).
        assert resolve_display_setting({}, "discord", "interim_assistant_messages") is True
        assert resolve_display_setting({}, "discord", "long_running_notifications") is True
        assert resolve_display_setting({}, "discord", "busy_ack_detail") is True

    def test_telegram_mobile_chatter_can_opt_in(self):
        """Per-platform config can re-enable Telegram busy-ack detail
        and re-disable the kept-on defaults."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "telegram": {
                        "interim_assistant_messages": False,
                        "long_running_notifications": False,
                        "busy_ack_detail": "on",
                    }
                }
            }
        }
        assert resolve_display_setting(config, "telegram", "interim_assistant_messages") is False
        assert resolve_display_setting(config, "telegram", "long_running_notifications") is False
        assert resolve_display_setting(config, "telegram", "busy_ack_detail") is True


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


class TestToolProgressGrouping:
    """resolve_display_setting() for the tool_progress_grouping knob."""

    def test_default_is_accumulate(self):
        """No config anywhere → global default 'accumulate'."""
        from gateway.display_config import resolve_display_setting

        assert (
            resolve_display_setting({}, "telegram", "tool_progress_grouping")
            == "accumulate"
        )

    def test_global_separate(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress_grouping": "separate"}}
        assert (
            resolve_display_setting(config, "discord", "tool_progress_grouping")
            == "separate"
        )

    def test_platform_override_wins(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress_grouping": "accumulate",
                "platforms": {"discord": {"tool_progress_grouping": "separate"}},
            }
        }
        assert (
            resolve_display_setting(config, "discord", "tool_progress_grouping")
            == "separate"
        )
        # Other platforms still get the global value.
        assert (
            resolve_display_setting(config, "telegram", "tool_progress_grouping")
            == "accumulate"
        )

    def test_invalid_value_falls_back_to_accumulate(self):
        """_normalise rejects anything outside accumulate|separate."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress_grouping": "bogus"}}
        assert (
            resolve_display_setting(config, "telegram", "tool_progress_grouping")
            == "accumulate"
        )

    def test_case_insensitive(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress_grouping": "SEPARATE"}}
        assert (
            resolve_display_setting(config, "telegram", "tool_progress_grouping")
            == "separate"
        )


class TestReasoningStyle:
    """Per-platform reasoning render style (code | blockquote | subtext)."""

    def test_discord_defaults_to_subtext(self):
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({}, "discord", "reasoning_style") == "subtext"

    def test_other_platforms_default_to_code(self):
        from gateway.display_config import resolve_display_setting

        for plat in ("telegram", "slack", "matrix", "api_server"):
            assert (
                resolve_display_setting({}, plat, "reasoning_style") == "code"
            ), plat

    def test_platform_override_wins(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"discord": {"reasoning_style": "blockquote"}}}}
        assert (
            resolve_display_setting(config, "discord", "reasoning_style") == "blockquote"
        )

    def test_global_override(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"reasoning_style": "subtext"}}
        assert (
            resolve_display_setting(config, "telegram", "reasoning_style") == "subtext"
        )

    def test_invalid_value_falls_back_to_code(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"reasoning_style": "bogus"}}
        assert resolve_display_setting(config, "telegram", "reasoning_style") == "code"

    def test_case_insensitive(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"reasoning_style": "SUBTEXT"}}
        assert resolve_display_setting(config, "telegram", "reasoning_style") == "subtext"
