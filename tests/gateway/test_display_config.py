"""Tests for gateway.display_config — per-platform display/verbosity resolver."""


# ---------------------------------------------------------------------------
# Resolver: resolution order
# ---------------------------------------------------------------------------

class TestToolProgressProvenance:
    def test_winning_source_controls_mode_and_intent(self):
        from gateway.display_config import resolve_tool_progress

        cases = [
            ({}, None, ("off", False)),
            ({}, "all", ("all", True)),
            ({"tool_progress": None}, "all", ("all", True)),
            ({"platforms": {"slack": {"tool_progress": None}}}, "off", ("off", True)),
            ({"tool_progress_overrides": {"slack": None}}, "new", ("new", True)),
            ({"tool_progress": False}, "all", ("off", True)),
            ({"tool_progress": "all", "platforms": {"slack": {"tool_progress": None}}}, "off", ("all", True)),
            ({"tool_progress": "off", "tool_progress_overrides": {"slack": "new"}}, "all", ("new", True)),
            ({"tool_progress_overrides": {"slack": "off"}, "platforms": {"slack": {"tool_progress": "all"}}}, None, ("all", True)),
        ]
        for display, env, expected in cases:
            assert resolve_tool_progress({"display": display}, "slack", env) == expected


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


# ---------------------------------------------------------------------------
# Built-in platform defaults (tier system)
# ---------------------------------------------------------------------------


def assert_keeps_platform_display_defaults(cfg):
    """Every platform resolves every tier key (and tool_progress) exactly as with no config at all.

    Shared by every config seeder's regression test: a seeded global ``display.<key>`` beats each
    platform tier, because the gateway loader merges no DEFAULT_CONFIG (#121230)."""
    from gateway.display_config import _PLATFORM_DEFAULTS, resolve_display_setting, resolve_tool_progress

    tier_keys = {key for tier in _PLATFORM_DEFAULTS.values() for key in tier}
    for platform in _PLATFORM_DEFAULTS:
        assert resolve_tool_progress(cfg, platform) == resolve_tool_progress({}, platform), platform
        for key in tier_keys:
            assert resolve_display_setting(cfg, platform, key) == resolve_display_setting({}, platform, key), (
                platform, key)


class TestInstallerSeededConfigThroughGatewayResolver:
    """Regression for #121230: a fresh install copies cli-config.yaml.example to config.yaml, and the
    gateway then rendered reasoning into QQBot/Telegram/... because the template pinned a global
    ``display.show_reasoning: true`` over every platform's ``False`` default.
    """

    @staticmethod
    def _seed_like_installer(home):
        import shutil
        from pathlib import Path

        template = Path(__file__).resolve().parents[2] / "cli-config.yaml.example"
        home.mkdir(parents=True, exist_ok=True)
        shutil.copy(template, home / "config.yaml")
        return home / "config.yaml"

    def test_shipped_template_keeps_every_platform_default(self, tmp_path):
        """The installers, the Docker first boot and ``doctor --fix`` copy cli-config.yaml.example
        verbatim, so an uncommented ``display.<key>`` there becomes an explicit global value."""
        from gateway.config import Platform
        from gateway.run import _load_gateway_config, _resolve_gateway_display_bool

        seeded = _load_gateway_config(self._seed_like_installer(tmp_path / "hermes-home"))
        assert "display" in seeded  # the loader fails open to {}, which would pass vacuously

        assert_keeps_platform_display_defaults(seeded)
        # ...and through the resolver the gateway turn actually calls (same arguments as gateway/run_turn.py).
        assert _resolve_gateway_display_bool(
            seeded, "qqbot", "show_reasoning", default=False, platform=Platform.QQBOT,
            require_platform_override_for={Platform.MATTERMOST},
        ) is False

    def test_explicit_global_opt_in_still_reaches_gateway_platforms(self, tmp_path):
        """Control: an operator who deliberately writes ``display.show_reasoning: true`` still gets it (#7148)."""
        from gateway.config import Platform
        from gateway.run import _load_gateway_config, _resolve_gateway_display_bool

        (tmp_path / "config.yaml").write_text("display:\n  show_reasoning: true\n")
        cfg = _load_gateway_config(tmp_path / "config.yaml")
        assert _resolve_gateway_display_bool(
            cfg, "qqbot", "show_reasoning", default=False, platform=Platform.QQBOT,
            require_platform_override_for={Platform.MATTERMOST},
        ) is True


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


# ---------------------------------------------------------------------------
# Streaming per-platform (None = follow global)
# ---------------------------------------------------------------------------

class TestStreamingPerPlatform:
    """Streaming per-platform override semantics."""


    def test_explicit_false_disables(self):
        """Explicit False disables streaming for that platform."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {"telegram": {"streaming": False}},
            }
        }
        assert resolve_display_setting(config, "telegram", "streaming") is False





# ---------------------------------------------------------------------------
# cleanup_progress — opt-in deletion of temporary progress bubbles
# ---------------------------------------------------------------------------

class TestCleanupProgress:
    """``cleanup_progress`` is off by default and resolvable per-platform."""



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










class TestAllowSilentReplies:
    """display.allow_silent_replies lets a bare silence marker answer a human turn."""

    def test_off_by_default_on_every_platform(self):
        from gateway.display_config import resolve_display_setting

        for platform in ("telegram", "discord", "slack", "email"):
            assert resolve_display_setting({}, platform, "allow_silent_replies") is False

    def test_global_and_per_platform_opt_in(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"telegram": {"allow_silent_replies": "true"}}}}
        assert resolve_display_setting(config, "telegram", "allow_silent_replies") is True
        assert resolve_display_setting(config, "discord", "allow_silent_replies") is False
        assert resolve_display_setting(
            {"display": {"allow_silent_replies": True}}, "discord", "allow_silent_replies") is True

    def test_gateway_helper_reads_the_setting(self, monkeypatch):
        import gateway.run as gateway_run
        from gateway.config import Platform
        from gateway.run_turn import GatewayTurnMixin
        from types import SimpleNamespace

        source = SimpleNamespace(platform=Platform.TELEGRAM)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        assert GatewayTurnMixin._silent_replies_allowed(source) is False
        monkeypatch.setattr(
            gateway_run, "_load_gateway_config", lambda: {"display": {"allow_silent_replies": True}})
        assert GatewayTurnMixin._silent_replies_allowed(source) is True


class TestAgentDisplayName:
    """display.agent_name names the agent in gateway notices."""

    def test_default_and_override(self, monkeypatch):
        import gateway.run as gateway_run
        from gateway.display_config import agent_display_name

        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        assert agent_display_name() == "Hermes"
        monkeypatch.setattr(
            gateway_run, "_load_gateway_config", lambda: {"display": {"agent_name": " Wintermute "}})
        assert agent_display_name() == "Wintermute"
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"agent_name": ""}})
        assert agent_display_name() == "Hermes"
