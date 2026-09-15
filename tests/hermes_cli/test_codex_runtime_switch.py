"""Tests for the /codex-runtime slash-command shared logic.

These cover the pure-Python state machine; CLI and gateway handlers are
tested separately because they involve config persistence and prompt
formatting that's surface-specific."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli import codex_runtime_switch as crs


class TestParseArgs:
    @pytest.mark.parametrize("arg,expected", [
        ("", None),
        ("   ", None),
        ("auto", "auto"),
        ("codex_app_server", "codex_app_server"),
        ("on", "codex_app_server"),
        ("off", "auto"),
        ("codex", "codex_app_server"),
        ("default", "auto"),
        ("hermes", "auto"),
        ("ENABLE", "codex_app_server"),  # case-insensitive
        ("DiSaBlE", "auto"),
    ])
    def test_valid_args(self, arg, expected):
        value, errors = crs.parse_args(arg)
        assert errors == []
        assert value == expected


class TestGetCurrentRuntime:
    def test_default_when_unset(self):
        assert crs.get_current_runtime({}) == "auto"
        assert crs.get_current_runtime({"model": {}}) == "auto"
        assert crs.get_current_runtime({"model": {"openai_runtime": ""}}) == "auto"

    def test_unrecognized_falls_back_to_auto(self):
        assert crs.get_current_runtime(
            {"model": {"openai_runtime": "garbage"}}
        ) == "auto"


class TestSetRuntime:
    def test_creates_model_section_if_missing(self):
        cfg = {}
        old = crs.set_runtime(cfg, "codex_app_server")
        assert old == "auto"
        assert cfg["model"]["openai_runtime"] == "codex_app_server"


    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            crs.set_runtime({}, "garbage")


class TestApply:


    def test_reapply_codex_app_server_runs_migration(self):
        """Re-applying codex_app_server when already enabled must still
        run the migration. Common footgun: user pre-sets
        `openai_runtime: codex_app_server` in config.yaml, then runs
        /codex-runtime codex_app_server expecting the migration. Without
        this, the slash command short-circuits with "already set" and
        ~/.codex/config.toml never gets the hermes-tools MCP callback
        or plugin migration — silent partial setup.
        """
        cfg = {
            "model": {"openai_runtime": "codex_app_server"},
            "mcp_servers": {
                "filesystem": {"command": "npx", "args": ["-y", "fs-server"]},
            },
        }
        persisted = {}

        def persist(c):
            persisted.update(c)

        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")), \
             patch("hermes_cli.codex_runtime_plugin_migration.migrate") as mig:
            mig.return_value.migrated = ["filesystem", "hermes-tools"]
            mig.return_value.migrated_plugins = []
            mig.return_value.plugin_query_error = None
            mig.return_value.wrote_permissions_default = ":workspace"
            mig.return_value.errors = []
            mig.return_value.target_path = "/fake/.codex/config.toml"
            r = crs.apply(cfg, "codex_app_server",
                          persist_callback=persist)
        assert r.success
        assert mig.called, "migration must run on reapply, not just first enable"
        # Re-apply should signal "already set" but still announce migration ran
        assert "already set" in r.message
        assert "re-applying migration" in r.message
        # Migration output still surfaces
        assert "Migrated 1 MCP server" in r.message
        assert "filesystem" in r.message
        assert "Default sandbox: :workspace" in r.message
        # No config write needed when value is unchanged — the persist
        # callback should NOT have fired (avoids spurious config.yaml mtimes
        # on every re-apply).
        assert persisted == {}, (
            "persist_callback fired despite no config-value change"
        )
        # Caller still needs a fresh session for the cached agent to pick
        # up any migration-driven changes.
        assert r.requires_new_session is True





    def test_enable_triggers_mcp_migration(self):
        """Enabling codex_app_server should auto-migrate Hermes mcp_servers
        to ~/.codex/config.toml so the spawned subprocess sees them."""
        cfg = {
            "mcp_servers": {
                "filesystem": {"command": "npx", "args": ["-y", "fs-server"]},
            }
        }

        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")), \
             patch("hermes_cli.codex_runtime_plugin_migration.migrate") as mig:
            mig.return_value.migrated = ["filesystem", "hermes-tools"]
            mig.return_value.migrated_plugins = []
            mig.return_value.plugin_query_error = None
            mig.return_value.wrote_permissions_default = ":workspace"
            mig.return_value.errors = []
            mig.return_value.target_path = "/fake/.codex/config.toml"
            r = crs.apply(cfg, "codex_app_server")
        assert r.success
        assert mig.called  # migration was triggered
        # User MCP servers are reported (excluding internal hermes-tools)
        assert "Migrated 1 MCP server" in r.message
        assert "filesystem" in r.message
        # Permissions default surfaces
        assert "Default sandbox: :workspace" in r.message
        # Hermes tool callback announcement
        assert "via MCP" in r.message

    def test_disable_does_not_trigger_migration(self):
        """Switching back to auto must not write to ~/.codex/."""
        cfg = {
            "model": {"openai_runtime": "codex_app_server"},
            "mcp_servers": {"x": {"command": "y"}},
        }
        with patch("hermes_cli.codex_runtime_plugin_migration.migrate") as mig:
            r = crs.apply(cfg, "auto")
        assert r.success
        assert not mig.called  # disabling does not migrate

    def test_migration_failure_does_not_block_enable(self):
        """If MCP migration raises, the runtime change still proceeds —
        users can manually re-run migration later."""
        cfg = {"mcp_servers": {"x": {"command": "y"}}}
        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")), \
             patch("hermes_cli.codex_runtime_plugin_migration.migrate",
                   side_effect=RuntimeError("disk full")):
            r = crs.apply(cfg, "codex_app_server")
        assert r.success  # change still applied
        assert r.new_value == "codex_app_server"
        assert "MCP migration skipped" in r.message
        assert "disk full" in r.message


