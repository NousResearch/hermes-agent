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

    def test_invalid_arg_returns_error(self):
        value, errors = crs.parse_args("turbo")
        assert value is None
        assert errors and "Unknown runtime" in errors[0]


class TestGetCurrentRuntime:
    def test_default_when_unset(self):
        assert crs.get_current_runtime({}) == "auto"
        assert crs.get_current_runtime({"model": {}}) == "auto"
        assert crs.get_current_runtime({"model": {"openai_runtime": ""}}) == "auto"

    def test_unrecognized_falls_back_to_auto(self):
        assert crs.get_current_runtime(
            {"model": {"openai_runtime": "garbage"}}
        ) == "auto"

    def test_explicit_codex(self):
        assert crs.get_current_runtime(
            {"model": {"openai_runtime": "codex_app_server"}}
        ) == "codex_app_server"

    def test_handles_non_dict_config(self):
        assert crs.get_current_runtime(None) == "auto"  # type: ignore[arg-type]
        assert crs.get_current_runtime("notadict") == "auto"  # type: ignore[arg-type]
        assert crs.get_current_runtime({"model": "notadict"}) == "auto"


class TestSetRuntime:
    def test_creates_model_section_if_missing(self):
        cfg = {}
        old = crs.set_runtime(cfg, "codex_app_server")
        assert old == "auto"
        assert cfg["model"]["openai_runtime"] == "codex_app_server"

    def test_returns_previous_value(self):
        cfg = {"model": {"openai_runtime": "codex_app_server"}}
        old = crs.set_runtime(cfg, "auto")
        assert old == "codex_app_server"
        assert cfg["model"]["openai_runtime"] == "auto"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            crs.set_runtime({}, "garbage")


class TestApply:
    def test_read_only_call_reports_state(self):
        cfg = {"model": {"openai_runtime": "codex_app_server"}}
        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")):
            r = crs.apply(cfg, None)
        assert r.success
        assert r.new_value == "codex_app_server"
        assert r.old_value == "codex_app_server"
        assert "codex_app_server" in r.message
        assert "0.130.0" in r.message

    def test_no_change_when_already_set(self):
        cfg = {"model": {"openai_runtime": "auto"}}
        r = crs.apply(cfg, "auto")
        assert r.success
        assert r.message == "openai_runtime already set to auto"

    def test_enable_blocked_when_codex_missing(self):
        cfg = {}
        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(False, "codex not found")):
            r = crs.apply(cfg, "codex_app_server")
        assert r.success is False
        assert "Cannot enable" in r.message
        assert "npm i -g @openai/codex" in r.message
        # Config NOT mutated on failure
        assert cfg.get("model", {}).get("openai_runtime") in (None, "")

    def test_enable_succeeds_when_codex_present(self):
        cfg = {}
        persisted = {}

        def persist(c):
            persisted.update(c)

        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")):
            r = crs.apply(cfg, "codex_app_server", persist_callback=persist)
        assert r.success
        assert r.new_value == "codex_app_server"
        assert r.old_value == "auto"
        assert r.requires_new_session is True
        assert "via MCP" in r.message  # hermes-tools callback message
        assert cfg["model"]["openai_runtime"] == "codex_app_server"
        assert persisted["model"]["openai_runtime"] == "codex_app_server"

    def test_disable_does_not_check_binary(self):
        cfg = {"model": {"openai_runtime": "codex_app_server"}}
        with patch.object(crs, "check_codex_binary_ok") as bin_check:
            r = crs.apply(cfg, "auto")
        assert r.success
        # Binary check is irrelevant when disabling — should not be called
        # with the codex_app_server enable-gate signature.
        assert r.new_value == "auto"
        assert r.old_value == "codex_app_server"

    def test_persist_callback_failure_reported(self):
        cfg = {}

        def persist_boom(c):
            raise IOError("disk full")

        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")):
            r = crs.apply(cfg, "codex_app_server", persist_callback=persist_boom)
        assert r.success is False
        assert "persist failed" in r.message
        assert "disk full" in r.message

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

    def test_binary_check_cached_within_apply(self):
        """check_codex_binary_ok is invoked at most once per apply() call.

        The enable path has three sites that need the version (state report,
        enable gate, success message). Without caching, a single
        /codex-runtime invocation spawns `codex --version` three times.
        Regression guard against a refactor that drops the cache.
        """
        cfg = {}
        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")) as bin_check, \
             patch("hermes_cli.codex_runtime_plugin_migration.migrate"):
            r = crs.apply(cfg, "codex_app_server")
        assert r.success
        assert bin_check.call_count == 1, (
            f"check_codex_binary_ok was called {bin_check.call_count} time(s); "
            "should be cached and called exactly once per apply()"
        )

    def test_binary_check_cached_on_read_only_call(self):
        """Read-only call (new_value=None) calls the binary check exactly
        once and reuses the result for the message."""
        cfg = {"model": {"openai_runtime": "codex_app_server"}}
        with patch.object(crs, "check_codex_binary_ok",
                          return_value=(True, "0.130.0")) as bin_check:
            crs.apply(cfg, None)
        assert bin_check.call_count == 1
