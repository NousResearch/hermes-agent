"""Tests for the /acp-client-runtime slash-command shared logic.

These cover the pure-Python state machine; CLI and gateway handlers are tested
separately because they involve config persistence and surface-specific
formatting."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli import acp_runtime_switch as ars


class TestParseArgs:
    @pytest.mark.parametrize("arg,expected", [
        ("", None),
        ("   ", None),
        ("auto", "auto"),
        ("off", "auto"),
        ("disable", "auto"),
        ("hermes", "auto"),
        ("default", "auto"),
        ("acp_client", "acp_client"),
        ("acp-client", "acp_client"),
        ("on", "acp_client"),
        ("enable", "acp_client"),
        ("acp", "acp_client"),
        ("ACP_CLIENT", "acp_client"),   # case-insensitive
        ("ON", "acp_client"),
    ])
    def test_valid_args(self, arg, expected):
        value, errors = ars.parse_args(arg)
        assert errors == []
        assert value == expected

    def test_invalid_arg_returns_error(self):
        value, errors = ars.parse_args("turbo")
        assert value is None
        assert errors
        assert "Unknown value" in errors[0]


class TestGetCurrentState:
    def test_default_when_unset(self):
        assert ars.get_current_state({}) == "auto"
        assert ars.get_current_state({"model": {}}) == "auto"
        assert ars.get_current_state({"model": {"provider": ""}}) == "auto"

    def test_unrecognized_falls_back_to_auto(self):
        assert ars.get_current_state(
            {"model": {"provider": "openrouter"}}
        ) == "auto"

    def test_acp_client_provider_detected(self):
        assert ars.get_current_state(
            {"model": {"provider": "acp-client"}}
        ) == "acp_client"

    def test_handles_non_dict_config(self):
        assert ars.get_current_state(None) == "auto"  # type: ignore[arg-type]
        assert ars.get_current_state("notadict") == "auto"  # type: ignore[arg-type]
        assert ars.get_current_state({"model": "notadict"}) == "auto"


class TestGetCurrentCommand:
    def test_returns_empty_when_unset(self):
        assert ars.get_current_command({}) == ""
        assert ars.get_current_command({"model": {}}) == ""

    def test_returns_configured_command(self):
        assert ars.get_current_command(
            {"model": {"acp_command": "claude-agent-acp"}}
        ) == "claude-agent-acp"

    def test_handles_non_dict_config(self):
        assert ars.get_current_command(None) == ""  # type: ignore[arg-type]


class TestGetCurrentArgs:
    def test_returns_empty_list_when_unset(self):
        assert ars.get_current_args({}) == []
        assert ars.get_current_args({"model": {}}) == []

    def test_returns_configured_args(self):
        assert ars.get_current_args(
            {"model": {"acp_args": ["--port", "9000"]}}
        ) == ["--port", "9000"]

    def test_handles_non_list_gracefully(self):
        assert ars.get_current_args({"model": {"acp_args": None}}) == []

    def test_handles_non_dict_config(self):
        assert ars.get_current_args(None) == []  # type: ignore[arg-type]


class TestEnableDisable:
    def test_enable_sets_all_keys(self):
        cfg = {}
        old = ars.enable_runtime(cfg, "claude-agent-acp", ["--verbose"])
        assert old == "auto"
        assert cfg["model"]["provider"] == "acp-client"
        assert cfg["model"]["acp_command"] == "claude-agent-acp"
        assert cfg["model"]["acp_args"] == ["--verbose"]

    def test_enable_creates_model_section_if_missing(self):
        cfg = {}
        ars.enable_runtime(cfg, "my-agent", [])
        assert "model" in cfg

    def test_disable_removes_acp_keys(self):
        cfg = {"model": {
            "provider": "acp-client",
            "acp_command": "claude-agent-acp",
            "acp_args": [],
        }}
        old = ars.disable_runtime(cfg)
        assert old == "acp_client"
        assert cfg["model"].get("provider") is None
        assert cfg["model"].get("acp_command") is None
        assert cfg["model"].get("acp_args") is None

    def test_disable_is_idempotent_on_empty_config(self):
        cfg = {}
        old = ars.disable_runtime(cfg)
        assert old == "auto"  # was already disabled


class TestApply:
    def test_read_only_reports_state(self):
        cfg = {"model": {"provider": "acp-client", "acp_command": "myagent"}}
        with patch.object(ars, "check_acp_command_ok", return_value=(True, "1.2.3")):
            r = ars.apply(cfg, None)
        assert r.success
        assert r.new_value == "acp_client"
        assert r.old_value == "acp_client"
        assert "myagent" in r.message

    def test_read_only_when_disabled(self):
        cfg = {}
        r = ars.apply(cfg, None)
        assert r.success
        assert r.new_value == "auto"

    def test_no_change_when_already_disabled(self):
        cfg = {}
        r = ars.apply(cfg, "auto")
        assert r.success
        assert r.new_value == "auto"
        assert "already disabled" in r.message

    def test_no_change_when_already_enabled(self):
        cfg = {"model": {"provider": "acp-client", "acp_command": "myagent"}}
        r = ars.apply(cfg, "acp_client")
        assert r.success
        assert "already enabled" in r.message

    def test_enable_fails_when_no_command(self):
        cfg = {}
        r = ars.apply(cfg, "acp_client")
        assert r.success is False
        assert "acp_command is required" in r.message

    def test_enable_blocked_when_binary_missing(self):
        cfg = {}
        with patch.object(ars, "check_acp_command_ok",
                          return_value=(False, "myagent not found on PATH")):
            r = ars.apply(cfg, "acp_client", acp_command="myagent")
        assert r.success is False
        assert "Cannot enable" in r.message
        assert "myagent" in r.message
        # Config must NOT be mutated on failure
        assert cfg.get("model", {}).get("provider") != "acp-client"

    def test_enable_succeeds_when_binary_present(self):
        cfg = {}
        persisted = {}

        def persist(c):
            persisted.update(c)

        with patch.object(ars, "check_acp_command_ok",
                          return_value=(True, "claude-agent-acp 0.3.1")):
            r = ars.apply(
                cfg,
                "acp_client",
                acp_command="claude-agent-acp",
                acp_args=["--verbose"],
                persist_callback=persist,
            )
        assert r.success
        assert r.new_value == "acp_client"
        assert r.old_value == "auto"
        assert r.requires_new_session is True
        assert "claude-agent-acp" in r.message
        assert cfg["model"]["provider"] == "acp-client"
        assert cfg["model"]["acp_command"] == "claude-agent-acp"
        assert cfg["model"]["acp_args"] == ["--verbose"]
        assert persisted["model"]["provider"] == "acp-client"

    def test_disable_succeeds(self):
        cfg = {"model": {
            "provider": "acp-client",
            "acp_command": "myagent",
            "acp_args": [],
        }}
        r = ars.apply(cfg, "auto")
        assert r.success
        assert r.new_value == "auto"
        assert r.old_value == "acp_client"
        assert r.requires_new_session is True
        assert cfg["model"].get("provider") != "acp-client"

    def test_enable_uses_existing_command_when_not_provided(self):
        """If no new command is given but one is already stored, reuse it."""
        cfg = {"model": {
            "provider": "acp-client",
            "acp_command": "stored-agent",
            "acp_args": [],
        }}
        with patch.object(ars, "check_acp_command_ok",
                          return_value=(True, "1.0")):
            r = ars.apply(cfg, "acp_client")
        assert r.success
        assert "already enabled" in r.message  # no-change path

    def test_persist_callback_failure_reported(self):
        cfg = {}

        def persist_boom(c):
            raise IOError("disk full")

        with patch.object(ars, "check_acp_command_ok",
                          return_value=(True, "1.0")):
            r = ars.apply(
                cfg,
                "acp_client",
                acp_command="myagent",
                persist_callback=persist_boom,
            )
        assert r.success is False
        assert "persist failed" in r.message
        assert "disk full" in r.message

    def test_disable_persist_callback_failure_reported(self):
        cfg = {"model": {"provider": "acp-client", "acp_command": "myagent"}}

        def persist_boom(c):
            raise IOError("disk full")

        r = ars.apply(cfg, "auto", persist_callback=persist_boom)
        assert r.success is False
        assert "persist failed" in r.message
