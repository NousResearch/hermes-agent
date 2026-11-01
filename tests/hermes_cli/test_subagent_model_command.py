"""/subagent model — parse/format/apply logic plus the CLI handler wiring.

The slash command pins ``delegation.provider`` + ``delegation.model`` through the repo's
config-write helper and prints the effective route on bare invocation. These tests cover the
shared module directly and the CLI handler with ``save_config_value`` / ``load_config``
stubbed, so nothing touches the real ``config.yaml``.
"""

from __future__ import annotations

import cli as cli_mod
from cli import HermesCLI
from hermes_cli import subagent_model as sm


class TestParseArgs:
    def test_bare_and_model_print_current(self):
        assert sm.parse_args("").errors  # /subagent alone is a usage error
        req = sm.parse_args("model")
        assert req.errors == []
        assert req.action == "show"

    def test_set_parses_provider_and_model(self):
        req = sm.parse_args("model set openrouter/google/gemini-3-flash-preview")
        assert req.errors == []
        assert req.action == "set"
        assert (req.provider, req.model) == ("openrouter", "google/gemini-3-flash-preview")

    def test_set_subcommand_is_case_insensitive(self):
        req = sm.parse_args("MODEL SET Nous/hermes-4")
        assert req.errors == []
        assert (req.provider, req.model) == ("Nous", "hermes-4")

    def test_unknown_subcommand_errors(self):
        assert sm.parse_args("tools").errors

    def test_unknown_verb_errors(self):
        assert sm.parse_args("model frobnicate openrouter/x").errors

    def test_missing_empty_and_malformed_values_error(self):
        for arg in ("model set", "model set openrouter", "model set /glm-5", "model set openrouter/"):
            req = sm.parse_args(arg)
            assert req.errors, arg
            assert req.action == "show" and not req.provider and not req.model


class TestFormatStatus:
    def test_unset_shows_inherit(self):
        lines = sm.format_status({})
        assert "inherit (parent agent's provider and model)" in lines[0]
        assert "hot_reload_model: false" in lines[3]

    def test_pinned_shows_route_and_hot_reload(self):
        lines = sm.format_status(
            {"delegation": {"provider": "nous", "model": "hermes-4", "hot_reload_model": True}}
        )
        assert "Subagent model: nous/hermes-4" in lines[0]
        assert "delegation.provider: nous" in lines[1]
        assert "delegation.model: hermes-4" in lines[2]
        assert "hot_reload_model: true" in lines[3]


class TestApply:
    def test_sets_both_keys_and_persists_the_mutated_config(self):
        cfg: dict = {}
        seen: list[dict] = []
        status = sm.apply(
            cfg, "openrouter", "google/gemini-3-flash-preview",
            persist_callback=lambda c: seen.append(c) or True,
        )
        assert status.success
        assert cfg["delegation"]["provider"] == "openrouter"
        assert cfg["delegation"]["model"] == "google/gemini-3-flash-preview"
        assert seen and seen[0]["delegation"]["provider"] == "openrouter"
        assert "openrouter" in status.message

    def test_persist_returning_false_is_a_failure(self):
        status = sm.apply({}, "nous", "hermes-4", persist_callback=lambda _c: False)
        assert not status.success
        assert "config.yaml" in status.message

    def test_persist_raising_is_a_failure_not_a_crash(self):
        def boom(_c):
            raise OSError("read-only")

        status = sm.apply({}, "nous", "hermes-4", persist_callback=boom)
        assert not status.success
        assert "read-only" in status.message


def _cli_with_capture(monkeypatch, config, written):
    printed: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)
    monkeypatch.setattr(
        cli_mod, "save_config_value",
        lambda key, value: (written.__setitem__(key, value), True)[1],
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    return HermesCLI.__new__(HermesCLI), printed


class TestHandlerWiring:
    def test_handler_persists_pinned_route(self, monkeypatch):
        written: dict = {}
        cmd, printed = _cli_with_capture(monkeypatch, {}, written)
        assert cmd._handle_subagent_command(
            "/subagent model set openrouter/google/gemini-3-flash-preview") is True
        assert written == {
            "delegation.provider": "openrouter",
            "delegation.model": "google/gemini-3-flash-preview",
        }
        assert any("✓" in line for line in printed)

    def test_handler_prints_current_route(self, monkeypatch):
        written: dict = {}
        cmd, printed = _cli_with_capture(
            monkeypatch,
            {"delegation": {"provider": "nous", "model": "hermes-4", "hot_reload_model": True}},
            written,
        )
        assert cmd._handle_subagent_command("/subagent model") is True
        assert written == {}
        assert any("nous/hermes-4" in line for line in printed)

    def test_handler_rejects_invalid_input(self, monkeypatch):
        written: dict = {}
        cmd, printed = _cli_with_capture(monkeypatch, {}, written)
        assert cmd._handle_subagent_command("/subagent model set openrouter") is True
        assert written == {}
        assert any("✗" in line for line in printed)
