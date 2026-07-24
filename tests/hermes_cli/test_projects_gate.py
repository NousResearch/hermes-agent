"""Tests for the ``projects.enabled`` config gate (issue #58588).

Covers:
- The central ``projects_enabled()`` predicate in ``hermes_cli.projects_gate``
- The CLI stub parser that surfaces the disabled message
- The RPC guard (``_E_PROJECTS_DISABLED``) in ``tui_gateway.server``
- The toolset fold gate in ``_load_enabled_toolsets()``

Default behaviour (no override) is preserved — these tests only assert
behaviour when the user has explicitly opted out.
"""

from __future__ import annotations

import argparse

import pytest


# ---------------------------------------------------------------------------
# Predicate tests
# ---------------------------------------------------------------------------


class TestProjectsEnabledPredicate:
    """The predicate is the single source of truth for the three call sites."""

    def test_default_when_no_projects_key(self):
        from hermes_cli.projects_gate import projects_enabled

        # Mirror a fresh install: no ``projects`` section in config.
        assert projects_enabled(cfg={}) is True

    def test_explicit_true(self):
        from hermes_cli.projects_gate import projects_enabled

        assert projects_enabled(cfg={"projects": {"enabled": True}}) is True

    def test_explicit_false(self):
        from hermes_cli.projects_gate import projects_enabled

        assert projects_enabled(cfg={"projects": {"enabled": False}}) is False

    def test_other_keys_ignored(self):
        from hermes_cli.projects_gate import projects_enabled

        # ``projects`` may grow new sub-keys; the toggle only honours
        # ``enabled``. This guards against accidental key shadowing.
        assert projects_enabled(cfg={"projects": {"enabled": False, "x": 1}}) is False
        assert projects_enabled(cfg={"projects": {"enabled": True, "y": 2}}) is True

    def test_mis_shaped_config_falls_back_to_default(self):
        from hermes_cli.projects_gate import projects_enabled

        # ``projects: "off"`` (string instead of dict) must not crash the
        # CLI parser or RPC handler. Falls back to the documented default.
        assert projects_enabled(cfg={"projects": "off"}) is True
        assert projects_enabled(cfg={"projects": None}) is True
        assert projects_enabled(cfg={"projects": 0}) is True

    def test_disabled_message_mentions_toggle(self):
        from hermes_cli.projects_gate import projects_disabled_message

        msg = projects_disabled_message()
        assert "projects" in msg
        assert "enabled" in msg
        assert "config.yaml" in msg


# ---------------------------------------------------------------------------
# CLI gate tests — verify that the parser registration honours the toggle
# ---------------------------------------------------------------------------


class TestCliParserGate:
    """Verify ``hermes project`` is wired up unconditionally and the handler gates."""

    def test_parser_registers_full_tree_when_enabled(self, monkeypatch):
        # Force the gate open regardless of any on-disk config the test
        # runner might have.
        from hermes_cli import projects_gate

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: True)

        # Re-import the build_parser via the public surface.
        from hermes_cli.projects_cmd import build_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = build_parser(sub)

        # Full tree present: ``create``, ``list``, ``show``, ...
        # We don't enumerate every subcommand (that's the existing
        # test_projects_cli.py's job); we just verify build_parser runs
        # without raising when the gate is open.
        assert p is not None

    def test_handler_returns_disabled_when_gate_off(self, monkeypatch, capsys):
        """cmd_project() short-circuits to the disabled message + exit 1."""
        from hermes_cli import projects_gate, main

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: False)

        rc = main.cmd_project(None)
        captured = capsys.readouterr()

        assert rc == 1
        assert "disabled" in captured.err.lower()
        # projects_command must NOT have been called.
        # (If it had been, ``None`` would crash inside the real handler.)

    def test_handler_dispatches_when_gate_on(self, monkeypatch):
        """cmd_project() delegates to projects_command() when the gate is open."""
        from hermes_cli import projects_gate, main
        import hermes_cli.projects_cmd as projects_cmd

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: True)

        sentinel_args = object()
        calls = []

        def _spy(args):
            calls.append(args)
            return 0

        # cmd_project re-imports ``projects_command`` from the module on every
        # call, so patch the module attribute (not a local binding).
        monkeypatch.setattr(projects_cmd, "projects_command", _spy)

        rc = main.cmd_project(sentinel_args)

        assert rc == 0
        assert calls == [sentinel_args]


# ---------------------------------------------------------------------------
# RPC guard tests — verify _projects_method rejects when disabled
# ---------------------------------------------------------------------------


class TestRpcGuard:
    """Verify all ``projects.*`` RPCs reject with ``_E_PROJECTS_DISABLED``."""

    def test_rpc_rejects_when_disabled(self, monkeypatch):
        """One RPC path covers the decorator; the decorator wraps all 11."""
        from hermes_cli import projects_gate
        import tui_gateway.server as server

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: False)

        # Pick the cheapest method (no DB side-effects): ``projects.list``.
        handler = server._methods["projects.list"]
        resp = handler(1, {})

        assert "error" in resp, f"expected rejection, got: {resp}"
        assert resp["error"]["code"] == server._E_PROJECTS_DISABLED
        assert "disabled" in resp["error"]["message"].lower()

    def test_rpc_succeeds_when_enabled(self, monkeypatch):
        """Regression guard: enabled-by-default path still works."""
        from hermes_cli import projects_gate
        import tui_gateway.server as server

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: True)

        handler = server._methods["projects.list"]
        resp = handler(1, {})

        assert "error" not in resp, resp.get("error")

    def test_error_code_constant_exists(self):
        """The new error code must be exported alongside the existing three."""
        import tui_gateway.server as server

        assert hasattr(server, "_E_PROJECTS_DISABLED")
        assert server._E_PROJECTS_DISABLED == 5064
        # Distinct from the other three.
        assert server._E_PROJECTS_DISABLED not in {
            server._E_PROJECTS,
            server._E_NO_PROJECT,
            server._E_PROJECT_ARG,
        }


# ---------------------------------------------------------------------------
# Toolset gate tests — verify _load_enabled_toolsets honours the toggle
# ---------------------------------------------------------------------------


class TestToolsetGate:
    """Verify the ``project`` toolset is not folded in when disabled."""

    def test_toolset_excluded_when_disabled(self, monkeypatch):
        from hermes_cli import projects_gate
        from tui_gateway.server import _load_enabled_toolsets

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: False)

        # The function reads HERMES_TUI_TOOLSETS env var; clear it so we
        # get the deterministic code path (env-empty + toolset recovery).
        monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

        enabled = _load_enabled_toolsets()
        if enabled is not None:
            assert "project" not in enabled, (
                f"project toolset leaked through disabled gate: {enabled}"
            )

    def test_toolset_included_when_enabled(self, monkeypatch):
        from hermes_cli import projects_gate
        from tui_gateway.server import _load_enabled_toolsets

        monkeypatch.setattr(projects_gate, "projects_enabled", lambda cfg=None: True)
        monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

        enabled = _load_enabled_toolsets()
        if enabled is not None:
            assert "project" in enabled, (
                f"project toolset missing under enabled gate: {enabled}"
            )