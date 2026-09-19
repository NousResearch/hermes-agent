"""Regression for #111408: ``_tui_init_run_state`` must be able to seed the
config-watcher signature.

The commit that widened the CLI config-watcher caches to
``utils.file_signature`` (salvage of #111113) rewrote the seed line in
``cli_tui_mixin._tui_init_run_state`` but did not bring ``file_signature``
into that module's namespace. Every sibling cache touched by the same commit
(``cli_info_mixin``, ``config``, ``model_tools``, ``tui_gateway.server``)
imports the name; only the mixin did not, so the expression raised
``NameError`` the first time a user started the interactive TUI.

The pre-existing watcher tests could not catch it: ``_make_cli`` builds the
object with ``object.__new__`` and assigns ``_config_sig`` directly, so the
seed line — the only place the mixin references the helper — never ran.

These tests call the real method against a real ``config.yaml``, which is the
path a user's first ``hermes`` invocation takes.
"""

from pathlib import Path

import pytest

import cli as cli_mod


def _bare_cli(config: dict) -> "cli_mod.HermesCLI":
    """A HermesCLI with only what ``_tui_init_run_state`` reads before the seed.

    ``object.__new__`` skips ``__init__``; the two flags below are what the
    method's plugin-callback leg would otherwise touch. Everything else the
    method sets is created by the method itself.
    """
    obj = object.__new__(cli_mod.HermesCLI)
    obj.config = config
    obj._tool_callbacks_installed = True
    obj._tirith_security_checked = True
    return obj


def test_tui_init_run_state_seeds_config_signature(tmp_path, monkeypatch):
    """The seed line runs and stores a real signature (regression for #111408).

    Red on the base commit: ``NameError: name 'file_signature' is not defined``.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp_servers: {}\n")

    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: cfg)

    obj = _bare_cli({"mcp_servers": {}})
    obj._tui_init_run_state()

    assert obj._config_sig is not None, "signature should be seeded from config.yaml"
    assert isinstance(obj._config_sig, tuple), "seed must be a file_signature tuple"
    assert len(obj._config_sig) == 4, "file_signature returns a 4-tuple"


def test_tui_init_run_state_signature_changes_when_config_changes(tmp_path, monkeypatch):
    """The seeded signature is the one the watcher compares against.

    A different config.yaml must produce a different signature, otherwise the
    auto-reload this cache exists for would never fire.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp_servers: {}\n")

    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: cfg)

    first = _bare_cli({"mcp_servers": {}})
    first._tui_init_run_state()

    cfg.write_text("mcp_servers:\n  github: {url: https://mcp.github.com}\n")

    second = _bare_cli({"mcp_servers": {}})
    second._tui_init_run_state()

    assert first._config_sig != second._config_sig


def test_tui_init_run_state_seed_is_none_without_config_file(tmp_path, monkeypatch):
    """A missing config.yaml seeds ``None`` rather than raising."""
    missing = tmp_path / "does-not-exist.yaml"

    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: missing)

    obj = _bare_cli({"mcp_servers": {}})
    obj._tui_init_run_state()

    assert obj._config_sig is None
