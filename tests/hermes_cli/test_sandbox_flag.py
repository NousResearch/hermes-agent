"""Tests for `hermes chat --sandbox` → HERMES_TERMINAL_JAIL_ENABLED bridge."""

from __future__ import annotations

import os
import sys
import types

import pytest


_VAR = "HERMES_TERMINAL_JAIL_ENABLED"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)
    yield
    os.environ.pop(_VAR, None)


def test_apply_sandbox_sets_env_when_flag_true():
    import hermes_cli.main as main_mod

    args = types.SimpleNamespace(sandbox=True)
    main_mod._apply_sandbox(args)

    assert os.environ[_VAR] == "true"


def test_apply_sandbox_noop_when_flag_false():
    import hermes_cli.main as main_mod

    args = types.SimpleNamespace(sandbox=False)
    main_mod._apply_sandbox(args)

    assert _VAR not in os.environ


def test_apply_sandbox_noop_when_attr_missing():
    """SimpleNamespace-based callers (e.g. Termux fast paths, tests) may not
    define `sandbox` at all — must not raise."""
    import hermes_cli.main as main_mod

    args = types.SimpleNamespace()
    main_mod._apply_sandbox(args)

    assert _VAR not in os.environ


def test_chat_parser_accepts_sandbox_flag():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(["chat", "--sandbox"])

    assert args.sandbox is True


def test_chat_parser_sandbox_absent_leaves_default():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(["chat"])

    # chat subparser uses argparse.SUPPRESS for inherited flags — attribute
    # falls back to the top-level parser's default (False).
    assert getattr(args, "sandbox", False) is False


def test_top_level_sandbox_propagates_to_chat_subcommand():
    """`hermes --sandbox chat -q ...` must not drop the flag (parent→subparser
    propagation, same contract as --yolo/--safe-mode)."""
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(["--sandbox", "chat"])

    assert args.sandbox is True


def test_prepare_agent_startup_applies_sandbox_before_plugin_discovery(monkeypatch):
    import hermes_cli.main as main_mod

    args = types.SimpleNamespace(command="chat", sandbox=True, tui=False)
    plugins = types.ModuleType("hermes_cli.plugins")

    def discover_plugins() -> None:
        assert os.environ[_VAR] == "true"

    setattr(plugins, "discover_plugins", discover_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)
    monkeypatch.setattr(main_mod, "_should_background_mcp_startup", lambda _args: False)
    monkeypatch.setattr(main_mod, "_command_has_dedicated_mcp_startup", lambda _args: True)

    main_mod._prepare_agent_startup(args)

    assert os.environ[_VAR] == "true"


def test_sandbox_flag_inherited_on_relaunch():
    """--sandbox must survive self-relaunch (sessions browse, setup wizard)."""
    from hermes_cli.relaunch import _extract_inherited_flags

    assert "--sandbox" in _extract_inherited_flags(["chat", "--sandbox", "-q", "hi"])
    assert "--sandbox" in _extract_inherited_flags(["--sandbox", "chat"])


def test_config_bridge_exports_true_only_when_enabled(monkeypatch, tmp_path):
    """terminal.jail_enabled: true → HERMES_TERMINAL_JAIL_ENABLED=true.
    False/absent → env var untouched (plugin default applies)."""
    from hermes_cli import config as config_mod

    cfg = {"terminal": {"jail_enabled": True}}
    env: dict[str, str] = {}
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {"terminal": {}})
    config_mod.apply_terminal_config_to_env(env=env, config=cfg, override=True)
    assert env[_VAR] == "true"

    env2: dict[str, str] = {}
    cfg_off = {"terminal": {"jail_enabled": False}}
    config_mod.apply_terminal_config_to_env(env=env2, config=cfg_off, override=True)
    assert _VAR not in env2


def test_config_bridge_does_not_clobber_cli_flag(monkeypatch):
    """--sandbox sets the env var directly; the config bridge with
    jail_enabled absent/false must not unset it."""
    from hermes_cli import config as config_mod

    env: dict[str, str] = {_VAR: "true"}  # as --sandbox would set
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {})
    config_mod.apply_terminal_config_to_env(
        env=env, config={"terminal": {"jail_enabled": False}}, override=False
    )
    assert env[_VAR] == "true"
