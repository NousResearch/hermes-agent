"""Behavioral coverage for ``hermes chat --no-tools``."""

import os
import sys
import types
from argparse import Namespace

import pytest


def test_chat_parser_accepts_no_tools():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["chat", "--no-tools", "--resume", "session-id"])

    assert args.no_tools is True
    assert args.resume == "session-id"


def test_no_tools_rejects_explicit_toolsets(monkeypatch):
    from hermes_cli.main import _apply_no_tools

    monkeypatch.delenv("HERMES_NO_TOOLS", raising=False)
    args = Namespace(no_tools=True, toolsets="terminal")

    with pytest.raises(SystemExit, match="cannot be used with --toolsets"):
        _apply_no_tools(args)

    assert "HERMES_NO_TOOLS" not in os.environ


def test_cmd_chat_forwards_no_tools(monkeypatch):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--no-tools"])
    captured = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: captured.update(kwargs)

    monkeypatch.delenv("HERMES_NO_TOOLS", raising=False)
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)

    main_mod.cmd_chat(args)

    assert os.environ["HERMES_NO_TOOLS"] == "1"
    assert captured["no_tools"] is True


def test_classic_cli_preserves_explicit_empty_tool_selection(monkeypatch):
    import cli as cli_mod

    captured = {}

    class FakeCLI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def show_banner(self):
            pass

        def show_tools(self):
            pass

    monkeypatch.delenv("HERMES_NO_TOOLS", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(no_tools=True, list_tools=True)

    assert exc_info.value.code == 0
    assert captured["toolsets"] == []


def test_no_tools_overrides_configured_and_worker_tools(monkeypatch):
    from model_tools import get_tool_definitions

    monkeypatch.setenv("HERMES_NO_TOOLS", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-id")

    assert get_tool_definitions(
        enabled_toolsets=["terminal"], quiet_mode=True
    ) == []


def test_tui_preserves_explicit_empty_tool_selection(monkeypatch):
    from tui_gateway.server import _load_enabled_toolsets

    monkeypatch.setenv("HERMES_NO_TOOLS", "1")
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "terminal")

    assert _load_enabled_toolsets() == []
