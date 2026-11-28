"""``hermes chat --source`` must not overwrite a launcher-provided session source (#154979).

The launcher's ``HERMES_SESSION_SOURCE`` records where a run came from; ``--source`` is a listing
tag. Replacing one with the other silently loses provenance, so a conflict is refused.
"""

from __future__ import annotations

import os
import sys
import types

import pytest


def _cmd_chat(monkeypatch, argv):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(argv)
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: None
    fake_banner = types.ModuleType("hermes_cli.banner")
    fake_banner.prefetch_update_check = lambda: None
    fake_skills_sync = types.ModuleType("tools.skills_sync")
    fake_skills_sync.sync_skills = lambda quiet=True: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
    monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    return main_mod.cmd_chat(args)


def test_conflicting_source_flag_is_refused_before_overwriting_the_launcher_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "integration")
    monkeypatch.delenv("HERMES_SESSION_SOURCE_EXPLICIT", raising=False)

    with pytest.raises(SystemExit) as exc:
        _cmd_chat(monkeypatch, ["chat", "--cli", "--source", "tool"])

    assert exc.value.code == 2
    assert os.environ["HERMES_SESSION_SOURCE"] == "integration"
    assert "HERMES_SESSION_SOURCE_EXPLICIT" not in os.environ
    err = capsys.readouterr().err
    assert "--source 'tool'" in err and "HERMES_SESSION_SOURCE='integration'" in err


def test_source_flag_sets_the_label_when_the_launcher_set_none(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE_EXPLICIT", raising=False)

    _cmd_chat(monkeypatch, ["chat", "--cli", "--source", "tool"])

    assert os.environ["HERMES_SESSION_SOURCE"] == "tool"
    assert os.environ["HERMES_SESSION_SOURCE_EXPLICIT"] == "1"


def test_source_flag_may_restate_the_launcher_value(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    monkeypatch.delenv("HERMES_SESSION_SOURCE_EXPLICIT", raising=False)

    _cmd_chat(monkeypatch, ["chat", "--cli", "--source", "tool"])

    assert os.environ["HERMES_SESSION_SOURCE"] == "tool"
    assert os.environ["HERMES_SESSION_SOURCE_EXPLICIT"] == "1"
