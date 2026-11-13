from types import SimpleNamespace

import pytest

from hermes_cli import main as main_mod
from hermes_cli._parser import build_top_level_parser


def test_chat_parser_accepts_opt_in_structured_result_record():
    parser, _subparsers, _chat_parser = build_top_level_parser()

    args = parser.parse_args(
        ["chat", "-q", "machine task", "-Q", "--emit-result-record"]
    )

    assert args.quiet is True
    assert args.emit_result_record is True


def test_result_record_rejects_tui_selected_by_configuration(monkeypatch, capsys):
    args = SimpleNamespace(
        emit_result_record=True,
        quiet=True,
        query="machine task",
    )
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: True)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert "classic CLI" in capsys.readouterr().err
