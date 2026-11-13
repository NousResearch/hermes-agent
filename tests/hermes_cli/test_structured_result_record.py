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


@pytest.mark.parametrize(
    ("quiet", "query"),
    [
        (False, "machine task"),
        (True, None),
    ],
)
def test_result_record_rejects_invalid_classic_mode_with_usage_exit(
    monkeypatch, capsys, quiet, query
):
    args = SimpleNamespace(
        emit_result_record=True,
        quiet=quiet,
        query=query,
        model=None,
        toolsets=None,
    )
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert "quiet single-query mode" in capsys.readouterr().err
