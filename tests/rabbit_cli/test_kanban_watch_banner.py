"""Watch identifies the board selected by the production command parser."""

import argparse

import pytest

from rabbit_cli import kanban_db as kb
from rabbit_cli import kanban_ops
from rabbit_cli.kanban import kanban_command
from rabbit_cli.kanban_parser import build_parser


@pytest.mark.parametrize(
    "environment,explicit,expected",
    [(None, None, "alpha"), ("beta", None, "beta"), ("alpha", "beta", "beta")],
)
def test_watch_names_resolved_board(tmp_path, monkeypatch, capsys, environment, explicit, expected):
    monkeypatch.setenv("RABBIT_HOME", str(tmp_path))
    monkeypatch.setenv("RABBIT_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("RABBIT_KANBAN_DB", raising=False)
    monkeypatch.delenv("RABBIT_KANBAN_BOARD", raising=False)
    kb.create_board("alpha")
    kb.create_board("beta")
    kb.set_current_board("alpha")
    if environment:
        monkeypatch.setenv("RABBIT_KANBAN_BOARD", environment)
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    argv = ["kanban"] + (["--board", explicit] if explicit else []) + ["watch"]

    def interrupt(_interval):
        raise KeyboardInterrupt

    monkeypatch.setattr(kanban_ops.time, "sleep", interrupt)
    assert kanban_command(parser.parse_args(argv)) == 0
    assert f"initial board '{expected}'" in capsys.readouterr().out
