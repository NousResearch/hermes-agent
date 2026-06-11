"""Tests for the 'kanban board' CLI subcommand: parsing + dispatch."""

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as k
from hermes_cli import kanban_db as kb


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse a kanban argv through the real build_parser path."""
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    k.build_parser(sub)
    return parser.parse_args(["kanban", *argv])


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_board_action_registered():
    ns = _parse(["board"])
    assert ns.kanban_action == "board"


def test_board_defaults():
    ns = _parse(["board"])
    assert ns.limit == 5
    assert ns.show_all is False
    assert ns.mine is False
    assert ns.assignee is None
    assert ns.json is False
    assert ns.tail is False
    assert ns.refresh == 5


def test_board_flags():
    ns = _parse([
        "board", "--limit", "10", "--show-all", "--mine",
        "--json", "--tail", "--refresh", "3",
    ])
    assert ns.limit == 10
    assert ns.show_all is True
    assert ns.mine is True
    assert ns.json is True
    assert ns.tail is True
    assert ns.refresh == 3


def test_board_assignee():
    ns = _parse(["board", "--assignee", "researcher"])
    assert ns.assignee == "researcher"


def test_board_global_board_flag_precedes_subcommand():
    # --board is a GLOBAL flag and must come before the subcommand.
    ns = _parse(["--board", "default", "board"])
    assert ns.board == "default"
    assert ns.kanban_action == "board"


def test_board_json_dispatch(kanban_home, capsys):
    with kb.connect() as conn:
        kb.create_task(conn, title="hello world")
    rc = k.kanban_command(_parse(["board", "--json"]))
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert any(t["title"] == "hello world" for t in data)


def test_board_layout_default_is_auto():
    assert _parse(["board"]).layout == "auto"


def test_board_layout_stack():
    assert _parse(["board", "--layout", "stack"]).layout == "stack"


def test_board_layout_columns():
    assert _parse(["board", "--layout", "columns"]).layout == "columns"


def test_board_layout_rejects_unknown():
    with pytest.raises(SystemExit):
        _parse(["board", "--layout", "grid"])
