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


# --- hermes-sweeper review: bounded args + slash-safe interactive modes ---

def test_board_limit_rejects_negative():
    # A negative --limit was truthy-but-not->0, silently becoming unlimited.
    with pytest.raises(SystemExit):
        _parse(["board", "--limit", "-1"])


def test_board_limit_zero_still_unlimited():
    # 0 remains the documented "unlimited" sentinel and must still parse.
    ns = _parse(["board", "--limit", "0"])
    assert ns.limit == 0


def test_board_refresh_rejects_zero():
    # --refresh 0 -> time.sleep(0) busy loop; must be rejected at parse time.
    with pytest.raises(SystemExit):
        _parse(["board", "--refresh", "0"])


def test_board_refresh_rejects_negative():
    with pytest.raises(SystemExit):
        _parse(["board", "--refresh", "-1"])


def _run_slash_no_dispatch(monkeypatch, rest):
    """Run a /kanban slash string with kanban_command mocked so an unbounded
    interactive command cannot actually hang the test; return (output, dispatched)."""
    dispatched = []
    monkeypatch.setattr(
        k, "kanban_command",
        lambda args: (dispatched.append(getattr(args, "kanban_action", None)), 0)[1],
    )
    out = k.run_slash(rest)
    return out, dispatched


def test_slash_rejects_board_tail(monkeypatch):
    # board --tail runs an unbounded live view; over the slash path run_slash
    # captures stdout synchronously, so it would hang the gateway worker.
    out, dispatched = _run_slash_no_dispatch(monkeypatch, "board --tail")
    assert not dispatched, "board --tail must not dispatch over slash (would hang)"
    assert "interactive terminal" in out.lower()


def test_slash_allows_board_without_tail(monkeypatch):
    # The bounded board render must still dispatch normally over slash.
    _out, dispatched = _run_slash_no_dispatch(monkeypatch, "board")
    assert dispatched == ["board"]


def test_slash_rejects_tail_and_watch(monkeypatch):
    # Pre-existing sibling hangs closed by the same guard.
    out_tail, disp_tail = _run_slash_no_dispatch(monkeypatch, "tail 5")
    assert not disp_tail and "interactive terminal" in out_tail.lower()
    out_watch, disp_watch = _run_slash_no_dispatch(monkeypatch, "watch")
    assert not disp_watch and "interactive terminal" in out_watch.lower()
