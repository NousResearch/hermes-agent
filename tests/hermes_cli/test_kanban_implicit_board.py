"""Board-less mutations must reveal routing through the shared current file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_ops import _implicit_board_guard


COMMANDS = (
    ("create", "new work"),
    ("dispatch", "--dry-run"),
    ("swarm", "goal", "--worker", "dev:part", "--verifier", "dev", "--synthesizer", "dev"),
    ("specify", "--all"),
    ("decompose", "--all"),
    ("gc",),
    ("daemon", "--force"),
    ("init",),
    ("repair",),
)
TASK_COUNT = {"create": 1, "swarm": 4}


@pytest.fixture
def switched_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_HOME", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    kb.create_board("other")
    kb.set_current_board("other")
    monkeypatch.setattr(kc, "_check_dispatcher_presence", lambda: (True, ""))
    monkeypatch.setattr(kbd, "run_daemon", lambda **kwargs: None)
    return home


def _strict(home, value=True):
    from hermes_cli.config import load_config

    (home / "config.yaml").write_text(
        f"kanban:\n  require_explicit_board: {str(value).lower()}\n", encoding="utf-8")
    load_config()  # Establish the loader's normal last-known-good backup before snapshots.


def _run(command, *, board=None):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    flags = ["--board", board] if board else []
    output = [] if command[0] in {"init", "gc", "daemon", "comment"} else ["--json"]
    return kc.kanban_command(parser.parse_args(["kanban", *flags, *command, *output]))


def _tasks(board):
    with kbc.connect_closing(board=board) as conn:
        return kb.list_tasks(conn)


def _snapshot(home):
    return {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}


@pytest.mark.parametrize("command", COMMANDS)
def test_implicit_board_warns_and_proceeds(switched_home, capsys, command):
    assert _run(command) == 0
    captured = capsys.readouterr()
    assert captured.err.count("was selected only by") == 1
    assert "board 'other'" in captured.err
    assert str(kb.current_board_path()) in captured.err
    assert "--board other" in captured.err
    assert "HERMES_KANBAN_BOARD" in captured.err
    assert len(_tasks("other")) == TASK_COUNT.get(command[0], 0)
    assert _tasks("default") == []


@pytest.mark.parametrize("command", COMMANDS)
def test_strict_refuses_before_any_mutation(switched_home, monkeypatch, capsys, command):
    _strict(switched_home)
    with kbc.connect_closing(board="other") as conn:
        tid = kb.create_task(conn, title="waiting", assignee="dev")
        before = kb.get_task(conn, tid)

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("strict refusal must happen before dispatch_once")

    def unexpected_repair(*args, **kwargs):
        pytest.fail("strict refusal must happen before repair_db")

    monkeypatch.setattr(kbd, "dispatch_once", unexpected_dispatch)
    monkeypatch.setattr(kbc, "repair_db", unexpected_repair)
    snapshot = _snapshot(switched_home)
    assert _run(command) == 2
    assert _snapshot(switched_home) == snapshot
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "board 'other'" in captured.err
    assert "kanban.require_explicit_board: false" in captured.err
    assert _tasks("default") == []
    assert _tasks("other") == [before]


@pytest.mark.parametrize("pin", ["flag", "env", "default"])
@pytest.mark.parametrize("command", COMMANDS)
def test_explicit_or_default_board_is_silent_even_when_strict(
    switched_home, monkeypatch, capsys, pin, command
):
    _strict(switched_home)
    if pin == "env":
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "other")
    if pin == "default":
        kb.set_current_board("default")
    assert _run(command, board="other" if pin == "flag" else None) == 0
    err = capsys.readouterr().err
    assert "was selected only by" not in err
    if command[0] != "daemon":
        assert err == ""
    landed = "default" if pin == "default" else "other"
    assert len(_tasks(landed)) == TASK_COUNT.get(command[0], 0)


@pytest.mark.parametrize("command", COMMANDS)
def test_strict_refuses_before_initializing_missing_db(switched_home, capsys, command):
    _strict(switched_home)
    kb.kanban_db_path().unlink()  # create_board eagerly initialises its temporary DB.
    assert not kb.kanban_db_path().exists()
    before = _snapshot(switched_home)
    assert _run(command) == 2
    assert _snapshot(switched_home) == before
    assert "board 'other'" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["specify", "decompose"])
def test_id_addressed_triage_verb_is_unguarded(switched_home, capsys, verb):
    _strict(switched_home)
    with kbc.connect_closing(board="other") as conn:
        tid = kb.create_task(conn, title="already specified", assignee="dev")
    assert _run((verb, tid)) == 1  # Not in triage, not a strict routing refusal.
    captured = capsys.readouterr()
    assert captured.err == ""
    outcome = json.loads(captured.out)
    assert outcome["task_id"] == tid
    assert not outcome["ok"]
    assert "triage" in outcome["reason"]


def test_daemon_without_force_is_unguarded(switched_home, capsys):
    _strict(switched_home)
    assert _run(("daemon",)) == 2
    err = capsys.readouterr().err
    assert "DEPRECATED" in err
    assert "was selected only by" not in err


def test_comment_on_implicit_board_is_unguarded(switched_home, capsys):
    _strict(switched_home)
    with kbc.connect_closing(board="other") as conn:
        tid = kb.create_task(conn, title="existing work", assignee="dev")
    assert _run(("comment", tid, "progress", "--author", "user")) == 0
    assert capsys.readouterr().err == ""
    with kbc.connect_closing(board="other") as conn:
        assert kb.list_comments(conn, tid)[-1].body == "progress"


@pytest.mark.parametrize("verb", ["list", "ls", "stats", "notify-list"])
def test_board_reads_are_unguarded(switched_home, capsys, verb):
    _strict(switched_home)
    assert _run((verb,)) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("config", [None, False])
def test_config_absent_or_false_warns_only(switched_home, capsys, config):
    if config is not None:
        _strict(switched_home, config)
    assert _run(COMMANDS[0]) == 0
    err = capsys.readouterr().err
    assert "board 'other'" in err
    assert "kanban.require_explicit_board: false" not in err
    assert [t.title for t in _tasks("other")] == ["new work"]


def test_blank_board_argument_is_not_explicit(switched_home):
    _strict(switched_home)
    for blank in ("", "   "):
        message, refused = _implicit_board_guard(blank)
        assert refused and "board 'other'" in message


def test_archived_board_override_is_explicit_intent(switched_home):
    _strict(switched_home)
    kb.create_board("old")
    kb.remove_board("old")  # archives: leaves a tombstone
    assert _implicit_board_guard()[1] is True
    with kb.scoped_current_board("old"):
        assert _implicit_board_guard() == (None, False)
