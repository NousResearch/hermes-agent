"""`hermes kanban comment` takes its body through `--body` as well as positionally (#106938).

`kanban create` documents its opening post as `--body`, while `comment` only took a
positional string, so one concept had two shapes and the wrong guess cost a failed
call plus a `--help` round trip. `--body` is now an alias for the positional text —
same stored body, same `--max-len` trim. A caller that supplies both is picking two
different bodies, so the command refuses instead of silently dropping one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _new_task() -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="PROBE", created_by="tester")


def _comment(argv: list[str]) -> int:
    """Drive the real ``hermes kanban comment`` parser into ``_cmd_comment``."""
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    return kc._cmd_comment(root.parse_args(["kanban", "comment", *argv]))


def _bodies(task_id: str) -> list[str]:
    with kbc.connect_closing() as conn:
        return [c.body for c in kb.list_comments(conn, task_id)]


def test_body_flag_stores_the_same_comment_as_the_positional_text(kanban_home, capsys):
    tid = _new_task()
    assert _comment([tid, "--body", "use the 2026 schema, not 2025"]) == 0
    # The positional spelling is the same request: same text, same stored row.
    assert _comment([tid, "use", "the", "2026", "schema,", "not", "2025"]) == 0
    # ... and --max-len caps the flag body through the same trim path.
    assert _comment([tid, "--body", "x" * 200, "--max-len", "40"]) == 0
    capsys.readouterr()
    bodies = _bodies(tid)
    assert bodies[:2] == ["use the 2026 schema, not 2025"] * 2
    assert len(bodies[2]) <= 40


def test_a_flag_before_the_positional_text_is_not_order_sensitive(kanban_home, capsys):
    tid = _new_task()
    # `comment <id> --author NAME "text"` puts an option ahead of the positional
    # text: the alias must resolve identically whatever the option/positional order.
    assert _comment([tid, "--author", "alice", "ship", "the", "2026", "schema"]) == 0
    # ... and an option ahead of it still reaches the same trim path.
    assert _comment([tid, "--max-len", "40", "--author", "bob", "y" * 200]) == 0
    capsys.readouterr()
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
    assert [(c.author, c.body) for c in comments][0] == ("alice", "ship the 2026 schema")
    assert comments[1].author == "bob"
    assert len(comments[1].body) <= 40


def test_two_bodies_at_once_are_refused_and_nothing_is_stored(kanban_home, capsys):
    tid = _new_task()
    # Flag + positional together is two bodies, not one: refuse the call outright.
    assert _comment([tid, "--body", "flag body", "positional", "body"]) == 2
    assert _comment([tid]) == 2
    capsys.readouterr()
    assert _bodies(tid) == []
