"""`kanban create --body-file` — shell-proof body delivery (#115432).

`--body` is a plain string: embedded newlines and trailing flag-like tokens
(e.g. a body line reading ``--json``) die in shell-to-Python marshalling
(MSYS git-bash / terminal shell=True). ``--body-file <path>|-`` reads the
body from a file (``-`` = stdin) so the bytes never cross the shell.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

BODY = "line one\nline two\n--json\n--body should survive verbatim\n"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _build_kanban_parser():
    root = argparse.ArgumentParser(prog="hermes")
    subs = root.add_subparsers()
    return kc.build_parser(subs)


def _make_create_ns(**overrides):
    ns = argparse.Namespace(
        title="t", body=None, body_file=None, assignee=None,
        created_by="user", workspace="scratch", branch=None, project=None,
        tenant=None, priority=0, parent=None, triage=False,
        idempotency_key=None, max_runtime=None, max_retries=None,
        skills=None, json=False, model_override=None,
        provider_override=None, goal_mode=False, goal_max_turns=None,
        completion_contract=None, initial_status="running",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _latest_body():
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn)
    assert tasks, "expected _cmd_create to have stored a task"
    return tasks[-1].body


def test_create_parser_accepts_body_file():
    parser = _build_kanban_parser()
    args = parser.parse_args(["create", "title here", "--body-file", "note.md"])
    assert args.body_file == "note.md"


def test_create_body_file_preserves_newlines_and_flag_like_lines(kanban_home, tmp_path, capsys):
    f = tmp_path / "body.md"
    f.write_text(BODY)
    rc = kc._cmd_create(_make_create_ns(body_file=str(f)))
    assert rc == 0
    capsys.readouterr()
    assert _latest_body() == BODY


def test_create_body_file_dash_reads_stdin(kanban_home, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(BODY))
    rc = kc._cmd_create(_make_create_ns(body_file="-"))
    assert rc == 0
    capsys.readouterr()
    assert _latest_body() == BODY


def test_create_body_and_body_file_conflict(kanban_home, tmp_path, capsys):
    f = tmp_path / "body.md"
    f.write_text(BODY)
    rc = kc._cmd_create(_make_create_ns(body="inline", body_file=str(f)))
    assert rc == 2
    capsys.readouterr()
