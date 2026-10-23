"""Claim-less transitions must leave live runs and their history unchanged.

The CLI and tool refusals must not suggest bypasses. Matching run ownership
and explicit operator overrides remain supported: this is an accidental-
mutation guard, not confinement of arbitrary local code. Reason comments
and transitions must commit or roll back together, including CLI unblock.
"""

from __future__ import annotations

import argparse
import os
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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    yield home


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser


def _live_running_task(kanban_home) -> tuple[str, int]:
    """A claimed running card whose worker process is alive (this process)."""
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="live", assignee="coder")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        kbd._set_worker_pid(conn, tid, os.getpid())
        run_id = kb._current_run_id(conn, tid)
    assert run_id is not None
    return tid, int(run_id)


def _task_status(tid: str) -> str:
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    return str(task.status)


def _state(tid):
    with kbc.connect_closing() as conn:
        return {
            "task": kb.get_task(conn, tid),
            "runs": kb.list_runs(conn, tid),
            "comments": kb.list_comments(conn, tid),
            "events": kb.list_events(conn, tid),
        }


@pytest.mark.parametrize("verb,reason_args", [
    ("block", ["child tried to change state"]),
    ("schedule", ["child tried to change state"]),
    ("unblock", ["--reason", "cannot resume"]),
    ("complete", ["--result", "not the owner"]),
    ("request-review", ["--summary", "not the owner"]),
    ("kanban_block", []),
    ("kanban_complete", []),
])
def test_refusal_preserves_live_claim_and_history(kanban_home, capsys, verb, reason_args):
    tid, run_id = _live_running_task(kanban_home)
    before = _state(tid)
    assert before["task"].status == "running"
    assert before["task"].claim_lock
    assert before["task"].worker_pid == os.getpid()
    assert before["runs"][0].id == run_id
    assert before["runs"][0].ended_at is None

    if verb.startswith("kanban_"):
        import json
        import tools.kanban_tools  # register the real handlers
        from tools.registry import registry

        result = registry.dispatch(verb, {"task_id": tid, "reason": "not the owner", "summary": "not the owner"})
        assert isinstance(result, str)
        err = json.loads(result)["error"]
    else:
        args = _parser().parse_args(["kanban", verb, tid, *reason_args])
        assert kc.kanban_command(args) != 0
        err = capsys.readouterr().err
    if verb == "unblock":
        assert "cannot unblock" in err
    else:
        assert "live worker" in err or "live claim" in err

    # The claim, open run, comments AND event history must all be unchanged.
    assert _state(tid) == before
    assert "force" not in err and "reclaim" not in err


@pytest.mark.parametrize("verb,status", [("block", "blocked"), ("schedule", "scheduled")])
@pytest.mark.parametrize("authority", ["owner", "force"])
@pytest.mark.parametrize("reason", ["operator note", None])
def test_cli_authorized_transition_commits_with_comment(kanban_home, monkeypatch, verb, status, authority, reason):
    tid, run_id = _live_running_task(kanban_home)
    before = _state(tid)
    argv = ["kanban", verb, tid, *([reason] if reason else [])]
    if authority == "owner":
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1))
        assert kc.kanban_command(_parser().parse_args(argv)) != 0
        assert _state(tid) == before  # stale owner fails the CAS without history
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    else:
        argv.append("--force")
    assert kc.kanban_command(_parser().parse_args(argv)) == 0

    after = _state(tid)
    assert after["task"].status == status
    assert after["task"].claim_lock is None
    assert after["runs"][0].id == run_id
    assert after["runs"][0].ended_at is not None
    assert after["runs"][0].outcome == status
    assert [c.body for c in after["comments"]] == ([f"{status.upper()}: {reason}"] if reason else [])
    added_events = after["events"][len(before["events"]):]
    assert [e.kind for e in added_events] == (["commented", status] if reason else [status])


def test_block_cli_without_live_worker_unchanged(kanban_home):
    """Human flow on a card nobody is running (or whose worker is gone) blocks as before."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="admin", assignee="coder")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        # no _set_worker_pid: a library/CLI claim protects no live run

    args = _parser().parse_args(["kanban", "block", tid, "manual"])
    assert kc.kanban_command(args) == 0
    assert _task_status(tid) == "blocked"


@pytest.mark.parametrize("verb,event", [("block", "blocked"), ("schedule", "scheduled"), ("unblock", "unblocked")])
@pytest.mark.parametrize("fail_at", ["commented", "transition"])
def test_cli_transition_and_comment_roll_back_together(kanban_home, monkeypatch, verb, event, fail_at):
    import sqlite3

    tid, _ = _live_running_task(kanban_home)
    if verb == "unblock":
        with kbc.connect_closing() as conn:
            assert kb.block_task(conn, tid, force=True)
    before = _state(tid)
    hooks = []
    monkeypatch.setattr(kb, "_fire_task_hook", lambda *a, **kw: hooks.append(a))
    failed_event = event if fail_at == "transition" else "commented"
    with kbc.connect_closing() as conn:
        # A real SQLite failure after the comment INSERT (and, for transition,
        # after its nested savepoint RELEASE) must undo the entire operation.
        conn.execute(
            "CREATE TRIGGER fail_history BEFORE INSERT ON task_events "
            f"WHEN NEW.kind = '{failed_event}' "
            "BEGIN SELECT RAISE(ABORT, 'injected history write failure'); END"
        )
    extra = ["--reason", "retry later"] if verb == "unblock" else ["retry later", "--force"]
    args = _parser().parse_args(["kanban", verb, tid, *extra])
    with pytest.raises(sqlite3.IntegrityError, match="injected history write failure"):
        kc.kanban_command(args)
    assert _state(tid) == before
    assert hooks == []


@pytest.mark.parametrize("verb,status,prefix", [
    ("block", "blocked", "BLOCKED"),
    ("schedule", "scheduled", "SCHEDULED"),
    ("unblock", "ready", "UNBLOCK"),
])
def test_cli_bulk_comments_only_successful_transitions(kanban_home, capsys, verb, status, prefix):
    rejected, _ = _live_running_task(kanban_home)
    before = _state(rejected)
    with kbc.connect_closing() as conn:
        accepted = kb.create_task(conn, title="operator card", assignee="coder")
        if verb == "unblock":
            assert kb.block_task(conn, accepted)
    accepted_before = _state(accepted)
    missing = "t_missing"
    if verb == "unblock":
        argv = ["kanban", verb, missing, rejected, accepted, "--reason", "bulk reason"]
    else:
        argv = ["kanban", verb, missing, "bulk reason", "--ids", rejected, accepted]
    assert kc.kanban_command(_parser().parse_args(argv)) == 1
    out = capsys.readouterr()
    assert accepted in out.out
    assert missing in out.err and rejected in out.err
    assert _state(rejected) == before
    assert _state(missing) == {"task": None, "runs": [], "comments": [], "events": []}
    after = _state(accepted)
    assert after["task"].status == status
    assert [c.body for c in after["comments"]] == [f"{prefix}: bulk reason"]
    added_events = after["events"][len(accepted_before["events"]):]
    assert [e.kind for e in added_events] == ["commented", "unblocked" if verb == "unblock" else status]
