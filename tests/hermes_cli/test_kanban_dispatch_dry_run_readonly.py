"""Tests: ``hermes kanban dispatch --dry-run`` is read-only (issue #94916).

``--dry-run`` advertises "just print what would happen", but
``_dispatch_once_locked`` called ``_run_reclaim_phase`` before it ever looked at
the flag, so a dry-run ran every mutating sweep the dispatcher has:
``release_stale_claims``, ``reconcile_orphaned_running``, ``detect_stale_running``,
``detect_crashed_workers``, ``enforce_max_runtime`` and ``recompute_ready`` --
three of which signal worker PIDs. On a Docker Compose stack whose containers
share a network namespace (and so a hostname) but not a PID namespace, a dry-run
from a sibling container closed four live runs as crashed "pid N not alive".

One board carries work for every sweep. The contract under test: a dry-run
tick leaves ``tasks`` / ``task_runs`` / ``task_events`` / ``task_comments``
byte-for-byte identical, sends no signal, still previews the spawn, and says
``dry_run=True`` in its result; the same board on a real tick moves every one
of those sweeps, which is what proves the fixture is not vacuous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty board and no crash grace window."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # 0 = reclaim immediately, so the fixtures below don't have to wait out the
    # freshly-spawned-worker grace window.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


class _Workers:
    """Fake process table: which worker PIDs look alive, and every signal sent.

    A signal "kills" the PID so the sweeps' exit polls return at once.
    """

    def __init__(self):
        self.alive: set[int] = set()
        self.sent: list[tuple[int, int]] = []

    def pid_alive(self, pid) -> bool:
        return int(pid) in self.alive

    def kill(self, pid, sig) -> None:
        self.sent.append((int(pid), int(sig)))
        self.alive.discard(int(pid))


@pytest.fixture
def workers(monkeypatch) -> _Workers:
    w = _Workers()
    monkeypatch.setattr(kb, "_pid_alive", w.pid_alive)
    monkeypatch.setattr(kbd, "_kill_fn", lambda signal_fn=None: w.kill)
    return w


def _spawn_fn(task, workspace, board=None):
    return 999999


def _host() -> str:
    return kb._claimer_id().split(":", 1)[0]


def _dump(conn, table: str) -> list[tuple]:
    return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def _board_checksum(conn) -> str:
    """sha256 over ``tasks`` + ``task_runs`` + ``task_events`` + ``task_comments``.

    Schema and rows, so a new column moves the digest too. ``task_comments`` is
    in the tuple because the reconcile sweep narrates itself through
    ``_kb._insert_comment``, which no other assertion here would notice.
    """
    digest = hashlib.sha256()
    for table in ("tasks", "task_runs", "task_events", "task_comments"):
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        digest.update(f"{table}({','.join(cols)})\n".encode())
        for row in _dump(conn, table):
            digest.update(repr(row).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _claimed_running(conn, *, title: str, max_runtime_seconds=None) -> tuple[str, int]:
    """A ``running`` task claimed by this host with a recorded worker PID."""
    tid = kb.create_task(
        conn, title=title, assignee="alice", max_runtime_seconds=max_runtime_seconds,
    )
    assert kb.claim_task(conn, tid, claimer=f"{_host()}:worker-{title}") is not None
    proc = subprocess.Popen(["true"])
    proc.wait()
    kbd._set_worker_pid(conn, tid, proc.pid)
    return tid, proc.pid


def _expire_claim(conn, tid: str) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 600, tid),
        )


def _backdate_start(conn, tid: str, seconds: int) -> None:
    then = int(time.time()) - seconds
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (then, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ? AND ended_at IS NULL",
            (then, tid),
        )


def _orphan_running(conn, tid: str) -> None:
    """``running`` with no claim bookkeeping at all: the reconcile sweep's case."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL WHERE id = ?",
            (tid,),
        )


def _mark_parent_done(conn, tid: str) -> None:
    """Flip a parent to ``done`` with raw SQL.

    ``complete_task`` promotes the children itself, which would leave nothing
    for ``recompute_ready`` to do inside the tick under test.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )


# ``detect_stale_running`` threshold for the fixture: the stale row is backdated
# well past it, the overrun row well short of it, so each sweep gets its own row.
_STALE_AFTER = 300


def _board_with_work_for_every_sweep(conn, workers: _Workers) -> dict[str, str]:
    """One row per sweep, plus a ready card for the spawn preview."""
    expired, _ = _claimed_running(conn, title="expired")          # release_stale_claims
    _expire_claim(conn, expired)
    orphan = kb.create_task(conn, title="orphan", assignee="alice")  # reconcile_orphaned_running
    _orphan_running(conn, orphan)
    stale, _ = _claimed_running(conn, title="stale")              # detect_stale_running
    _backdate_start(conn, stale, _STALE_AFTER * 2)
    dead, _ = _claimed_running(conn, title="dead")                # detect_crashed_workers
    overrun, overrun_pid = _claimed_running(                      # enforce_max_runtime
        conn, title="overrun", max_runtime_seconds=1,
    )
    _backdate_start(conn, overrun, 60)
    workers.alive.add(overrun_pid)  # alive, so it is the timeout sweep that must stop it
    parent = kb.create_task(conn, title="parent", assignee="alice")  # recompute_ready
    child = kb.create_task(conn, title="child", assignee="alice", parents=[parent])
    _mark_parent_done(conn, parent)
    ready = kb.create_task(conn, title="ready", assignee="alice")    # spawn preview
    return {
        "expired": expired, "orphan": orphan, "stale": stale, "dead": dead,
        "overrun": overrun, "child": child, "ready": ready,
    }


def _tick(conn, *, dry_run: bool) -> kbd.DispatchResult:
    return kbd.dispatch_once(
        conn, spawn_fn=_spawn_fn, dry_run=dry_run, stale_timeout_seconds=_STALE_AFTER,
    )


def test_dry_run_tick_is_read_only(conn, workers, all_assignees_spawnable):
    """A dry-run previews the spawn and moves nothing else: no row, no signal."""
    ids = _board_with_work_for_every_sweep(conn, workers)
    before = _board_checksum(conn)

    res = _tick(conn, dry_run=True)

    assert _board_checksum(conn) == before, "dry-run mutated the board"
    assert workers.sent == []
    assert res.skipped_locked is False
    # The sweeps' counters are empty because they did not run, and the result
    # says so rather than looking like a healthy board.
    assert res.dry_run is True
    assert res.reclaimed == 0 and res.promoted == 0
    assert res.reconciled_orphans == res.stale == res.crashed == res.timed_out == []
    # The preview still works, without claiming: the card stays unclaimed.
    assert res.spawned == [(ids["ready"], "alice", "")]
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?", (ids["ready"],)
    ).fetchone()
    assert (row["status"], row["claim_lock"], row["worker_pid"]) == ("ready", None, None)


def test_real_tick_moves_every_sweep_on_the_same_board(conn, workers, all_assignees_spawnable):
    """Control: the fixture is not vacuous — without the flag each sweep fires."""
    ids = _board_with_work_for_every_sweep(conn, workers)
    before = _board_checksum(conn)

    res = _tick(conn, dry_run=False)

    assert res.dry_run is False
    assert _board_checksum(conn) != before
    assert res.reclaimed == 1
    assert res.reconciled_orphans == [ids["orphan"]]
    assert res.stale == [ids["stale"]]
    assert res.crashed == [ids["dead"]]
    assert res.timed_out == [ids["overrun"]]
    assert res.promoted >= 1
    assert ids["ready"] in [tid for (tid, _who, _ws) in res.spawned]
    # The part that made the dry-run destructive rather than misleading: real
    # ticks signal workers, so a dry-run that ran these sweeps would too.
    assert workers.sent != []


def test_dry_run_json_carries_the_marker(kanban_home, capsys):
    """``dispatch --dry-run --json`` says it was a preview, in-band.

    A script reading ``reclaimed: 0, crashed: []`` from a dry-run must be able
    to tell "sweeps skipped" from "nothing to reclaim".
    """
    for flag in (True, False):
        args = argparse.Namespace(dry_run=flag, max=None, json=True)
        assert kanban_ops._cmd_dispatch(args) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["dry_run"] is flag
