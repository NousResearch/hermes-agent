"""Tests: a fresh-process dispatcher still books a clean-exit worker as a
protocol violation.

``_classify_worker_exit`` reads ``_recent_worker_exits``, an in-process dict
filled by ``reap_worker_zombies()`` → ``os.waitpid(-1, ...)`` — only children
of the *calling* process can land there. The gateway-embedded dispatcher owns
its workers, but ``hermes kanban dispatch`` from an operator timer is a new
process every tick: its registry is empty, so a worker that exited rc=0
without a terminal board call classified as ``unknown`` → plain ``crashed``
("pid N not alive"), never a ``protocol_violation``. ``consecutive_failures``
stayed untouched and ``_protocol_violation_streak`` never accumulated, so the
breaker never tripped and the same card could re-spawn 15+ times.

The durable witness is the worker's own log: the CLI prints its exit summary
(``Resume this session with:``) only after ``app.run()`` returned 0 (or a
Ctrl-C 130). A fresh-process sweep reads that tail and books the same
protocol violation the registry hit would.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
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


def _dead_worker_row(conn, host: str, tid: str) -> int:
    kb.claim_task(conn, tid, claimer=f"{host}:w")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,)
    )
    conn.commit()
    return dead.pid


def test_fresh_process_clean_exit_booked_as_protocol_violation(conn, kanban_home):
    """Registry miss + exit-summary marker in the log → the run closes as a
    protocol violation with the corrective error, not bare ``crashed``."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="fresh-proc", assignee="w")
    pid = _dead_worker_row(conn, host, tid)
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "worker chatter\n\nResume this session with:\n  hermes --resume s1\n\n"
        "Duration:       3s\nMessages:       1 (1 user, 0 tool calls)\n",
        encoding="utf-8",
    )
    # Fresh dispatcher process: the registry has no exit status for this pid.
    kbd._recent_worker_exits.pop(pid, None)

    crashed = kbd.detect_crashed_workers(conn)

    assert tid in crashed
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["outcome"] == "crashed"
    assert "protocol violation" in run["error"]
    meta = kb._json_dict(run["metadata"])
    assert meta.get("protocol_violation") is True
    assert meta.get("exit_code") == 0
    assert kbd._protocol_violation_streak(conn, tid) == 1


def test_fresh_process_clean_exit_streak_trips_breaker(conn, kanban_home):
    """Three consecutive clean-exit violations close the card as blocked via
    the same violation-only budget the gateway dispatcher uses."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="breaker", assignee="w")
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "Run\nResume this session with:\n  hermes --resume s\n", encoding="utf-8"
    )

    for i in range(kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
        pid = _dead_worker_row(conn, host, tid)
        kbd._recent_worker_exits.pop(pid, None)
        kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        if i < kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT - 1:
            assert task.status == "ready", (
                f"run {i}: still within budget, got {task.status}"
            )

    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.consecutive_failures >= 1


def test_fresh_process_no_marker_stays_plain_crash(conn, kanban_home):
    """Registry miss + no exit summary (genuine crash, truncated log) keeps
    the plain ``crashed`` booking — the log probe must not invent violations."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="no-marker", assignee="w")
    pid = _dead_worker_row(conn, host, tid)
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Traceback (most recent call last):\n  boom\n", encoding="utf-8")
    kbd._recent_worker_exits.pop(pid, None)

    crashed = kbd.detect_crashed_workers(conn)

    assert tid in crashed
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["outcome"] == "crashed"
    assert "protocol violation" not in run["error"]
    meta = kb._json_dict(run["metadata"])
    assert not meta.get("protocol_violation")
    assert kbd._protocol_violation_streak(conn, tid) == 0


def test_registry_hit_still_wins_over_missing_log(conn, kanban_home):
    """A nonzero reaped exit keeps its honest classification even when the log
    happens to be gone — the durable probe only upgrades ``unknown``."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reaped", assignee="w")
    pid = _dead_worker_row(conn, host, tid)
    kbd._record_worker_exit(pid, 1 << 8)  # nonzero exit → plain crash

    crashed = kbd.detect_crashed_workers(conn)

    assert tid in crashed
    run = conn.execute(
        "SELECT outcome, error FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert "exited with code 1" in run["error"]
    assert "protocol violation" not in run["error"]
