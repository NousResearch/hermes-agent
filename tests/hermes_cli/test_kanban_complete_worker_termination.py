"""Invariant: an EXTERNAL close of a running card ends the owning executor
synchronously (2026-11-05 zombie-writer incident, fleet card t_2da8548a).

``complete_task`` used to be pure bookkeeping: a force/supersede close from an
orchestrator or the dashboard flipped the card to ``done`` while the headless
executor kept running against a workspace nothing tracked anymore — two such
zombies wiped a live sibling's untracked helper mid-incident. The close now
SIGTERMs the displaced worker (TERM→KILL rails in ``_terminate_reclaimed_worker``)
and records a ``complete_worker_termination`` audit event, sharing one helper
with ``archive_task`` (#76196).

Never signalled: a self-completion. ``expected_run_id`` is proof of ownership
(the caller IS that run), and the own-worker guard (pid / ``HERMES_KANBAN_TASK``)
catches the in-process tool and the CLI-child paths — a worker must not be killed
inside its own post-completion finalisation. A worker that hangs AFTER its own
completion stays the deferred reaper's job (#111791).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_termination import close_initiated_by_own_worker
from tests.waiting import wait_until


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Event-based startup sync (HX013): the spawn fingerprint must be readable
    # before _set_worker_pid snapshots it.
    wait_until(lambda: kb._pid_alive(proc.pid), timeout=10, what="sleeper startup")
    return proc


def _running_card_with_worker(conn, proc, *, claimer=None):
    tid = kb.create_task(conn, title="in-flight", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=claimer or kb._claimer_id()) is not None
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    return tid, run_id


def _termination_event(conn, tid):
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'complete_worker_termination'", (tid,),
    ).fetchone()
    return json.loads(row["payload"]) if row else None


def test_force_close_of_running_card_terminates_the_live_worker(conn):
    """The incident path: an operator/orchestrator force-closes a running card;
    the executor must actually die, not just lose its DB row."""
    proc = _sleeper()
    try:
        tid, _run_id = _running_card_with_worker(conn, proc)
        assert kb.complete_task(conn, tid, result="closed externally", force=True) is True
        assert proc.wait(timeout=15) == -signal.SIGTERM
        payload = _termination_event(conn, tid)
        assert payload is not None
        assert payload["prev_pid"] == proc.pid
        assert payload["host_local"] is True
        assert payload["termination_attempted"] is True
        assert payload["terminated"] is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_self_completion_with_run_ownership_never_signals_the_worker(conn):
    """``expected_run_id`` is proof of ownership — a self-completion. Killing it
    would cut the worker's own finalisation; a hang afterwards belongs to the
    deferred reaper (#111791), which reads the pid evidence off the closed run row."""
    proc = _sleeper()
    try:
        tid, run_id = _running_card_with_worker(conn, proc)
        assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
        assert proc.poll() is None, "self-completion must not signal the worker"
        assert _termination_event(conn, tid) is None
        run = conn.execute(
            "SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["worker_pid"] == proc.pid, "reaper evidence must survive the close"
    finally:
        proc.kill()
        proc.wait()


def test_close_by_own_worker_env_is_never_signalled(conn, monkeypatch):
    """A worker that force-closes its OWN card (``HERMES_KANBAN_TASK`` rides the
    whole process tree, so the CLI-child path is covered too) keeps running its
    finalisation."""
    proc = _sleeper()
    try:
        tid, _run_id = _running_card_with_worker(conn, proc)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        assert kb.complete_task(conn, tid, result="closed by owner", force=True) is True
        assert proc.poll() is None
        assert _termination_event(conn, tid) is None
    finally:
        proc.kill()
        proc.wait()


def test_cross_host_close_records_audit_without_signalling(conn):
    """Only the worker's own host may kill it; a foreign claim is audited and the
    signalling is left to that host's dispatcher sweep."""
    proc = _sleeper()
    try:
        tid, _run_id = _running_card_with_worker(conn, proc, claimer="other-host:4242")
        assert kb.complete_task(conn, tid, result="closed externally", force=True) is True
        assert proc.poll() is None
        payload = _termination_event(conn, tid)
        assert payload is not None
        assert payload["host_local"] is False
        assert payload["termination_attempted"] is False
    finally:
        proc.kill()
        proc.wait()


def test_close_without_proof_of_a_dead_worker_audits_without_kill(conn, monkeypatch):
    """A no-proof close of a running card whose worker is already gone passes the
    live-claim fence and records the (already dead) termination — the audit trail
    exists even when there was nothing to signal."""
    tid = kb.create_task(conn, title="abandoned", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    kbd._set_worker_pid(conn, tid, 54321)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    signalled = []
    assert kb.complete_task(
        conn, tid, result="closed", signal_fn=lambda pid, sig: signalled.append((pid, sig)),
    ) is True
    assert signalled == []
    payload = _termination_event(conn, tid)
    assert payload is not None and payload["prev_pid"] == 54321


def test_own_worker_guard_recognises_pid_and_env(monkeypatch):
    import os
    tid = "t_deadbeef00"
    assert close_initiated_by_own_worker(tid, os.getpid()) is True
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert close_initiated_by_own_worker(tid, os.getpid() + 100000) is False
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert close_initiated_by_own_worker(tid, None) is True
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_other")
    assert close_initiated_by_own_worker(tid, None) is False
