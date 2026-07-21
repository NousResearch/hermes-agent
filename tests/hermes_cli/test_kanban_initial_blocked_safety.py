"""Regression tests for Kanban initial blocked safety.

Covers the Control Room 2026-05-30 smoke where a card created with
``initial_status='blocked'`` was auto-promoted, claimed, and spawned.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def _event_payloads(conn, task_id: str, kind: str) -> list[dict]:
    payloads: list[dict] = []
    for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall():
        payloads.append(json.loads(row["payload"] or "{}"))
    return payloads


def test_initial_status_blocked_without_parents_is_sticky_and_not_promoted(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked smoke should remain inert",
            assignee="kanban-coordenador",
            initial_status="blocked",
        )

        assert kb.get_task(conn, tid).status == "blocked"
        assert "blocked" in _event_kinds(conn, tid)

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"

        events = _event_kinds(conn, tid)
        assert "promoted" not in events
        assert "claimed" not in events
        assert "spawned" not in events
        assert kb.claim_task(conn, tid) is None


def test_initial_status_blocked_with_done_parent_waits_for_explicit_unblock(kanban_home: Path) -> None:
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(
            conn,
            title="manual blocked child",
            parents=[parent],
            initial_status="blocked",
        )
        kb.complete_task(conn, parent, result="parent done")

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"
        assert "promoted" not in _event_kinds(conn, child)

        assert kb.unblock_task(conn, child)
        assert kb.get_task(conn, child).status == "ready"


def test_initial_status_blocked_explicit_unblock_releases_parentless_task(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual block", initial_status="blocked")
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid)

        assert kb.get_task(conn, tid).status == "ready"
        events = _event_kinds(conn, tid)
        assert "unblocked" in events
        assert "promoted" not in events  # unblock releases directly to ready


def test_circuit_breaker_gave_up_without_blocked_event_still_auto_recovers(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="circuit breaker recovery")
        task = kb.claim_task(conn, tid)
        assert task is not None

        assert kb._record_spawn_failure(conn, tid, "boom", failure_limit=1)
        assert kb.get_task(conn, tid).status == "blocked"
        assert "gave_up" in _event_kinds(conn, tid)
        assert "blocked" not in _event_kinds(conn, tid)

        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"


def test_archive_running_task_records_archived_atomically_then_worker_terminated(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_terminate(pid, claim_lock, **kwargs):
        calls.append((pid, claim_lock, kwargs))
        return {
            "prev_pid": pid,
            "prev_claim_lock": claim_lock,
            "host_local": True,
            "pid_is_child": True,
            "termination_attempted": True,
            "terminated": True,
            "reaped": True,
            "sigkill": False,
        }

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="archive active worker")
        task = kb.claim_task(conn, tid, claimer="host:test-claim")
        assert task is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (424242, tid),
        )
        conn.commit()

        assert kb.archive_task(conn, tid)

        assert calls == [(424242, "host:test-claim", {})]
        archived = kb.get_task(conn, tid)
        assert archived.status == "archived"
        assert archived.worker_pid is None
        assert archived.claim_lock is None

        events = _event_kinds(conn, tid)
        assert events.index("archived") < events.index("worker_terminated")
        archived_payload = _event_payloads(conn, tid, "archived")[-1]
        assert archived_payload["worker_termination"]["pending"] is True
        terminated_payload = _event_payloads(conn, tid, "worker_terminated")[-1]
        assert terminated_payload["worker_termination"]["terminated"] is True


def test_archive_running_real_child_terminates_and_reaps(kanban_home: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX waitpid/reap test")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="archive real child")
            task = kb.claim_task(conn, tid, claimer=kb._claimer_id())
            assert task is not None
            conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (proc.pid, tid))
            conn.commit()

            assert kb.archive_task(conn, tid)

            info = _event_payloads(conn, tid, "worker_terminated")[-1]["worker_termination"]
            assert info["termination_attempted"] is True
            assert info["terminated"] is True
            assert info["reaped"] is True
            assert proc.poll() is not None
            with pytest.raises(ChildProcessError):
                os.waitpid(proc.pid, os.WNOHANG)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_terminate_reclaimed_worker_refuses_stale_or_recycled_non_child_pid() -> None:
    if os.name == "nt":
        pytest.skip("POSIX waitpid/reap test")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        claim_lock = kb._claimer_id()
        # Reap first so the PID is no longer our child. This simulates the
        # safety-relevant stale/recycled case: the stored pid is not a live
        # child owned by this claim anymore, so the terminator must not signal.
        proc.terminate()
        proc.wait(timeout=5)
        info = kb._terminate_reclaimed_worker(proc.pid, claim_lock)
        assert info["termination_attempted"] is False
        assert info["terminated"] is False
        assert info.get("refused_reason") == "pid_not_child_or_already_reaped"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_archive_with_no_worker_pid_is_noop_and_double_archive_does_not_kill(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_terminate(pid, claim_lock, **kwargs):
        calls.append((pid, claim_lock, kwargs))
        return {"prev_pid": pid, "terminated": False}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="archive no pid")
        assert kb.archive_task(conn, tid)
        assert calls == [(None, None, {})]
        archived_payload = _event_payloads(conn, tid, "archived")[-1]
        assert archived_payload["worker_termination"]["pending"] is False

        assert kb.archive_task(conn, tid) is False
        assert calls == [(None, None, {})]
        assert len(_event_payloads(conn, tid, "archived")) == 1
