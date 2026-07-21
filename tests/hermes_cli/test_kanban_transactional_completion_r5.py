"""Adversarial regressions for transactional completion/reclaim generation fences."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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


def _claim_snapshot(conn, task_id: str, *, claimer: str | None = None):
    claimed = kb.claim_task(conn, task_id, claimer=claimer)
    assert claimed is not None
    assert claimed.current_run_id is not None
    assert claimed.claim_lock
    return claimed


def _bind_worker(conn, task_id: str, pid: int) -> None:
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    assert task.claim_lock
    assert kb._set_worker_pid(
        conn,
        task_id,
        pid,
        expected_run_id=task.current_run_id,
        expected_claim_lock=task.claim_lock,
    )


def test_expired_reclaim_reservation_recovers_and_closes_exact_run(
    kanban_home,
    monkeypatch,
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="orphaned reservation", assignee="worker")
        host = kb._claimer_id().split(":", 1)[0]
        claimed = _claim_snapshot(conn, task_id, claimer=f"{host}:worker")
        run_id = claimed.current_run_id
        assert run_id is not None
        _bind_worker(conn, task_id, 424242)
        task = kb.get_task(conn, task_id)
        assert task is not None

        reservation = kb._reserve_reclaim_generation(
            conn,
            task_id,
            expected_run_id=run_id,
            expected_claim_lock=task.claim_lock,
            expected_worker_pid=task.worker_pid,
            expected_claim_expires=task.claim_expires,
            expected_last_heartbeat_at=task.last_heartbeat_at,
        )
        assert reservation is not None
        conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=?",
            (int(time.time()) - 1, task_id),
        )
        conn.commit()

        assert kb.release_stale_claims(conn) == 1
        recovered = kb.get_task(conn, task_id)
        assert recovered is not None
        assert recovered.status == "ready"
        assert recovered.current_run_id is None
        run = kb.get_run(conn, reservation.run_id)
        assert run is not None
        assert run.ended_at is not None
        assert run.outcome == "reclaimed"
        kinds = [event.kind for event in kb.list_events(conn, task_id)]
        assert "reclaim_reserved" in kinds
        assert "reclaimed" in kinds


def test_reclaim_reservation_fences_completion_ownership(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="completion fence", assignee="worker")
        claimed = _claim_snapshot(conn, task_id)
        run_id = claimed.current_run_id
        assert run_id is not None
        task = kb.get_task(conn, task_id)
        assert task is not None
        reservation = kb._reserve_reclaim_generation(
            conn,
            task_id,
            expected_run_id=run_id,
            expected_claim_lock=task.claim_lock,
            expected_worker_pid=task.worker_pid,
            expected_claim_expires=task.claim_expires,
            expected_last_heartbeat_at=task.last_heartbeat_at,
        )
        assert reservation is not None

        assert not kb.complete_task(
            conn,
            task_id,
            summary="stale worker completion",
            expected_run_id=run_id,
        )
        current = kb.get_task(conn, task_id)
        assert current is not None
        assert current.status == "running"
        assert current.current_run_id == run_id
        assert current.claim_lock == reservation.marker


def test_termination_without_birth_identity_never_signals_pid(
    kanban_home,
    monkeypatch,
):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(kb, "_process_identity", lambda _pid: "current-birth")
    host = kb._claimer_id().split(":", 1)[0]

    result = kb._terminate_reclaimed_worker(
        os.getpid(),
        f"{host}:worker",
        worker_birth_identity=None,
        signal_fn=lambda pid, sig: sent.append((pid, sig)),
    )

    assert sent == []
    assert result["termination_attempted"] is False
    assert result["termination_error"] == "missing_process_identity"
    assert result["worker_alive"] is True


def test_legacy_spawn_callbacks_cannot_target_new_retry(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="spawn callback fence", assignee="worker")
        stable_lock = "dispatcher:stable"
        first = _claim_snapshot(conn, task_id, claimer=stable_lock)
        assert kb.reclaim_task(conn, task_id, expected_run_id=first.current_run_id)
        second = _claim_snapshot(conn, task_id, claimer=stable_lock)
        assert second.current_run_id != first.current_run_id
        second_run_id = second.current_run_id
        assert second_run_id is not None

        assert kb._record_spawn_failure(conn, task_id, "late failure") is False
        assert kb._set_worker_pid(conn, task_id, 333333) is False

        current = kb.get_task(conn, task_id)
        assert current is not None
        assert current.status == "running"
        assert current.current_run_id == second_run_id
        assert current.worker_pid is None
        second_run = kb.get_run(conn, second_run_id)
        assert second_run is not None
        assert second_run.ended_at is None
        assert second_run.worker_pid is None


def test_blank_archive_payload_is_invalid_completion_evidence(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="archive evidence", assignee="worker")
        assert kb.complete_task(conn, task_id, summary="valid manual completion")
        assert kb.archive_task(conn, task_id)
        conn.execute(
            "UPDATE task_events SET payload='' WHERE id=("
            "SELECT id FROM task_events WHERE task_id=? AND kind='archived' "
            "ORDER BY id DESC LIMIT 1)",
            (task_id,),
        )
        conn.commit()

        valid, reason = kb.completion_validity(conn, task_id)
        assert valid is False
        assert "blank or malformed" in reason
