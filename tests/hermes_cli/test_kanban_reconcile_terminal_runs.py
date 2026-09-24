from __future__ import annotations

import argparse
import json
import sqlite3

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    return home


def _legacy_open_terminal_run(conn, *, terminal_status: str = "done", worker_pid: int | None = None):
    task_id = kb.create_task(conn, title=f"legacy {terminal_status}", assignee="worker")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL WHERE id = ?",
            (terminal_status, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (worker_pid, run_id),
        )
    return task_id, run_id


def _local_host() -> str:
    return kb._claimer_id().rsplit(":", 1)[0]


def test_reconcile_terminal_runs_dry_run_is_read_only(board):
    with kbc.connect() as conn:
        task_id, run_id = _legacy_open_terminal_run(conn, worker_pid=876543)
        report = kb.reconcile_terminal_runs(
            conn, dry_run=True, reason="audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        assert report == {
            "dry_run": True,
            "eligible": [{"run_id": run_id, "task_id": task_id, "task_status": "done", "worker_pid": 876543}],
            "reconciled": [],
            "skipped_live": [],
            "skipped_unverifiable": [],
            "skipped_current": [],
        }
        row = conn.execute("SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert tuple(row) == ("running", None, None)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'run_ledger_reconciled'", (task_id,)
        ).fetchone()[0] == 0


def test_reconcile_terminal_runs_closes_only_dead_detached_terminal_runs(board):
    with kbc.connect() as conn:
        done_id, done_run = _legacy_open_terminal_run(conn, terminal_status="done", worker_pid=111)
        archived_id, archived_run = _legacy_open_terminal_run(conn, terminal_status="archived", worker_pid=None)
        live_id, live_run = _legacy_open_terminal_run(conn, terminal_status="done", worker_pid=222)

        active_id = kb.create_task(conn, title="active", assignee="worker")
        active = kb.claim_task(conn, active_id)
        assert active is not None and active.current_run_id is not None

        # Malformed terminal/current linkage must fail closed rather than rewrite the active run.
        current_terminal_id = kb.create_task(conn, title="terminal but current", assignee="worker")
        current_terminal = kb.claim_task(conn, current_terminal_id)
        assert current_terminal is not None and current_terminal.current_run_id is not None
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (current_terminal_id,))

        report = kb.reconcile_terminal_runs(
            conn,
            reason="close historical ledger",
            trusted_claim_host=_local_host(),
            worker_alive_fn=lambda pid, _started: pid == 222,
            now=4321,
        )

        assert [entry["run_id"] for entry in report["reconciled"]] == [done_run, archived_run]
        assert report["skipped_live"] == [
            {"run_id": live_run, "task_id": live_id, "task_status": "done", "worker_pid": 222}
        ]
        assert report["skipped_current"] == [{
            "run_id": current_terminal.current_run_id,
            "task_id": current_terminal_id,
            "task_status": "done",
            "worker_pid": None,
        }]

        for task_id, run_id, task_status in (
            (done_id, done_run, "done"), (archived_id, archived_run, "archived"),
        ):
            row = conn.execute(
                "SELECT status, outcome, ended_at, claim_expires, error, metadata FROM task_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            assert tuple(row[:4]) == ("reconciled", "ledger_reconciled", 4321, None)
            assert row["error"] == "historical_terminal_run_reconciled: close historical ledger"
            assert json.loads(row["metadata"]) == {
                "reason": "close historical ledger",
                "task_status": task_status,
            }
            event = conn.execute(
                "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = 'run_ledger_reconciled'",
                (task_id,),
            ).fetchone()
            assert event["run_id"] == run_id
            assert json.loads(event["payload"]) == {
                "reason": "close historical ledger",
                "run_id": run_id,
                "task_status": task_status,
            }
            task = conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
            assert tuple(task) == (task_status, None)

        assert conn.execute("SELECT status FROM task_runs WHERE id = ?", (live_run,)).fetchone()[0] == "running"
        assert conn.execute(
            "SELECT status FROM task_runs WHERE id = ?", (active.current_run_id,)
        ).fetchone()[0] == "running"
        assert conn.execute(
            "SELECT status FROM task_runs WHERE id = ?", (current_terminal.current_run_id,)
        ).fetchone()[0] == "running"


def test_reconcile_terminal_runs_cli_json_and_dry_run(board, capsys, monkeypatch):
    with kbc.connect() as conn:
        task_id, run_id = _legacy_open_terminal_run(conn)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban", "reconcile-runs", "--dry-run", "--json", "--reason", "historical cleanup",
        "--claim-host", _local_host(),
    ])
    assert kc.kanban_command(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["eligible"] == [
        {"run_id": run_id, "task_id": task_id, "task_status": "done", "worker_pid": None}
    ]


def test_reconcile_terminal_runs_fails_closed_for_foreign_and_unknown_workers(board):
    with kbc.connect() as conn:
        foreign_id, foreign_run = _legacy_open_terminal_run(conn, worker_pid=701)
        unknown_id, unknown_run = _legacy_open_terminal_run(conn, worker_pid=702)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET claim_lock = 'foreign-host:7', worker_started_at = 'epoch|7' WHERE id = ?",
                (foreign_run,),
            )
            conn.execute(
                "UPDATE task_runs SET claim_lock = ?, worker_started_at = 'epoch|8' WHERE id = ?",
                (kb._claimer_id(), unknown_run),
            )

        report = kb.reconcile_terminal_runs(
            conn, reason="audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: None, now=1234,
        )
        assert report["eligible"] == []
        assert [row["run_id"] for row in report["skipped_unverifiable"]] == [foreign_run, unknown_run]
        assert conn.execute("SELECT status FROM task_runs WHERE id = ?", (foreign_run,)).fetchone()[0] == "running"
        assert conn.execute("SELECT status FROM task_runs WHERE id = ?", (unknown_run,)).fetchone()[0] == "running"
        assert {foreign_id, unknown_id} == {
            row["task_id"] for row in report["skipped_unverifiable"]
        }


def test_reconcile_terminal_runs_preserves_binary_error_and_metadata(board):
    with kbc.connect() as conn:
        task_id, run_id = _legacy_open_terminal_run(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET error = ?, metadata = ? WHERE id = ?",
                (sqlite3.Binary(b"bad-error-\xff"), sqlite3.Binary(b"bad-json-\xfe"), run_id),
            )
        report = kb.reconcile_terminal_runs(
            conn, reason="binary audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        assert [row["run_id"] for row in report["reconciled"]] == [run_id]
        metadata = json.loads(conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()[0])
        assert metadata["prior_error_raw"] == {
            "storage_type": "blob", "encoding": "base64", "data": "YmFkLWVycm9yLf8=",
        }
        assert metadata["prior_metadata_raw"] == {
            "storage_type": "blob", "encoding": "base64", "data": "YmFkLWpzb24t/g==",
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'run_ledger_reconciled'", (task_id,),
        ).fetchone()[0] == 1


def test_reconcile_terminal_runs_rolls_back_run_when_audit_insert_fails(board):
    with kbc.connect() as conn:
        _task_id, run_id = _legacy_open_terminal_run(conn)
        conn.execute(
            "CREATE TRIGGER reject_reconciliation_event BEFORE INSERT ON task_events "
            "WHEN NEW.kind = 'run_ledger_reconciled' BEGIN SELECT RAISE(ABORT, 'audit rejected'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="audit rejected"):
            kb.reconcile_terminal_runs(
                conn, reason="rollback audit", trusted_claim_host=_local_host(),
                worker_alive_fn=lambda _pid, _started: False, now=1234,
            )
        row = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()
        assert tuple(row) == ("running", None, None)


def test_reconcile_runs_cli_dry_run_skips_init_and_legacy_backfill(board, capsys, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="active legacy pointer", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        original_run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,),
        ).fetchone()[0]
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,))

    monkeypatch.setattr(kb, "init_db", lambda: pytest.fail("dry-run must not initialize or migrate"))
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban", "reconcile-runs", "--dry-run", "--json", "--reason", "read-only audit",
        "--claim-host", _local_host(),
    ])
    assert kc.kanban_command(args) == 0
    json.loads(capsys.readouterr().out)

    # Open without Hermes initialization so this assertion cannot itself backfill.
    db_path = kb.kanban_db_path()
    raw = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        assert raw.execute("SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] is None
        assert raw.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0] == original_run_count
    finally:
        raw.close()


def test_reconcile_runs_denied_in_delegated_child(board, monkeypatch):
    from agent import delegation_context
    monkeypatch.setattr(delegation_context, "kanban_path_is_fenced", lambda _path: True)
    with kbc.connect() as conn:
        with pytest.raises(PermissionError):
            kb.reconcile_terminal_runs(conn, dry_run=True)


def test_worker_liveness_holds_live_pid_when_fingerprint_probe_is_unknown(monkeypatch):
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: None)
    assert kbd._worker_liveness(123, "epoch|123") is None
    assert kbd._worker_alive(123, "epoch|123") is True


def test_readonly_connector_rejects_writes(board):
    with kbc.connect_readonly_closing() as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE tasks SET title = title")


def test_reconcile_terminal_runs_rejects_pidless_foreign_claim(board):
    with kbc.connect() as conn:
        _task_id, run_id = _legacy_open_terminal_run(conn, worker_pid=None)
        conn.execute("UPDATE task_runs SET claim_lock = 'foreign-host:9' WHERE id = ?", (run_id,))
        report = kb.reconcile_terminal_runs(
            conn, reason="foreign audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        assert report["eligible"] == []
        assert [row["run_id"] for row in report["skipped_unverifiable"]] == [run_id]


def test_reconcile_terminal_runs_defaults_claim_host_to_local(board):
    with kbc.connect() as conn:
        _task_id, run_id = _legacy_open_terminal_run(conn)
        report = kb.reconcile_terminal_runs(
            conn, reason="implicit local host", worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        assert [row["run_id"] for row in report["reconciled"]] == [run_id]


def test_reconcile_terminal_runs_preserves_invalid_and_empty_storage_bytes(board):
    with kbc.connect() as conn:
        _task1, run1 = _legacy_open_terminal_run(conn)
        _task2, run2 = _legacy_open_terminal_run(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET error=CAST(X'FF' AS TEXT), metadata=CAST(X'FE' AS TEXT) WHERE id=?",
                (run1,),
            )
            conn.execute(
                "UPDATE task_runs SET error=?, metadata=? WHERE id=?",
                (sqlite3.Binary(b""), sqlite3.Binary(b""), run2),
            )
        kb.reconcile_terminal_runs(
            conn, reason="raw audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        one = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run1,)).fetchone()[0])
        two = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run2,)).fetchone()[0])
        assert one["prior_error_raw"] == {"storage_type": "text", "encoding": "base64", "data": "/w=="}
        assert one["prior_metadata_raw"] == {"storage_type": "text", "encoding": "base64", "data": "/g=="}
        assert two["prior_error_raw"] == {"storage_type": "blob", "encoding": "base64", "data": ""}
        assert two["prior_metadata_raw"] == {"storage_type": "blob", "encoding": "base64", "data": ""}


def test_pid_and_fingerprint_probe_uncertainty_never_proves_death(monkeypatch):
    import subprocess
    monkeypatch.setattr(kbd.sys, "platform", "darwin")
    monkeypatch.setattr(kbd, "_primary_pid_liveness", lambda _pid: True)
    monkeypatch.setattr(
        kbd.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 9, stdout="", stderr="probe failed"),
    )
    assert kbd._pid_liveness(123) is None
    assert kbd._worker_liveness(123, "epoch|100") is None

    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot-time:123|250")
    assert kbd._worker_liveness(123, "boot-time:123|100") is True
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot-time:456|100")
    assert kbd._worker_liveness(123, "boot-time:123|100") is False


def test_reconcile_terminal_runs_rejects_missing_claim_provenance(board):
    with kbc.connect() as conn:
        _task_id, run_id = _legacy_open_terminal_run(conn, worker_pid=123)
        conn.execute("UPDATE task_runs SET claim_lock = NULL WHERE id = ?", (run_id,))
        report = kb.reconcile_terminal_runs(
            conn, reason="missing provenance", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        assert report["eligible"] == []
        assert [row["run_id"] for row in report["skipped_unverifiable"]] == [run_id]


def test_reconcile_terminal_runs_preserves_valid_json_text_byte_exactly(board):
    with kbc.connect() as conn:
        _task_id, run_id = _legacy_open_terminal_run(conn)
        raw = b'{"first":1, "first":2, "number":1.00}'
        conn.execute("UPDATE task_runs SET metadata=CAST(? AS TEXT) WHERE id=?", (raw, run_id))
        kb.reconcile_terminal_runs(
            conn, reason="lexical audit", trusted_claim_host=_local_host(),
            worker_alive_fn=lambda _pid, _started: False, now=1234,
        )
        metadata = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()[0])
        assert metadata["prior_metadata_raw"] == {
            "storage_type": "text", "encoding": "base64",
            "data": "eyJmaXJzdCI6MSwgImZpcnN0IjoyLCAibnVtYmVyIjoxLjAwfQ==",
        }


def test_primary_pid_probe_errors_are_unknown(monkeypatch):
    monkeypatch.setattr(kbd, "_primary_pid_liveness", lambda _pid: None)
    assert kbd._pid_liveness(123) is None


def test_apply_cli_skips_init_and_backfill(board, capsys, monkeypatch):
    with kbc.connect() as conn:
        task_id, _run_id = _legacy_open_terminal_run(conn)
    monkeypatch.setattr(kb, "init_db", lambda: pytest.fail("apply must not initialize or migrate"))
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban", "reconcile-runs", "--json", "--reason", "apply without migration",
        "--claim-host", _local_host(),
    ])
    assert kc.kanban_command(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reconciled"]
    with kbc.connect() as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == "done"


def test_apply_connector_refuses_missing_database(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        with kbc.connect_existing_closing(missing):
            pass
    assert not missing.exists()


def test_unverified_fingerprint_never_authorizes_signal(monkeypatch):
    signals = []
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    info = kbd._terminate_reclaimed_worker(
        123, f"{_local_host()}:worker", started_at=kbd.UNVERIFIED_WORKER_FINGERPRINT,
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )
    assert signals == []
    assert info["signal_refused"] is True
    assert info["terminated"] is False


def test_termination_unknown_after_sigterm_refuses_sigkill(monkeypatch):
    signals = []
    states = iter([True] + [True] * 10 + [None])
    monkeypatch.setattr(kbd, "_worker_liveness", lambda _pid, _started: next(states))
    monkeypatch.setattr(kbd.time, "sleep", lambda _seconds: None)
    info = kbd._terminate_reclaimed_worker(
        123, f"{_local_host()}:worker", started_at="epoch|100",
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )
    assert len(signals) == 1
    assert info["signal_refused"] is True
    assert info["sigkill"] is False
    assert info["terminated"] is False


def test_process_fingerprint_requires_complete_epoch(monkeypatch):
    monkeypatch.setattr(kbd, "_worker_instantiation_epoch", lambda _epoch: None)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda _pid: 123)
    assert kbd._process_fingerprint(99) is None


def test_max_runtime_failed_termination_keeps_claim(board, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="live timeout", assignee="worker", max_runtime_seconds=1)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        old = 1
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid=123, worker_started_at='epoch|100', started_at=? WHERE id=?",
                (old, task_id),
            )
            conn.execute("UPDATE task_runs SET started_at=? WHERE id=?", (old, claimed.current_run_id))
        monkeypatch.setattr(kbd, "_worker_liveness", lambda _pid, _started: True)

        def denied(_pid, _signal):
            raise PermissionError("signal denied")

        assert kbd.enforce_max_runtime(conn, signal_fn=denied) == []
        task = kb.get_task(conn, task_id)
        assert task.status == "running" and task.worker_pid == 123 and task.claim_lock is not None


def test_legacy_null_fingerprint_never_authorizes_signal(monkeypatch):
    signals = []
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    info = kbd._terminate_reclaimed_worker(
        123, f"{_local_host()}:worker", started_at=None,
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )
    assert signals == []
    assert info["signal_refused"] is True
    assert info["terminated"] is False


def test_incomplete_recorded_epoch_is_unknown(monkeypatch):
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot-time:123|500")
    assert kbd._worker_liveness(123, "|500") is None


def test_boot_time_epoch_drift_is_same_worker(monkeypatch):
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot-time:10150|500")
    assert kbd._worker_liveness(123, "boot-time:10000|500") is True


def test_legacy_start_only_fingerprint_is_unknown(monkeypatch):
    monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
    assert kbd._worker_liveness(123, 500) is None


def test_claim_host_attestation_must_match_local_host(board):
    with kbc.connect() as conn:
        _task_id, _run_id = _legacy_open_terminal_run(conn)
        with pytest.raises(ValueError, match="must equal this host"):
            kb.reconcile_terminal_runs(
                conn, dry_run=True, reason="foreign attestation",
                trusted_claim_host="another-host", worker_alive_fn=lambda *_: False,
            )


def test_manual_reclaim_holds_claim_when_worker_not_terminated(board, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="manual live", assignee="worker")
        kb.claim_task(conn, task_id)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid=123, worker_started_at='unverified' WHERE id=?",
                (task_id,),
            )
        monkeypatch.setattr(kbd, "_pid_liveness", lambda _pid: True)
        assert kb.reclaim_task(conn, task_id, reason="operator") is False
        task = kb.get_task(conn, task_id)
        assert task.status == "running" and task.worker_pid == 123 and task.claim_lock is not None
