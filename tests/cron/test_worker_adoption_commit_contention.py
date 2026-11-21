"""An adoption rejected at COMMIT must be retried before any job side effect (#135487)."""

import json
import sqlite3
import time
from unittest.mock import Mock


def test_worker_retries_busy_commit_before_ack_and_runs_once(tmp_path, monkeypatch):
    from cron import executions, scheduler
    from hermes_cli.sqlite_util import open_db

    db = tmp_path / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db)

    def connection(initialize=None):
        # A rollback-journal reader permits UPDATE but prevents its COMMIT.
        # This tests the supported DELETE fallback, not WAL reader behavior.
        return open_db(db, db_label="test cron ledger", wal=False,
                       busy_timeout_ms=0, initialize=initialize)

    monkeypatch.setattr(executions, "_connect",
                        lambda: connection(executions._initialize_schema))
    record = executions.create_execution("commit-busy-job", source="builtin")
    assert executions.mark_execution_handoff_pending(record["id"])
    # The schema is already initialized: contention must occur during adoption.
    monkeypatch.setattr(executions, "_connect", connection)
    reader = sqlite3.connect(db)
    assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    reader.execute("BEGIN")
    assert reader.execute("SELECT status FROM executions WHERE id=?",
                          (record["id"],)).fetchone()[0] == "claimed"

    payload, ack = tmp_path / "payload.json", tmp_path / "ready.json"
    payload.write_text(json.dumps({
        "job": {"id": "commit-busy-job", "execution_id": record["id"]},
        "profile_home": str(tmp_path / "profile"),
        "adoption_deadline": time.monotonic() + 120,
    }), encoding="utf-8")
    real_fetch = executions._fetch
    uncommitted_states = []

    def fetch(conn, execution_id):
        row = real_fetch(conn, execution_id)
        uncommitted_states.append(row["status"])
        return row

    monkeypatch.setattr(executions, "_fetch", fetch)
    real_adopt = executions.adopt_claimed_execution
    busy_errors = []

    def adopt(execution_id):
        try:
            return real_adopt(execution_id)
        except sqlite3.OperationalError as exc:
            busy_errors.append(exc.sqlite_errorcode)
            # UPDATE and SELECT succeeded in the writer, but COMMIT rolled back.
            assert uncommitted_states == ["running"]
            assert not ack.exists()
            run.assert_not_called()
            assert reader.execute("SELECT status, handoff_pending FROM executions WHERE id=?",
                                  (execution_id,)).fetchone() == ("claimed", 1)
            reader.rollback()  # release only after the real failed commit
            raise

    monkeypatch.setattr(executions, "adopt_claimed_execution", adopt)
    def run_after_commit(*args, **kwargs):
        assert ack.exists()
        assert reader.execute("SELECT status FROM executions WHERE id=?",
                              (record["id"],)).fetchone()[0] == "running"
        return True

    run = Mock(side_effect=run_after_commit)
    monkeypatch.setattr(scheduler, "run_one_job", run)
    try:
        try:
            result = scheduler._run_external_worker_payload(payload, ack)
        except sqlite3.OperationalError:
            result = False
    finally:
        reader.close()

    assert busy_errors == [sqlite3.SQLITE_BUSY]
    assert result is True, "worker abandoned occurrence after a recoverable commit contention"
    assert uncommitted_states == ["running", "running"]
    run.assert_called_once()
    assert json.loads(ack.read_text(encoding="utf-8"))["execution_id"] == record["id"]
    assert executions.get_execution(record["id"])["status"] == "running"
    assert real_adopt(record["id"]) is None
