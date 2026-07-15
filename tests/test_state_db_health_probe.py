"""Bounded health probing for the SQLite session store."""

import argparse
import sqlite3

import pytest

import hermes_state


def _build_minimal_state_db(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE sessions ("
            "id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()


def test_routine_probe_installs_and_clears_progress_handler(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)

    events = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def set_progress_handler(self, progress_handler, n):
            events.append(("progress", progress_handler, n))
            return super().set_progress_handler(progress_handler, n)

        def close(self):
            events.append(("close",))
            return super().close()

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, factory=TrackingConnection, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracking_connect)

    assert hermes_state._db_opens_cleanly(db_path) is None
    assert [event[0] for event in events] == ["progress", "progress", "close"]
    assert callable(events[0][1]) and events[0][2] > 0
    assert events[1] == ("progress", None, 0)


def test_simulated_slow_integrity_check_is_cancelled_without_waiting(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    phase = {"sql": None}
    real_connect = sqlite3.connect

    class SlowIntegrityConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            phase["sql"] = sql
            return super().execute(sql, parameters)

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, factory=SlowIntegrityConnection, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracking_connect)

    def clock():
        return 106.0 if phase["sql"] == "PRAGMA integrity_check" else 100.0

    result = hermes_state._probe_db_health(
        db_path,
        budget_seconds=5.0,
        _clock=clock,
        _progress_ops=1,
    )

    assert result.status == "skipped"
    assert result.category == "timeout"
    assert "5" in result.reason


def test_timeout_clears_handler_before_closing_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)

    events = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def set_progress_handler(self, progress_handler, n):
            events.append(("progress", progress_handler, n))
            return super().set_progress_handler(progress_handler, n)

        def close(self):
            events.append(("close",))
            return super().close()

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, factory=TrackingConnection, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracking_connect)
    ticks = iter((0.0, 6.0))

    result = hermes_state._probe_db_health(
        db_path,
        budget_seconds=5.0,
        _clock=lambda: next(ticks, 6.0),
        _progress_ops=1,
    )

    assert result.status == "skipped"
    assert [event[0] for event in events] == ["progress", "progress", "close"]
    assert events[1] == ("progress", None, 0)


def test_locked_database_is_skipped_not_corrupt(tmp_path):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    locker = sqlite3.connect(str(db_path), isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        result = hermes_state._probe_db_health(
            db_path,
            _busy_timeout_seconds=0.01,
        )
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    assert result.status == "skipped"
    assert result.category == "locked"
    assert "locked" in result.reason.lower()


def test_locked_classification_supports_python_without_sqlite_error_constants(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    locker = sqlite3.connect(str(db_path), isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    monkeypatch.delattr(hermes_state.sqlite3, "SQLITE_BUSY", raising=False)
    monkeypatch.delattr(hermes_state.sqlite3, "SQLITE_LOCKED", raising=False)
    try:
        result = hermes_state._probe_db_health(
            db_path,
            _busy_timeout_seconds=0.01,
        )
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    assert result.status == "skipped"
    assert result.category == "locked"


def test_corrupt_database_is_reported_unhealthy(tmp_path):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"not a sqlite database")

    result = hermes_state._probe_db_health(db_path)

    assert result.status == "unhealthy"
    assert result.category == "database_error"
    assert "not a database" in result.reason.lower()


def test_malformed_schema_is_reported_unhealthy(tmp_path):
    db_path = tmp_path / "state.db"
    db = hermes_state.SessionDB(db_path=db_path)
    db.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()

    result = hermes_state._probe_db_health(db_path)

    assert result.status == "unhealthy"
    assert result.category == "malformed_schema"
    assert "malformed database schema" in result.reason.lower()


def test_fts_trigger_corruption_is_reported_unhealthy(tmp_path):
    db_path = tmp_path / "state.db"
    db = hermes_state.SessionDB(db_path=db_path)
    session_id = db.create_session(session_id="health-probe", source="cli")
    db.append_message(session_id, role="user", content="health probe")
    db.close()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEF'")
    conn.close()

    result = hermes_state._probe_db_health(db_path)

    assert result.status == "unhealthy"
    assert result.category in {"fts_index", "fts_write"}
    assert result.reason


def test_database_error_from_fts_write_probe_keeps_fts_classification(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    real_connect = sqlite3.connect

    class FailingFTSConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            if sql.startswith("INSERT INTO messages "):
                raise sqlite3.DatabaseError("database disk image is malformed")
            return super().execute(sql, parameters)

    def failing_connect(*args, **kwargs):
        return real_connect(*args, factory=FailingFTSConnection, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", failing_connect)

    result = hermes_state._probe_db_health(db_path)

    assert result.status == "unhealthy"
    assert result.category == "fts_write"
    assert "malformed" in result.reason


def test_explicit_full_probe_runs_without_a_progress_budget(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    events = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def set_progress_handler(self, progress_handler, n):
            events.append((progress_handler, n))
            return super().set_progress_handler(progress_handler, n)

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, factory=TrackingConnection, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracking_connect)

    result = hermes_state._probe_db_health(db_path, full_integrity=True)

    assert result.status == "healthy"
    assert result.check_mode == "full"
    assert events == []


def test_repair_does_not_touch_database_when_health_probe_is_skipped(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    monkeypatch.setattr(
        hermes_state,
        "_probe_db_health",
        lambda _path: hermes_state.DBHealthResult(
            status="skipped", category="timeout", reason="budget expired"
        ),
    )
    monkeypatch.setattr(
        hermes_state,
        "_backup_db_file",
        lambda _path: pytest.fail("skipped health checks must not start repair"),
    )

    report = hermes_state.repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert report["skipped"] is True
    assert "budget expired" in report["error"]
    assert report["backup_path"] is None


def test_repair_stops_when_post_strategy_verification_is_skipped(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    results = iter(
        (
            hermes_state.DBHealthResult(
                status="unhealthy", category="fts_write", reason="bad FTS index"
            ),
            hermes_state.DBHealthResult(
                status="skipped", category="locked", reason="database is locked"
            ),
        )
    )
    monkeypatch.setattr(hermes_state, "_probe_db_health", lambda _path: next(results))

    report = hermes_state.repair_state_db_schema(db_path, backup=False)

    assert report["repaired"] is False
    assert report["skipped"] is True
    assert "verification" in report["error"]
    assert "locked" in report["error"]


def test_repair_does_not_treat_general_integrity_failure_as_fts_damage(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    monkeypatch.setattr(
        hermes_state,
        "_probe_db_health",
        lambda _path: hermes_state.DBHealthResult(
            status="unhealthy", category="integrity", reason="page 7 is corrupt"
        ),
    )
    monkeypatch.setattr(
        hermes_state,
        "_backup_db_file",
        lambda _path: pytest.fail("unsupported corruption must not start FTS repair"),
    )

    report = hermes_state.repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert report["skipped"] is False
    assert "not an FTS/schema repair target" in report["error"]


def test_doctor_parser_exposes_explicit_full_state_db_check():
    from hermes_cli.subcommands.doctor import build_doctor_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_doctor_parser(subparsers, cmd_doctor=lambda _args: None)

    assert parser.parse_args(["doctor"]).full_state_db_check is False
    assert parser.parse_args(["doctor", "--full-state-db-check"]).full_state_db_check is True
    doctor_help = parser._subparsers._group_actions[0].choices["doctor"].format_help()
    assert "unbounded" in doctor_help.lower()
    assert "state.db" in doctor_help


def test_console_sessions_repair_does_not_repair_a_skipped_probe(
    tmp_path, monkeypatch
):
    from hermes_cli.console_engine import _sessions_repair

    db_path = tmp_path / "state.db"
    _build_minimal_state_db(db_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(
        hermes_state,
        "_probe_db_health",
        lambda _path: hermes_state.DBHealthResult(
            status="skipped", category="locked", reason="database is locked"
        ),
    )
    monkeypatch.setattr(
        hermes_state,
        "repair_state_db_schema",
        lambda *_args, **_kwargs: pytest.fail("skipped probe must not invoke repair"),
    )

    output = _sessions_repair(None, [])

    assert "health check skipped" in output.lower()
    assert "locked" in output.lower()
