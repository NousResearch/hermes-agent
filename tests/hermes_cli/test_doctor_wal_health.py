"""Tests for the state.db WAL health check in hermes doctor (#96976).

A large WAL whose frames are all checkpointed is healthy: its on-disk size is
just the high-water mark SQLite keeps for reuse, and "repairing" it is a
no-op. Doctor must classify WAL health by checkpoint state, not raw size.
"""

import re
import sqlite3

import hermes_cli.doctor as doctor_mod


def _build_db(tmp_path):
    """Create state.db with an open keeper connection and return its parts.

    The keeper must stay open for the lifetime of the test: SQLite deletes
    the -wal file when the last connection closes cleanly, and the scenario
    under test is a live DB (e.g. held by the gateway) whose WAL persists
    after a full checkpoint.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    db_path = hermes_home / "state.db"
    keeper = sqlite3.connect(str(db_path))
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
    keeper.commit()
    return keeper, db_path, hermes_home / "state.db-wal", hermes_home


def _grow_wal(db_path, wal_path, target_bytes=55 * 1024 * 1024):
    """Write enough rows in one transaction to push the WAL past target_bytes."""
    chunk = b"x" * (64 * 1024)  # 64 KB per row → ~1024-byte overhead per frame
    conn = sqlite3.connect(str(db_path))
    while wal_path.stat().st_size < target_bytes:
        conn.executemany("INSERT INTO t (blob) VALUES (?)", [(chunk,)] * 100)
    conn.commit()
    conn.close()
    assert wal_path.stat().st_size > target_bytes


def test_uncheckpointed_frames_zero_when_fully_checkpointed(tmp_path):
    """A checkpointed WAL reports 0 un-checkpointed frames."""
    keeper, db_path, wal_path, _ = _build_db(tmp_path)
    try:
        _grow_wal(db_path, wal_path)
        # Checkpoint everything (keeper stays open so the WAL file persists).
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()

        assert doctor_mod._wal_uncheckpointed_frames(db_path) == 0
        # The WAL stays large after PASSIVE (no truncation) — the false positive.
        assert wal_path.stat().st_size > 50 * 1024 * 1024
    finally:
        keeper.close()


def test_large_checkpointed_wal_is_not_reported_as_issue(tmp_path):
    """A >50MB fully checkpointed WAL must be healthy, not flagged (#96976)."""
    keeper, db_path, wal_path, hermes_home = _build_db(tmp_path)
    try:
        _grow_wal(db_path, wal_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()

        issues: list[str] = []
        fixed = doctor_mod._check_wal_health(hermes_home, db_path, should_fix=False, issues=issues)

        assert fixed == 0
        assert issues == []
    finally:
        keeper.close()


def test_large_checkpointed_wal_fix_truncates(tmp_path):
    """--fix reclaims the space a PASSIVE checkpoint leaves behind."""
    keeper, db_path, wal_path, hermes_home = _build_db(tmp_path)
    try:
        _grow_wal(db_path, wal_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        conn.close()

        issues: list[str] = []
        fixed = doctor_mod._check_wal_health(hermes_home, db_path, should_fix=True, issues=issues)

        assert fixed == 1
        assert issues == []
        assert wal_path.stat().st_size < 50 * 1024 * 1024
    finally:
        keeper.close()


def test_large_uncheckpointed_wal_is_reported_as_issue(tmp_path):
    """A WAL with frames a concurrent reader holds back still warns."""
    keeper, db_path, wal_path, hermes_home = _build_db(tmp_path)
    try:
        # An open read transaction pins the WAL start, so a PASSIVE checkpoint
        # cannot backfill the frames written after the reader's snapshot.
        reader = sqlite3.connect(str(db_path))
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM t").fetchone()

        _grow_wal(db_path, wal_path, target_bytes=55 * 1024 * 1024)

        try:
            issues: list[str] = []
            fixed = doctor_mod._check_wal_health(hermes_home, db_path, should_fix=False, issues=issues)
            uncheckpointed = doctor_mod._wal_uncheckpointed_frames(db_path)

            assert uncheckpointed is None or uncheckpointed > 0
            assert fixed == 0
            assert any("Large WAL file" in i for i in issues)
        finally:
            reader.rollback()
            reader.close()
    finally:
        keeper.close()


def test_probe_fails_closed_for_unreadable_db(tmp_path):
    """An unreadable/locked DB degrades to None (size-based verdict path)."""
    keeper, db_path, _, _ = _build_db(tmp_path)
    try:

        class BoomConn:
            def execute(self, *_a, **_kw):
                raise sqlite3.OperationalError("database is locked")

            def close(self):
                pass

        real_connect = sqlite3.connect

        def fake_connect(*_a, **_kw):
            return BoomConn()

        sqlite3.connect = fake_connect
        try:
            assert doctor_mod._wal_uncheckpointed_frames(db_path) is None
        finally:
            sqlite3.connect = real_connect
    finally:
        keeper.close()


def test_wal_fix_counts_toward_doctor_fix_summary(monkeypatch, tmp_path):
    """Regression: a WAL truncation performed by --fix must be tallied in the
    "Fixed N issue(s)" summary (fixed_count), not silently dropped."""
    from tests.hermes_cli.test_doctor_command_install import _run_doctor, _setup_doctor_env

    home, project, _ = _setup_doctor_env(monkeypatch, tmp_path)

    # Grow a fully checkpointed WAL in the doctor env's state.db.
    db_path = home / "state.db"
    keeper = sqlite3.connect(str(db_path))
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
    keeper.commit()
    wal_path = home / "state.db-wal"
    chunk = b"x" * (64 * 1024)
    conn = sqlite3.connect(str(db_path))
    while wal_path.stat().st_size < 55 * 1024 * 1024:
        conn.executemany("INSERT INTO t (blob) VALUES (?)", [(chunk,)] * 100)
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    conn.close()
    try:
        out = _run_doctor(fix=True)
        # The fixture env itself triggers 7 fixes; the WAL truncation must
        # contribute exactly 1 more (regression: the return value of
        # _check_wal_health() was dropped from fixed_count).
        m = re.search(r"Fixed (\d+) issue\(s\)", out)
        assert m is not None, out
        assert int(m.group(1)) == 8
        assert wal_path.stat().st_size < 50 * 1024 * 1024
    finally:
        keeper.close()
