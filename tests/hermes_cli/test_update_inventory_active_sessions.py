"""Pre-update impact: sessions with recent activity per profile, read-only."""
import sqlite3
import time

from hermes_cli.update_inventory import (
    _collect_active_sessions,
    _count_recent_sessions,
)


def _write_state_db(home, rows) -> None:
    """rows: [(last_activity_at|None, started_at|None)] — a minimal sessions table."""
    home.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(home / "state.db")
    try:
        con.execute("CREATE TABLE sessions (id TEXT, last_activity_at REAL, started_at REAL)")
        con.executemany(
            "INSERT INTO sessions (id, last_activity_at, started_at) VALUES (?, ?, ?)",
            [(f"s{i}", last, started) for i, (last, started) in enumerate(rows)],
        )
        con.commit()
    finally:
        con.close()


def test_counts_only_recently_active_sessions(tmp_path):
    home = tmp_path / "profiles" / "default"
    now = time.time()
    _write_state_db(home, [
        (now - 60, now - 3600),      # active via heartbeat
        (None, now - 120),           # active via recent start
        (now - 86400, now - 90000),  # stale
        (None, None),                # never active
    ])
    assert _count_recent_sessions(home) == 2


def test_missing_db_is_unknown_not_zero(tmp_path):
    assert _count_recent_sessions(tmp_path / "profiles" / "ghost") is None


def test_unreadable_db_is_unknown_not_zero(tmp_path):
    home = tmp_path / "profiles" / "broken"
    home.mkdir(parents=True)
    (home / "state.db").write_bytes(b"not a database")
    assert _count_recent_sessions(home) is None


def test_collect_skips_profiles_without_a_readable_db(tmp_path):
    live = tmp_path / "profiles" / "live"
    _write_state_db(live, [(time.time() - 30, time.time() - 3600)])
    counts = _collect_active_sessions([
        ("live", live),
        ("ghost", tmp_path / "profiles" / "ghost"),
    ])
    assert counts == {"live": 1}
