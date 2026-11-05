"""In-process state.db readers register with the tracked-connection registry.

``has_live_connection()`` / ``offline_file_access()`` decide when raw byte I/O on a live database
is safe: closing any raw descriptor cancels every POSIX lock the process holds on the file. A
reader opened with a bare ``sqlite3.connect`` is a live holder those checks cannot see, so each
of these sites must open through ``connect_tracked``.
"""

import sqlite3
from pathlib import Path

import pytest

import hermes_cli.sqlite_safe_read as safe_read


@pytest.fixture
def state_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, api_content TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('s1')")
        conn.execute("INSERT INTO messages (content, api_content) VALUES ('hello', NULL)")
    conn.close()
    return db


@pytest.fixture
def tracked(monkeypatch):
    """Keys registered as live while the code under test ran."""
    seen: list[str] = []
    real = safe_read._track_key

    def _spy(key, delta=1):
        if delta > 0:
            seen.append(key)
        return real(key, delta)

    monkeypatch.setattr(safe_read, "_track_key", _spy)
    return seen


def _key(path: Path) -> str:
    return str(path.resolve())


def test_readiness_probe_is_tracked(state_db, tracked):
    from gateway.readiness import _probe_state_db
    _probe_state_db(state_db.parent)
    assert _key(state_db) in tracked


def test_readiness_probe_opens_the_tracked_file_with_reserved_home_characters(
    state_db, tmp_path, monkeypatch, tracked,
):
    """URI parsing must not redirect a tracked readiness open to a different file."""
    from gateway.readiness import _probe_state_db

    home = tmp_path / "profile #? with spaces"
    state_db.parent.rename(home)
    db_path = home / "state.db"
    opened_paths: list[str] = []
    real_connect = safe_read.connect_tracked

    def _observe_open(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        row = conn.execute("PRAGMA database_list").fetchone()
        opened_paths.append(_key(Path(row[2])))
        return conn

    monkeypatch.setattr(safe_read, "connect_tracked", _observe_open)
    assert _probe_state_db(home)["status"] == "ok"
    assert opened_paths == [_key(db_path)]
    assert _key(db_path) in tracked


def test_integrity_check_and_journal_mode_are_tracked(state_db, tracked):
    from gateway.lifecycle_ledger import _state_db_journal_mode, check_state_db_integrity
    check_state_db_integrity(state_db.parent)
    _state_db_journal_mode(state_db)
    assert tracked.count(_key(state_db)) >= 2


def test_spillover_reference_scan_is_tracked(state_db, tracked):
    from tools.tool_result_storage import referenced_spillover_names
    spill_dir = state_db.parent / "a" / "b"
    spill_dir.mkdir(parents=True)
    referenced_spillover_names(spill_dir)
    assert _key(state_db) in tracked


def test_doctor_session_count_and_damage_check_are_tracked(state_db, tracked):
    from hermes_cli.doctor_state import _session_count
    from hermes_state_repair import state_db_has_structural_damage
    assert _session_count(state_db) == 1
    state_db_has_structural_damage(state_db)
    assert tracked.count(_key(state_db)) >= 2


def test_shared_metrics_first_session_read_is_tracked(state_db, tracked):
    import hermes_cli.observability.shared_metrics_snapshot as snap
    with sqlite3.connect(state_db) as conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN started_at REAL")
        conn.execute("UPDATE sessions SET started_at = 1.0")
    conn.close()
    snap._first_session_at.pop(str(state_db.parent), None)
    assert snap._first_session_started_at(state_db.parent) == 1.0
    assert _key(state_db) in tracked


def test_worktree_liveness_reads_are_tracked(state_db, tracked, tmp_path):
    """Cron's worktree sweep runs these inside the gateway process, not only from the CLI."""
    from hermes_cli.worktree_ops import _retained_tree_is_live, _tree_still_served
    tree = tmp_path / "tree"
    tree.mkdir()
    _retained_tree_is_live(str(state_db), str(tree))
    _tree_still_served(str(state_db), str(tree))
    assert tracked.count(_key(state_db)) >= 2


def test_backup_integrity_query_is_tracked(state_db, tracked):
    from hermes_cli.backup import _query_ro_sqlite
    value, exc = _query_ro_sqlite(state_db, lambda c: c.execute("SELECT count(*) FROM sessions").fetchone()[0])
    assert (value, exc) == (1, None)
    assert _key(state_db) in tracked


def test_approvals_suggest_scan_is_tracked(state_db, tracked):
    from hermes_cli.approvals_suggest import _connect_readonly
    _connect_readonly(state_db).close()
    assert _key(state_db) in tracked


def test_restore_destination_connection_is_tracked(state_db, tracked, tmp_path):
    from hermes_cli.backup_restore import _restore_db_pages
    snapshot = tmp_path / "snapshot.db"
    with sqlite3.connect(snapshot) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.close()
    assert _restore_db_pages(snapshot, state_db)
    assert _key(state_db) in tracked


def test_set_journal_mode_switch_connection_is_tracked(state_db, tracked, monkeypatch):
    import argparse
    from hermes_cli.sessions_cmd_journal_mode import cmd_set_journal_mode
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", state_db)
    args = argparse.Namespace(sessions_action="set-journal-mode", mode="delete", db=None, force=True)
    assert cmd_set_journal_mode(args) == 0
    assert _key(state_db) in tracked

