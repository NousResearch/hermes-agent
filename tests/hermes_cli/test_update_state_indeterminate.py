"""Regression coverage for issue #147938's update-time false corruption alarms."""

from contextlib import closing
import sqlite3

from hermes_cli import backup, sqlite_safe_read
from hermes_cli.update_cmd_maint import (
    _restore_state_db_from_snapshot,
    _verify_and_restore_one_state_db,
)
from hermes_cli.update_state_integrity import verify_update_state_db


def _make_db(path, count=2):
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
        db.executemany(
            "INSERT INTO sessions(name) VALUES(?)",
            [(f"message-{i}",) for i in range(count)],
        )
        db.commit()


def _rows(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT name FROM sessions ORDER BY id").fetchall()


def test_unreadable_live_header_uses_sqlite_and_keeps_new_messages(tmp_path, monkeypatch, capsys):
    """Windows OSError/own live handle must not trigger an avoidable restore."""
    state = tmp_path / "state.db"
    _make_db(state)
    snapshot_id = backup.create_quick_snapshot(hermes_home=tmp_path)
    assert snapshot_id
    with closing(sqlite3.connect(state)) as db:
        db.execute("INSERT INTO sessions(name) VALUES('new-after-snapshot')")
        db.commit()

    original = sqlite_safe_read.read_header_bytes_preopen
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen",
        lambda path, **kw: None if path == state else original(path, **kw),
    )
    assert verify_update_state_db(state)["valid"] is True
    _verify_and_restore_one_state_db(tmp_path, label="default home", snapshot_id=snapshot_id)

    assert len(_rows(state)) == 3
    assert _rows(state)[-1] == ("new-after-snapshot",)
    assert "Auto-restored" not in capsys.readouterr().out


def test_locked_sqlite_probe_defers_even_with_fresh_snapshot(tmp_path, monkeypatch, capsys):
    """No evidence of corruption means no data-destructive auto-restore."""
    state = tmp_path / "state.db"
    _make_db(state)
    snapshot_id = backup.create_quick_snapshot(hermes_home=tmp_path)
    assert snapshot_id
    before = state.read_bytes()
    original = backup._query_ro_sqlite
    monkeypatch.setattr(
        backup, "_query_ro_sqlite",
        lambda path, fn: (None, sqlite3.OperationalError("database is locked"))
        if path == state else original(path, fn),
    )

    assert verify_update_state_db(state)["valid"] is None
    _verify_and_restore_one_state_db(tmp_path, label="default home", snapshot_id=snapshot_id)

    assert state.read_bytes() == before
    output = capsys.readouterr().out
    assert "verification deferred" in output
    assert "Auto-restored" not in output


def test_genuine_sqlite_corruption_is_still_detected(tmp_path):
    state = tmp_path / "state.db"
    _make_db(state)
    with closing(sqlite3.connect(state)) as db:
        page = db.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
    with state.open("r+b") as handle:
        handle.seek((page - 1) * page_size)
        handle.write(b"\0")
    verdict = verify_update_state_db(state)
    assert verdict["valid"] is False


def test_successful_restore_with_unreadable_header_is_not_misreported(tmp_path, monkeypatch, capsys):
    """After the copy, the same Windows-style header refusal must not claim failure."""
    state = tmp_path / "state.db"
    _make_db(state)
    snap = tmp_path / "snapshot.db"
    _make_db(snap, count=1)
    original = sqlite_safe_read.read_header_bytes_preopen
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen",
        lambda path, **kw: None if path == state else original(path, **kw),
    )

    assert _restore_state_db_from_snapshot(state, snap) is True
    assert len(_rows(state)) == 1
    assert "verification" not in capsys.readouterr().out.lower()


def test_restore_that_applied_but_cannot_be_verified_reports_truthfully(tmp_path, monkeypatch, capsys):
    """A completed write with an indeterminate check is NOT a failed copy."""
    from hermes_cli import update_state_integrity

    state = tmp_path / "state.db"
    _make_db(state)
    snap = tmp_path / "snapshot.db"
    _make_db(snap, count=1)
    monkeypatch.setattr(
        update_state_integrity, "verify_update_state_db",
        lambda path: {"valid": None, "message": "database is locked"},
    )
    assert _restore_state_db_from_snapshot(state, snap) is None
    assert len(_rows(state)) == 1
    out = capsys.readouterr().out
    assert "Snapshot was applied" in out
    assert "did not complete" not in out
