"""A locked live destination must not strand /snapshot restore or hermes import."""

import json
import sqlite3
import threading

from hermes_cli import backup, backup_restore


def _make_db(path, value):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))


def _call_in_thread(fn, timeout):
    """Run *fn* on a worker so a regression fails an assertion instead of hanging pytest."""
    result = {}

    def run():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced through result["error"]
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    return worker, result


def test_restore_busy_destination_fails_bounded_without_replacing_live_db(tmp_path, monkeypatch):
    src, dst = tmp_path / "snap.db", tmp_path / "state.db"
    _make_db(src, "snapshot")
    _make_db(dst, "live")
    if hasattr(backup_restore, "_RESTORE_STALL_SECONDS"):
        monkeypatch.setattr(backup_restore, "_RESTORE_STALL_SECONDS", 0.1)

    # BEGIN EXCLUSIVE holds the write lock in a distinct live connection. A
    # backup() with no progress callback retries SQLITE_BUSY indefinitely, so
    # the unbounded variant never returns and the join below times out.
    holder = sqlite3.connect(dst, timeout=0)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("UPDATE t SET v='uncommitted'")
    try:
        worker, result = _call_in_thread(lambda: backup_restore._safe_restore_db(src, dst), timeout=3)
        assert not worker.is_alive(), "restore did not return while the destination stayed locked"
        assert "error" not in result, result["error"]
        assert result.get("value") is False
    finally:
        holder.rollback()
        holder.close()
    with sqlite3.connect(dst) as conn:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "live"


def test_restore_unlocked_destination_still_copies_snapshot(tmp_path):
    src, dst = tmp_path / "snap.db", tmp_path / "state.db"
    _make_db(src, "snapshot")
    _make_db(dst, "live")
    assert backup_restore._safe_restore_db(src, dst) is True
    with sqlite3.connect(dst) as conn:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "snapshot"


def test_quick_restore_reports_locked_database_as_incomplete(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    snap = home / "state-snapshots" / "locked"
    snap.mkdir(parents=True)
    _make_db(home / "state.db", "live")
    _make_db(snap / "state.db", "snapshot")
    (home / "config.yaml").write_text("old: true\n", encoding="utf-8")
    (snap / "config.yaml").write_text("new: true\n", encoding="utf-8")
    manifest = {
        "id": "locked",
        "files": {
            "state.db": (snap / "state.db").stat().st_size,
            "config.yaml": (snap / "config.yaml").stat().st_size,
        },
    }
    (snap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(backup_restore, "_RESTORE_STALL_SECONDS", 0.1)

    holder = sqlite3.connect(home / "state.db", timeout=0)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("UPDATE t SET v='uncommitted'")
    failed = []
    try:
        restored = backup.restore_quick_snapshot(
            "locked", hermes_home=home, failed_paths=failed
        )
    finally:
        holder.rollback()
        holder.close()

    assert restored is False
    assert failed == ["state.db"]
    assert (home / "config.yaml").read_text(encoding="utf-8") == "new: true\n"
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "live"
