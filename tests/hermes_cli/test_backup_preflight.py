"""Preflight must work while the app is broken and its database is still live."""
import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path


def test_preflight_finishes_while_another_connection_keeps_writing(tmp_path):
    # A stepped sqlite3 backup restarts on every foreign write and never finishes under load.
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    script = Path(__file__).resolve().parents[2] / "hermes_cli" / "backup_sqlite.py"
    with sqlite3.connect(db) as setup:
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("CREATE TABLE messages (body TEXT)")
        setup.executemany("INSERT INTO messages VALUES (?)", [("x" * 4000,)] * 4000)
    stop = threading.Event()

    def write_continuously():
        with sqlite3.connect(db, timeout=5) as writer:
            while not stop.is_set():
                writer.execute("INSERT INTO messages VALUES ('live turn')")
                writer.commit()
                stop.wait(0.005)

    thread = threading.Thread(target=write_continuously)
    thread.start()
    try:
        result = subprocess.run([sys.executable, "-I", "-S", str(script), str(home)],
                                capture_output=True, text=True, timeout=20, check=False)
    finally:
        stop.set()
        thread.join()
    assert result.returncode == 0, result.stderr
    with sqlite3.connect(json.loads(result.stdout)["path"]) as snapshot:
        assert snapshot.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_preflight_captures_committed_wal_without_application_imports(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    script = Path(__file__).resolve().parents[2] / "hermes_cli" / "backup_sqlite.py"
    runner = """
import runpy, sys
class NoApplicationImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('hermes', 'utils', 'yaml')):
            raise ImportError('application imports are broken')
sys.meta_path.insert(0, NoApplicationImports())
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE messages (body TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO messages VALUES ('committed only in WAL')")
        writer.commit()
        assert Path(str(db) + "-wal").stat().st_size > 0
        # The control proves a main-file copy loses committed rows.
        control = tmp_path / "raw.db"
        control.write_bytes(db.read_bytes())
        with sqlite3.connect(control) as raw:
            assert raw.execute("SELECT * FROM messages").fetchall() == []
        for _ in range(3):
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", runner, str(script), str(home)],
                capture_output=True, text=True, timeout=20,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            backup = Path(json.loads(result.stdout)["path"])
            with sqlite3.connect(backup) as snapshot:
                assert snapshot.execute("SELECT * FROM messages").fetchall() == [("committed only in WAL",)]
                assert snapshot.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert len(list(home.glob("state.db.pre-update-emergency-*.bak"))) == 2
    writer.close()
    previous = sorted(home.glob("state.db.pre-update-emergency-*.bak"))
    db.write_bytes(b"broken database")
    result = subprocess.run([sys.executable, "-I", "-S", str(script), str(home)], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode != 0
    assert sorted(home.glob("state.db.pre-update-emergency-*.bak")) == previous
    assert not list(home.glob("*.partial"))


def test_rollback_journal_copy_lets_a_live_writer_commit_between_steps(tmp_path, monkeypatch):
    # A rollback-journal reader blocks every commit while it holds the read lock, so a
    # one-step copy of a large state.db stalls the live backend's writes for the whole copy.
    from hermes_cli import backup_sqlite

    db = tmp_path / "state.db"
    with sqlite3.connect(db) as setup:
        setup.execute("PRAGMA journal_mode=DELETE")
        setup.execute("CREATE TABLE messages (body TEXT)")
        setup.executemany("INSERT INTO messages VALUES (?)", [("x" * 4000,)] * 2000)
    writer = sqlite3.connect(db, timeout=0, isolation_level=None)  # a held lock fails at once
    unread_pages_at_commit, write_errors = [], []
    real_connect = sqlite3.connect

    class WriterCommitsBetweenSteps(sqlite3.Connection):
        def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.25):
            def between_steps(status, remaining, total):
                if status == sqlite3.SQLITE_OK:
                    try:
                        writer.execute("INSERT INTO messages VALUES ('live turn')")
                        unread_pages_at_commit.append(remaining)
                    except sqlite3.OperationalError as exc:
                        write_errors.append(str(exc))
                if progress is not None:
                    progress(status, remaining, total)
            return super().backup(target, pages=pages, progress=between_steps, name=name, sleep=sleep)

    def connect(database, *args, **kwargs):
        if "mode=ro" in str(database):
            kwargs["factory"] = WriterCommitsBetweenSteps
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(backup_sqlite.sqlite3, "connect", connect)
    snapshot = tmp_path / "snapshot.db"
    try:
        copied = backup_sqlite._safe_copy_db(db, snapshot)
    finally:
        writer.close()

    assert write_errors == []
    # The writer got in before the copy had read the whole database.
    assert any(remaining > 0 for remaining in unread_pages_at_commit)
    # Each commit restarts the copy; the restarts are bounded and end in a snapshot or a refusal.
    if copied:
        with real_connect(snapshot) as copy:
            assert copy.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    else:
        assert not snapshot.exists()
