"""Preflight must work while the app is broken and its database is still live."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


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
    result = subprocess.run([sys.executable, "-I", "-S", str(script), str(home)], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert sorted(home.glob("state.db.pre-update-emergency-*.bak")) == previous
    assert not list(home.glob("*.partial"))
