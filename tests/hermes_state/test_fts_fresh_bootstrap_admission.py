"""Fresh state.db FTS bootstrap must honor cross-process rebuild admission."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import hermes_state_common
from hermes_state import SessionDB


_HOLD_ADMISSION_SCRIPT = """
import sys
import time
from pathlib import Path

from hermes_state_common import fts_rebuild_admission

with fts_rebuild_admission(Path(sys.argv[1]), timeout_seconds=0) as acquired:
    assert acquired
    print("locked", flush=True)
    time.sleep(30)
"""


def _trigram_supported() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE trigram_probe USING fts5(content, tokenize='trigram')")
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return True


def test_fresh_fts_bootstrap_does_not_publish_schema_without_admission(tmp_path, monkeypatch):
    """A fresh opener that loses admission leaves no partially initialized FTS surface."""
    db_path = tmp_path / "state.db"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_ADMISSION_SCRIPT, str(db_path)],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        monkeypatch.setattr(hermes_state_common, "_FTS_REBUILD_LOCK_TIMEOUT_SECONDS", 0.1)

        deferred = SessionDB(db_path=db_path)
        try:
            assert deferred._fts_stale is True
            assert deferred._fts_enabled is False
            assert deferred._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts'"
            ).fetchone() is None
        finally:
            deferred.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)

    recovered = SessionDB(db_path=db_path)
    try:
        assert recovered._fts_stale is False
        assert recovered._fts_enabled is True
        if _trigram_supported():
            assert recovered._trigram_available is True
            assert recovered._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_trigram'"
            ).fetchone() is not None
        recovered.create_session("s1", source="test")
        recovered.append_message("s1", "user", "fresh bootstrap recovered")
        assert recovered.search_messages("recovered")
    finally:
        recovered.close()
