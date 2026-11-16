"""DELETE-journal readers wait out another process's commit instead of raising ``database is locked``.

Under rollback-journal mode a reader needs a SHARED lock that a sibling process's commit blocks for
the length of its journal+db fsyncs. Readers used to fail after the writer connection's 1 s busy
timeout, so the dashboard's /api/sessions returned 503 and ``hermes sessions list`` crashed while a
gateway was writing. The holder here is a separate process, as in production.
"""

import subprocess
import sys

import pytest
import hermes_yaml as yaml

from hermes_state import SessionDB

_HOLD_S = 2.0  # longer than the old 1 s read budget, well inside the new one

_HOLDER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN EXCLUSIVE")
print("held", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
"""


@pytest.fixture()
def delete_mode_db(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    assert db._wal_active is False
    assert db._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="hello")
    yield db
    db.close()


@pytest.mark.parametrize("read_only", [True, False], ids=["read_only_handle", "writer_handle"])
def test_delete_mode_read_waits_for_other_process_commit(delete_mode_db, read_only):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(delete_mode_db.db_path), str(_HOLD_S)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        reader = SessionDB(db_path=delete_mode_db.db_path, read_only=True) if read_only else delete_mode_db
        try:
            assert [s["id"] for s in reader.list_sessions_rich(limit=10)] == ["s1"]
            assert reader.session_count() == 1
        finally:
            if read_only:
                reader.close()
    finally:
        assert holder.wait(timeout=30) == 0
    # Writes keep their short busy timeout: they retry at application level with jitter.
    assert delete_mode_db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
