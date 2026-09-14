"""The per-job byte cap holds across independent notepad writers."""

import sqlite3
import subprocess
import sys


def test_concurrent_writes_cannot_exceed_job_capacity(tmp_path, monkeypatch):
    from cron import notepad

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    value = "x" * (notepad.MAX_VALUE_BYTES - 1)
    for key in "abc":
        notepad.set_note("job", key, value)
    attempts = []

    def write_competitor():
        result = subprocess.run(
            [sys.executable, "-c", """
import sqlite3
import sys
from cron import notepad
try:
    notepad.set_note('job', 'e', 'x' * (notepad.MAX_VALUE_BYTES - 1))
except sqlite3.OperationalError as exc:
    if 'locked' not in str(exc):
        raise
    sys.exit(75)
except ValueError:
    sys.exit(76)
"""], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode in (0, 75, 76), result.stderr
        return result.returncode

    class InterleavedCursor(sqlite3.Cursor):
        def fetchone(self):
            row = super().fetchone()
            # Finish the aggregate SELECT so its read cursor alone does not
            # block a writer in DELETE-journal mode. Keep its observed total.
            super().fetchall()
            attempts.append(write_competitor())
            return row

    class InterleavedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "SELECT COALESCE(SUM" in sql:
                return self.cursor(factory=InterleavedCursor).execute(sql, parameters)
            return super().execute(sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(notepad, "_connect", lambda: sqlite3.connect(
            notepad._current_notepad_file(), timeout=5, factory=InterleavedConnection))
        notepad.set_note("job", "d", value)

    assert attempts
    if attempts == [75]:
        assert write_competitor() == 76
    rows = notepad.list_notes("job")
    total = sum(len(row["key"].encode()) + len(row["value"].encode()) for row in rows)
    assert total <= notepad.MAX_JOB_TOTAL_BYTES
    assert notepad.get_note("job", "d") == value
    assert notepad.get_note("job", "e") is None
