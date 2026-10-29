"""Incident read/modify/write operations must serialize with other processes."""

import json
import sqlite3
import subprocess
import sys

import pytest

from cron import incidents


@pytest.mark.parametrize("operation", ["insert", "alert", "resolve"])
def test_concurrent_incident_mutations(tmp_path, monkeypatch, operation):
    db = tmp_path / "executions.db"
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", db)
    monkeypatch.setattr(incidents._executions, "EXECUTIONS_FILE", None)
    incident_id, _ = incidents.upsert_incident("job", "provider unavailable")
    if operation == "insert":
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM cron_incidents")

    child = """
import json, sqlite3, sys
from pathlib import Path
from cron import incidents
incidents.EXECUTIONS_FILE = Path(sys.argv[1])
connect = incidents._connect
def impatient_connect():
    conn = connect()
    conn.execute('PRAGMA busy_timeout=0')
    return conn
incidents._connect = impatient_connect
try:
    result = (incidents.upsert_incident('job', 'provider unavailable')
              if sys.argv[2] == 'insert' else incidents.ack_incident(sys.argv[3]))
    print(json.dumps(result))
except sqlite3.OperationalError as exc:
    if 'locked' not in str(exc):
        raise
    sys.exit(75)
"""
    attempts = []

    def compete():
        result = subprocess.run(
            [sys.executable, "-c", child, str(db), operation, incident_id],
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode in (0, 75), result.stderr
        return result

    connect = sqlite3.connect

    class InterleavedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith(("SELECT id, state FROM cron_incidents", "SELECT state FROM cron_incidents")):
                # A second process mutates the same incident after the lookup,
                # bypassing the first process's RLock. No timing sleeps needed.
                attempts.append(compete())
            return cursor

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=InterleavedConnection))
        if operation == "insert":
            incidents.upsert_incident("job", "provider unavailable")
        else:
            incidents.set_incident_state(incident_id, "alerted" if operation == "alert" else "resolved")

    assert len(attempts) == 1
    result = attempts[0]
    if result.returncode == 75:
        result = compete()  # Retry once the competing writer releases its transaction.
        assert result.returncode == 0, result.stderr
    if operation == "insert":
        assert json.loads(result.stdout) == [incident_id, False]
        assert incidents.count_incidents() == 1
    else:
        assert incidents.get_incident(incident_id)["state"] == "closed"
        assert not incidents.set_incident_state(incident_id, "alerted")
