"""A live ledger writer must survive an unavailable process-start fingerprint."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_OWNER = """
import json
import sys
from cron import executions, delivery_queue
import gateway.status
if sys.argv[1] == 'recorded':
    gateway.status.get_process_start_time = lambda pid: None
row = executions.create_execution('live-job', source='builtin')
executions.mark_execution_running(row['id'])
delivery_queue.enqueue(row['id'], {'id': 'live-job'}, 'result')
delivery_queue.claim_next()
print(json.dumps(row), flush=True)
sys.stdin.read(1)
finished = executions.finish_execution(row['id'], success=True)
delivered = delivery_queue._finish(row['id'], error=None)
print(json.dumps({'execution': finished, 'delivered': delivered}), flush=True)
"""


@pytest.mark.parametrize("missing", ["recorded", "current"])
def test_live_owner_can_finish_both_ledgers_when_fingerprint_is_unavailable(tmp_path, monkeypatch, missing):
    from cron import executions, delivery_queue
    import gateway.status

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", None)
    env = dict(os.environ, HERMES_HOME=str(home))
    repo = Path(__file__).resolve().parents[2]
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER, missing], cwd=repo, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        row = json.loads(owner.stdout.readline())
        assert owner.poll() is None
        assert gateway.status._pid_exists(owner.pid)
        if missing == "current":
            assert row["process_started_at"] is not None
            assert not executions._owner_is_live(owner.pid, row["process_started_at"] + 1)
            assert executions.recover_interrupted_executions() == 0
            assert delivery_queue.recover_abandoned() == 0
            # Fault at the OS metadata boundary; real PID probing, processes and SQLite remain active.
            monkeypatch.setattr(gateway.status, "get_process_start_time", lambda pid: None)
        else:
            assert row["process_started_at"] is None
        recovered = (executions.recover_interrupted_executions(), delivery_queue.recover_abandoned())
        output, error = owner.communicate("x", timeout=15)
        assert owner.returncode == 0, error
        outcome = json.loads(output)
        assert recovered == (0, 0), outcome
        assert outcome["execution"]["status"] == "completed"
        assert outcome["delivered"] is True
        assert delivery_queue.get_status(row["id"])["status"] == "delivered"
    finally:
        if owner.poll() is None:
            owner.communicate("x", timeout=15)
