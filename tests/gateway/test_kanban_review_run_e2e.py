"""End-to-end: one real review-lane run completes on a disposable home.

The probe (``fixtures/kanban_review_run_probe.py``) boots a real ``gateway.run`` daemon, hands a
card to the ``review`` column and lets the real dispatcher, worker client and managed child run it
against a loopback model. This is the acceptance t_5ae5058d could not meet while ``worker.persist``
died on an unevaluable liveness probe: the run must be booked ``completed``, the card must reach
``done``, and the worker client must exit 0 instead of reporting
``Kanban execution unknown; retained by owner``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_a_review_lane_run_completes_end_to_end(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'),
               HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo))
    # child_env() is a whitelist, so the probe's own negative-control switch rides past it here.
    if os.environ.get('KANBAN_REVIEW_PROBE_MODE'):
        env['REVIEW_PROBE_MODE'] = os.environ['KANBAN_REVIEW_PROBE_MODE']
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_review_run_probe.py')],
                            env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt, indent=2))

    # The review lane really claimed and ran this card: not a no-op dispatcher tick.
    assert len(receipt['spawned']) == 1, receipt
    assert receipt['ran'], receipt
    assert receipt['review_requested'], receipt

    # ... and it finished through the board protocol with a clean worker exit.
    assert receipt['run_outcome'] == 'completed', receipt
    assert not receipt['run_error'], receipt
    assert receipt['task_status'] == 'done', receipt
    assert receipt['failures'] == 0, receipt
    assert tuple(receipt['client_exit']) == ('clean_exit', 0), receipt
