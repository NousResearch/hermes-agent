"""End-to-end: a managed worker that dies at start leaves a diagnosable record and a usable card.

The fixture (``fixtures/kanban_lost_worker_probe.py``) runs the real stack in a disposable home:
a real ``gateway.run`` daemon, the real dispatcher, the real worker client and the real managed
child process, against a real board and the real profile logs. The card requires a skill nobody
installed, so the child dies in its first turn with ``ValueError: kanban_skills_unavailable`` —
the same shape as the orca run observed on 2026-11-05, four times over.

What the operator must be able to see afterwards (t_925a816f):

  * the crash record names the failing step and the exception, not just ``managed_worker_failed``;
  * the profile's errors.log carries the same cause;
  * the card is claimable again without a manual ``hermes sessions discard``.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_a_managed_worker_that_dies_at_start_is_diagnosable_and_leaves_a_usable_card(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'),
               HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_lost_worker_probe.py')],
                            env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt, indent=2))

    # The attempt really ran and really died: nothing here is faked by the dispatcher alone.
    assert receipt['spawned'], receipt
    assert receipt['ran'], receipt

    # 1. The crash record names WHY (step, exception, source line) and carries the child's exit code.
    cause = receipt['cause']
    assert cause and cause['step'] == 'execute' and cause['exc_type'] == 'ValueError', receipt
    assert cause['detail'].startswith('ValueError: kanban_skills_unavailable'), receipt
    assert cause['where'].startswith('session_kanban.py:'), receipt
    assert cause['exit_code'] == 1, receipt
    assert cause['log'].endswith('managed-worker.log'), receipt

    # 2. The booked run error quotes that cause, so the board alone is enough to diagnose it.
    assert 'execute: ValueError: kanban_skills_unavailable' in receipt['run_error'], receipt
    assert 'child exit 1' in receipt['run_error'], receipt

    # 3. The profile's errors.log says the same thing on one line.
    assert 'Managed worker lost: managed_worker_failed' in receipt['errors_log'], receipt
    assert 'execute: ValueError: kanban_skills_unavailable' in receipt['errors_log'], receipt

    # 4. The card is usable again with no manual ``hermes sessions discard``.
    assert receipt['task_status'] == 'ready', receipt
    assert receipt['claimable'] is True, receipt
