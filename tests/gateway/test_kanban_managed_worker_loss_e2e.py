"""End-to-end: a managed worker that dies at start leaves a diagnosable record and a parked card.

The fixture (``fixtures/kanban_lost_worker_probe.py``) runs the real stack in a disposable home:
a real ``gateway.run`` daemon, the real dispatcher, the real worker client and the real managed
child process, against a real board and the real profile logs. The card requires a skill nobody
installed, so the child dies in its first turn with ``ValueError: kanban_skills_unavailable`` —
the same shape as the orca run observed on 2026-11-05, four times over.

What the operator must be able to see afterwards (t_925a816f, t_5c02fad2):

  * the crash record names the failing step and the exception, not just ``managed_worker_failed``;
  * the profile's errors.log carries the same cause;
  * the card is NOT left for another blind retry: the guard parks it ``blocked``/``capability`` on
    the first attempt, with a reason naming the skill the profile could not resolve.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_a_managed_worker_that_dies_at_start_is_diagnosable_and_parks_the_card(tmp_path):
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

    # 1. The attempt is diagnosable. The guard parks the card *before* the loss bookkeeping, so the
    #    board's record is the `blocked` event (assertion 3) and the exception detail (step,
    #    exception, skill, source line, exit code) rides the profile's errors.log. A structured
    #    `worker_lost` cause is only recorded while the card is still `running`, so it is optional
    #    here — but if one exists it must still name the exception.
    cause = receipt['cause']
    if cause is not None:
        assert cause['detail'].startswith('ValueError: kanban_skills_unavailable'), receipt

    # 2. The profile's errors.log names the step, the exception, the skill and the exit code.
    assert 'Managed worker lost: managed_worker_failed' in receipt['errors_log'], receipt
    assert 'execute: ValueError: kanban_skills_unavailable' in receipt['errors_log'], receipt
    assert 'absent-skill-t925' in receipt['errors_log'], receipt
    assert 'at session_kanban.py:' in receipt['errors_log'], receipt
    assert 'child exit 1' in receipt['errors_log'], receipt

    # 3. The card is parked for a human on the FIRST attempt, with the skill named: no blind retry
    #    of a profile that cannot start a worker turn (t_5c02fad2 retried four times).
    assert receipt['task_status'] == 'blocked', receipt
    block = receipt['block']
    assert block and block['kind'] == 'capability', receipt
    assert 'absent-skill-t925' in block['reason'], receipt
    assert "profile 'default'" in block['reason'], receipt
    assert receipt['claimable'] is False, receipt
