"""Native Stop must end real background effects, not just interrupt the model loop."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.mark.parametrize('persist', [False, True])
def test_native_gui_stop_ends_terminal_effects_and_preserves_neighbors(tmp_path, persist):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    command = [sys.executable, str(root / 'tests/gateway/fixtures/native_stop_process_peer.py')]
    if persist:
        command.append('--persist')
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                            timeout=100, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['process']['persist_on_release'] is persist
    assert receipt['process']['owner_task_id'] == receipt['session_id']
    assert receipt['completed_native']['completion_reason'] == 'killed', receipt
    assert receipt['delayed_exists'] is False, receipt
    assert receipt['same_process_alive'] is False, receipt
    assert receipt['neighbors_completed'] is True, receipt
    assert receipt['completion_consumed'] is True and receipt['request_count'] == 2, receipt
    assert receipt['admissions'] == [{'status': 'terminal', 'outcome': 'interrupted',
                                      'generation': receipt['generation']}], receipt
