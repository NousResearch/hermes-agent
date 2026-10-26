"""Native block ends registered background work before a replacement executes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.mark.platforms('linux', 'macos')
@pytest.mark.parametrize('persist', [False, True])
@pytest.mark.parametrize('early', [False, True], ids=['after-worker-exit', 'immediate-resume'])
def test_native_kanban_stop_reaps_background_before_successor(tmp_path, persist, early):
    root = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(tmp_path / 'user'), USERPROFILE=str(tmp_path / 'user'),
        HERMES_HOME=str(tmp_path / 'state'), HERMES_KANBAN_HOME=str(tmp_path / 'state'),
        HERMES_PROFILE='default', PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    probe = root / 'tests/gateway/fixtures/kanban_background_stop_probe.py'
    result = subprocess.run([sys.executable, str(probe), str(int(persist)), str(root), str(int(early))],
                            cwd=root, env=env, text=True, capture_output=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert not receipt['background_alive_after_workers_gone']
    assert not receipt['background_alive_at_successor_admission']
    assert not receipt['old_effect_after_stop']
    assert receipt['task_status'] == 'done'
