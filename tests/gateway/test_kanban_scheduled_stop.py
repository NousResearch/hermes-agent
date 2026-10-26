"""A Stop of idle scheduled work must not wait for the next attempt to cancel its jobs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.mark.platforms('linux', 'macos')
@pytest.mark.parametrize('stop_later', [False, True], ids=['planned-retirement', 'idle-stop-before-resume'])
def test_idle_stop_reaps_scheduled_job_before_any_resume(tmp_path, stop_later):
    root = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(tmp_path / 'user'), USERPROFILE=str(tmp_path / 'user'),
        HERMES_HOME=str(tmp_path / 'state'), HERMES_KANBAN_HOME=str(tmp_path / 'state'),
        HERMES_PROFILE='default', PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    probe = root / 'tests/gateway/fixtures/kanban_scheduled_stop_probe.py'
    result = subprocess.run([sys.executable, str(probe), str(int(stop_later)), str(root)],
                            cwd=root, env=env, text=True, capture_output=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['background_alive_after_planned_worker_exit']
    assert receipt['effect_after_gate'] is (not stop_later)
    if stop_later:
        assert receipt['no_resume_stop_proof']
        assert not receipt['background_alive_after_explicit_stop']
        assert not receipt['background_alive_at_successor_admission']
