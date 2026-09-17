"""Real native bootstrap and dashboard login share one canonical worker, not identities."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize('creator', ['native', 'dashboard'])
def test_operator_cross_surface_real_auth_fifo_and_attribution(tmp_path, creator):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    # Do not inherit credentials, plugins, provider configuration, or user state.
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run(
        [sys.executable, str(root / 'tests/gateway/fixtures/operator_cross_surface_peer.py'), creator],
        cwd=root, env=env, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'operator-receipt.json').read_text())
    assert receipt['creator'] == creator
    assert receipt['model_requests'] == 2
    assert receipt['distinct_attribution'] is True
    assert receipt['disconnect_survived'] is True
    assert receipt['negative_boundaries'] is True
