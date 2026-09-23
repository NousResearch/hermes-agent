"""Canonical observers read owner state without bootstrapping legacy services."""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def probe(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/ancillary_reads_peer.py'), mode],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['mode'] == mode and receipt['authenticated_ws']
    print(json.dumps(receipt))


def test_control_snapshot_is_authorized_read_only_owner_projection(tmp_path):
    probe(tmp_path, 'control')


def test_activity_snapshot_requires_current_owner_objects(tmp_path):
    probe(tmp_path, 'activity')
