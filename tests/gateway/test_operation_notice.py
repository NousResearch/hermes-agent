"""Operation notices traverse authenticated session controls, never a local adapter ACK."""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_authenticated_notice_is_visible_and_exact_retry_deduplicated(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/operation_notice_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['authenticated_ws'] and receipt['notified']
