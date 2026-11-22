"""Desktop's subagent Steer / Stop are served by the shared owner (dokterdok N56)."""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_desktop_subagent_steer_and_stop_reach_only_this_sessions_children(tmp_path):
    """Before: the owner answered -32601 and the legacy fallback ran the sidecar handler, which
    never holds an authority session, so Steer/Stop always read as rejected."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/subagent_controls_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['owned'] == {'steer': {'status': 'queued', 'subagent_id': 'owned', 'text': 'check owned'},
                                'stop': {'found': True, 'subagent_id': 'owned'}}, receipt
    assert receipt['sibling'] == {'steer': {'status': 'rejected', 'subagent_id': 'sibling', 'text': 'check sibling'},
                                  'stop': {'found': False, 'subagent_id': 'sibling'}}, receipt
    assert receipt['delivered'] == {'owned': [['check owned'], 1], 'sibling': [[], 0]}, receipt
    assert receipt['late']['status'] == 'rejected', receipt
