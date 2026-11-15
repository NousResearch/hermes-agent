"""A browser pick attaches through the shared owner, copied only from a profile it may reach.

The Webapp's upload route stages a browser file and hands the renderer a ``staged_upload`` receipt
(``hermes_cli/staged_uploads.py``); once the browser is an attach client of the gateway, that
receipt reaches the owner's canonical ``file.attach``, which only knew ``path``/``data_url`` and
answered ``invalid_params`` for every browser file drop.
"""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_a_staged_browser_upload_attaches_only_from_its_own_backend_and_profile(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/staged_file_attach_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert isinstance(receipt['own'], dict) and receipt['own']['attached'] is True, receipt
    assert receipt['own']['ref_text'].startswith('@file:'), receipt
    assert receipt['own_bytes'] == 'BROWSER PAYLOAD', receipt
    # A replacement backend's receipt and a path outside the upload dir are refused by the source
    # checks; a profile the owner does not serve never reaches them.
    assert receipt['foreign_install'] == 'invalid_params', receipt
    assert receipt['outside_uploads'] == 'invalid_params', receipt
    assert receipt['unserved_profile'] == 'permission_denied', receipt
    assert receipt['copied'] == ['BROWSER PAYLOAD'], receipt
