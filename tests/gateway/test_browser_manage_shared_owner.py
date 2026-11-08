"""/browser connect|disconnect through the real WS legacy fallback (dokterdok N25, #134889 CLI-9).

Ink, Desktop and the classic CLI all send /browser here. On an owner serving one profile or several,
connect/disconnect run in the session's profile: the live endpoint and the browser reap are that
profile's, so a browser one profile connects never reaches another profile's sessions."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.fixture(scope='module')
def receipt(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    tmp = tmp_path_factory.mktemp('browser-scope')
    home, user = tmp / 'state', tmp / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/browser_scope_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((home / 'receipt.json').read_text())


@pytest.mark.parametrize('phase', ['one', 'two'])
def test_connect_and_disconnect_are_the_sessions_profiles_alone(receipt, phase):
    """One served profile or several: connect publishes THIS profile's endpoint and reaps only its
    browsers; the operator's env endpoint and the other profile's own live connection stay."""
    url, key = f"http://127.0.0.1:{receipt['one:connect']['cdp_port']}", receipt['home_key']
    # The other profile keeps its own live connection, or its own browser.cdp_url; never this
    # profile's, and never the launch process's BROWSER_CDP_URL (that is this profile's env).
    bee = 'http://127.0.0.1:9444' if phase == 'two' else 'ws://bee.invalid/devtools/browser/PROFILE-B'
    connect = receipt[f'{phase}:connect']
    assert connect['code'] is None and connect['result'] == {'connected': True, 'url': url}, connect
    assert connect['own'] == url and connect['reaps'] == [key, key], connect  # never 'ALL'
    assert connect['env'] == 'http://127.0.0.1:9333' and connect['bee'] == bee, connect
    assert receipt[f'{phase}:status']['result']['url'] == url
    disconnect = receipt[f'{phase}:disconnect']
    assert disconnect['code'] is None and disconnect['result'] == {'connected': False}, disconnect
    assert disconnect['own'] is None and disconnect['reaps'] == [key] * 4, disconnect
    assert disconnect['env'] == 'http://127.0.0.1:9333' and disconnect['bee'] == bee, disconnect
    # Back on the operator's env endpoint (a missing action means status).
    assert receipt[f'{phase}:missing']['result']['url'] == 'http://127.0.0.1:9333'
