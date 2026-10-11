"""/browser connect|disconnect from one TUI or Desktop chat never rewrites the shared owner's
process-wide CDP endpoint or reaps other chats' browsers (dokterdok N25); /browser status and
/browser use still answer. One row per browser.manage action reaching the legacy fallback."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env

_OTHER_CHAT = 'http://127.0.0.1:9333'


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


@pytest.mark.parametrize('action,refused', [
    ('connect', True), ('disconnect', True), ('status', False), ('missing', False)])
def test_browser_manage_on_the_shared_owner_keeps_process_wide_cdp(receipt, action, refused):
    row = receipt[action]
    assert row['cdp'] == _OTHER_CHAT and row['reaps'] == 0, row  # nothing process-wide changed
    if refused:
        assert row['code'] == 4030 and row['result'] is None, row  # refused in words, not -32601
    else:
        assert row['code'] is None and row['result']['url'] == _OTHER_CHAT, row
