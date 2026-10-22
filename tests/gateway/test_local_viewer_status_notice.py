"""A local viewer sees the agent's status lines and notices, as main's TUI and CLI showed them.

On the shared owner a local route's ``LocalSessionAdapter.send`` publishes nothing, so a warning
status or an ``AgentNotice`` raised during a viewer's turn reached no viewer. Real owner, real
AIAgent turn over a loopback model, authenticated WebSocket viewer, in a disposable process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SHOWN = [['status.update', {'kind': 'warn', 'text': 'PROBE_WARNING_STATUS'}],
         ['status.update', {'kind': 'lifecycle', 'text': 'PROBE_LIFECYCLE_STATUS'}],
         ['notification.show', {'text': 'PROBE_NOTICE', 'level': 'warn', 'kind': 'sticky', 'ttl_ms': None,
                                'key': 'probe', 'id': 'probe'}]]
CLEARED = ['notification.clear', {'key': 'probe'}]


@pytest.mark.parametrize('mode', ['default', 'suppress'])
def test_local_viewer_receives_agent_status_and_notices(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures' / 'local_status_notice_peer.py'), mode],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100, check=False)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'receipt.json').read_text())
    # display.suppress_warning_notifications still hides the diagnostics (warn status, warn notice).
    # The recovery clear always reaches the viewer, as main's TUI emitted it (clearing nothing is a no-op).
    expected = (SHOWN if mode == 'default' else [SHOWN[1]]) + [CLEARED]
    assert receipt['events'] == expected
    assert receipt['cli_stderr'].split() == [payload['text'] for _, payload in expected if 'text' in payload]
