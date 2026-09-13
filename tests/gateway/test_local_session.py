"""Fresh sessions use the composed authority, real TurnRunner and local tools."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('kind', ['approval', 'clarify'])
def test_fresh_local_session_owns_execution_across_viewers(tmp_path, kind):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures' / 'local_session_peer.py'), kind],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'receipt.json').read_text())
    assert receipt['human_response'] and receipt['same_agent'] and receipt['detached_pending']
    if kind == 'approval':
        assert receipt['terminal_effect']
    assert receipt['negative_controls'] and receipt['reconnected_identity']
    print(json.dumps(receipt))
