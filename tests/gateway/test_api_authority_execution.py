"""API and authenticated WS share the existing selected cached agent."""
import json
import os
from pathlib import Path
import subprocess
import sys


def _probe(tmp_path, *, advanced=False, runs=False):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    if advanced:
        env['API_ADVANCED'] = '1'
    if runs:
        env['API_RUN'] = str(runs)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'api_authority_peer.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'receipt.json').read_text())
    assert receipt['same_agent'], receipt
    print(json.dumps(receipt))


def test_api_and_ws_use_one_turnrunner(tmp_path):
    _probe(tmp_path)
