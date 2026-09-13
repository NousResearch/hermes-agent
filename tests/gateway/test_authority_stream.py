"""Canonical execution stream through production API binding and real model SSE."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["ordinary", pytest.param("blocked", marks=pytest.mark.linux_only)])
def test_authority_stream_has_one_order_and_rejects_retired_callbacks(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo), STREAM_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'authority_stream_peer.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'receipt.json').read_text())
    print(json.dumps(receipt))
    assert receipt['real_terminal_effect'] and receipt['same_stamps'] and receipt['stale_callback_inert']
