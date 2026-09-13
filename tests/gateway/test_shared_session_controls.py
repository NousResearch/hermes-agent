"""Shared control behavior through real execution and transport boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_shared_approval_outlives_viewer_and_bypasses_fifo(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'authority_controls_peer.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((state / 'receipt.json').read_text())
    print(json.dumps(receipt))
    assert receipt['real_terminal_effect'] and receipt['fifo_bypassed'] and receipt['detach_kept_pending']


def test_control_identity_rejects_foreign_and_retired_workers(tmp_path, monkeypatch):
    import asyncio
    import runpy
    fixture = runpy.run_path(str(Path(__file__).parent / 'fixtures' / 'control_boundaries.py'))
    asyncio.run(fixture['exercise_control_boundaries'](tmp_path, monkeypatch))
