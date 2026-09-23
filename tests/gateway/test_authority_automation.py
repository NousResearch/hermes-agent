"""Completion producers must commit to the same FIFO as human input."""
import os
from pathlib import Path
import subprocess
import sys


def _probe(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env['PYTHONPATH'] = str(root)
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/automation_peer.py'),
                             str(tmp_path), mode], env=env, cwd=root, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=200)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def test_terminal_completion_commits_while_ordinary_daemon_model_is_held(tmp_path):
    _probe(tmp_path, 'terminal')


def test_heartbeat_timer_commits_while_ordinary_daemon_model_is_held(tmp_path):
    _probe(tmp_path, 'heartbeat')
