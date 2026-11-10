"""Local automation belongs to the owner, never an attached client."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _probe(tmp_path, source):
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env['PYTHONPATH'] = str(root)
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/local_automation_peer.py'),
        str(tmp_path), source], cwd=root, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui'])
def test_local_completion_without_an_observer_is_a_note_the_next_turn_sees(tmp_path, source):
    _probe(tmp_path, source)  # #134889 LC-7: no unattended turn


def test_local_watch_match_without_an_observer_is_a_note(tmp_path):
    _probe(tmp_path, 'watch')


def test_local_completion_survives_owner_kill_without_duplicate(tmp_path):
    _probe(tmp_path, 'restart')
