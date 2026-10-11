"""Ink ``!cmd`` runs for its own local session, in that session's directory (dokterdok N22)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.mark.platforms("posix")
def test_shell_exec_runs_in_the_session_cwd_behind_the_safety_gates(tmp_path):
    """The owner serves ``shell.exec`` with the ``session_id`` Ink sends: the command runs in the
    session's frozen launch cwd (not the daemon's), its exit code and output come back, the
    owner's credentials are scrubbed from the child, and a dangerous or sessionless command is
    refused."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/shell_exec_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['owner_cwd'] != receipt['session_cwd']
    ran = receipt['ran']
    assert isinstance(ran, dict) and ran.get('code') == 3, receipt
    assert Path(ran['stdout'].strip()).resolve() == Path(receipt['session_cwd']).resolve(), receipt
    assert ran['stderr'] == 'key=', receipt
    assert receipt['dangerous'].get('message') == 'dangerous_command', receipt
    assert receipt['sessionless'] == 4001, receipt
