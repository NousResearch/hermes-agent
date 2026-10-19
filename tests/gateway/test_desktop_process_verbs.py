"""Desktop's /stop, per-process Stop, MCP reload and file attach are served by the shared owner (dokterdok N27)."""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_desktop_process_kill_stop_and_mcp_reload_reach_only_this_session(tmp_path):
    """``process.kill`` stops this chat's own process and refuses another chat's; ``process.stop``
    with the resolved session id stops the rest of this chat's; ``reload.mcp`` is served (a
    profile-scoped reconcile), not the -32601 the Desktop reported as "MCP reload failed"."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/desktop_process_verbs_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['kill'] == 'killed', receipt
    # Not ``not_found``, which Desktop's session-gone latch reads as a dead chat (dokterdok N64).
    assert receipt['foreign_kill'] == 'process_not_found', receipt
    assert receipt['stop'] == {'killed': 1, 'delegations': 0}, receipt
    assert receipt['list_fields'] == ['completion_reason', 'exited_at', 'started_epoch'], receipt
    assert receipt['list_ids'] is True, receipt
    # tools.show is served for an owner session (the classic CLI's /tools), not -32601.
    assert set(receipt['tools_show']) == {'sections', 'total'}, receipt
    assert receipt['tools_show']['total'] == sum(len(s['tools']) for s in receipt['tools_show']['sections']), receipt
    # Scoped to THIS session's toolsets (created with toolsets=[]), not the launch profile's catalog.
    assert receipt['tools_show']['total'] == 0, receipt
    assert receipt['skills_reload'] == ['zeta-skill'], receipt
    assert receipt['exited'] == [True, True, False], receipt
    assert receipt['reload_mcp'] == 'reloaded', receipt
    # file.attach: the sidecar answered ``session not found`` for every authority session.
    assert [ref.startswith('@file:') for ref in receipt['file_attach']] == [True, True], receipt
    assert receipt['staged'] == ['hello', 'hi'], receipt
