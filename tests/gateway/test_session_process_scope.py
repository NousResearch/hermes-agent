"""Process-global sidecar verbs never reach the shared owner's other sessions (dokterdok N2)."""
import json
from pathlib import Path
import subprocess
import sys

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_stop_is_session_scoped_and_global_verbs_keep_the_owner_refusal(tmp_path):
    """Ink ``/stop`` kills only its own session's processes (a persisted one included); another
    chat's processes, persisted or not, survive. ``delegation.pause``, ``reload.env``,
    and ``agents.list`` keep the authority's -32601 instead of reaching the legacy
    handler that pauses every spawn or rewrites the shared environ."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/process_scope_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['stop'] == {'killed': 2}, receipt
    assert receipt['exited'] == [True, True, False, False], receipt
    assert receipt['sessionless_stop'] == 4001, receipt
    assert receipt['exited_after_sessionless'] == [True, True, False, False], receipt
    assert receipt['pause'] == -32601 and receipt['spawn_paused'] is False, receipt
    assert receipt['reload_env'] == -32601 and receipt['environ_rewritten'] is False, receipt
    # reload.mcp is now served by the owner, profile-scoped (test_desktop_process_verbs.py).
    assert receipt['agents_list'] == -32601, receipt
