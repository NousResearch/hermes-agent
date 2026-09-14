"""Clarification response survives viewer detach through the real gateway worker."""
import os
from pathlib import Path
import subprocess
import sys


def test_real_shared_clarify_survives_detach(tmp_path):
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    repo = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures' / 'authority_clarify_peer.py')],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (state / 'clarify-passed').read_text() == 'reply reached real model; stale reply rejected'


def test_expired_clarify_disappears_before_turn_settlement():
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    from tools import clarify_gateway

    events = SessionEvents()
    controls = PendingControls(events)
    entry = clarify_gateway.register('owned-expiring-question', 'owned-expiring-route', 'Choose', ['a', 'b'])
    try:
        controls.register_clarify('owned-session', 1, entry)
        assert controls.snapshot('owned-session', 1)
        assert clarify_gateway.wait_for_response(entry.clarify_id, timeout=.01) is None
        assert controls.snapshot('owned-session', 1) == ()
        assert controls.respond('owned-session', 1, entry.clarify_id, {'answer': 'a'}, kind='clarify')['status'] == 'already_resolved'
    finally:
        clarify_gateway.clear_session(entry.session_key)
