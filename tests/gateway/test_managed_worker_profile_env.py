"""A served secondary profile's managed worker never inherits launch-profile residue."""
import os
from types import SimpleNamespace

from agent.secret_scope import set_multiplex_active


def test_secondary_worker_env_drops_launch_terminal_policy_and_env_settings(tmp_path, monkeypatch):
    """Same contract as the cron and kanban dispatchers: a worker for served profile X gets the
    env a standalone ``hermes -p X`` gateway would build — X's HERMES_HOME and X's ``.env`` over a
    base without the launch profile's ``TERMINAL_*`` policy or launch-only ``.env`` settings."""
    from gateway.session_managed_worker import _worker_env
    root = tmp_path / '.hermes'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (root / '.env').write_text('HERMES_MODEL=launch-model\nTERMINAL_ENV=docker\n')
    (beta / '.env').write_text('BETA_ONLY_TOKEN=beta-secret\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_MODEL', 'launch-model')
    monkeypatch.setenv('TERMINAL_ENV', 'docker')
    monkeypatch.setenv('TERMINAL_CWD', str(root))
    set_multiplex_active(True)
    try:
        env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert env['HERMES_HOME'] == str(beta)
    assert env['BETA_ONLY_TOKEN'] == 'beta-secret'
    assert not {k for k in env if k.startswith('TERMINAL_')}, env
    assert 'HERMES_MODEL' not in env


def test_launch_scrub_runs_before_the_turn_context_is_injected(tmp_path, monkeypatch):
    """The launch residue strip runs on the RAW environ, before ``build_subprocess_env`` bridges this
    turn's own session context: a launch ``.env`` that names ``HERMES_SESSION_ID`` must not erase the
    id derived for the secondary's turn (#109338 ordering)."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import scoped_current_session_id
    from gateway.session_managed_worker import _worker_env
    root = tmp_path / '.hermes'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (root / '.env').write_text('HERMES_SESSION_ID=launch-session\nLAUNCH_ONLY=launch-value\n')
    (beta / '.env').write_text('BETA_ONLY_TOKEN=beta-secret\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_SESSION_ID', 'launch-session')
    monkeypatch.setenv('LAUNCH_ONLY', 'launch-value')
    before = dict(os.environ)
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(beta), scoped_current_session_id('owned-session'):
            env = _worker_env(SimpleNamespace(profile_id=str(beta)))
    finally:
        set_multiplex_active(False)
    assert env['HERMES_SESSION_ID'] == 'owned-session'
    assert 'LAUNCH_ONLY' not in env
    assert env['BETA_ONLY_TOKEN'] == 'beta-secret'
    assert dict(os.environ) == before
