"""A served secondary profile's managed worker never inherits launch-profile residue."""
import os
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize('target_session', [None, 'fixture-target-env-session'])
def test_launch_scrub_preserves_constructed_context_then_target_overlay(tmp_path, monkeypatch, target_session):
    from agent.secret_scope import is_multiplex_active
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import scoped_current_session_id
    from gateway.session_managed_worker import _worker_env
    root = tmp_path / 'root'
    target = root / 'profiles' / 'member'
    target.mkdir(parents=True)
    (root / '.env').write_text('HERMES_SESSION_ID=fixture-launch-session\nLAUNCH_CREDENTIAL=fixture-launch-only\n')
    (target / '.env').write_text('TARGET_CREDENTIAL=fixture-target-only\n' +
        (f'HERMES_SESSION_ID={target_session}\n' if target_session else ''))
    monkeypatch.setattr(os, 'environ', {'PATH': os.defpath, 'HOME': str(tmp_path), 'HERMES_HOME': str(root),
        'HERMES_SESSION_ID': 'fixture-launch-session', 'LAUNCH_CREDENTIAL': 'fixture-launch-only',
        'LANG': 'C.UTF-8', 'HERMES_GATEWAY_LOCK_DIR': 'fixture-lock-dir'})
    before = dict(os.environ)
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(target), scoped_current_session_id('fixture-owned-session'):
            env = _worker_env(SimpleNamespace(profile_id=str(target)))
    finally:
        set_multiplex_active(previous)
    assert env['HERMES_SESSION_ID'] == (target_session or 'fixture-owned-session')
    assert env['TARGET_CREDENTIAL'] == 'fixture-target-only'
    assert 'LAUNCH_CREDENTIAL' not in env
    assert env['HERMES_HOME'] == str(target)
    assert env['LANG'] == before['LANG']
    assert env['HERMES_GATEWAY_LOCK_DIR'] == before['HERMES_GATEWAY_LOCK_DIR']
    assert os.environ == before
