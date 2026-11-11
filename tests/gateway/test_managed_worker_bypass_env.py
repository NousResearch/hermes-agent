"""Bypass policy must reach the parent's real env builder before managed-worker spawn."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


class SpawnCaptured(Exception):
    """Transport sentinel: no worker or ownership acknowledgement is fabricated."""


@pytest.fixture
def worker_owner(tmp_path, monkeypatch):
    import hermes_constants as hc
    from agent import secret_scope

    launch = tmp_path / '.hermes'
    real = tmp_path / 'real'
    real.mkdir()
    launch.mkdir()
    (launch / '.env').write_text('LAUNCH_PRIVATE=launch-only\nOPENAI_API_KEY=launch-provider\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(hc, '_PINNED_PROCESS_HERMES_HOME', str(launch))
    for key in ('TMPDIR', 'TMP', 'TEMP', 'HERMES_SCRATCH_DIR'):
        monkeypatch.delenv(key, raising=False)
    for key, value in {'HOME': str(real), 'HERMES_REAL_HOME': str(real), 'HERMES_HOME': str(launch),
                       'TERMINAL_HOME_MODE': 'auto', 'LAUNCH_PRIVATE': 'launch-only',
                       'OPENAI_API_KEY': 'launch-provider',
                       'AUXILIARY_VISION_API_KEY': 'internal-never-forward'}.items():
        monkeypatch.setenv(key, value)
    multiplex = secret_scope.is_multiplex_active()
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(launch),
                                                profile_home=str(launch))
    secret_scope.set_multiplex_active(True)
    try:
        yield launch, real
    finally:
        secret_scope.set_multiplex_active(multiplex)
        secret_scope.reset_secret_scope(secret_token)


def _exercise_spawn(monkeypatch, worker_owner, *, routed, policy, yaml, outer, captured):
    import hermes_constants as hc
    from agent import secret_scope
    from gateway import session_managed_worker as smw
    from tools.terminal_scope import get_terminal_scope, reset_terminal_scope, set_terminal_scope

    launch, real = worker_owner
    target = launch / 'profiles' / 'worker' if routed else launch
    (target / 'home').mkdir(parents=True)
    if routed:
        (target / '.env').write_text('OPENAI_API_KEY=target-provider\nTARGET_PRIVATE=target-only\n')
    (target / 'config.yaml').write_text(yaml)
    authority = SimpleNamespace(profile_id=str(target))
    ref = SimpleNamespace(session_id='sid')
    row = {'generation': 1, 'payload': {'text': 'plain recovery message'}}
    def capture(argv, **kwargs):
        captured.append((argv, kwargs))
        raise SpawnCaptured('transport reached')

    monkeypatch.setattr(smw.subprocess, 'Popen', capture)
    home_token = hc.set_hermes_home_override(str(target))
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(target),
                                                profile_home=str(target))
    token = set_terminal_scope(outer)
    before = dict(os.environ)
    try:
        try:
            asyncio.run(smw.execute_managed(authority, ref, row, policy))
        finally:
            assert get_terminal_scope() is outer
            assert hc.get_hermes_home() == target
            assert dict(os.environ) == before
            assert authority._managed_spawning == {}
    finally:
        reset_terminal_scope(token)
        secret_scope.reset_secret_scope(secret_token)
        hc.reset_hermes_home_override(home_token)
    return captured, target, real


@pytest.mark.parametrize('mode', ['safe', 'config'])
@pytest.mark.parametrize('routed', [False, True], ids=['launch-multiplex', 'routed'])
@pytest.mark.parametrize('yaml_kind', ['malformed', 'opposing-home'])
@pytest.mark.parametrize('outer_refusal', [False, True], ids=['defaults', 'outer-refusal'])
def test_bypass_parent_env_reaches_transport(tmp_path, monkeypatch, worker_owner,
                                             mode, routed, yaml_kind, outer_refusal):
    import hermes_constants as hc
    from gateway.session_local import _bypass_policy
    from tools.terminal_scope import TerminalPolicyRefusal, default_terminal_scope

    policy = _bypass_policy({'cwd': str(tmp_path), 'model': 'safe-fixture', 'provider': 'custom',
                             'base_url': 'http://127.0.0.1:9/v1',
                             'safe_mode': mode == 'safe', 'ignore_user_config': True}, private_secrets={})
    frozen = policy.terminal_json
    assert json.loads(frozen).get('TERMINAL_HOME_MODE', 'auto') == 'auto'
    # auto is profile HOME in a container, real HOME on a native host. Ignored YAML opposes it.
    yaml = ('terminal: [unterminated\n' if yaml_kind == 'malformed' else
            f'terminal: {{home_mode: {"real" if hc.is_container() else "profile"}}}\n')
    outer = TerminalPolicyRefusal('outer profile unavailable') if outer_refusal else default_terminal_scope()
    captured = []
    # The real env construction must finish before the transport sentinel is raised.
    with pytest.raises(SpawnCaptured, match='transport reached'):
        _exercise_spawn(monkeypatch, worker_owner, routed=routed, policy=policy,
                        yaml=yaml, outer=outer, captured=captured)
    assert len(captured) == 1
    argv, kwargs = captured[0]
    assert argv[1:] == ['-m', 'agent.managed_worker']
    child = kwargs['env']
    launch, real = worker_owner
    target = launch / 'profiles' / 'worker' if routed else launch
    assert child['HOME'] == str(target / 'home' if hc.is_container() else real)
    assert child['HERMES_HOME'] == str(target)
    assert child['HERMES_REAL_HOME'] == str(real)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        assert child[key] == str(target / 'cache' / 'scratch')
    assert 'AUXILIARY_VISION_API_KEY' not in child
    assert child['OPENAI_API_KEY'] == ('target-provider' if routed else 'launch-provider')
    if routed:
        assert 'LAUNCH_PRIVATE' not in child
        assert child['TARGET_PRIVATE'] == 'target-only'
    assert policy.terminal_json == frozen


@pytest.mark.parametrize('routed', [False, True], ids=['launch-multiplex', 'routed'])
def test_ordinary_parent_env_still_refuses_malformed_yaml(tmp_path, monkeypatch, worker_owner, routed):
    from gateway.session_policy import build_policy
    from tools.terminal_scope import TerminalPolicyUnavailable, default_terminal_scope

    policy = build_policy({'cwd': str(tmp_path), 'model': 'ordinary'}, {}, profile_terminal=False)
    captured = []
    with pytest.raises(TerminalPolicyUnavailable, match='config.yaml'):
        _exercise_spawn(monkeypatch, worker_owner, routed=routed, policy=policy,
                        yaml='terminal: [unterminated\n', outer=default_terminal_scope(), captured=captured)
    assert captured == []
