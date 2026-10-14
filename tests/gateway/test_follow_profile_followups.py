"""follow_profile_config follow-ups (andrexibiza C14-C18, C03): one profile read per turn, explicit
create-time choices over followed defaults, a held pick still follows the rotated key, followed secrets
are never recovered from the frozen receipt, and the profile's toolsets are followed. Real SessionStore /
SessionAuthority and a real config.yaml; ``_load_gateway_config`` is only wrapped, never replaced."""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_follow_profile_config import _create, _owner, _route


async def _create_bot_chat(owner, authority, home, request_id, **extra):
    """Desktop's Bot Chat create: no toolsets param (the surface's own selection)."""
    response = await owner.dispatch({'id': request_id, 'method': 'session.create', 'params': dict(
        request_id=request_id, source='gui', cwd=str(home), follow_profile_config=True, **extra)})
    assert 'result' in response, response
    return authority.sessions[response['result']['session_id']]


def _profile(home, model, port, key, **extra):
    (home / 'config.yaml').write_text(json.dumps({'model': {
        'provider': 'custom', 'base_url': f'http://127.0.0.1:{port}/v1', 'default': model, 'api_key': key}, **extra}))


@pytest.fixture
def home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


@pytest.mark.asyncio
async def test_rotated_key_reaches_follow_and_held_pick_on_a_fresh_owner_and_routes_are_never_mixed(home, monkeypatch):
    from gateway import run
    from gateway.session_policy import policy_for_source, policy_scope
    _profile(home, 'old', 1, 'k-old')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        follow = await _create(owner, authority, home, 'follow')
        pick = await _create(owner, authority, home, 'pick', model='picked')
        _profile(home, 'old', 1, 'k-new')  # rotate only the key
        assert _route(runner, follow) == ('old', 'http://127.0.0.1:1/v1', 'k-new')
        # C15: the pick keeps its model and follows the rotated key, like the no-pick control.
        assert _route(runner, pick) == ('picked', 'http://127.0.0.1:1/v1', 'k-new')
        # C14: a fresh owner (no in-memory config secrets) enters both sessions' scopes and resolves.
        authority._local_config_secrets = {}
        for live in (follow, pick):
            with policy_scope(policy_for_source(runner, live.source), authority=authority):
                assert _route(runner, live)[2] == 'k-new'
        # C18: the profile moves A -> B between two reads of one resolution (the file the credential pool
        # reads stays A): the route is whole A or whole B, never B's model on A's endpoint and key.
        _profile(home, 'a', 3, 'key-a')
        real, reads = run._load_gateway_config, []
        def moved_after_first_read(path=None):
            reads.append(path)
            if len(reads) == 1:
                return real(path)
            return {'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:4/v1', 'default': 'b', 'api_key': 'key-b'}}
        monkeypatch.setattr(run, '_load_gateway_config', moved_after_first_read)
        assert _route(runner, follow) in {('a', 'http://127.0.0.1:3/v1', 'key-a'), ('b', 'http://127.0.0.1:4/v1', 'key-b')}
        monkeypatch.setattr(run, '_load_gateway_config', real)
        # Inside a turn every resolution uses the turn's one read, whatever the file says meanwhile.
        _profile(home, 'a', 5, 'key-a')
        with policy_scope(policy_for_source(runner, follow.source), authority=authority):
            _profile(home, 'b', 6, 'key-b')
            assert _route(runner, follow) == ('a', 'http://127.0.0.1:5/v1', 'key-a')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_explicit_create_choices_win_and_session_info_reports_the_next_turn(home):
    from gateway.session_local import local_session_info
    from gateway.session_policy import policy_for_source
    _profile(home, 'old', 1, 'k', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        plain = await _create(owner, authority, home, 'plain')
        explicit = await _create(owner, authority, home, 'explicit', reasoning='high', service_tier='priority')
        _profile(home, 'old', 1, 'k', agent={'reasoning_effort': 'minimal', 'service_tier': 'normal'})
        # C16: explicit create-time reasoning/tier are applied after the followed defaults.
        policy = policy_for_source(runner, explicit.source)
        assert policy.current_reasoning_config(runner)['effort'] == 'high'
        assert policy.current_service_tier(runner, None) == 'priority'
        # C17: lazy session.info reports what the next turn resolves, for both sessions.
        for live, effort, tier in ((explicit, 'high', 'priority'), (plain, 'minimal', '')):
            policy = policy_for_source(runner, live.source)
            info = local_session_info(authority, _ref(authority, live))
            assert (info['reasoning_effort'], info['service_tier']) == (effort, tier)
            assert policy.current_reasoning_config(runner)['effort'] == effort
            assert (policy.current_service_tier(runner, None) or '') == tier
    finally:
        await owner.close()
        store.close_all_db_handles()


def _ref(authority, live):
    sid = next(sid for sid, entry in authority.sessions.items() if entry is live)
    return SimpleNamespace(session_id=sid)


@pytest.mark.asyncio
async def test_follow_session_runs_the_profiles_current_toolsets_and_a_frozen_one_does_not(home):
    """C03 (main #124211): ``hermes tools disable`` reaches a following Bot Chat's next turn."""
    from gateway.session_policy import active_policy, policy_for_source, policy_scope
    _profile(home, 'old', 1, 'k')
    store, runner, authority, owner = _owner(home)
    try:
        follow = await _create_bot_chat(owner, authority, home, 'follow-tools')
        pinned = await _create_bot_chat(owner, authority, home, 'pinned-tools', toolsets=['web'])
        before = {}
        for name, live in (('follow', follow), ('pinned', pinned)):
            with policy_scope(policy_for_source(runner, live.source), authority=authority):
                before[name] = set(active_policy().toolsets)
        assert 'web' in before['follow']
        _profile(home, 'old', 1, 'k', agent={'disabled_toolsets': ['web']})
        policy = policy_for_source(runner, follow.source)
        with policy_scope(policy, authority=authority):
            assert set(active_policy().toolsets) == before['follow'] - {'web'}
        # The toolsets the turn's agent is built with (``_run_agent_display_settings``).
        assert set(policy.current_toolsets(policy.config(authority))) == before['follow'] - {'web'}
        with policy_scope(policy_for_source(runner, pinned.source), authority=authority):
            assert set(active_policy().toolsets) == before['pinned'] == {'web'}
    finally:
        await owner.close()
        store.close_all_db_handles()
