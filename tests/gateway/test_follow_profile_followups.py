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


async def _pick_reasoning(owner, authority, live, effort):
    """Desktop's reasoning selector, Ink's and the classic CLI's ``/reasoning <level>``: the owner's verb."""
    sid = _ref(authority, live).session_id
    row = authority.db.get_session(sid)
    reply = await owner.dispatch({'id': 'r', 'method': 'session.mutate', 'params': {
        'session_id': sid, 'request_id': 'reasoning-' + effort, 'operation': 'reasoning', 'payload': {'effort': effort},
        'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation']}})
    assert reply['result']['effort'] == effort, reply
    return sid


@pytest.mark.asyncio
async def test_a_session_reasoning_pick_wins_over_the_followed_default_after_a_restart(home):
    """Desktop's reasoning picker and ``/reasoning <level>`` on a Bot Chat (created without ``reasoning``):
    the acknowledged level is the next turn's, through a profile edit and a fresh owner; an unpicked
    following chat still runs the profile's current level."""
    from gateway.session_local import local_session_info
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import policy_for_source
    _profile(home, 'm', 1, 'k', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda key, value: None
    try:
        picked = await _create_bot_chat(owner, authority, home, 'picked')
        plain = await _create_bot_chat(owner, authority, home, 'plain')
        sid = await _pick_reasoning(owner, authority, picked, 'high')
        _profile(home, 'm', 1, 'k', agent={'reasoning_effort': 'minimal'})
        for live, effort in ((picked, 'high'), (plain, 'minimal')):
            info = local_session_info(authority, _ref(authority, live))
            assert (policy_for_source(runner, live.source).current_reasoning_config(runner)['effort'],
                    info['reasoning_effort']) == (effort, effort)
    finally:
        await owner.close()
        store.close_all_db_handles()
    store, runner, authority, owner = _owner(home)  # a fresh owner: only the durable receipt carries the pick
    runner._peek_session_state = lambda key: None
    try:
        restore_local_session(authority, sid)
        assert policy_for_source(runner, authority.sessions[sid].source).current_reasoning_config(runner)['effort'] == 'high'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('method,params,expected', [
    ('config.set', {'key': 'reasoning', 'value': 'minimal', 'scope': 'global'}, 'minimal'),
    ('slash.exec', {'command': '/reasoning minimal'}, 'minimal'),
    ('config.set', {'key': 'reasoning', 'value': 'reset'}, 'medium'),
    ('slash.exec', {'command': '/reasoning reset'}, 'medium'),
], ids=['config.set-global', 'slash.exec-level', 'config.set-reset', 'slash.exec-reset'])
async def test_a_later_reasoning_command_replaces_the_pick_after_a_restart(home, tmp_path, monkeypatch, method, params,
                                                                          expected):
    """After a pick, ``/reasoning <level> [--global]`` or ``/reasoning reset`` from Desktop, Ink or
    ``slash.exec`` is the session's level from then on: a fresh owner and a resume keep the command's level,
    and a reset follows the profile (here moved to ``medium``), on a Bot Chat and on a frozen session."""
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import policy_for_source
    from gateway.session_policy_refresh import commit_refresh

    from gateway import session_local

    async def reply(event):  # the messaging handler: its reply and --global's config.yaml write
        return 'ok'
    monkeypatch.setattr(session_local, 'authorize_local_source', lambda runner, source: True)
    monkeypatch.chdir(tmp_path)  # the commands' profile scope is ``_owner``'s relative profile id
    _profile(home, 'm', 1, 'k', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda key, value: None
    runner._apply_reasoning_selection = lambda *args, **kwargs: 'ok'
    runner._command_handler_table = lambda names: {'reasoning': reply}
    try:
        follow = await _create_bot_chat(owner, authority, home, 'follow')
        created = await owner.dispatch({'id': 'c', 'method': 'session.create', 'params': {
            'request_id': 'frozen', 'source': 'gui', 'cwd': str(home)}})
        sids = [await _pick_reasoning(owner, authority, live, 'high')
                for live in (follow, authority.sessions[created['result']['session_id']])]
        for sid in sids:
            answer = await owner.dispatch({'id': 'x', 'method': method, 'params': {'session_id': sid, **params}})
            assert 'result' in answer, answer
    finally:
        await owner.close()
        store.close_all_db_handles()
    _profile(home, 'm', 1, 'k', agent={'reasoning_effort': 'medium'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda key, value: None
    try:
        for sid in sids:
            restore_local_session(authority, sid)
            await commit_refresh(authority, SessionRef(authority.profile_id, sid))  # the resume's re-read
        assert [policy_for_source(runner, authority.sessions[sid].source).current_reasoning_config(runner)['effort']
                for sid in sids] == [expected, expected]
    finally:
        await owner.close()
        store.close_all_db_handles()


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
