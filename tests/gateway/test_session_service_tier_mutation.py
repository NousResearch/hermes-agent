"""Retained speed controls share the policy receipt and next-turn projection."""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_policy_verbs import _mutation, _owner


@pytest.fixture
def session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    from gateway.session_local import create_local_session
    from gateway.session_state import SessionState
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-only')
    store, db, authority, owner, _ = _owner(tmp_path, monkeypatch, None)
    profile = {'model': {'default': 'gpt-6.1-sol', 'provider': 'openai-api',
                         'base_url': 'https://api.openai.com/v1', 'api_key': 'fixture-only'},
               'agent': {'reasoning_effort': 'low', 'service_tier': 'normal'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: json.loads(json.dumps(profile)))
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    runner = authority.runner
    runner.session_authority = authority
    runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
    states = {}
    runner._session_state = lambda route: states.setdefault(route, SessionState())
    runner._peek_session_state = states.get
    for name in ('_set_session_reasoning_override', '_set_session_service_tier_override'):
        setattr(runner, name, getattr(GatewayConfigLoadersMixin, name).__get__(runner))
    ref = create_local_session(authority, owner.actor, dict(
        request_id='speed', source='gui', cwd=str(tmp_path), toolsets=[], follow_profile_config=True))
    live = authority.sessions[ref.session_id]
    model, runtime = runner._resolve_session_agent_runtime(source=live.source, session_key=live.route)
    assert (model, runtime['provider'], runtime['base_url']) == (
        'gpt-6.1-sol', 'openai-api', 'https://api.openai.com/v1')
    return store, db, authority, owner, ref, profile


def _info(authority, ref):
    from gateway.session_local import local_session_info
    return local_session_info(authority, ref)


@pytest.mark.asyncio
async def test_app_server_refuses_unsendable_tier_without_committing(session, monkeypatch):
    store, db, authority, owner, ref, _profile = session
    monkeypatch.setattr(authority.runner, '_resolve_session_agent_runtime', lambda **kwargs: (
        'gpt-6.1-sol', {'provider': 'openai-codex', 'base_url': 'https://chatgpt.com/backend-api/codex',
                        'api_mode': 'codex_app_server'}))
    monkeypatch.setattr('agent.codex_service_tiers.ultrafast_offered', lambda *args, **kwargs: True)
    try:
        before = db.get_session(ref.session_id)
        refused = await owner.dispatch(_mutation(db, ref, 'unsendable', 'service_tier', {'service_tier': 'ultrafast'}))
        assert refused['error']['data']['reason'] == 'fast_unavailable', refused
        assert db.get_session(ref.session_id) == before
        assert not db.list_meta_prefix('gateway.mutation.v1.')
        # The app-server does have a word for priority; do not block its supported fast control.
        applied = await owner.dispatch(_mutation(db, ref, 'supported', 'service_tier', {'service_tier': 'fast'}))
        assert applied['result']['service_tier'] == 'priority', applied
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_tier_receipt_retry_cold_restore_fork_and_reset(session):
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import policy_for_source
    store, db, authority, owner, ref, profile = session
    runner, live = authority.runner, authority.sessions[ref.session_id]
    try:
        db.append_message(ref.session_id, role='user', content='cached prefix')
        db._execute_write(lambda conn: conn.execute('UPDATE sessions SET system_prompt=? WHERE id=?',
                                                     ('frozen system prompt', ref.session_id)))
        before = db.get_messages(ref.session_id)
        prompt = db.get_session(ref.session_id)['system_prompt']
        launch_request = policy_for_source(runner, live.source).request_json
        request = _mutation(db, ref, 'speed-1', 'service_tier', {'service_tier': 'cold'})
        runner._set_session_service_tier_override(live.route, 'priority')
        applied = await owner.dispatch(request)
        assert 'result' in applied, applied
        assert applied['result']['service_tier'] == 'cold'
        revision = db.get_session(ref.session_id)['runtime_revision']
        assert (await owner.dispatch(request))['result'] == applied['result']
        assert db.get_session(ref.session_id)['runtime_revision'] == revision
        assert _info(authority, ref)['service_tier'] == 'cold'
        resumed = await owner.dispatch({'id': 'resume', 'method': 'session.resume',
                                        'params': {'session_id': ref.session_id}})
        assert resumed['result']['info']['service_tier'] == 'cold'
        assert db.get_messages(ref.session_id) == before
        assert db.get_session(ref.session_id)['system_prompt'] == prompt
        # Followed profile edits cannot erase a committed session pick.
        profile['agent']['service_tier'] = 'normal'
        authority.sessions.pop(ref.session_id)
        runner.adapters[live.source.platform].policies.pop(live.source.chat_id)
        restore_local_session(authority, ref.session_id)
        assert _info(authority, ref)['service_tier'] == 'cold'
        branch = await owner.dispatch(_mutation(db, ref, 'fork', 'branch', {}))
        child = SessionRef(authority.profile_id, branch['result']['branched_session_id'])
        assert _info(authority, child)['service_tier'] == 'cold'
        changed = await owner.dispatch(_mutation(db, child, 'child-speed', 'service_tier', {'service_tier': 'auto'}))
        assert 'result' in changed, changed
        assert _info(authority, child)['service_tier'] == 'auto'
        assert _info(authority, ref)['service_tier'] == 'cold'
        reset = await owner.dispatch(_mutation(db, ref, 'reset', 'reset', {}))
        assert 'result' in reset, reset
        policy = policy_for_source(runner, authority.sessions[ref.session_id].source)
        assert policy.current_service_tier(runner, live.route) == 'cold'
        assert policy.request_json == launch_request
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_tier_refusal_is_atomic_and_normal_can_disable_on_unsupported_route(session, monkeypatch):
    store, db, authority, owner, ref, profile = session
    try:
        # Resolve the followed route, not the obsolete frozen creation route.
        profile['model']['base_url'] = 'https://proxy.example/v1'
        for value, reason in [('fast', 'fast_unavailable'), ('auto', 'fast_unavailable'),
                              ('cold', 'fast_unavailable'), ('ultrafast', 'fast_unavailable'),
                              ('typo', 'invalid_params'), (True, 'invalid_params')]:
            before = db.get_session(ref.session_id)
            result = await owner.dispatch(_mutation(db, ref, str(value), 'service_tier', {'service_tier': value}))
            assert result['error']['data']['reason'] == reason, result
            assert db.get_session(ref.session_id) == before
            assert not db.list_meta_prefix('gateway.mutation.v1.')
        result = await owner.dispatch(_mutation(db, ref, 'disable', 'service_tier', {'service_tier': 'normal'}))
        assert 'result' in result, result
        assert _info(authority, ref)['service_tier'] == ''
        profile['model']['base_url'] = 'https://api.openai.com/v1'
        monkeypatch.setattr('agent.codex_service_tiers.ultrafast_offered', lambda *a, **k: False)
        before = db.get_session(ref.session_id)
        result = await owner.dispatch(_mutation(db, ref, 'no-entitlement', 'service_tier', {'service_tier': 'ultrafast'}))
        assert result['error']['data']['reason'] == 'fast_unavailable'
        assert db.get_session(ref.session_id) == before
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_followed_account_eligibility_is_owner_scoped_and_other_session_is_unchanged(session, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home
    from gateway.session_local import create_local_session
    store, db, authority, owner, ref, profile = session
    sibling = create_local_session(authority, owner.actor, dict(
        request_id='untouched', source='gui', cwd=str(tmp_path), toolsets=[], follow_profile_config=True))
    before = db.get_session(sibling.session_id)
    reads = []

    def offered(*args, **kwargs):
        reads.append((str(get_hermes_home()), kwargs.get('access_token')))
        return kwargs.get('access_token') == 'eligible-account'

    monkeypatch.setattr('agent.codex_service_tiers.ultrafast_offered', offered)
    try:
        profile['model']['api_key'] = 'eligible-account'
        applied = await owner.dispatch(_mutation(db, ref, 'eligible', 'service_tier', {'service_tier': 'ultrafast'}))
        assert 'result' in applied, applied
        profile['model']['api_key'] = 'ineligible-account'
        refused = await owner.dispatch(_mutation(db, ref, 'ineligible', 'service_tier', {'service_tier': 'ultrafast'}))
        assert refused['error']['data']['reason'] == 'fast_unavailable', refused
        assert reads == [(authority.profile_id, 'eligible-account'), (authority.profile_id, 'ineligible-account')]
        assert db.get_session(sibling.session_id) == before
        assert _info(authority, sibling)['service_tier'] == ''
        assert _info(authority, ref)['service_tier'] == 'ultrafast'
    finally:
        await owner.close()
        store.close_all_db_handles()
