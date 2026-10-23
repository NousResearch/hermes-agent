"""A bare ``custom`` provider is the billing class every named endpoint reports, never a route.

The gateway freezes, persists and syncs the configured ``custom:<name>`` entry it ran on: a launch
pick echoing ``custom`` back, a ``/model`` switch persisting it, and a turn syncing the adapter's
label. Entries sharing one relay base_url are told apart by the model they serve.

Also: a profile-following session's model pick with an inline secret outside the followed
sections, and ``--source desktop``. No endpoint is contacted.
"""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_follow_profile_followups import _profile
from tests.gateway.test_session_follow_profile_config import _owner

RELAY = 'http://127.0.0.1:8080/v1'
SIBLINGS = {'providers': {
    'glm': {'api': RELAY, 'key_env': 'RELAY_KEY', 'default_model': 'glm-5.3'},
    'deepseek': {'api': RELAY, 'key_env': 'RELAY_KEY', 'default_model': 'deepseek-v4.1-flash'}}}


@pytest.fixture
def home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    profile = get_hermes_home()
    profile.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', profile)
    return profile


@pytest.mark.parametrize('pick, provider', [
    ({}, 'custom:glm'),
    ({'base_url': RELAY, 'model': 'deepseek-v4.1-flash'}, 'custom:deepseek'),
], ids=['profile-entry', 'sibling-on-shared-url'])
def test_a_bare_custom_launch_pick_freezes_the_configured_entry(tmp_path, pick, provider):
    from gateway.session_policy import build_policy
    config = {'model': {'default': 'glm-5.3', 'provider': 'custom:glm', 'base_url': RELAY,
                        'key_env': 'RELAY_KEY'}, **SIBLINGS}
    params = dict(cwd=str(tmp_path), provider='custom', **({'model': 'unlisted-model'} | pick))
    model = build_policy(params, config).config()['model']
    assert (model['provider'], model.get('base_url')) == (provider, RELAY)


def test_a_bare_custom_route_is_persisted_as_its_entry(home):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_state import SessionDB
    (home / 'config.yaml').write_text(json.dumps(SIBLINGS))
    db = SessionDB(db_path=home / 'state.db')
    try:
        for sid in ('switched', 'synced'):
            db.create_session(sid, source='cli', model='glm-5.3')
        # A /model switch on the gateway persists the resolved provider it got back.
        db.update_session_model('switched', 'deepseek-v4.1-flash', 'custom', base_url=RELAY)
        # A turn syncs the route its agent ran on, whose provider is the adapter's label.
        agent = SimpleNamespace(model='deepseek-v4.1-flash', provider='custom', base_url=RELAY,
                                api_mode='chat_completions')
        runner = SimpleNamespace(_session_db=SimpleNamespace(_db=db))
        GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, 'synced', agent)
        stored = {sid: json.loads(db.get_session(sid)['model_config']) for sid in ('switched', 'synced')}
        assert stored['switched']['provider'] == stored['switched']['gateway_runtime']['provider'] == 'custom:deepseek'
        assert stored['synced']['gateway_runtime']['provider'] == 'custom:deepseek'
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_following_session_switches_model_with_an_inline_secret_outside_its_followed_sections(home):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session_bot import _target
    from gateway.session_policy import policy_for_source
    _profile(home, 'old-model', 1, 'old-key', tts={'openai': {'api_key': 'tts-key'}})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._resolve_session_agent_runtime = (
        lambda **kwargs: GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, **kwargs))
    try:
        ref, live, _entry = await _target(authority, owner.actor)
        snap = authority.db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': dict(
            session_id=ref.session_id, request_id='pick', operation='model', payload={'model': 'new-model'},
            expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])})
        assert 'result' in response, response
        policy = policy_for_source(runner, live.source)
        assert policy.model == 'new-model'
        assert policy.config(authority)['tts']['openai']['api_key'] == 'tts-key'
    finally:
        await owner.close()
        store.close_all_db_handles()


def test_source_desktop_launches_the_desktop_surface(tmp_path):
    from gateway.session_policy import build_policy
    policy = build_policy(dict(cwd=str(tmp_path), source='desktop'), {})
    assert (policy.source, policy.platform) == ('gui', 'desktop')
