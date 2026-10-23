"""A bare ``custom`` provider is the billing class every named endpoint reports, never a route.

The gateway freezes, persists and syncs the configured ``custom:<name>`` entry it ran on: a launch
pick echoing ``custom`` back, a ``/model`` switch persisting it, and a turn syncing the adapter's
label. A turn records the entry its runtime resolved; only an unknown entry is told apart from
siblings on one relay base_url by the model they serve.

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


def test_a_disabled_sibling_serving_the_model_never_names_the_route():
    from hermes_cli.runtime_provider_custom import routed_custom_identity
    config = {'providers': {**SIBLINGS['providers'],
                            'deepseek': {**SIBLINGS['providers']['deepseek'], 'enabled': False}}}
    assert routed_custom_identity(RELAY, 'deepseek-v4.1-flash', config) == 'custom:glm'


SPLIT_KEYS = {'providers': {name: {**entry, 'key_env': f'{name.upper()}_KEY'}
                            for name, entry in SIBLINGS['providers'].items()}}


def test_a_bare_custom_switch_resumes_on_the_key_that_served_it(home, monkeypatch):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_cli.model_switch import switch_model
    from hermes_state import SessionDB
    from tui_gateway import server
    monkeypatch.setenv('GLM_KEY', 'glm-key')
    monkeypatch.setenv('DEEPSEEK_KEY', 'deepseek-key')
    (home / 'config.yaml').write_text(json.dumps(SPLIT_KEYS))
    # A glm session (the adapter reports bare ``custom``) runs /model deepseek-v4.1-flash --provider custom.
    switched = switch_model('deepseek-v4.1-flash', 'custom', 'glm-5.3', RELAY, 'glm-key',
                            explicit_provider='custom', user_providers=SPLIT_KEYS['providers'])
    assert switched.success and switched.api_key == 'glm-key', switched
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('s', source='cli', model='glm-5.3')
        db.update_session_model('s', switched.new_model, switched.target_provider, base_url=switched.base_url)
        # The next turn syncs the switched agent, whose requested provider is bare ``custom`` too.
        agent = SimpleNamespace(model=switched.new_model, provider='custom', requested_provider='custom',
                                base_url=RELAY, api_mode='chat_completions')
        GatewayTurnPrepareMixin._sync_session_model_from_agent(
            SimpleNamespace(_session_db=SimpleNamespace(_db=db)), 's', agent)
        # Resume reads the stored route back and resolves its credentials.
        overrides = server._stored_session_runtime_overrides(db.get_session('s'))
        _model, runtime = server._resolve_agent_model_runtime(
            overrides['model_override'], overrides.get('provider_override'))
        assert runtime['api_key'] == 'glm-key'
    finally:
        db.close()


@pytest.mark.parametrize('requested, model', [('custom:glm', 'deepseek-v4.1-flash'), ('custom:deepseek', 'glm-5.3')],
                         ids=['glm-running-the-siblings-model', 'deepseek-running-the-siblings-model'])
def test_a_turn_on_a_named_entry_persists_that_entry_not_the_sibling_listing_its_model(home, requested, model):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_state import SessionDB
    (home / 'config.yaml').write_text(json.dumps(SIBLINGS))
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('s', source='cli', model=model)
        agent = SimpleNamespace(model=model, provider='custom', requested_provider=requested, base_url=RELAY,
                                api_mode='chat_completions')
        GatewayTurnPrepareMixin._sync_session_model_from_agent(
            SimpleNamespace(_session_db=SimpleNamespace(_db=db)), 's', agent)
        assert json.loads(db.get_session('s')['model_config'])['gateway_runtime']['provider'] == requested
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('requested, model', [('custom:glm', 'deepseek-v4.1-flash'), ('custom:deepseek', 'glm-5.3')],
                         ids=['glm-running-the-siblings-model', 'deepseek-running-the-siblings-model'])
async def test_the_model_picker_marks_the_entry_a_session_runs_on_current(home, monkeypatch, requested, model):
    from gateway import session_config
    (home / 'config.yaml').write_text(json.dumps(SIBLINGS))
    agent = SimpleNamespace(model=model, provider='custom', requested_provider=requested, base_url=RELAY)
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(home), agent=lambda ref: agent))
    monkeypatch.setattr(session_config, '_authorize', lambda connection, ref, params: None)
    monkeypatch.setattr('hermes_cli.inventory.build_model_options_payload',
                        lambda ctx, **kwargs: {'provider': ctx.current_provider})
    result = await session_config.model_options(connection, SimpleNamespace(session_id='s'), {})
    assert result == {'provider': requested}


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


def test_source_desktop_is_sent_as_the_gui_surface_the_create_contract_lists(monkeypatch):
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems
    monkeypatch.setenv('HERMES_SESSION_SOURCE', 'desktop')
    monkeypatch.setenv('HERMES_SESSION_SOURCE_EXPLICIT', '1')
    sent = {}

    def call(method, **params):
        if method == 'session.create':
            sent.update(params)
        return {'session_create': {'parameters': ['cwd', 'source', 'row_cwd']}, 'stored_session_id': 'sid'}
    GatewaySessionStore(SimpleNamespace(call=call, state=lambda sid: SimpleNamespace(
        adopt_snapshot=lambda snapshot: None))).create({})
    assert sent['source'] == 'gui'
    assert canonical_param_problems(CANONICAL_METHODS['session.create'], {'request_id': 'r', 'source': 'gui'}) == []
