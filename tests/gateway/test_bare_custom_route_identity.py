"""A bare ``custom`` provider is the billing class every named endpoint reports, never a route.

A launch pick echoing ``custom`` back freezes the configured ``custom:<name>`` entry, a disabled entry
never names a route, and the model picker marks the entry the live agent resolved, not the first
sibling on a shared relay base_url.

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


def test_a_disabled_entry_on_the_url_never_names_the_route(home):
    from hermes_cli.runtime_provider_custom import session_provider_identity
    (home / 'config.yaml').write_text(json.dumps({'providers': {
        'glm': {**SIBLINGS['providers']['glm'], 'enabled': False}, 'deepseek': SIBLINGS['providers']['deepseek']}}))
    assert session_provider_identity('custom', 'custom', RELAY, 'deepseek-v4.1-flash') == 'custom:deepseek'


@pytest.mark.asyncio
async def test_the_model_picker_marks_the_entry_a_session_runs_on_current(home, monkeypatch):
    from gateway import session_config
    (home / 'config.yaml').write_text(json.dumps(SIBLINGS))
    # deepseek runs glm's model: the first entry on the shared URL (and the one listing it) is not the route.
    agent = SimpleNamespace(model='glm-5.3', provider='custom', requested_provider='custom:deepseek', base_url=RELAY)
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(home), agent=lambda ref: agent))
    monkeypatch.setattr(session_config, '_authorize', lambda connection, ref, params: None)
    monkeypatch.setattr('hermes_cli.inventory.build_model_options_payload',
                        lambda ctx, **kwargs: {'provider': ctx.current_provider})
    result = await session_config.model_options(connection, SimpleNamespace(session_id='s'), {})
    assert result == {'provider': 'custom:deepseek'}


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
