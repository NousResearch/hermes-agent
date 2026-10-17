"""SPEC R1/R2/R3 lifecycle regressions; temp-home configs only, no network.

Ownership fails closed: an extra_body value crosses a route boundary (fallback, the voice route, a
restore after /model) only when it is positively the caller's. What a config channel placed (the
constructor body while config defines extra_body, custom_providers merges, a /model switch, gateway
turn routes) is recorded when it is placed and never crosses, whatever config says later. A value
only partly config-owned is ambiguous, and a body no ledger accounts for has unknown provenance;
neither crosses. A /model switch drops the whole previous body, like upstream.
"""
from copy import deepcopy
import socket
from types import SimpleNamespace as N
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from agent.agent_init import _custom_provider_extra_body_for_agent as lookup
from agent.agent_init import _merge_custom_provider_extra_body as merge
from agent.chat_completion_helpers import _rescope_fallback_extra_body as rescope
from agent.chat_completion_helpers import build_api_kwargs
from agent.agent_runtime_helpers import _apply_switched_provider_request_overrides as switch
from hermes_cli.config_providers import get_compatible_custom_providers as view
from hermes_cli.config_providers import _custom_provider_entry_to_provider_config as convert
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent

M = 'huihui-qwen3.8-27b-abliterated'
U = 'http://127.0.0.1:1234/v1'
E = dict(name='LM Studio LAN', provider='lmstudio', model=M, base_url=U,
         extra_body={'temperature': 0, 'top_k': 20})


def registered_agent(caller):
    """The registration's body merged (and recorded) first; the caller's values are set afterwards."""
    eb = {'caller': True}
    if caller is not None:
        eb['temperature'] = caller
    a = N(provider='lmstudio', model=M, base_url=U, _custom_providers=[E], request_overrides={'speed': 'fast'})
    merge(a, [E])
    a.request_overrides['extra_body'] = {**a.request_overrides['extra_body'], **eb}
    return a, eb


@pytest.mark.parametrize('destination', ['lmstudio', 'openrouter'])
@pytest.mark.parametrize('caller', [None, 0, 0.4])
def test_fallback_keeps_only_caller_owned_values(destination, caller):
    a, eb = registered_agent(caller)
    a.model, a.provider = 'OTHER', destination
    rescope(a, M, 'lmstudio', U)
    # top_k was placed by the registration and recorded; a temperature equal to what the route
    # being left sets is config-owned too (upstream's rule). Only 0.4 is positively the caller's.
    expected = eb if caller == 0.4 else {'caller': True}
    assert a.request_overrides == {'extra_body': expected, 'speed': 'fast'}


@pytest.mark.parametrize('destination', ['lmstudio', 'openrouter'])
@pytest.mark.parametrize('caller', [None, 0, 0.4])
def test_switch_drops_the_whole_previous_body(destination, caller):
    """S1: a /model switch keeps none of the old route's extra_body, caller values included."""
    a, _ = registered_agent(caller)
    a.model, a.provider = 'OTHER', destination
    switch(a, destination)
    assert a.request_overrides == {'speed': 'fast'}


@pytest.mark.parametrize('model, body', [(M, E['extra_body']), ('OTHER', None)])
def test_switch_records_only_the_new_routes_body(model, body):
    a, _ = registered_agent(0.4)
    a.model = model
    switch(a, 'lmstudio')
    assert a.request_overrides == ({'extra_body': body, 'speed': 'fast'} if body else {'speed': 'fast'})
    assert a._custom_provider_injected_extra_body == (body or {})
    if body:
        a.request_overrides['extra_body']['top_k'] = 1  # the live body never aliases the ledger
        assert a._custom_provider_injected_extra_body == body


def test_differing_override_after_injection_survives():
    a = N(provider='lmstudio', model=M, base_url=U, _custom_providers=[E], request_overrides={})
    merge(a, [E])
    a.request_overrides['extra_body']['temperature'] = 0.7
    a.model = 'OTHER'
    rescope(a, M, 'lmstudio', U)
    assert a.request_overrides == {'extra_body': {'temperature': 0.7}}


@pytest.mark.parametrize('boundary', ['fallback', 'fallback_after_merge', 'switch'])
def test_unknown_provenance_never_crosses(boundary):
    """No ledger records where this body came from, so none of it is positively the caller's; the
    first config channel that records on such an agent records the body it found as well."""
    a = N(provider='lmstudio', model=M, base_url=U, _custom_providers=[E],
          request_overrides={'extra_body': {'temperature': 0.4, 'mine': {'a': 1}}, 'speed': 'fast'})
    if boundary == 'fallback_after_merge':
        merge(a, [E])
        assert a.request_overrides['extra_body'] == {'temperature': 0.4, 'top_k': 20, 'mine': {'a': 1}}
    a.model, a.provider = 'OTHER', 'openrouter'
    if boundary == 'switch':
        switch(a, 'openrouter')
    else:
        rescope(a, M, 'lmstudio', U)
    assert a.request_overrides == {'speed': 'fast'}


@pytest.mark.parametrize('mixed', [
    [{'name': 'lookup', 'mode': 'strict', 'mine': 1}],  # a configured element plus a caller key
    [{'name': 'lookup', 'mode': 'strict'}, {'name': 'mine'}],  # a configured element plus a caller element
])
def test_partly_config_owned_list_element_never_crosses(mixed):
    entry = {**E, 'extra_body': {'tools': [{'name': 'lookup', 'mode': 'strict'}]}}
    a = N(provider='lmstudio', model=M, base_url=U, _custom_providers=[entry], request_overrides={})
    merge(a, [entry])
    a.request_overrides['extra_body'] = {**a.request_overrides['extra_body'], 'tools': mixed, 'mine': 1}
    a.model = 'OTHER'
    rescope(a, M, 'lmstudio', U)
    assert a.request_overrides == {'extra_body': {'mine': 1}}


@pytest.mark.parametrize('migrated', [False, True])
def test_marker_conversion_write_payload_roundtrip(migrated):
    original = {'custom_providers': [deepcopy(E)]}
    snapshot = deepcopy(original)
    if migrated:
        from hermes_cli.config_migrations import _migrate_to_12
        with (patch('hermes_cli.config_migrations.read_raw_config', return_value=deepcopy(original)),
              patch('hermes_cli.config_migrations._persist_migration') as persist):
            _migrate_to_12({}, quiet=True)
        persist.assert_called_once()
        rewritten = persist.call_args.args[0]
        assert 'custom_providers' not in rewritten
    else:
        rewritten = {'providers': {'lan': convert(E)}}
    assert original == snapshot
    for cfg in (original, rewritten):
        entries = view(cfg)
        assert entries[0]['provider'] == 'lmstudio'
        assert lookup(provider='lmstudio', model=M, base_url=U, custom_providers=entries) == E['extra_body']


@pytest.mark.parametrize('provider', ['lmstudio', 'custom', 'custom:lm studio lan', 'custom:lan'])
@pytest.mark.parametrize('model', ['OTHER', M.upper(), '', '*'])
@pytest.mark.parametrize('representation', ['legacy', 'providers'])
def test_explicit_local_alias_cannot_bypass_pin(provider, model, representation):
    entry = {**E, 'models': ['OTHER', M.upper()]}
    cfg = {'custom_providers': [entry]} if representation == 'legacy' else {'providers': {'lan': convert(entry)}}
    assert lookup(provider=provider, model=model, base_url=U, custom_providers=view(cfg)) is None


@pytest.mark.parametrize('pin', ['', None, '*'])
def test_explicit_local_requires_nonempty_real_pin(pin):
    assert lookup(provider='custom', model=M, base_url=U,
                  custom_providers=view({'custom_providers': [{**E, 'model': pin, 'models': [M]}]})) is None


def test_legacy_non_local_catalog_case_matching_unchanged():
    entry = {**E, 'provider': 'custom', 'models': ['OTHER']}
    assert lookup(provider='custom', model='other', base_url=U,
                  custom_providers=view({'custom_providers': [entry]})) == E['extra_body']


# Canonical construction path: the CLI, TUI, cron and gateway build AIAgent from
# resolve_runtime_provider(), whose request_overrides ARE the custom entry's extra_body.
LAN_URL = 'http://127.0.0.1:59999/v1'
LAN_BODY = {'temperature': 0, 'cache': {'no-cache': True}, 'stop': ['</s>']}


def lan_entry(body, name='lan2'):
    return {'name': name, 'base_url': LAN_URL, 'api_key': 'test', 'model': 'm1', 'extra_body': body}


def named_config(body=LAN_BODY):
    return {'providers': {'lan2': {'name': 'lan2', 'base_url': LAN_URL, 'api_key': 'test', 'default_model': 'm1',
                                   'discover_models': False, 'models': {'m1': {'context_length': 64000}},
                                   'extra_body': body}},
            'model': {'default': 'm1', 'provider': 'custom:lan2'}}


RUNTIME_ROUTES = {
    # shape: (requested provider, model, config.yaml, the entry's extra_body)
    'named': ('custom:lan2', 'm1', named_config(), LAN_BODY),
    # LM Studio's default port with an exact-model pin; preload and probes are stubbed below.
    'lmstudio': ('custom:lm studio lan', M, {'custom_providers': [{**E, 'api_key': 'test'}]}, E['extra_body']),
}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    client = N(base_url='https://fallback.invalid/v1', api_key='test')
    with (patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')),
          # Hermes' happy-eyeballs racer connects via connect_ex.
          patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network forbidden')),
          patch('model_tools.get_tool_definitions', return_value=[]),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('agent.process_bootstrap.OpenAI'),
          patch('agent.auxiliary_client.resolve_provider_client', side_effect=lambda provider, model, **kw: (client, model)),
          patch('agent.model_metadata.get_model_context_length', return_value=128_000),
          patch('agent.agent_runtime_helpers._rebuild_primary_client'),
          patch.object(AIAgent, '_ensure_lmstudio_runtime_loaded', return_value=None)):
        def resolve(shape, config=None):
            requested, model, default_config, _ = RUNTIME_ROUTES[shape]
            (home / 'config.yaml').write_text(yaml.safe_dump(config or default_config))
            return resolve_runtime_provider(requested=requested, target_model=model)

        def build(shape, mixed=None, config=None):
            """``mixed`` joins the runtime body in the constructor's request_overrides."""
            model = RUNTIME_ROUTES[shape][1]
            rt = resolve(shape, config)
            overrides = deepcopy(rt.get('request_overrides') or {})
            if mixed:
                overrides['extra_body'] = {**overrides.get('extra_body', {}), **mixed}
            overrides['seed'] = 7
            agent = AIAgent(model=model, provider=rt['provider'], requested_provider=rt.get('requested_provider'),
                            api_key=rt['api_key'], base_url=rt['base_url'], api_mode=rt.get('api_mode'),
                            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
                            skip_background_review=True, request_overrides=overrides,
                            fallback_model=[{'provider': 'zai', 'model': 'glm-x'}])
            return rt, agent
        yield N(resolve=resolve, build=build)


def wire_body(agent):
    return build_api_kwargs(agent, [{'role': 'user', 'content': 'offline'}], []).get('extra_body') or {}


def switch_to_openrouter(agent):
    agent.switch_model(new_model='gpt-x', new_provider='openrouter', api_key='test',
                       base_url='https://openrouter.ai/api/v1', api_mode='chat_completions')


@pytest.mark.parametrize('shape', ['named', 'lmstudio'])
def test_runtime_resolved_body_stays_config_owned(runtime, shape):
    body = RUNTIME_ROUTES[shape][3]
    rt, agent = runtime.build(shape)
    assert rt['request_overrides'] == {'extra_body': body}
    assert wire_body(agent) == body
    assert agent._try_activate_fallback()
    assert not body.keys() & wire_body(agent).keys()
    assert agent._restore_primary_runtime()
    assert wire_body(agent) == body
    switch_to_openrouter(agent)
    assert 'extra_body' not in agent.request_overrides
    assert not body.keys() & wire_body(agent).keys()


@pytest.mark.parametrize('shape', ['named', 'lmstudio'])
@pytest.mark.parametrize('provenance', ['config', 'constructor', 'after_init'])
def test_switch_leaves_what_upstream_leaves(runtime, shape, provenance):
    """S1 on the CLI/TUI path: after /model the agent holds exactly what upstream's switch leaves,
    the old request_overrides minus extra_body, however the body's values got there."""
    body = RUNTIME_ROUTES[shape][3]
    _, agent = runtime.build(shape, mixed={'temperature': 0.4} if provenance == 'constructor' else None)
    if provenance == 'after_init':
        agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], 'temperature': 0.4, 'mine': 1}
    upstream = {key: value for key, value in agent.request_overrides.items() if key != 'extra_body'}
    switch_to_openrouter(agent)
    assert agent.request_overrides == upstream == {'seed': 7}
    assert not (body.keys() | {'mine'}) & wire_body(agent).keys()


@pytest.mark.parametrize('shape', ['named', 'lmstudio'])
def test_constructor_value_mixed_into_runtime_body_is_config_owned(runtime, shape):
    """S2: a constructor body can't be told apart from runtime-resolved config, so a value in it never
    crosses, even one that differs from every configured entry."""
    body = RUNTIME_ROUTES[shape][3]
    _, agent = runtime.build(shape, mixed={'temperature': 0.4})
    assert wire_body(agent) == {**body, 'temperature': 0.4}
    assert agent._try_activate_fallback()
    assert not body.keys() & wire_body(agent).keys()
    assert agent._restore_primary_runtime()
    assert wire_body(agent) == {**body, 'temperature': 0.4}
    assert agent._try_activate_fallback()
    assert not body.keys() & wire_body(agent).keys()


@pytest.mark.parametrize('shape', ['named', 'lmstudio'])
def test_value_set_after_construction_is_the_callers(runtime, shape):
    body = RUNTIME_ROUTES[shape][3]
    _, agent = runtime.build(shape)
    agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], 'temperature': 0.4, 'mine': {'a': 1}}
    assert agent._try_activate_fallback()
    crossed = wire_body(agent)
    assert (crossed.get('temperature'), crossed.get('mine')) == (0.4, {'a': 1})
    assert not (body.keys() - {'temperature'}) & crossed.keys()


def test_caller_value_equal_to_config_is_config_owned(runtime):
    _, agent = runtime.build('named')
    agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], 'temperature': 0}
    assert agent._try_activate_fallback()
    assert 'temperature' not in wire_body(agent)


def test_gateway_reload_between_turns_never_crosses(runtime):
    """S2: the gateway reuses its cached agent and re-applies the freshly resolved body each turn. A
    config reload in between must not make the reloaded (or the original) values caller-owned."""
    from gateway.run_turn_runner import TurnRunner
    _, agent = runtime.build('named')
    reloaded = {'temperature': 0.3, 'top_p': 0.9, 'cache': {'no-cache': True, 'ttl': 5}, 'stop': ['</s>', 'END']}
    rt = runtime.resolve('named', named_config(reloaded))
    TurnRunner._merge_turn_request_overrides(agent, {'request_overrides': rt['request_overrides']})
    assert wire_body(agent) == reloaded
    for _ in range(2):
        assert agent._try_activate_fallback()
        assert not (reloaded.keys() | LAN_BODY.keys()) & wire_body(agent).keys()
        assert agent._restore_primary_runtime()


@pytest.mark.parametrize('shape', ['named', 'lmstudio'])
def test_reloaded_entries_never_make_recorded_values_the_callers(runtime, shape):
    """S2: switch_model refreshes the agent's custom_providers from a reloaded config. Values placed
    under the old config stay config-owned although the reloaded entry no longer sets them."""
    body = RUNTIME_ROUTES[shape][3]
    _, agent = runtime.build(shape)
    reloaded = dict.fromkeys(body, 0.3)
    entries = view(named_config(reloaded)) if shape == 'named' else view(
        {'custom_providers': [{**E, 'api_key': 'test', 'extra_body': reloaded}]})
    agent._custom_providers = entries
    assert agent._try_activate_fallback()
    assert not body.keys() & wire_body(agent).keys()


def test_differing_entries_never_cross(runtime):
    """S2: runtime resolution picks lan2 by name while the agent's lookup matches an earlier entry for
    the same endpoint and model; neither body may cross."""
    shadow = {'temperature': 0.9, 'top_k': 5}
    config = {'custom_providers': [lan_entry(shadow, name='shadow'), lan_entry(LAN_BODY)]}
    rt, agent = runtime.build('named', config=config)
    assert rt['request_overrides'] == {'extra_body': LAN_BODY}
    assert wire_body(agent) == {**shadow, **LAN_BODY}
    assert agent._try_activate_fallback()
    assert not (shadow.keys() | LAN_BODY.keys()) & wire_body(agent).keys()


@pytest.mark.parametrize('key, mixed', [
    ('cache', {'no-cache': True, 'ttl': 5}),  # configured dict plus a caller key
    ('stop', ['</s>', 'END']),  # configured list plus a caller element
])
def test_partly_config_owned_value_never_crosses(runtime, key, mixed):
    _, agent = runtime.build('named')
    agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], key: mixed, 'mine': 1}
    assert agent._try_activate_fallback()
    crossed = wire_body(agent)
    assert crossed.get('mine') == 1
    assert key not in crossed


@pytest.mark.parametrize('config, owned', [
    ({}, False),  # no extra_body anywhere: the constructor body is the caller's, as upstream treats it
    ({'auxiliary': {'vision': {'extra_body': {'x': 1}}}}, True),  # config could be its source
])
def test_constructor_body_owner_depends_on_config_extra_body(runtime, config, owned):
    rt, agent = runtime.build('named', mixed={'reasoning': {'effort': 'high'}}, config={**named_config({}), **config})
    assert not rt.get('request_overrides')
    assert agent._try_activate_fallback()
    assert ('reasoning' in wire_body(agent)) is not owned
