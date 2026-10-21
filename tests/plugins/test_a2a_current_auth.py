"""Production plugin discovery/dispatch of request-local, read-only authentication."""
import json

import pytest


@pytest.mark.parametrize('args', [
    {'peer': 'operator'}, {'method': 'peer_bearer'}, {'task_id': 'other'},
    {'context_id': 'other'}, {'a2a_auth': {'peer': 'operator'}},
])
def test_model_arguments_cannot_supply_or_select_authentication(args):
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    discover_plugins()
    result = json.loads(registry.dispatch('a2a_current_authentication', args))
    assert result == {'available': False, 'authorization_granted': False,
                      'error': 'This tool accepts no arguments.'}


def test_declaration_is_closed_and_config_gated(monkeypatch):
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    discover_plugins()
    name = 'a2a_current_authentication'
    entry = registry.get_entry(name)
    assert entry.toolset == 'a2a'
    assert entry.schema['parameters'] == {
        'type': 'object', 'properties': {}, 'additionalProperties': False,
    }
    import sys
    module = sys.modules[entry.handler.__module__]
    monkeypatch.setattr(module, '_get_scoped_secret', lambda _: '')
    monkeypatch.setattr(module, '_load_config', dict)
    assert entry.check_fn() is False
    monkeypatch.setattr(module, '_load_config', lambda: {'platforms': {'a2a': {'enabled': True}}})
    assert entry.check_fn() is True
    assert registry.get_definitions({name}, quiet=True)[0]['function']['name'] == name
    from hermes_cli.plugins import get_plugin_manager
    manifest = next(p.manifest for p in get_plugin_manager()._plugins.values()
                    if p.manifest.name == 'a2a-platform')
    assert name in manifest.provides_tools
    from toolsets import resolve_toolset
    assert name in resolve_toolset('a2a')
    assert name not in resolve_toolset('file')


from gateway import session_context as sc
from plugins.platforms.a2a.security import A2ASecurityContext


@pytest.mark.parametrize('mode', ['absent', 'forged', 'shared', 'local'])
def test_absent_and_non_named_identity_never_become_named_proof(mode, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    discover_plugins()
    name = 'a2a_current_authentication'
    monkeypatch.setenv('HERMES_A2A_PEER', 'operator')
    monkeypatch.setenv('HERMES_A2A_AUTH_METHOD', 'peer_bearer')
    auth = None
    if mode == 'forged':
        auth = {'peer': 'operator', 'method': 'peer_bearer'}
    elif mode == 'shared':
        auth = A2ASecurityContext('secret', (), frozenset(), False, '127.0.0.1', '').authenticate_request('Bearer secret')
    elif mode == 'local':
        auth = A2ASecurityContext('', (), frozenset(), False, '127.0.0.1', '').authenticate_request(None)
    try:
        sc.set_session_vars(platform='a2a', user_id='operator', a2a_auth=auth)
        result = json.loads(registry.dispatch(name, {}, task_id='spoof', session_id='spoof',
                                             a2a_auth={'peer': 'operator'}))
        if mode in ('absent', 'forged'):
            assert result == {'available': False, 'authorization_granted': False}
        else:
            assert result == {'available': True, 'peer': auth.peer, 'method': auth.method.value,
                              'named_peer': False, 'authorization_granted': False}
        # A worker without a copied turn must not borrow the parent's identity.
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert json.loads(pool.submit(registry.dispatch, name, {}).result()) == {
                'available': False, 'authorization_granted': False}
    finally:
        sc.clear_session_vars([])


def test_discovered_tool_reads_only_current_server_context(monkeypatch):
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    discover_plugins()
    name = 'a2a_current_authentication'
    assert registry.get_entry(name) is not None
    auth = A2ASecurityContext('', (('secret', 'alice'),), frozenset(), False, '127.0.0.1', '').authenticate_request('Bearer secret')
    try:
        sc.set_session_vars(platform='a2a', a2a_auth=auth)
        assert json.loads(registry.dispatch(name, {'peer': 'operator', 'task_id': 'other'})) == {
            'available': False, 'authorization_granted': False,
            'error': 'This tool accepts no arguments.'}
        assert json.loads(registry.dispatch(name, {}, a2a_auth={'peer': 'operator'})) == {
            'available': True, 'peer': 'alice', 'method': 'peer_bearer',
            'named_peer': True, 'authorization_granted': False,
        }
    finally:
        sc.clear_session_vars([])
