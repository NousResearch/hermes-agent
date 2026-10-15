"""Saved selections remain usable through the classic CLI gateway boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest


def _build(monkeypatch, config, explicit=None, coding=None):
    import cli
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from gateway.session_policy import _resolve_toolsets

    store = object.__new__(GatewaySessionStore)
    store.connection = None
    resolved = []

    def call(self, method, **params):
        assert method == 'session.create'
        resolved.append(_resolve_toolsets(params, deepcopy(config), 'cli', False))
        return {'stored_session_id': 'gateway-session'}

    def construct(**kwargs):
        return SimpleNamespace(_session_db=store, model='fixture',
                               enabled_toolsets=kwargs['toolsets'], session_id='fixture',
                               _resumed=False, ignore_rules=True)

    monkeypatch.setattr(cli, 'CLI_CONFIG', config)
    monkeypatch.setattr(cli, 'HermesCLI', construct)
    monkeypatch.setattr('agent.coding_context.coding_selection', lambda **kw: coding)
    monkeypatch.setattr(GatewaySessionStore, 'call', call)
    monkeypatch.setattr(GatewaySessionStore, '_adopt', lambda self, snapshot: snapshot['stored_session_id'])
    monkeypatch.setattr('hermes_cli.gateway_tools.bind_session', lambda *args: None)
    host = cli._build_cli_from_args(
        model=None, toolsets=explicit, provider=None, reasoning=None, api_key=None,
        base_url=None, max_turns=None, run_budget=None, verbose=False, compact=False,
        resume=None, checkpoints=False, pass_session_id=False, ignore_rules=True, skills=None)
    return host, resolved, CLIInitMixin


def test_saved_stale_selection_reaches_gateway(monkeypatch):
    config = {'platform_toolsets': {'cli': ['terminal', 'mcp-deleted-fixture']}}
    host, resolved, mixin = _build(monkeypatch, config)
    try:
        mixin._ensure_gateway_session(host)
    except ValueError as exc:
        pytest.fail(f'Saved stale selection blocked gateway creation: {exc}')
    assert 'terminal' in resolved[0]
    assert 'mcp-deleted-fixture' not in host.enabled_toolsets
    assert config['platform_toolsets']['cli'] == ['terminal', 'mcp-deleted-fixture']


@pytest.mark.parametrize('saved', [[], ['mcp-deleted-fixture']])
def test_empty_saved_selection_never_expands(monkeypatch, saved):
    host, resolved, mixin = _build(monkeypatch, {'platform_toolsets': {'cli': saved}})
    mixin._ensure_gateway_session(host)
    assert host.enabled_toolsets == []
    assert resolved == [set()]
    from datetime import datetime
    mixin._new_session_id(host, datetime.now())
    assert resolved == [set(), set()]


def test_explicit_unknown_toolset_still_refused(monkeypatch):
    host, _, mixin = _build(monkeypatch, {}, explicit='terminal,mcp-deleted-fixture')
    with pytest.raises(ValueError, match='invalid_params'):
        mixin._ensure_gateway_session(host)


@pytest.mark.parametrize('name', ['fixture', 'mcp-fixture'])
def test_configured_mcp_names_are_not_discarded(monkeypatch, name):
    config = {'platform_toolsets': {'cli': ['terminal', name]},
              'mcp_servers': {'fixture': {'command': 'fixture-not-run'}}}
    host, _, _ = _build(monkeypatch, config)
    assert name in host.enabled_toolsets


def test_coding_posture_is_unchanged(monkeypatch):
    host, _, _ = _build(monkeypatch, {}, coding=['terminal', 'file'])
    assert host.enabled_toolsets == ['terminal', 'file']


def test_new_uses_sanitized_selection(monkeypatch):
    from datetime import datetime
    config = {'platform_toolsets': {'cli': ['terminal', 'mcp-deleted-fixture']}}
    host, resolved, mixin = _build(monkeypatch, config)
    mixin._ensure_gateway_session(host)
    assert mixin._new_session_id(host, datetime.now()) == 'gateway-session'
    assert resolved == [{'terminal'}, {'terminal'}]


def test_plugin_and_portable_mcp_names_are_preserved(monkeypatch):
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, 'get_plugin_toolset_keys_nowait', lambda: {'plugin-fixture'})
    monkeypatch.setattr(plugins, 'get_portable_mcp_server_names_nowait', lambda: {'portable-fixture'})
    names = ['plugin-fixture', 'mcp-portable-fixture', 'portable-fixture']
    host, _, _ = _build(monkeypatch, {'platform_toolsets': {'cli': names}})
    assert set(names) <= set(host.enabled_toolsets)


def test_missing_config_keeps_default_selection(monkeypatch):
    from hermes_cli.tools_config import _get_platform_tools
    expected = sorted(_get_platform_tools({}, 'cli'))
    host, _, mixin = _build(monkeypatch, {})
    assert host.enabled_toolsets == expected
    mixin._ensure_gateway_session(host)


def test_validation_failure_is_not_silently_treated_as_stale(monkeypatch):
    def broken(config):
        raise RuntimeError('fixture discovery failure')
    monkeypatch.setattr('hermes_cli.toolset_validation.saved_toolset_resolver', broken)
    with pytest.raises(RuntimeError, match='fixture discovery failure'):
        _build(monkeypatch, {'platform_toolsets': {'cli': ['terminal']}})
