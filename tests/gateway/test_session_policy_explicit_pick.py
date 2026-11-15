"""An explicit ``-t`` pick is the session's whole toolset selection (#147832).

``_get_platform_tools`` turns on every plugin toolset missing from ``known_plugin_toolsets`` and every
enabled MCP server; a pick that names neither must not inherit them through the gateway create path.
"""
import pytest

from gateway.session_policy import build_policy


@pytest.fixture
def extensions(tmp_path, monkeypatch):
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_toolset_keys_nowait', lambda: {'fake_coding_plugin'})
    servers = {'srv': {'command': str(tmp_path / 'absent')}, 'other': {'command': str(tmp_path / 'absent')}}
    return {'platform_toolsets': {'cli': ['hermes-cli']}, 'mcp_servers': servers}


def _toolsets(tmp_path, cfg, pick):
    return set(build_policy({'source': 'cli', 'cwd': str(tmp_path), 'toolsets': pick}, cfg).toolsets)


def test_a_pick_without_extensions_gets_no_plugin_toolset_or_mcp_server(tmp_path, extensions):
    assert _toolsets(tmp_path, extensions, ['session_search']) == {'session_search'}


@pytest.mark.parametrize('server', ['srv', 'mcp-srv'])
def test_a_named_mcp_server_stays_and_the_others_do_not(tmp_path, extensions, server):
    toolsets = _toolsets(tmp_path, extensions, ['session_search', server])
    assert 'other' not in toolsets and 'fake_coding_plugin' not in toolsets, toolsets
    assert toolsets & {'srv', 'mcp-srv'}, toolsets


def test_a_named_plugin_toolset_stays(tmp_path, extensions, monkeypatch):
    monkeypatch.setattr('toolsets.validate_toolset', lambda name: True)
    monkeypatch.setattr('hermes_cli.toolset_validation.saved_toolset_resolver', lambda _cfg: lambda name: True)
    toolsets = _toolsets(tmp_path, extensions, ['session_search', 'fake_coding_plugin'])
    assert toolsets == {'session_search', 'fake_coding_plugin'}, toolsets


def test_no_pick_still_gets_the_profile_extensions(tmp_path, extensions):
    toolsets = set(build_policy({'source': 'cli', 'cwd': str(tmp_path)}, extensions).toolsets)
    assert {'fake_coding_plugin', 'srv', 'other'} <= toolsets, toolsets


@pytest.mark.parametrize('pick', ['all', '*'])
def test_an_all_pick_keeps_every_plugin_toolset_and_mcp_server(tmp_path, extensions, pick):
    assert {'fake_coding_plugin', 'srv', 'other'} <= _toolsets(tmp_path, extensions, [pick])


@pytest.fixture
def builtin_named_plugin_keys(monkeypatch):
    # The issue's leaked set: plugin keys that are also built-in toolset names.
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_toolset_keys_nowait', lambda: {'coding', 'delegation', 'fake_plug'})
    return {'platform_toolsets': {'cli': ['hermes-cli']}}


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui', 'acp'])
def test_a_plugin_key_named_like_a_builtin_does_not_ride_along(tmp_path, builtin_named_plugin_keys, source):
    params = {'source': source, 'cwd': str(tmp_path), 'toolsets': ['session_search']}
    assert set(build_policy(params, builtin_named_plugin_keys).toolsets) == {'session_search'}


def test_a_builtin_named_plugin_key_stays_when_picked(tmp_path, builtin_named_plugin_keys):
    assert _toolsets(tmp_path, builtin_named_plugin_keys, ['session_search', 'delegation']) == {'session_search', 'delegation'}


@pytest.mark.parametrize('pick', ['hermes-api-server', 'hermes-webhook'])
def test_a_composite_keeps_a_builtin_that_a_server_is_named_like(tmp_path, pick):
    # Both composites expand into built-in names; this MCP server is also called ``web``, and the pick needs web.
    cfg = {'platform_toolsets': {'cli': ['hermes-cli']}, 'mcp_servers': {'web': {'command': str(tmp_path / 'absent')}}}
    assert 'web' in _toolsets(tmp_path, cfg, [pick])
    assert _toolsets(tmp_path, cfg, ['session_search']) == {'session_search'}


def test_a_composite_keeps_file_when_a_server_is_named_file(tmp_path):
    # ``hermes-api-server`` needs the built-in file tools; an MCP server called ``file`` must not cost it them.
    cfg = {'platform_toolsets': {'cli': ['hermes-cli']}, 'mcp_servers': {'file': {'command': str(tmp_path / 'absent')}}}
    assert {'file', 'terminal'} <= _toolsets(tmp_path, cfg, ['hermes-api-server'])
    # ``hermes-webhook`` has no file tools, so the unnamed server stays out.
    assert 'file' not in _toolsets(tmp_path, cfg, ['hermes-webhook'])


@pytest.mark.parametrize('pick', ['hermes-api-server', 'hermes-webhook'])
def test_a_wider_plugin_key_is_not_put_back_for_a_composite(tmp_path, builtin_named_plugin_keys, pick):
    # The ``coding`` key also holds web_search, but putting it back would add terminal and file tools.
    cfg = {**builtin_named_plugin_keys, 'mcp_servers': {'web': {'command': str(tmp_path / 'absent')}}}
    toolsets = _toolsets(tmp_path, cfg, [pick])
    assert 'web' in toolsets and 'coding' not in toolsets, toolsets
    # A pick that already has web through ``coding`` puts nothing back, so the server stays out.
    assert 'web' not in _toolsets(tmp_path, cfg, ['coding'])


def test_a_pick_covering_a_builtin_named_plugin_key_keeps_it(tmp_path, builtin_named_plugin_keys):
    # ``coding`` already holds delegate_task, so the ``delegation`` key adds nothing the pick did not grant.
    assert _toolsets(tmp_path, builtin_named_plugin_keys, ['coding']) == {'coding', 'delegation'}
