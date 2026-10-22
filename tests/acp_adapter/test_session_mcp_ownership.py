"""The gateway owns each session's editor MCP servers as session state (#134889 ACP-8/ACP-10).

A changed editor list re-freezes through one owner path (equal schemas keep the manifest), and a
viewed session keeps its connections between executions instead of respawning them per turn.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.session_policy import policy_scope
from hermes_state_runtime import RuntimeStoreError
from tests.acp_adapter.test_session_mcp_policy import owner, policy


def _editor_servers():
    from tools.mcp_tool import _servers
    return {key: server for key, server in _servers.items() if key[-1].startswith('editor_')}


def test_changed_credentials_with_equal_schemas_keep_the_manifest(server_spec, tmp_path):
    from gateway.session_local_mcp import attach_editor_mcp, rebind_editor_mcp
    from tools.registry import registry
    authority = owner()
    before = policy(authority, 'rebind', server_spec, tmp_path)
    changed = json.loads(json.dumps(server_spec))
    changed['env'][0]['value'] = 'PRIVATE_B'
    after, tools_changed = rebind_editor_mcp(authority, 'rebind', before, [changed])
    old, new = json.loads(before.editor_mcp_json), json.loads(after.editor_mcp_json)
    # The prompt-cache prefix (tool manifest + toolsets) is byte-stable; only the fingerprint moved.
    assert not tools_changed
    assert new['schemas'] == old['schemas'] and new['names'] == old['names'] and new['scope'] == old['scope']
    assert after.toolsets == before.toolsets
    assert new['fingerprint'] != old['fingerprint']
    assert 'PRIVATE_B' not in after.editor_mcp_json
    attach_editor_mcp(authority, after, [changed])
    with policy_scope(after, authority=authority):
        names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
        assert 'PRIVATE_B' in registry.dispatch(names[0], {})


def test_changed_schema_is_reported_as_a_manifest_change(server_spec, tmp_path):
    from gateway.session_local_mcp import rebind_editor_mcp
    authority = owner()
    before = policy(authority, 'schema', server_spec, tmp_path)
    script = tmp_path / 'owned_mcp.py'
    script.write_text(script.read_text().replace('owned echo', 'renamed echo'))
    after, tools_changed = rebind_editor_mcp(authority, 'schema', before, [server_spec])
    assert tools_changed
    schemas = json.loads(after.editor_mcp_json)['schemas']
    assert 'renamed echo' in json.dumps(schemas)
    assert after.toolsets == before.toolsets  # same server, same ledger name


def test_viewed_session_keeps_editor_connections_between_executions(server_spec, tmp_path):
    from gateway.session_local_mcp import release_editor_mcp
    authority = owner()
    live = SimpleNamespace(subscribers={'editor': object()})
    authority.sessions = {'viewed': live}
    p = policy(authority, 'viewed', server_spec, tmp_path)
    with policy_scope(p, authority=authority):
        first = _editor_servers()
    assert first, 'a viewed session keeps its editor servers connected after the execution'
    with policy_scope(p, authority=authority):
        assert _editor_servers() == first  # the same connection objects: no respawn per turn
    # Another session with the same editor config never adopts these connections.
    authority.sessions['other'] = SimpleNamespace(subscribers={'editor': object()})
    other = policy(authority, 'other', server_spec, tmp_path)
    with policy_scope(other, authority=authority):
        assert set(_editor_servers()) - set(first)
    live.subscribers.clear()
    authority.sessions['other'].subscribers.clear()
    release_editor_mcp(authority, ['viewed', 'other'])
    assert not _editor_servers()


def test_unviewed_session_releases_editor_connections_after_execution(server_spec, tmp_path):
    authority = owner()
    authority.sessions = {'quiet': SimpleNamespace(subscribers={})}
    p = policy(authority, 'quiet', server_spec, tmp_path)
    with policy_scope(p, authority=authority):
        assert _editor_servers()
    assert not _editor_servers()


def test_editor_scopes_are_per_session_and_per_profile():
    from gateway.session_local_mcp import session_scope
    one, other = owner(), SimpleNamespace(profile_id='other-profile')
    assert session_scope(one, 'parent') != session_scope(one, 'child')
    assert session_scope(one, 'parent') != session_scope(other, 'parent')


@pytest.mark.parametrize(('actor_profile', 'capabilities', 'reason'), [
    ('owned-profile', {'session:read', 'session:submit'}, 'permission_denied'),
    ('foreign-profile', {'session:read', 'session:submit', 'session:control'}, 'profile_mismatch'),
])
def test_editor_mcp_change_requires_control_in_the_owning_profile(server_spec, actor_profile, capabilities, reason):
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_mutations import _mutate_session
    authority = owner()
    actor = Principal('viewer', actor_profile, frozenset(capabilities), 'viewer-transport')
    params = {'session_id': 'held', 'request_id': 'r', 'expected_revision': 0, 'expected_generation': 0,
              'operation': 'editor_mcp', 'payload': {'mcp_servers': [server_spec]}}
    with pytest.raises(RuntimeStoreError, match=reason):
        asyncio.run(_mutate_session(authority, actor, SessionRef('owned-profile', 'held'), params))
    assert not _editor_servers()  # refused before any discovery


def test_store_payload_carries_only_the_fingerprint(server_spec):
    from gateway.session_local_mcp import fingerprint, private_mutation_payload
    from hermes_state_mutations import validate_action
    payload, servers = private_mutation_payload('editor_mcp', {'mcp_servers': [server_spec]})
    assert servers == [server_spec]
    assert payload == {'mcp_servers': {'fingerprint': fingerprint([server_spec])}}
    assert 'PRIVATE_A' not in json.dumps(payload)
    validate_action('editor_mcp', payload)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        validate_action('editor_mcp', {'mcp_servers': [server_spec]})
    branch, _ = private_mutation_payload('branch', {'mcp_servers': [server_spec]})
    validate_action('branch', branch)
