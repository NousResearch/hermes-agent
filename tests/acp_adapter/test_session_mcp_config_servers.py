"""A config MCP server stays callable in a session whose editor supplies MCP servers (#143502).

The editor's servers put every turn of the session inside its own tool scope; the profile's
config servers, connected outside any session as at gateway boot, must still dispatch there.
"""
from gateway.session_policy import policy_scope
from tests.acp_adapter.test_session_mcp_policy import owner, policy


def test_a_config_mcp_server_stays_callable_in_an_editor_session(server_spec, tmp_path):
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    config = {'command': server_spec['command'], 'args': server_spec['args'],
              'env': {'BORROWED_CREDENTIAL': 'FROM_PROFILE'}}
    register_mcp_servers({'profile': config})
    try:
        authority = owner()
        session = policy(authority, 'editor', server_spec, tmp_path)
        with policy_scope(session, authority=authority):
            assert 'FROM_PROFILE' in registry.dispatch('mcp__profile__echo', {})
    finally:
        shutdown_mcp_servers(names={'profile'})
