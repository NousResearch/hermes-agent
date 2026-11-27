"""A disabled MCP server whose key matches a built-in toolset must not drop that toolset.

Refreshing a local session rewrites toolsets from the profile's ``mcp_servers`` map.
Subtracting every server key removed ``memory`` when ``mcp_servers.memory`` was
disabled, so Desktop chats and ``hermes -z`` lost the built-in memory tool and the
memory nudge never fired. The server's own toolset name still leaves the policy.
"""
import json

from gateway.session_policy import LocalSessionPolicy
from gateway.session_policy_refresh import refreshed_policy


def _policy(toolsets, stored):
    return LocalSessionPolicy(
        source='cli', platform='cli', cwd='/', model='m', toolsets=tuple(toolsets),
        config_json=json.dumps(stored), request_json=json.dumps({'source': 'cli'}),
        terminal_json=json.dumps({'TERMINAL_ENV': 'local'}))


def test_disabled_mcp_memory_keeps_the_builtin_in_a_refreshed_local_policy():
    """A disabled ``mcp_servers.memory`` entry leaves the built-in ``memory`` toolset
    in place. A disabled server that does not share a built-in name (``scratch``)
    still loses its own toolset, and so does the colliding server's ``mcp-memory`` name."""
    stored = {
        'platform_toolsets': {'cli': ['memory', 'terminal']},
        'mcp_servers': {'scratch': {'command': 'scratch-mcp'}},
    }
    live = {
        'platform_toolsets': {'cli': ['memory', 'terminal']},
        'mcp_servers': {
            'memory': {'command': 'npx', 'args': ['@modelcontextprotocol/server-memory'], 'enabled': False},
            'scratch': {'command': 'scratch-mcp', 'enabled': False},
        },
    }
    policy = _policy(('memory', 'mcp-memory', 'scratch', 'terminal'), stored)
    terminal = {'TERMINAL_ENV': 'local'}

    reloaded, _, _ = refreshed_policy(policy, live, terminal, mcp_only=True)
    assert 'memory' in reloaded.toolsets
    assert 'terminal' in reloaded.toolsets
    assert 'scratch' not in reloaded.toolsets
    assert 'mcp-memory' not in reloaded.toolsets

    resumed, _, _ = refreshed_policy(policy, live, terminal)
    assert 'memory' in resumed.toolsets
    assert 'scratch' not in resumed.toolsets
