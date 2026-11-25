"""Which editor MCP servers a session freezes, and under which names (#134889 ACP-8, ACP-9)."""
import json
import sys
import time

from gateway.session_policy import build_policy, bind_launch_key, policy_scope
from tests.acp_adapter.test_session_mcp_policy import owner


def _policy(authority, sid, specs, tmp_path):
    private = {}
    result = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'local',
        'toolsets': [], 'editor': {'mcp_servers': specs}}, {}, private_secrets=private)
    return bind_launch_key(authority, sid, result, None, config_secrets=private)


def _editor_tools():
    from tools.registry import registry
    return [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]


def test_a_broken_editor_server_is_left_out_of_the_session(server_spec, tmp_path):
    from tools.registry import registry
    from tools.mcp_tool import _servers
    silent = {'name': 'silent', 'command': sys.executable, 'args': ['-c', 'import time; time.sleep(120)'], 'env': []}
    authority = owner()
    started = time.monotonic()
    p = _policy(authority, 'mixed', [silent, server_spec], tmp_path)
    assert time.monotonic() - started < 25
    started = time.monotonic()
    with policy_scope(p, authority=authority):
        names = _editor_tools()
        assert len(names) == 1 and 'PRIVATE_A' in registry.dispatch(names[0], {})
    # The execution never respawned the broken server (that alone would cost its connect timeout).
    assert time.monotonic() - started < 4
    assert not any(key[-1].startswith('editor_') for key in _servers)


def test_editor_tool_names_keep_the_server_name(server_spec, tmp_path, monkeypatch):
    authority = owner()
    p = _policy(authority, 'named', [server_spec], tmp_path)
    with policy_scope(p, authority=authority):
        (name,) = _editor_tools()
    assert name.startswith('mcp__editor_same_editor_name_') and name.endswith('__echo')
    # A manifest frozen before names were recorded keeps executing under the names it froze.
    import gateway.session_local_mcp as local_mcp
    import hashlib
    def legacy(scope, servers):
        prefix = hashlib.sha256(scope.encode()).hexdigest()[:20]
        return {i: f'editor_{prefix}_{i}' for i in range(len(servers))}
    with monkeypatch.context() as patched:
        patched.setattr(local_mcp, '_server_names', legacy)
        old = _policy(authority, 'legacy', [server_spec], tmp_path)
    data = json.loads(old.editor_mcp_json)
    del data['names']
    from dataclasses import replace
    old = replace(old, editor_mcp_json=json.dumps(data))
    with policy_scope(old, authority=authority):
        (legacy_name,) = _editor_tools()
    assert legacy_name in data['schemas']
