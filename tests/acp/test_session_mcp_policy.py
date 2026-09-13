"""Editor MCP credentials are borrowed and tools are frozen per canonical owner."""
import concurrent.futures
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import pytest

from gateway.session_policy import build_policy, bind_launch_key, policy_scope, restore_policy
from hermes_state_runtime import RuntimeStoreError


@pytest.fixture
def server_spec(tmp_path):
    script = tmp_path / 'owned_mcp.py'
    script.write_text('''import json, os, sys
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'owned-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':os.environ['BORROWED_CREDENTIAL']}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    return {'name': 'same-editor-name', 'command': sys.executable, 'args': [str(script)],
            'env': [{'name': 'BORROWED_CREDENTIAL', 'value': 'PRIVATE_A'}]}


def owner():
    return SimpleNamespace(instance_id='owned', epoch=1, profile_id='owned-profile')


def policy(authority, sid, spec, tmp_path):
    private = {}
    result = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'local',
        'toolsets': [], 'editor': {'mcp_servers': [spec]}}, {}, private_secrets=private)
    return bind_launch_key(authority, sid, result, None, config_secrets=private)


def test_opposing_editors_freeze_tools_and_release_only_their_session(server_spec, tmp_path):
    from tools.registry import registry
    authority = owner()
    left = policy(authority, 'a', server_spec, tmp_path)
    other = json.loads(json.dumps(server_spec))
    other['env'][0]['value'] = 'PRIVATE_B'
    right = policy(authority, 'b', other, tmp_path)
    import threading
    ready = threading.Barrier(2)
    released = threading.Event()
    def run(p, marker):
        with policy_scope(p, authority=authority):
            names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
            assert len(names) == 1
            ready.wait(timeout=20)
            if marker == 'PRIVATE_B':
                assert released.wait(20)
            result = registry.dispatch(names[0], {})
            assert marker in result
        if marker == 'PRIVATE_A':
            released.set()
        return names
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        a = pool.submit(run, left, 'PRIVATE_A')
        b = pool.submit(run, right, 'PRIVATE_B')
        an, bn = a.result(40), b.result(40)
    assert an != bn
    assert not any(registry.get_entry(n) for n in an + bn)
    durable = json.dumps(asdict(left))
    assert 'PRIVATE_A' not in durable and str(tmp_path / 'owned_mcp.py') not in durable


def test_cold_editor_rebind_requires_exact_borrowed_config(server_spec, tmp_path):
    from gateway.session_local_mcp import attach_editor_mcp
    from tools.registry import registry
    original = policy(owner(), 'cold', server_spec, tmp_path)
    cold = restore_policy(json.loads(json.dumps(asdict(original))))
    authority = owner()
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        with policy_scope(cold, authority=authority):
            pass
    changed = json.loads(json.dumps(server_spec))
    changed['env'][0]['value'] = 'CHANGED_SECRET'
    with pytest.raises(RuntimeStoreError, match='acp_mcp_policy_conflict'):
        attach_editor_mcp(authority, cold, [changed])
    attach_editor_mcp(authority, cold, [server_spec])
    with policy_scope(cold, authority=authority):
        names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
        assert 'PRIVATE_A' in registry.dispatch(names[0], {})
    assert cold.toolsets == original.toolsets
