"""An editor chat keeps working while its editor's MCP servers are not attached (#145567).

Only a fingerprint and the schema manifest of an editor's MCP servers are durable; their
specification (argv, env) lives in the owner's memory, and only the editor resupplies it, when it
opens the chat. After a gateway restart, or for a turn from another client, every turn and every
fork of the chat was refused with the API-key copy (``launch_credentials_unavailable``).
"""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gateway.session_policy import restore_policy
from tests.acp_adapter.test_session_mcp_policy import owner, policy


def _offered(toolsets):
    """The tools an agent built for *toolsets* here can reach: its catalog, before ``tool_search``
    collapses MCP tools behind its bridge."""
    from model_tools import get_tool_definitions
    return [d['function']['name'] for d in get_tool_definitions(
        enabled_toolsets=list(toolsets), quiet_mode=True, skip_tool_search_assembly=True)]


def test_turn_after_restart_runs_without_editor_tools_and_says_how_to_get_them_back(
        server_spec, tmp_path, monkeypatch):
    from gateway import session_authorities as authorities, session_policy
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_local_mcp import attach_editor_mcp
    frozen = policy(owner(), 'editor-chat', server_spec, tmp_path)
    cold = restore_policy(json.loads(json.dumps(asdict(frozen))))  # a restarted owner restores it
    restarted = owner()  # holds no editor servers
    runners = [TurnRunner(SimpleNamespace(), SimpleNamespace(source=None, session_id='editor-chat', session_key=None))
               for _ in range(2)]
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: cold)
    monkeypatch.setattr(authorities, 'active_authority', lambda runner: restarted)
    turns, notices = [], []
    monkeypatch.setattr(TurnRunner, '_notice_callback_sync', lambda self, notice: notices.append(notice))

    def run_turn(self):
        turns.append(_offered(cold.toolsets))
        return {'final_response': 'ran', 'messages': [], 'api_calls': 1, 'completed': True}
    monkeypatch.setattr(TurnRunner, '_run_sync_scoped', run_turn)

    result = runners[0].run_sync()
    assert result.get('completed') and not result.get('failed'), result
    assert turns and not [name for name in turns[0] if name.startswith('mcp__')], turns
    assert len(notices) == 1 and notices[0].level == 'warn', notices
    text = notices[0].text
    assert 'MCP' in text and 'Reopen this chat from your editor' in text and 'API key' not in text, text
    assert (cold.editor_mcp_json, cold.toolsets) == (frozen.editor_mcp_json, frozen.toolsets)  # nothing re-frozen

    attach_editor_mcp(restarted, cold, [server_spec])  # the editor opens the chat again
    del turns[:], notices[:]
    assert runners[1].run_sync().get('completed')
    assert [name for name in turns[0] if name.startswith('mcp__')] and not notices, (turns, notices)


class _Bridge:
    def __init__(self, connection):
        self.connection = connection

    async def rpc(self, method, **params):
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        return reply['error']['message'] if 'error' in reply else reply['result']


@pytest.fixture
def editor_chat(tmp_path, monkeypatch, server_spec):
    """An ACP chat created with the editor's MCP server, on a real authority and SQLite."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    built = []

    async def build():
        authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
        built.append(authority)
        connection = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})
        ref = create_local_session(authority, connection.actor, {
            'request_id': 'editor', 'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': [],
            'editor': {'mcp_servers': [server_spec]}})
        return authority, _Bridge(connection), ref.session_id
    try:
        yield build
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        for authority in built:
            for scope in list(getattr(authority, '_editor_mcp_held', {})):
                shutdown_mcp_servers(scope=scope)
        store.close_all_db_handles()


def _stored(authority, sid):
    from hermes_state_local import local_receipt
    return local_receipt(authority.db, sid)['policy']


@pytest.mark.asyncio
async def test_fork_after_restart_starts_without_editor_servers(editor_chat, server_spec):
    authority, client, sid = await editor_chat()
    parent = _stored(authority, sid)
    assert parent['editor_mcp_json'] and any(name.startswith('mcp-editor_') for name in parent['toolsets'])
    authority._local_editor_mcp.clear()  # what a gateway restart leaves: the specification is gone
    row = authority.db.get_session(sid)
    forked = await client.rpc('session.mutate', session_id=sid, request_id='fork', operation='branch', payload={},
                              expected_revision=row['runtime_revision'],
                              expected_generation=row['runtime_generation'])
    assert isinstance(forked, dict), forked  # main: 'launch_credentials_unavailable'
    child = _stored(authority, forked['branched_session_id'])
    assert not child.get('editor_mcp_json'), child
    assert not [name for name in child['toolsets'] if name.startswith('mcp-editor_')], child['toolsets']
    assert _stored(authority, sid) == parent  # the parent keeps its frozen selection for its editor
