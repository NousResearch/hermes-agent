"""Inert SDK boundary for actual AIAgent context/persistence/loop tests."""
import copy
from types import MethodType
from unittest.mock import MagicMock

def inert_agent(monkeypatch, db, sid):
    from run_agent import AIAgent
    from agent.conversation_loop import run_conversation
    from tests.agent.test_run_agent import _mock_response
    # Construction cannot discover tools, connect a client, title, or fetch metadata.
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda *a, **k: [])
    monkeypatch.setattr('model_tools.check_toolset_requirements', lambda *a, **k: {})
    monkeypatch.setattr('agent.process_bootstrap.OpenAI', MagicMock())
    monkeypatch.setattr('agent.agent_init.fetch_model_metadata', lambda *a, **k: {})
    monkeypatch.setattr('agent.model_metadata.get_model_context_length', lambda *a, **k: 256000)
    def no_socket(*a, **k):
        raise AssertionError('inert persistence test attempted network I/O')
    monkeypatch.setattr('socket.socket.connect', no_socket)
    monkeypatch.setattr('socket.socket.connect_ex', no_socket)
    monkeypatch.setattr('hermes_cli.plugins.invoke_hook', lambda *a, **k: [])
    monkeypatch.setattr('agent.turn_context._maybe_title_session_at_turn_start', lambda *a: None)
    agent = AIAgent(api_key='inert-key', base_url='https://inert.invalid/v1',
        provider='openai-compat', model='inert-model', enabled_toolsets=[], max_iterations=3,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        session_db=db, session_id=sid, save_trajectories=False)
    agent._model_supports_vision = lambda: True
    agent._cached_system_prompt = 'Frozen system prefix'
    agent._skip_mcp_refresh = True
    agent.compression_enabled = False
    agent.tool_delay = 0
    agent._use_prompt_caching = False
    agent._cleanup_task_resources = lambda *a: None
    agent._try_refresh_env_client_credentials = lambda: None
    agent._restore_primary_runtime = lambda: None
    agent._cleanup_dead_connections = lambda: False
    agent.run_conversation = MethodType(run_conversation, agent)
    sent, at_provider = [], []
    def provider(**kwargs):
        sent.append(copy.deepcopy(kwargs['messages']))
        at_provider.append([dict(r) for r in db._conn.execute(
            'SELECT * FROM messages WHERE session_id=? ORDER BY id', (sid,))])
        return _mock_response(content='inert answer', finish_reason='stop')
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = provider
    return agent, sent, at_provider
