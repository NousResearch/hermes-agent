"""The integrated tool executor retains read proof beside plugin and late LSP notes."""
import json
import threading
from hashlib import sha256
from pathlib import Path

import pytest

from tests.agent.test_run_agent import agent, _mock_tool_call, _mock_assistant_msg
from tests.gateway.test_managed_source_context import inert_runtime


@pytest.fixture(autouse=True)
def inert_file_reads(monkeypatch, inert_runtime):
    from tools.file_operations import ReadResult
    class FileOps:
        env = None
        @staticmethod
        def _add_line_numbers(text, offset):
            return '\n'.join(f'{n}|{line}' for n, line in enumerate(text.removesuffix('\n').split('\n'), offset))
        def read_file(self, path, offset, limit):
            text = Path(path).read_text()
            selected = ''.join(text.splitlines(keepends=True)[offset-1:offset-1+limit])
            return ReadResult(content=self._add_line_numbers(selected, offset),
                              total_lines=len(text.splitlines()), file_size=len(text.encode()))
    monkeypatch.setattr('tools.file_tools._get_file_ops', lambda *a: FileOps())
    monkeypatch.setattr('agent.tool_executor.get_active_env', lambda *a: None)
    monkeypatch.setattr('hermes_cli.plugins.discover_plugins', lambda: None)
    monkeypatch.setattr('tools.env_probe.warm_environment_probe_async', lambda: None)
    done = threading.Event()
    done.set()
    monkeypatch.setattr('run_agent._openrouter_prewarm_done', done)


@pytest.mark.parametrize('note_kind', ['plugin', 'lsp'])
@pytest.mark.parametrize('case', ['ok', 'prefix', 'suffix', 'forged_prefix', 'source_change', 'private', 'secret', 'replay'])
def test_read_file_proof_survives_integrated_notes(agent, tmp_path, monkeypatch, note_kind, case):
    from agent.llm_egress_runtime import dispatch_provider_request
    from agent.llm_egress_firewall import EgressBlocked
    from agent.source_provenance import SourceProvenanceError
    from agent.plugin_live_notes import queue_plugin_notice
    from agent.lsp import deferred
    from tools.file_tools import read_file_tool

    source = tmp_path / 'source.py'
    source.write_text('value = 1\n')
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_1234abcd')
    agent.provider, agent.base_url = 'nous', 'https://inference-api.nousresearch.com/v1'
    agent._base_url_lower = agent.base_url.lower()
    agent.session_id, agent._current_turn_id = 'notes-session', 'notes-turn'
    agent._current_api_request_id = 'notes-turn:api:1'
    agent._llm_egress_policy_digest = sha256(b'policy').hexdigest()
    agent._llm_egress_state_dir = tmp_path / 'egress'
    note = '[System note: plugin fixture is live.]' if note_kind == 'plugin' else '[LSP: fixture check complete.]'
    if case == 'private': note += ' /Users/fixture/private/notes.txt'
    if case == 'secret': note += ' sk-proj-' + 'x' * 64
    if case == 'private': monkeypatch.delenv('HERMES_KANBAN_TASK')
    def read(*a, **k):
        from agent.source_provenance import active_source_provenance
        assert active_source_provenance() is not None
        if note_kind == 'plugin': queue_plugin_notice(agent, 'fixture', note)
        else: deferred.post(agent.session_id, str(source), note, None)
        return read_file_tool(str(source), task_id='notes-read')
    monkeypatch.setattr('model_tools.handle_function_call', read)
    messages = [{'role': 'assistant', 'tool_calls': [{'id': 'read-note', 'type': 'function',
        'function': {'name': 'read_file', 'arguments': '{"path":"source.py"}'}}]}]
    call = _mock_tool_call(name='read_file', arguments=json.dumps({'path': str(source)}), call_id='read-note')
    agent._execute_tool_calls(_mock_assistant_msg(content='', tool_calls=[call]), messages, 'notes-read')
    [message] = [m for m in messages if m.get('role') == 'tool']
    assert message.get('_source_provenance') is not None
    assert message['content'].endswith(note)
    assert note_kind != 'plugin' or not agent._plugin_live_notes
    agent._current_api_request_id = 'notes-turn:api:2'
    kwargs = agent._build_api_kwargs(messages, tools_for_api=[])
    if case in {'prefix', 'forged_prefix'}:
        kwargs['messages'][-1]['content'] = kwargs['messages'][-1]['content'].replace('value = 1', 'value = 2')
    if case == 'suffix': kwargs['messages'][-1]['content'] += ' substituted note'
    if case == 'forged_prefix':
        proof = kwargs['_hermes_source_provenance'][0]
        text = kwargs['messages'][-1]['content']
        proof['content_sha256'] = sha256(text.encode()).hexdigest()
        proof['presentation_sha256'] = sha256(text[:proof['presentation_chars']].encode()).hexdigest()
    if case == 'source_change': source.write_text('value = 2\n')
    if case == 'replay':
        agent._current_turn_id, agent._current_api_request_id = 'other-turn', 'other-turn:api:1'
    captured = []
    if case != 'ok':
        with pytest.raises((EgressBlocked, SourceProvenanceError)):
            dispatch_provider_request(agent, kwargs, lambda body: captured.append(body))
        assert captured == []
        return
    dispatch_provider_request(agent, kwargs, lambda body: captured.append(body))
    assert captured[0]['messages'][-1]['content'] == message['content']
    receipt = json.loads((tmp_path / 'egress' / 'llm-egress-receipts.jsonl').read_text().splitlines()[-1])
    assert receipt['source_grant_count'] == receipt['source_segment_count'] == 1
