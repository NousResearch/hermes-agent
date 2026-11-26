"""Native Goal continuation must receive a useful turn before terminal-stop coaching.

Real agent loop, stop gates, finalizer, SQLite history/board and lifecycle tools;
only model/judge transports and unrelated discovery are inert.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway import session_kanban
from hermes_cli import goals, kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


@pytest.fixture
def worker(tmp_path, monkeypatch, request):
    params = getattr(request, 'param', {})
    goal_mode = params.get('goal_mode', True)
    import agent.kanban_stop
    import agent.turn_stop_gates
    import agent.turn_finalizer
    import hermes_cli.cli_single_query
    source = Path(__file__).resolve().parents[2]
    for module in (agent.kanban_stop, agent.turn_stop_gates, agent.turn_finalizer,
                   session_kanban, goals, hermes_cli.cli_single_query):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _make_tool_defs

    for key in tuple(os.environ):
        if key.startswith('HERMES_KANBAN_') or key == 'HERMES_DELEGATED_CHILD_CONTEXT':
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    monkeypatch.setattr('hermes_cli.plugins.discover_plugins', lambda: None)
    monkeypatch.setattr('hermes_cli.plugins.PluginManager.discover_and_load', lambda self, force=False: None)
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **kw: _make_tool_defs(
        'read_file', 'write_file', 'kanban_request_review', 'kanban_complete', 'kanban_block'))
    monkeypatch.setattr('model_tools.check_toolset_requirements', lambda **kw: {})
    monkeypatch.setattr('agent.process_bootstrap.OpenAI', MagicMock())
    monkeypatch.setattr(session_kanban, '_watch_submitter', lambda *a: None)
    def no_network(*a, **kw):
        raise AssertionError('network disabled in behavioral fixture')
    monkeypatch.setattr('socket.socket.connect', no_network)
    monkeypatch.chdir(tmp_path)
    kb.init_db()
    with connect_closing() as conn:
        tid = kb.create_task(conn, title='Write a two-section handoff document',
                             body='Both sections must exist before review.', assignee='default',
                             goal_mode=goal_mode, goal_max_turns=3)
        task = kb.claim_task(conn, tid, claimer='test-worker')
    context = dict(task_id=tid, board='default', db=str(kb.kanban_db_path()),
                   run_id=task.current_run_id, claim_lock=task.claim_lock,
                   workspace=str(tmp_path), workspaces_root=str(tmp_path), profile='default',
                   branch=None, tenant=None, goal_mode=goal_mode, goal_max_turns=3,
                   max_runtime_seconds=None, skills=[], context='', goal_text=task.title + '\n' + task.body)
    frame = {'text': 'Write the handoff document.',
             'policy': {'source': 'kanban', 'kanban_json': json.dumps(context)}}
    # Binder owns process-global values; monkeypatch restores them after this private-exec seam.
    for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_DB',
                'HERMES_KANBAN_RUN_ID', 'HERMES_KANBAN_CLAIM_LOCK', 'HERMES_KANBAN_WORKSPACE',
                'HERMES_KANBAN_WORKSPACES_ROOT', 'HERMES_KANBAN_GOAL_MODE', 'HERMES_PROFILE',
                'HERMES_SESSION_SOURCE', 'HERMES_KANBAN_CLAIM_TTL_SECONDS'):
        monkeypatch.setenv(key, '')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', params.get('inherited_goal', ''))
    session_kanban.bind_worker_context(frame)
    db = SessionDB(db_path=tmp_path / 'home' / 'state.db')
    agent = AIAgent(api_key='inert-test-key', base_url='https://inert.invalid/v1',
                    model='test/model', provider='openrouter', session_id='artifact-session',
                    session_db=db, quiet_mode=True, skip_context_files=True,
                    skip_memory=True, skip_background_review=True, max_iterations=10)
    agent.client = MagicMock()
    agent._cached_system_prompt = 'Write and verify the requested artifact.'
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    agent._stall_guards = False
    yield SimpleNamespace(agent=agent, frame=frame, context=context, db=db,
                          artifact=tmp_path / 'handoff.md')
    agent.release_clients()
    db.close()


def _run_caller(worker, caller, monkeypatch, *, expected_exit=0, notify_linger=None, follow_up=None):
    if caller == 'sdk':
        result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
        assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == expected_exit
        if expected_exit == 0:
            assert result['completed'] and not result.get('failed')
        return result
    import cli
    from hermes_cli.cli_single_query import _run_quiet_single_query, _run_single_query_mode

    surface = SimpleNamespace(agent=worker.agent, session_id=worker.agent.session_id,
        conversation_history=[], _last_turn_result=None, _single_query_mode=False,
        _claim_active_session=lambda *a, **kw: True,
        console=SimpleNamespace(print=lambda *a, **kw: None),
        _show_security_advisories=lambda: None, _print_exit_summary=lambda **kw: None)

    def chat(prompt, **kw):
        if follow_up is not None and surface.conversation_history:
            return follow_up(surface, prompt)
        result = surface.agent.run_conversation(prompt, conversation_history=surface.conversation_history)
        surface.conversation_history = result['messages']
        surface._last_turn_result = result
        return result['final_response']

    surface.chat = chat
    monkeypatch.setattr(cli, '_should_seed_interactive', lambda *a, **kw: False)
    monkeypatch.setattr(cli, '_collect_query_images', lambda query, image: (query, []))
    monkeypatch.setattr(cli, '_collect_kanban_task_images', lambda images: [])
    monkeypatch.setattr(cli, '_finalize_single_query', lambda surface: None)
    monkeypatch.setattr('hermes_cli.quiet_single_query.continue_quiet_notify_completions',
                        notify_linger or (lambda *a, **kw: None))
    with pytest.raises(SystemExit) as exited:
        if caller == 'cli-quiet':
            _run_quiet_single_query(surface, worker.frame['text'])
        else:
            _run_single_query_mode(surface, worker.frame['text'], None, False, True)
    assert exited.value.code == expected_exit


@pytest.mark.parametrize('caller', ['sdk', 'cli-quiet', 'cli-chat'])
def test_goal_receives_partial_then_continues_same_session(worker, monkeypatch, caller):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    agent, artifact = worker.agent, worker.artifact
    partial = 'Section one is written and verified; section two is not present.'
    second = 'Both sections are written and verified. Ready for review.'
    requests = []
    requests_at_continuation = []
    judged = []

    def judge(goal, response=None, last_response=None, **kw):
        response = response if response is not None else last_response
        judged.append(response)
        return ('continue' if response == partial else 'done', 'check both sections', False, None, False)

    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr('tools.kanban_tools.judge_goal', judge)
    monkeypatch.setattr('agent.auxiliary_client.get_text_auxiliary_client', lambda *a, **kw: (object(), 'inert-judge'))
    script = [
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'write_file', json.dumps({'path': str(artifact), 'content': '# First\nVerified first section.\n'}))]),
        _mock_response(content=partial),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'read_file', json.dumps({'path': str(artifact)}))]),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'write_file', json.dumps({'path': str(artifact), 'content': '# First\nVerified first section.\n# Second\nVerified second section.\n'}))]),
        _mock_response(content=second),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'kanban_request_review', json.dumps({'summary': 'Both document sections written and verified.',
                                                'artifacts': [str(artifact)]}))]),
        _mock_response(content='Both sections are preserved for review.'),
    ]

    def transport(**kw):
        requests.append((agent.session_id, copy.deepcopy(kw['messages'])))
        if len(requests) == 3:
            requests_at_continuation.append((list(judged), artifact.read_text()))
        assert script, 'unexpected extra model entry'
        return script.pop(0)

    agent.client.chat.completions.create.side_effect = transport
    _run_caller(worker, caller, monkeypatch)
    assert requests_at_continuation == [([partial], '# First\nVerified first section.\n')], (
        'premature terminal nudge bypassed native Goal judgment', requests[2][1][-1])
    assert requests[2][1][-1]['content'] == goals.KANBAN_GOAL_CONTINUATION_TEMPLATE.format(
        reason='check both sections')
    assert any(m.get('content') == partial for m in requests[2][1])
    assert any(m.get('content') == second for m in requests[5][1])
    assert judged[:2] == [partial, second]
    assert len(requests) == 7 and not script
    assert {sid for sid, _ in requests} == {'artifact-session'}
    assert getattr(agent, '_kanban_stop_nudges', 0) == 0
    history = worker.db.get_messages_as_conversation(agent.session_id)
    assert any(m.get('content') == partial for m in history)
    assert '# Second' in artifact.read_text()
    with connect_closing() as conn:
        assert kb.get_task(conn, worker.context['task_id']).status == 'review'
        run = kb.latest_run(conn, worker.context['task_id'])
        assert run is not None and run.status == 'review'
        assert [row[0] for row in conn.execute('SELECT id FROM task_runs WHERE task_id=?',
                                               (worker.context['task_id'],))] == [worker.context['run_id']]
    if caller == 'sdk':
        assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == 0


@pytest.mark.parametrize(('tool', 'args', 'status'), [
    ('kanban_complete', {'summary': 'Both sections verified.'}, 'done'),
    ('kanban_request_review', {'summary': 'Both sections ready for review.'}, 'review'),
    ('kanban_block', {'reason': 'Only the customer can provide the source document.',
                      'kind': 'needs_input'}, 'blocked'),
])
def test_first_turn_handoff_is_not_reopened(worker, monkeypatch, tool, args, status):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    def unexpected_judge(*args, **kwargs):
        pytest.fail('The outer loop judged a run that had already settled')

    monkeypatch.setattr(goals, 'judge_goal', unexpected_judge)
    judgments = []

    def judge(goal, last_response, **kwargs):
        judgments.append(last_response)
        return ('blocked' if tool == 'kanban_block' else 'done', 'handoff accepted', False, None, False)

    monkeypatch.setattr('tools.kanban_tools.judge_goal', judge)
    monkeypatch.setattr('agent.auxiliary_client.get_text_auxiliary_client',
                        lambda *a, **kw: (object(), 'inert-judge'))
    transport = worker.agent.client.chat.completions.create
    transport.side_effect = [
        _mock_response(content='', finish_reason='tool_calls',
                       tool_calls=[_mock_tool_call(tool, json.dumps(args))]),
        _mock_response(content='Handed off.'),
    ]

    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])

    assert result['final_response'] == 'Handed off.'
    assert transport.call_count == 2
    assert len(judgments) == 1
    with connect_closing() as conn:
        task_id = worker.context['task_id']
        task = kb.get_task(conn, task_id)
        assert task.status == status and task.current_run_id is None
        assert conn.execute('SELECT id FROM task_runs WHERE task_id=?',
                            (task_id,)).fetchall()[0][0] == worker.context['run_id']
        assert conn.execute('SELECT COUNT(*) FROM task_runs WHERE task_id=?',
                            (task_id,)).fetchone()[0] == 1


@pytest.mark.parametrize('kind', ['needs_input', 'dependency'])
def test_premature_block_cannot_bypass_goal_judgment(worker, monkeypatch, kind):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    reason = 'Only the first section is written; I would prefer that someone else writes the rest.'
    partial = 'The first section is ready, and I can still write the second.'
    handoff_checks, loop_checks = [], []

    def check_block(goal, last_response):
        handoff_checks.append(last_response)
        return 'continue', 'No external blocker identified', False, None, False

    def check_partial(goal, last_response):
        loop_checks.append(last_response)
        return 'blocked', 'test stops the attempt after observing the partial', False, None, False

    monkeypatch.setattr('tools.kanban_tools.judge_goal', check_block)
    monkeypatch.setattr(goals, 'judge_goal', check_partial)
    monkeypatch.setattr('agent.auxiliary_client.get_text_auxiliary_client',
                        lambda *a, **kw: (object(), 'inert-judge'))
    worker.agent.client.chat.completions.create.side_effect = [
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'kanban_block', json.dumps({'kind': kind, 'reason': reason}))]),
        _mock_response(content=partial),
    ]

    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
    assert loop_checks == [partial], 'worker block stopped the run before the partial was judged'
    tool_result = next(m for m in result['messages'] if m['role'] == 'tool' and m.get('name') == 'kanban_block')
    assert json.loads(tool_result['content']).get('error')
    if kind == 'needs_input':
        assert len(handoff_checks) == 1 and reason in handoff_checks[0]
    with connect_closing() as conn:
        assert kb.get_task(conn, worker.context['task_id']).status == 'blocked'


@pytest.mark.parametrize('caller', ['sdk', 'cli-quiet', 'cli-chat'])
def test_unavailable_goal_judge_is_not_worker_success(worker, monkeypatch, caller):
    from tests.agent.test_run_agent import _mock_response

    # Isolate the outage from the separately tested pre-judge stop nudge.
    monkeypatch.setenv('HERMES_KANBAN_STOP_NUDGE', '0')
    partial = 'The first section is done; the second is not.'
    judged = []

    def unavailable(goal, last_response):
        judged.append(last_response)
        return 'continue', 'auxiliary provider unavailable', False, None, True

    monkeypatch.setattr(goals, 'judge_goal', unavailable)
    transport = worker.agent.client.chat.completions.create
    transport.side_effect = [_mock_response(content=partial)]

    _run_caller(worker, caller, monkeypatch, expected_exit=1)

    assert judged == [partial]
    assert transport.call_count == 1
    if caller == 'sdk':
        receipt = session_kanban.worker_result(worker.context['db'], worker.context)
        assert receipt['last_output'] == partial
    with connect_closing() as conn:
        task = kb.get_task(conn, worker.context['task_id'])
        assert task.status == 'running' and task.current_run_id == worker.context['run_id']


@pytest.mark.parametrize('path', ['partial', 'needs_input'])
def test_judge_import_failure_is_an_outage_not_a_continue_verdict(worker, monkeypatch, path):
    from tests.agent.test_run_agent import _mock_response
    from tools import kanban_tools

    real_judge, judged = goals.judge_goal, []

    def judge(*args, **kwargs):
        # The real judge, with its auxiliary client unimportable only while it runs.
        with pytest.MonkeyPatch.context() as broken:
            broken.setitem(sys.modules, 'agent.auxiliary_client', None)
            judged.append(real_judge(*args, **kwargs))
        return judged[-1]

    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr(kanban_tools, 'judge_goal', judge)
    monkeypatch.setenv('HERMES_KANBAN_STOP_NUDGE', '0')
    tid, run_id = worker.context['task_id'], worker.context['run_id']
    if path == 'partial':
        transport = worker.agent.client.chat.completions.create
        transport.side_effect = [_mock_response(content='Section one is ready. Section two remains.')]
        _run_caller(worker, 'sdk', monkeypatch, expected_exit=1)
        assert transport.call_count == 1 and len(judged) == 1
        expected = ('running', run_id)
    else:
        reason = 'Only the operator can choose the required export format.'
        assert json.loads(kanban_tools._handle_block({'kind': 'needs_input', 'reason': reason}))['ok']
        assert len(judged) == 1
        expected = ('blocked', None)
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        assert (task.status, task.current_run_id) == expected and run.id == run_id
        if path == 'needs_input':
            assert 'judge unavailable' in run.summary.lower()


@pytest.mark.parametrize('action', ['complete', 'request_review'])
@pytest.mark.parametrize('surface', ['tool', 'cli'])
@pytest.mark.parametrize('importable', [True, False])
def test_unimportable_judge_does_not_reject_handoff(worker, monkeypatch, action, surface, importable):
    from argparse import Namespace
    from agent import auxiliary_client as aux
    from hermes_cli import kanban as kanban_cli
    from tests.agent.test_run_agent import _mock_response
    from tools import kanban_tools

    # A configured judge that rejects: only the failed import can let the handoff through (#83610).
    verdict = _mock_response(content='{"verdict":"continue","reason":"The second section is missing."}')
    client = SimpleNamespace(base_url='https://inert.invalid/v1', chat=SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock(return_value=verdict))))
    monkeypatch.setattr(aux, '_get_auxiliary_task_config',
                        lambda task: {'provider': 'custom:judge', 'model': 'judge-model'} if task == 'goal_judge' else {})
    monkeypatch.setattr(aux, 'resolve_provider_client', lambda name, *a, **kw: (client, 'judge-model'))
    tid, run_id = worker.context['task_id'], worker.context['run_id']
    summary = 'Both sections were written and verified.'
    with pytest.MonkeyPatch.context() as entry:
        if not importable:
            entry.setitem(sys.modules, 'agent.auxiliary_client', None)
        if surface == 'tool':
            accepted = json.loads(getattr(kanban_tools, '_handle_' + action)({'task_id': tid, 'summary': summary})).get('ok', False)
        else:
            accepted = getattr(kanban_cli, '_cmd_' + action)(Namespace(
                task_id=tid, task_ids=[tid], summary=summary, result=None, force=False)) == 0
    assert accepted is not importable
    assert client.chat.completions.create.call_count == int(importable)
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        landed = 'running' if importable else 'done' if action == 'complete' else 'review'
        assert task.status == landed and run.id == run_id


def test_failed_gateway_continuation_is_not_judged_as_progress(worker, monkeypatch):
    from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    from tests.agent.test_run_agent import _mock_response

    monkeypatch.setenv('HERMES_KANBAN_STOP_NUDGE', '0')
    partial = 'The first section is done; the second is not.'
    failed = {'final_response': '', 'completed': False, 'failed': True, 'failure_reason': 'auth'}
    judged, turns = [], []

    def judge(goal, last_response):
        judged.append(last_response)
        return 'continue', 'write section two', False, None, False

    monkeypatch.setattr(goals, 'judge_goal', judge)
    worker.agent.client.chat.completions.create.side_effect = [_mock_response(content=partial)]
    run = worker.agent.run_conversation

    def run_with_failed_continuation(prompt, **kwargs):
        turns.append(prompt)
        if len(turns) == 1:
            return run(prompt, **kwargs)
        assert len(turns) == 2, 'the gateway exposed a failed turn as normal progress'
        return failed

    monkeypatch.setattr(worker.agent, 'run_conversation', run_with_failed_continuation)
    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])

    assert len(turns) == 2 and judged == [partial]
    assert result['failure_reason'] == 'auth'
    assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE


@pytest.mark.parametrize('write_lands', [True, False])
def test_one_turn_budget_judges_before_blocking(worker, monkeypatch, write_lands):
    from tests.agent.test_run_agent import _mock_response

    partial = 'Section one is ready. Section two is missing.'
    judged = []

    def judge(goal, last_response):
        judged.append(last_response)
        return 'continue', 'write section two', False, None, False

    worker.context['goal_max_turns'] = 1
    worker.frame['policy']['kanban_json'] = json.dumps(worker.context)
    monkeypatch.setenv('HERMES_KANBAN_STOP_NUDGE', '0')
    monkeypatch.setattr(goals, 'judge_goal', judge)
    if not write_lands:
        monkeypatch.setattr(kb, 'block_task', lambda *a, **kw: False)
    transport = worker.agent.client.chat.completions.create
    transport.side_effect = [_mock_response(content=partial)]

    _run_caller(worker, 'sdk', monkeypatch, expected_exit=0 if write_lands else 1)

    assert transport.call_count == 1 and judged == [partial]
    with connect_closing() as conn:
        task = kb.get_task(conn, worker.context['task_id'])
        run = kb.latest_run(conn, task.id)
        assert task.status == ('blocked' if write_lands else 'running')
        assert run.id == worker.context['run_id']
        if write_lands:
            assert 'exhausted its turn budget (1/1)' in run.summary


@pytest.mark.parametrize('action', ['block', 'schedule', 'request-review', 'complete'])
def test_cli_worker_terminal_rejection_keeps_claim(worker, monkeypatch, action):
    from argparse import Namespace
    from hermes_cli import kanban as kanban_cli

    tid = worker.context['task_id']
    monkeypatch.setattr(goals, 'judge_goal', lambda **kw: ('continue', 'keep working', False, None, False))
    handler = getattr(kanban_cli, '_cmd_' + action.replace('-', '_'))
    assert handler(Namespace(task_id=tid, task_ids=[tid], kind='needs_input', reason=['unfinished'],
                             summary='The first section is written.', result=None)) == 1
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == 'running' and task.current_run_id == worker.context['run_id']
        assert not kb.list_comments(conn, tid)


@pytest.mark.parametrize('caller', ['sdk', 'cli-quiet', 'cli-chat'])
@pytest.mark.parametrize('claim_change', ['expired', 'scheduled'])
def test_claim_loss_during_judge_prevents_continuation(worker, monkeypatch, caller, claim_change):
    from tests.agent.test_run_agent import _mock_response

    partial = 'Section one is ready. Section two remains.'
    judged = []

    def judge(goal, last_response):
        judged.append(last_response)
        with connect_closing() as conn:
            if claim_change == 'scheduled':
                assert kb.schedule_task(conn, worker.context['task_id'], reason='Operator postponed work.')
            else:
                with kb.write_txn(conn):
                    conn.execute('UPDATE tasks SET claim_expires = 0 WHERE id = ?', (worker.context['task_id'],))
        return 'continue', 'write section two', False, None, False

    monkeypatch.setenv('HERMES_KANBAN_STOP_NUDGE', '0')
    monkeypatch.setattr(goals, 'judge_goal', judge)
    transport = worker.agent.client.chat.completions.create
    transport.side_effect = [_mock_response(content=partial), _mock_response(content='Must not run.')]
    _run_caller(worker, caller, monkeypatch, expected_exit=1)
    assert judged == [partial] and transport.call_count == 1
    if claim_change == 'scheduled':
        with connect_closing() as conn:
            task = kb.get_task(conn, worker.context['task_id'])
            assert task is not None and task.status == 'scheduled'


@pytest.mark.parametrize('parents', ['open', 'finish_during_judge', 'none'])
def test_dependency_block_requires_current_judged_parents(worker, monkeypatch, parents):
    from tools import kanban_tools

    tid = worker.context['task_id']
    judged = []
    with connect_closing() as conn:
        parent = kb.create_task(conn, title='Produce source data', assignee='default')
        other = kb.create_task(conn, title='Review unrelated notes', assignee='default')
        if parents != 'none':
            for pid in (parent, other):
                kb.link_tasks(conn, pid, tid, expected_child_run_id=worker.context['run_id'])

    def judge(goal, last_response):
        judged.append(last_response)
        assert parent in last_response and other in last_response
        if parents == 'finish_during_judge':
            with connect_closing() as conn:
                assert kb.complete_task(conn, parent, summary='Source data is ready.')
        return 'blocked', 'source data is required', False, None, False

    monkeypatch.setattr(kanban_tools, 'judge_goal', judge)
    response = json.loads(kanban_tools._handle_block({'kind': 'dependency', 'reason': f'Need source data from {parent}.'}))
    assert len(judged) == (0 if parents == 'none' else 1)
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        if parents == 'open':
            assert response['ok'] and task.status == 'todo' and task.block_kind == 'dependency'
        else:
            assert 'error' in response and task.status == 'running'
            assert task.current_run_id == worker.context['run_id']


@pytest.mark.parametrize('surface', ['tool', 'cli'])
@pytest.mark.parametrize('expire', [False, True])
@pytest.mark.parametrize(('action', 'status', 'outcome'), [
    ('block', 'blocked', 'blocked'),
    ('complete', 'done', 'completed'),
    ('request_review', 'review', 'review_requested'),
])
def test_goal_worker_handoff_requires_live_lease_at_write(worker, monkeypatch, surface, expire,
                                                        action, status, outcome):
    from argparse import Namespace
    from hermes_cli import kanban as kanban_cli
    from tools import kanban_tools

    tid, run_id = worker.context['task_id'], worker.context['run_id']
    evidence = ('Need the operator to select the required export format.' if action == 'block'
                else 'Both sections are written and verified.')
    judged = []

    def judge(goal, last_response):
        judged.append(last_response)
        if expire:
            with connect_closing() as conn, kb.write_txn(conn):
                conn.execute('UPDATE tasks SET claim_expires = 0 WHERE id = ?', (tid,))
                conn.execute('UPDATE task_runs SET claim_expires = 0 WHERE id = ?', (run_id,))
        return ('blocked' if action == 'block' else 'done', 'handoff accepted', False, None, False)

    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr(kanban_tools, 'judge_goal', judge)
    fields = {'kind': 'needs_input', 'reason': evidence} if action == 'block' else {'summary': evidence}
    if surface == 'tool':
        result = json.loads(getattr(kanban_tools, '_handle_' + action)(fields))
        accepted = result.get('ok', False)
    else:
        if action == 'block':
            fields['reason'] = [evidence]
        handler = getattr(kanban_cli, '_cmd_' + action)
        accepted = handler(Namespace(task_id=tid, task_ids=[tid], result=None, **fields)) == 0
    assert len(judged) == 1 and evidence in judged[0]
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        assert task is not None and run is not None
        assert run.id == run_id
        if expire:
            assert task.status == 'running' and task.current_run_id == run_id and run.outcome is None
        else:
            assert task.status == status and run.outcome == outcome and run.summary == evidence
    assert accepted is not expire


@pytest.mark.parametrize('worker', [{'goal_mode': False}], indirect=True)
def test_ordinary_worker_handoff_keeps_its_lease_rules(worker):
    from tools import kanban_tools

    tid, run_id = worker.context['task_id'], worker.context['run_id']
    with connect_closing() as conn, kb.write_txn(conn):
        conn.execute('UPDATE tasks SET claim_expires = 0 WHERE id = ?', (tid,))
        conn.execute('UPDATE task_runs SET claim_expires = 0 WHERE id = ?', (run_id,))
    # Outside goal mode an unreaped worker still completes its own run, as before.
    assert json.loads(kanban_tools._handle_complete({'summary': 'Both sections verified.'}))['ok']
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        assert task.status == 'done' and run.id == run_id and run.outcome == 'completed'


@pytest.mark.parametrize('run_pin', ['', 'invalid'])
def test_cli_worker_block_without_run_pin_is_refused(worker, monkeypatch, run_pin):
    from argparse import Namespace
    from hermes_cli.kanban import _cmd_block

    tid = worker.context['task_id']
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', run_pin)
    monkeypatch.setattr(goals, 'judge_goal', lambda **kw: ('blocked', 'requires input', False, None, False))
    assert _cmd_block(Namespace(task_id=tid, task_ids=[], kind='needs_input', reason=['Need input'])) == 1
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None and task.current_run_id == worker.context['run_id']


@pytest.mark.parametrize('surface', ['tool', 'cli'])
@pytest.mark.parametrize('route_failed', [True, False])
def test_goal_human_input_request_distinguishes_outage_from_continue(worker, monkeypatch, surface, route_failed):
    from argparse import Namespace
    from hermes_cli.kanban import _cmd_block
    from tools import kanban_tools

    tid, run_id = worker.context['task_id'], worker.context['run_id']
    reason = 'Only the operator can choose the required export format.'
    judgments = []

    def judge(goal, last_response):
        judgments.append(last_response)
        detail = 'no judge route responded' if route_failed else 'another export format is available'
        return 'continue', detail, False, None, route_failed

    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr(kanban_tools, 'judge_goal', judge)
    if surface == 'tool':
        result = json.loads(kanban_tools._handle_block({'kind': 'needs_input', 'reason': reason}))
        accepted = result.get('ok', False)
    else:
        accepted = _cmd_block(Namespace(task_id=tid, task_ids=[tid], kind='needs_input', reason=[reason])) == 0

    assert len(judgments) == 1 and reason in judgments[0]
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        assert task is not None and run is not None
        assert run.id == run_id
        if route_failed:
            assert task.status == 'blocked' and task.block_kind == 'needs_input'
            assert task.current_run_id is None and run.outcome == 'blocked'
            assert run.summary and reason in run.summary and 'judge unavailable' in run.summary.lower()
        else:
            assert task.status == 'running' and task.current_run_id == run_id and run.outcome is None
    assert accepted is route_failed


def test_quiet_goal_reports_judge_failure_without_notification_reentry(worker, monkeypatch):
    from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, read_turn_report
    from tests.agent.test_run_agent import _mock_response

    report_path = str(worker.artifact.with_suffix('.json'))
    monkeypatch.setenv(TURN_REPORT_FILE_ENV, report_path)
    partial = 'Section one is ready. Section two remains.'
    worker.agent.client.chat.completions.create.side_effect = [_mock_response(content=partial)]

    def judge(goal, last_response):
        assert last_response == partial
        assert read_turn_report(report_path, os.getpid()) is None
        return 'continue', 'judge offline', False, None, True

    monkeypatch.setattr(goals, 'judge_goal', judge)
    def unexpected_reentry(*args, **kwargs):
        pytest.fail('notification linger ran before the Goal outcome was settled')

    _run_caller(worker, 'cli-quiet', monkeypatch, expected_exit=1, notify_linger=unexpected_reentry)
    report = read_turn_report(report_path, os.getpid())
    assert report is not None
    assert report['exit_code'] == 1 and report['reply'] == partial
    assert 'judge transport failure' in report['error']


def test_chat_goal_continuation_cannot_reuse_previous_result_on_auth_failure(worker, monkeypatch):
    from cli import HermesCLI
    from tests.agent.test_run_agent import _mock_response

    partial = 'Section one is done. The second is pending.'
    judged = []

    def judge(goal, last_response):
        judged.append(last_response)
        return 'continue', 'finish section two', False, None, False

    def follow_up(surface, prompt):
        def require_login():
            surface._credentials_terminal = True
            return False

        surface._secret_capture_callback = None
        surface._ensure_runtime_credentials = require_login
        return HermesCLI.chat(surface, prompt)

    monkeypatch.setattr(goals, 'judge_goal', judge)
    worker.agent.client.chat.completions.create.side_effect = [_mock_response(content=partial)]
    _run_caller(worker, 'cli-chat', monkeypatch, expected_exit=kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
                follow_up=follow_up)
    assert judged == [partial]
    with connect_closing() as conn:
        assert kb.goal_run_status(conn, worker.context['task_id'], worker.context['run_id']) == 'running'


def test_operator_can_block_goal_card_while_judge_unavailable(worker, monkeypatch):
    from tools import kanban_tools

    tid = worker.context['task_id']
    monkeypatch.delenv('HERMES_KANBAN_TASK')
    monkeypatch.setattr(kanban_tools, 'judge_goal',
                        lambda **kw: ('continue', 'judge offline', False, None, True))
    result = json.loads(kanban_tools._handle_block({
        'task_id': tid, 'kind': 'needs_input', 'reason': 'Customer must choose scope.'}))
    assert result['ok']
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == 'blocked' and task.block_kind == 'needs_input'


@pytest.mark.parametrize('tool', ['kanban_complete', 'kanban_request_review'])
@pytest.mark.parametrize('surface', ['tool', 'cli'])
@pytest.mark.parametrize('actor, backup', [('worker', True), ('worker', False), ('operator', False)])
def test_goal_handoff_uses_configured_judge_fallback(worker, monkeypatch, tool, surface, actor, backup):
    from argparse import Namespace
    from agent import auxiliary_client as aux
    from hermes_cli import kanban as kanban_cli
    from tests.agent.test_run_agent import _mock_response
    from tools import kanban_tools

    provider, model = 'custom:judge-backup', 'judge-backup-model'
    config = {'provider': 'anthropic', 'model': 'missing-primary'}
    if backup:
        config['fallback_chain'] = [{'provider': provider, 'model': model}]
    monkeypatch.setattr(aux, '_get_auxiliary_task_config', lambda task: config if task == 'goal_judge' else {})
    client = SimpleNamespace(base_url='https://inert.invalid/v1', chat=SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock(return_value=_mock_response(
            content='{"verdict":"done","reason":"Acceptance evidence is complete."}')))))

    def resolve(name, *args, **kwargs):
        if name == 'anthropic':
            return None, None
        assert backup and name == provider
        return client, model

    monkeypatch.setattr(aux, 'resolve_provider_client', resolve)
    if actor == 'operator':
        monkeypatch.delenv('HERMES_KANBAN_TASK')
    tid, run_id = worker.context['task_id'], worker.context['run_id']
    summary = 'Both sections were written and verified.'
    action = tool.removeprefix('kanban_')
    if surface == 'tool':
        reply = json.loads(getattr(kanban_tools, '_handle_' + action)({'task_id': tid, 'summary': summary}))
        accepted = reply.get('ok', False)
    else:
        handler = getattr(kanban_cli, '_cmd_' + action)
        accepted = handler(Namespace(task_id=tid, task_ids=[tid], summary=summary, result=None,
                                     force=actor == 'operator')) == 0
    # #83610: the explicit handoff keeps working even when no judge route responds.
    assert accepted
    assert client.chat.completions.create.call_count == int(backup)
    if backup:
        assert summary in str(client.chat.completions.create.call_args.kwargs['messages'])
    with connect_closing() as conn:
        task, run = kb.get_task(conn, tid), kb.latest_run(conn, tid)
        expected = 'done' if action == 'complete' else 'review'
        assert task.status == expected and run.id == run_id
        assert run.ended_at is not None


@pytest.mark.parametrize('action, actor, initial, expected', [
    ('complete', 'operator', 'review', 'done'),
    ('complete', 'operator', 'ready', 'ready'),
    ('request_review', 'operator', 'ready', 'review'),
    ('complete', 'worker', 'running', 'running'),
    ('request_review', 'worker', 'running', 'running'),
])
def test_cli_goal_handoff_without_new_evidence(worker, monkeypatch, action, actor, initial, expected):
    from argparse import Namespace
    from agent import auxiliary_client
    from hermes_cli import kanban as kanban_cli

    resolve = MagicMock(return_value=(None, None))
    monkeypatch.setattr(auxiliary_client, 'resolve_provider_client', resolve)
    tid = worker.context['task_id']
    if actor == 'operator':
        monkeypatch.delenv('HERMES_KANBAN_TASK')
        with connect_closing() as conn:
            tid = kb.create_task(conn, title='Operator Goal', assignee='default', goal_mode=True)
            if initial == 'review':
                assert kb.request_review(conn, tid)
            assert kb.get_task(conn, tid).status == initial
    handler = getattr(kanban_cli, '_cmd_' + action)
    assert handler(Namespace(task_id=tid, task_ids=[tid], summary=None, result=None)) == int(initial == expected)
    with connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == expected
        if actor == 'worker':
            assert task.current_run_id == worker.context['run_id']
    resolve.assert_not_called()


@pytest.mark.parametrize('surface', ['cli', 'tool'])
@pytest.mark.parametrize('summary', [None, ' \t '], ids=['omitted', 'blank'])
@pytest.mark.parametrize('verdict, offline, accepted', [
    ('continue', False, False), ('done', False, True), ('continue', True, True),
], ids=['partial', 'done', 'operator-recovery'])
def test_goal_completion_judges_result_when_summary_is_blank(
        worker, monkeypatch, surface, summary, verdict, offline, accepted):
    from argparse import Namespace
    from hermes_cli.kanban import _cmd_complete
    from tools import kanban_tools

    monkeypatch.delenv('HERMES_KANBAN_TASK')
    with connect_closing() as conn:
        tid = kb.create_task(conn, title='Write both sections', assignee='default', goal_mode=True)
    judge = MagicMock(return_value=(verdict, 'judge response', False, None, offline))
    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr(kanban_tools, 'judge_goal', judge)
    result = 'Only one section is written.' if not accepted else 'Both sections are written.'
    args = {'task_id': tid, 'summary': summary, 'result': result}
    if surface == 'cli':
        complete = _cmd_complete(Namespace(task_ids=[tid], **args)) == 0
    else:
        complete = json.loads(kanban_tools._handle_complete(args)).get('ok', False)
    judge.assert_called_once_with(goal='Write both sections', last_response=result)
    assert complete == accepted
    with connect_closing() as conn:
        assert kb.get_task(conn, tid).status == ('done' if accepted else 'ready')


@pytest.mark.parametrize('tool', ['kanban_complete', 'kanban_request_review'])
def test_worker_handoff_is_rejected_by_a_genuine_continue_verdict(worker, monkeypatch, tool):
    from tools import kanban_tools

    monkeypatch.setattr(kanban_tools, 'judge_goal',
                        lambda **kw: ('continue', 'second section missing', False, None, False))
    handler = kanban_tools._handle_complete if tool == 'kanban_complete' else kanban_tools._handle_request_review
    reply = json.loads(handler({'summary': 'Only one section is written.'}))
    assert 'second section missing' in reply.get('error', '')
    with connect_closing() as conn:
        task = kb.get_task(conn, worker.context['task_id'])
        assert task is not None
        assert task.status == 'running' and task.current_run_id == worker.context['run_id']


@pytest.mark.parametrize('worker', [{'goal_mode': False, 'inherited_goal': '1'}], indirect=True)
def test_false_assignment_overwrites_inherited_goal_and_keeps_terminal_guard(worker, monkeypatch):
    from agent.turn_stop_gates import apply_stop_gates

    assert worker.context['goal_mode'] is False
    assert os.environ['HERMES_KANBAN_TASK'] == worker.context['task_id']
    assert os.environ['HERMES_KANBAN_GOAL_MODE'] == '0'
    messages = [{'role': 'user', 'content': 'Write the document.'}]
    verdict = apply_stop_gates(worker.agent, {'role': 'assistant', 'content': 'First section written.'},
        final_response='First section written.', messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn
    assert messages[-1]['_kanban_stop_synthetic'] is True
    assert worker.context['task_id'] in messages[-1]['content']


@pytest.mark.parametrize(('verdict', 'budget', 'entries', 'reason'), [
    ('blocked', 3, 1, 'unachievable'),
    ('done', 3, 2, 'never'),
    ('continue', 1, 1, 'budget'),
])
def test_goal_controller_still_blocks_without_terminal_handoff(
        worker, monkeypatch, verdict, budget, entries, reason):
    from tests.agent.test_run_agent import _mock_response

    worker.context['goal_max_turns'] = budget
    worker.frame['policy']['kanban_json'] = json.dumps(worker.context)
    monkeypatch.setattr(goals, 'judge_goal', lambda *a, **kw: (verdict, 'criteria missing', False, None, False))
    worker.agent.client.chat.completions.create.side_effect = lambda **kw: _mock_response(
        content='The first section is available; the second section is missing.')
    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
    assert result['completed'] and not result.get('failed')
    assert worker.agent.client.chat.completions.create.call_count == entries
    assert getattr(worker.agent, '_kanban_stop_nudges', 0) == 0
    with connect_closing() as conn:
        task = kb.get_task(conn, worker.context['task_id'])
        assert task is not None and task.status == 'blocked'
        event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
                             (worker.context['task_id'],)).fetchone()
        assert event is not None and reason in json.loads(event[0])['reason'].lower()


def test_initial_iteration_exhaustion_cannot_enter_goal_controller(worker, monkeypatch):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    worker.agent.max_iterations = 1
    monkeypatch.setattr(goals, 'judge_goal', lambda *a, **kw: pytest.fail('unfinished initial turn entered Goal'))
    calls = []
    def transport(**kw):
        calls.append(copy.deepcopy(kw['messages']))
        if len(calls) == 1:
            return _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
                'write_file', json.dumps({'path': str(worker.artifact), 'content': 'First section only.\n'}))])
        return _mock_response(content='Only the first section was written before the iteration limit.')
    worker.agent.client.chat.completions.create.side_effect = transport
    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
    assert result['completed'] is False
    assert 'max_iterations' in result['turn_exit_reason'] or 'budget' in result['turn_exit_reason']
    assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == 1
    assert getattr(worker.agent, '_kanban_stop_nudges', 0) == 0


def _stop_agent():
    return SimpleNamespace(session_id='policy-session', _turn_file_mutation_paths=set(),
        _interim_content_was_streamed=lambda text: False,
        _emit_diagnostic_status=lambda text: None,
        _emit_interim_assistant_message=lambda msg: None,
        _flush_messages_to_session_db=lambda *a: None)


def _apply_stop(agent):
    from agent.turn_stop_gates import apply_stop_gates
    messages = [{'role': 'user', 'content': 'Write the document.'}]
    verdict = apply_stop_gates(agent, {'role': 'assistant', 'content': 'First section written.'},
        final_response='First section written.', messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    return verdict, messages


@pytest.mark.parametrize('mode', [None, '0', 'true', '1 ', 'yes'])
def test_ordinary_modes_keep_real_stop_guard(monkeypatch, mode):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.delenv('HERMES_KANBAN_STOP_NUDGE', raising=False)
    if mode is None:
        monkeypatch.delenv('HERMES_KANBAN_GOAL_MODE', raising=False)
    else:
        monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', mode)
    verdict, messages = _apply_stop(_stop_agent())
    assert verdict.continue_turn
    assert messages[-1]['_kanban_stop_synthetic'] is True
    assert messages[-2]['finish_reason'] == 'kanban_terminal_required'


@pytest.mark.parametrize('kind', ['delegate', 'cron', 'descendant', 'no-task'])
def test_nonowned_context_never_acquires_worker_stop_policy(monkeypatch, kind):
    from contextlib import nullcontext
    from agent.delegation_context import (delegated_child_context, non_dispatcher_owned_context,
                                           owned_kanban_task, delegated_child_subprocess_env)
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '0')
    if kind == 'descendant':
        monkeypatch.setenv('HERMES_DELEGATED_CHILD_CONTEXT', str(Path.cwd()))
    if kind == 'no-task':
        monkeypatch.delenv('HERMES_KANBAN_TASK')
        monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '1')
    scope = {'delegate': delegated_child_context, 'cron': non_dispatcher_owned_context}.get(kind, nullcontext)
    with scope():
        assert owned_kanban_task() == ''
        verdict, messages = _apply_stop(_stop_agent())
        assert not verdict.continue_turn
        assert len(messages) == 1
        if kind == 'delegate':
            env = delegated_child_subprocess_env(dict(os.environ, HERMES_KANBAN_GOAL_MODE='1'))
            assert 'HERMES_KANBAN_GOAL_MODE' not in env
            assert 'HERMES_KANBAN_TASK' not in env
            assert env['HERMES_DELEGATED_CHILD_CONTEXT']


def test_goal_exemption_does_not_skip_other_stop_gates(monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '1')
    monkeypatch.setattr('agent.verification_stop.verify_on_stop_enabled', lambda: True)
    monkeypatch.setattr('agent.verification_stop.build_verify_on_stop_nudge', lambda **kw: 'Verify the document.')
    verdict, messages = _apply_stop(_stop_agent())
    assert verdict.continue_turn
    assert messages[-1]['_verification_stop_synthetic'] is True
    assert messages[-2]['finish_reason'] == 'verification_required'
