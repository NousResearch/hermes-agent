"""Failed managed attempts retain a safe signature, not private exception text."""
import json
import os
from types import SimpleNamespace

import pytest
import run_agent  # Activate bootstrap before the per-test home-I/O guard.
from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


def test_missing_skill_signature_is_recorded_as_a_safe_code(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title='synthetic', assignee='default')
        task = kb.claim_task(conn, task_id, claimer='fixture:owner')
        context = {'db': str(path), 'task_id': task_id, 'run_id': task.current_run_id,
                   'claim_lock': task.claim_lock, 'skills': ['synthetic-skill']}
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'worker_bound',
                             {'pid': os.getpid(), 'claim_lock': task.claim_lock}, run_id=task.current_run_id)
    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt',
                        lambda *a, **kw: ('', [], ['synthetic-skill']))
    frame = {'policy': {'kanban_json': json.dumps(context)}}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban.run_worker_turns(SimpleNamespace(session_id='synthetic'), frame, [])
    result = session_kanban.worker_result(path, context)
    assert result['exit_code'] == 1
    assert result['last_output'] == 'Kanban worker failed: kanban_skills_unavailable'
    with connect_closing(path) as conn:
        assert kb.get_task(conn, task_id).status == 'running'  # No new block/retry policy.
    assert session_kanban.worker_result(path, context | {'claim_lock': 'foreign'}) == {}


def test_unclassified_exception_payload_is_not_copied_to_board(monkeypatch):
    recorded = []

    def fail(*args):
        raise RuntimeError('synthetic-private-payload-not-for-board')

    monkeypatch.setattr(session_kanban, '_run_task_turns', fail)
    monkeypatch.setattr(session_kanban, '_record_worker_result',
                        lambda context, code, text: recorded.append((code, text)))
    frame = {'policy': {'kanban_json': '{}'}}
    with pytest.raises(RuntimeError, match='synthetic-private-payload'):
        session_kanban._run_worker_turns(SimpleNamespace(), frame, [])
    # The type name (a code identifier) replaces the empty diagnostic; the message does not.
    assert recorded == [(1, 'Kanban worker failed: RuntimeError')]
    assert 'synthetic-private-payload' not in recorded[0][1]
