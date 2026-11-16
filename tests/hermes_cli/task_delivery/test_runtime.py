"""The existing agent Todo path cannot close a managed task before acceptance."""
import json
from types import SimpleNamespace

from agent.inline_tool_executors import InlineToolContext, _delivery_todo
from hermes_cli.task_delivery.runtime import recovery_context
from hermes_cli.task_delivery.store import WorkLedger
from hermes_state import SessionDB
from tools.todo_tool import TodoStore


def test_todo_completion_gate_and_recovery_read_persistent_facts(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('parent', source='gui')
        agent = SimpleNamespace(_session_db=db, session_id='parent')
        agent._todo_store = TodoStore(delivery_context=lambda: recovery_context(agent))
        work = WorkLedger(db)
        work.register('parent', 'fix', title='Fix customer flow', acceptance='Real request succeeds')
        ctx = InlineToolContext(effective_task_id='task')
        result = json.loads(_delivery_todo(agent, {'todos': [
            {'id': 'fix', 'content': 'Fix customer flow', 'status': 'in_progress'}]}, ctx))
        assert result['delivery'][0]['accepted'] is False
        refused = json.loads(_delivery_todo(agent, {'todos': [
            {'id': 'fix', 'content': 'Fixed', 'status': 'completed'}]}, ctx))
        assert 'acceptance' in refused['error']
        assert agent._todo_store.read()[0]['status'] == 'in_progress'
        snapshot = agent._todo_store.snapshot()
        agent._todo_store = TodoStore(delivery_context=lambda: recovery_context(agent))
        agent._todo_store.restore(**snapshot)
        recovered = agent._todo_store.format_for_injection()
        assert 'Persistent work delivery facts' in recovered
        assert 'not_reviewed' in recovered and '"accepted": false' in recovered
        # A planner replacing its checklist cannot erase unfinished delivery facts.
        _delivery_todo(agent, {'todos': []}, ctx)
        assert '"task_id": "fix"' in agent._todo_store.format_for_injection()


def test_normalized_todo_id_and_status_cannot_bypass_acceptance(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('owner', source='gui')
        agent = SimpleNamespace(_session_db=db, session_id='owner', _todo_store=TodoStore())
        WorkLedger(db).register('owner', 'fix', title='Fix', acceptance='Verified customer behavior')
        reply = json.loads(_delivery_todo(agent, {'todos': [
            {'id': ' fix ', 'content': 'Fixed', 'status': ' COMPLETED '}]}, InlineToolContext('t')))
        assert 'acceptance' in reply['error']
        assert agent._todo_store.read() == []


def test_recovery_follows_compression_but_not_an_explicit_branch(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('owner', source='cli')
        WorkLedger(db).register('owner', 'fix', title='Fix', acceptance='Verified customer behavior')
        db.end_session('owner', end_reason='compression')
        db.create_session('compressed', source='cli', parent_session_id='owner')
        agent = SimpleNamespace(_session_db=db, session_id='compressed')
        assert '"task_id": "fix"' in recovery_context(agent)
        db.create_session('branch', source='cli', parent_session_id='owner',
                          model_config={'_branched_from': 'owner'})
        agent.session_id = 'branch'
        assert recovery_context(agent) == ''
