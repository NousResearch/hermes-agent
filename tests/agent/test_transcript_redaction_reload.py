"""An idle native client must not reintroduce erased history on its next turn."""
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.run_agent.test_cross_process_turn_lease import _agent_with_db


def test_idle_agent_reloads_only_affected_history_before_an_uncontended_turn(tmp_path, monkeypatch):
    monkeypatch.setattr('agent.turn_liveness.resolve_turn_liveness_settings', lambda _: (None, 1))
    db = SessionDB(tmp_path / 'state.db')
    other_writer = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('s1', source='cli')
        target = db.append_message('s1', 'user', 'forgettoken source')
        answer = db.append_message('s1', 'assistant', 'forgettoken linked answer')
        kept = db.append_message('s1', 'user', 'Keep the unrelated task open')
        db.create_session('other', source='cli')
        db.append_message('other', 'user', 'Unrelated other session')
        history = db.get_messages_as_conversation('s1', include_row_ids=True)
        agent = _agent_with_db(db, session_id='s1', platform='cli')
        agent._session_messages = history
        agent._db_flush_scan_prefix = history[:]

        def model_loop(_agent, _message, _system, supplied, *_args, **_kwargs):
            return {'final_response': 'no inference in this fixture', 'messages': supplied, 'failed': False}

        monkeypatch.setattr('agent.conversation_loop.run_conversation', model_loop)

        def admit():
            return AIAgent.run_conversation(agent, 'Continue the accepted task',
                task_id='accepted-task', conversation_history=history)

        first = admit()
        assert first['messages'] is history
        # No held lease and no contention callback: another process erased while this agent was idle.
        expected = other_writer.get_message_redaction_snapshot('s1', [target, answer])
        other_writer.redact_message_payloads('s1', expected)
        second = admit()
        assert second['messages'] is not history
        assert 'forgettoken' not in str(second['messages'])
        assert agent._session_messages is second['messages']
        assert agent._db_flush_scan_prefix is None
        assert second['messages'][-1]['_row_id'] == kept
        assert second['messages'][-1]['content'] == 'Keep the unrelated task open'
        assert agent.session_id == 's1'
        # A subsequent real native compaction uses the refreshed payload, not the idle copy.
        db.archive_and_compact('s1', second['messages'], tail_count=len(second['messages']))
        assert 'forgettoken' not in str(db.get_messages('s1', include_inactive=True))

        # Neither an unchanged revision nor redaction in a different session reloads this turn.
        history = second['messages']
        other_id = db.get_messages('other')[0]['id']
        other_writer.redact_message_payloads('other', other_writer.get_message_redaction_snapshot('other', [other_id]))
        third = admit()
        assert third['messages'] is history
    finally:
        other_writer.close()
        db.close()
