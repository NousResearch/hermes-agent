"""Interrupted real Files turns keep a canonical closing row and safe control metadata."""
import copy
import json
from contextlib import nullcontext
from types import MethodType

import pytest

from agent.files_live_context import files_result_boundary, safe_files_result
from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent

PRIVATE = '/private/receiver-custody/document.txt'
SAFE = 'Review this document\n\n[Attached document: "document.txt"]'


@pytest.mark.parametrize('files', [True, False], ids=['files', 'ordinary'])
@pytest.mark.parametrize('during_recovery', [False, True], ids=['provider-call', 'error-recovery'])
def test_real_facade_interrupt_closes_canonical_history_before_safe_result_export(tmp_path, monkeypatch, files, during_recovery):
    from run_agent import AIAgent
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'interrupted')
        db.create_session(session_id=agent.session_id, source='api_server')
        agent._session_db_created = True
        agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
        provider = agent.client.chat.completions.create.side_effect
        def stop_during_request(**kwargs):
            provider(**kwargs)
            if during_recovery:
                import httpx
                import openai
                raise openai.InternalServerError('inert stream failure',
                    response=httpx.Response(500, request=httpx.Request('POST', 'https://inert.invalid/v1')),
                    body={'error': {'message': 'inert stream failure'}})
            agent.interrupt(hard_cancel=True)
            raise InterruptedError('explicit provider-call stop')
        if during_recovery:
            from tests.agent.test_files_error_export_privacy import _isolate_recovery
            _isolate_recovery(agent)
            def stop_on_recovery(**kwargs):
                agent.interrupt(hard_cancel=True)
                return False, kwargs['has_retried_429']
            agent._recover_with_credential_pool = stop_on_recovery
        agent.client.chat.completions.create.side_effect = stop_during_request
        scope = files_user_message_persistence(agent, SAFE) if files else nullcontext(None)
        with files_result_boundary(agent), scope as transcript:
            result = agent.run_conversation(PRIVATE if files else 'ordinary prompt', persist_user_message=transcript)
            checked = safe_files_result(agent, result, force=files)
        assert checked['interrupted'] is True
        assert checked['completed'] is False and not checked.get('failed')
        assert [row['role'] for row in checked['messages']] == ['user', 'assistant']
        assert checked['messages'] == agent._session_messages
        assert [row['role'] for row in db.get_messages(agent.session_id)] == ['user', 'assistant']
        if files:
            assert PRIVATE in json.dumps(sent)
            assert PRIVATE not in json.dumps(checked)
            assert PRIVATE not in json.dumps(agent._session_messages)
            assert PRIVATE not in json.dumps(db.get_messages(agent.session_id))
            assert checked['current_turn_user_idx'] == 0
            assert checked['turn_id'] == agent._current_turn_id
            assert checked['messages'] is not agent._session_messages
            checked['messages'][0]['content'] = 'returned snapshot mutation'
            assert agent._session_messages[0]['content'] == SAFE
            refused = safe_files_result(agent, {**result, 'messages': sent[0]}, force=True)
            assert refused['failed'] is True and refused['messages'] == []
    finally:
        db.close()


@pytest.mark.parametrize('alteration', ['foreign-provider', 'stale-snapshot'])
def test_foreign_result_history_is_never_adopted_by_the_files_closing_producer(tmp_path, monkeypatch, alteration):
    from run_agent import AIAgent
    import agent.conversation_loop as loop
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'current-owner')
        db.create_session(session_id=agent.session_id, source='api_server')
        agent._session_db_created = True
        agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
        provider = agent.client.chat.completions.create.side_effect
        def stop_during_request(**kwargs):
            provider(**kwargs)
            agent.interrupt(hard_cancel=True)
            raise InterruptedError('explicit provider-call stop')
        agent.client.chat.completions.create.side_effect = stop_during_request
        real_turn = loop._run_conversation_turn
        def alternate_result(*args, **kwargs):
            result = real_turn(*args, **kwargs)
            result['messages'] = copy.deepcopy(sent[0] if alteration == 'foreign-provider' else result['messages'])
            if alteration == 'stale-snapshot':
                result['messages'][0]['content'] = 'Earlier safe Files turn'
                result['turn_id'] = 'earlier-turn'
            return result
        monkeypatch.setattr(loop, '_run_conversation_turn', alternate_result)
        with files_result_boundary(agent), files_user_message_persistence(agent, SAFE) as transcript:
            result = agent.run_conversation(PRIVATE, persist_user_message=transcript)
        assert result['failed'] is True and result['messages'] == []
        assert not result.get('interrupted')
        assert [row['role'] for row in agent._session_messages] == ['user']
        rows = db.get_messages(agent.session_id)
        assert len(rows) == 1 and rows[0]['content'] == SAFE
        assert PRIVATE not in json.dumps(rows)
    finally:
        db.close()
