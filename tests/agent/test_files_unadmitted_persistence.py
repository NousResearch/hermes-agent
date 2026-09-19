"""Scope checkpoint: real public-facade lease-wait interruption before prologue.

No coordinator/worker runs: a conflicting SQLite lease plus an already-set
interrupt makes the actual facade return through carry_unadmitted_user_message.
"""
import json
from types import MethodType

import pytest

from hermes_state import SessionDB
from agent.session_persistence import files_user_message_persistence
from tests.agent.files_persistence_fixtures import inert_agent


@pytest.mark.parametrize('native', [False, True])
def test_unadmitted_files_carry_is_safe_before_followup_flush(tmp_path, monkeypatch, native, record_property):
    from run_agent import AIAgent
    from agent.conversation_loop import run_conversation
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'lease')
        db.create_session(session_id='lease', source='api_server')
        agent._session_db_created = True
        assert db.acquire_session_turn_lease('lease', 'other-test-holder', ttl_seconds=30, wait_seconds=0)
        agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
        # This consumer is before the coordinator. Fail the test if it reaches one.
        monkeypatch.setattr('agent.relay_runtime.SESSION_COORDINATOR.acquire_conversation',
                            lambda **k: pytest.fail('must not start coordinator'))
        agent.interrupt('following input')
        content = '/private/unadmitted-file.txt'
        if native:
            content = [{'type': 'text', 'text': content},
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        safe = 'accepted prompt\n\n[Attached file: "a.txt"]'
        with files_user_message_persistence(agent, safe) as projection:
            result = agent.run_conversation(content, persist_user_message=projection,
                persist_user_timestamp=1234.5, persist_user_platform_id='carried-files',
                persist_user_display_kind='internal_notification',
                persist_user_display_metadata={'origin': 'lease-test'})
        assert result['interrupted'] is True
        assert sent == []
        record_property('unadmitted_result', json.dumps(result))
        db.release_session_turn_lease('lease', 'other-test-holder')
        agent.run_conversation = MethodType(run_conversation, agent)
        agent.run_conversation('following input', conversation_history=result['messages'])
        rows = db.get_messages('lease')
        record_property('followup_sql', json.dumps(rows, default=lambda b: b.hex()))
        assert rows[0]['content'] == safe
        assert rows[0]['api_content'] is None
        assert rows[0]['timestamp'] == 1234.5
        assert rows[0]['platform_message_id'] == 'carried-files'
        assert rows[0]['display_kind'] == 'internal_notification'
        assert rows[0]['display_metadata'] == {'origin': 'lease-test'}
        assert 'data:image/' not in json.dumps(result)
        assert '/private/unadmitted-file.txt' not in json.dumps(result)
        assert '/private/unadmitted-file.txt' not in json.dumps(db.get_messages_as_conversation('lease'))
    finally:
        db.close()


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('hard_stop', [False, True])
def test_unadmitted_ordinary_carry_and_hard_stop_unchanged(tmp_path, monkeypatch, native, hard_stop):
    from run_agent import AIAgent
    from agent.conversation_loop import run_conversation
    db = SessionDB(tmp_path / 'ordinary.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'ordinary-lease')
        db.create_session(session_id=agent.session_id, source='api_server')
        agent._session_db_created = True
        assert db.acquire_session_turn_lease(agent.session_id, 'other', ttl_seconds=30, wait_seconds=0)
        agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
        monkeypatch.setattr('agent.relay_runtime.SESSION_COORDINATOR.acquire_conversation',
                            lambda **k: pytest.fail('must not start coordinator'))
        agent.interrupt('following ordinary input', hard_cancel=hard_stop)
        content = 'ordinary /private/reference.txt'
        if native:
            content = [{'type': 'text', 'text': content},
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        result = agent.run_conversation(content, persist_user_message='ordinary caption',
            persist_user_timestamp=2345.5, persist_user_platform_id='ordinary-message',
            persist_user_display_kind='internal_notification',
            persist_user_display_metadata={'origin': 'ordinary'})
        assert result['interrupted'] is True and sent == []
        db.release_session_turn_lease(agent.session_id, 'other')
        if hard_stop:
            assert result['messages'] == []
            assert db.get_messages(agent.session_id) == []
            return
        carried = result['messages'][0]
        assert carried['content'] == (content if native else 'ordinary caption')
        assert carried.get('api_content') == (None if native else content)
        agent.run_conversation = MethodType(run_conversation, agent)
        agent.run_conversation('following ordinary input', conversation_history=result['messages'])
        row = db.get_messages(agent.session_id)[0]
        assert row['content'] == ('ordinary /private/reference.txt\n[screenshot]' if native else 'ordinary caption')
        assert row['api_content'] == (None if native else content)
        assert row['timestamp'] == 2345.5
        assert row['platform_message_id'] == 'ordinary-message'
        assert row['display_kind'] == 'internal_notification'
        assert row['display_metadata'] == {'origin': 'ordinary'}
    finally:
        db.close()
