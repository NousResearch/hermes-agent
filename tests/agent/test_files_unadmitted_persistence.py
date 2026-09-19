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
            result = agent.run_conversation(content, persist_user_message=projection)
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
        assert '/private/unadmitted-file.txt' not in json.dumps(result)
        assert '/private/unadmitted-file.txt' not in json.dumps(db.get_messages_as_conversation('lease'))
    finally:
        db.close()
