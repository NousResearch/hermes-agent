"""Native Files is explicitly unsupported, never an implicit label-only turn."""
import copy
import json
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent
from tests.agent.test_files_error_export_privacy import PRIVATE, SAFE, DATA


@pytest.mark.parametrize('files', [True, False], ids=['files', 'ordinary'])
@pytest.mark.parametrize('native_image', [False, True], ids=['text', 'image'])
def test_native_dispatch_refuses_prepared_files_only(tmp_path, monkeypatch, record_property, files, native_image):
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'native-owner')
        agent.api_mode = 'codex_app_server'
        calls = []
        def native(**kw):
            calls.append(copy.deepcopy(kw))
            return {'final_response': 'native sentinel', 'messages': kw['messages'], 'completed': True}
        agent._run_codex_app_server_turn = native
        # All tests intercept before the real native runtime; no app-server can launch.
        monkeypatch.setattr('subprocess.Popen', Mock(side_effect=AssertionError('native process forbidden')))
        content = ([{'type': 'text', 'text': PRIVATE},
                    {'type': 'image_url', 'image_url': {'url': DATA}}] if native_image else PRIVATE)
        scope = files_user_message_persistence(agent, SAFE) if files else nullcontext(None)
        with scope as transcript:
            result = agent.run_conversation(content, persist_user_message=transcript)
        record_property('native_boundary', json.dumps({'calls': calls, 'result': result,
            'sql': db.get_messages('native-owner')}, default=str))
        assert sent == []
        if files:
            assert calls == []
            assert result['failed'] is True and result['completed'] is False
            assert result['failure_reason'] == 'prepared_files_unsupported'
            assert result['failure_retryable'] is False
            assert 'codex_app_server' in result['error']
            assert PRIVATE not in json.dumps(result) and DATA not in json.dumps(result)
            assert PRIVATE not in json.dumps(db.get_messages('native-owner'))
            # A following ordinary turn on this same owner still dispatches.
            following = agent.run_conversation('ordinary next turn', conversation_history=[])
            assert len(calls) == 1 and calls[0]['user_message'] == 'ordinary next turn'
            assert following['final_response'] == 'native sentinel'
        else:
            assert len(calls) == 1
            assert calls[0]['user_message'] == content
            assert result['final_response'] == 'native sentinel'
    finally:
        db.close()
