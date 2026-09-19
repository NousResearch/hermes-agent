"""Scope checkpoint: actual provider-error exports outside approved owners.

These are acceptance tests, not xfails. The SDK is inert; error classification,
error hook, dump and returned terminal result are real. No retries/fallback or
coordinator are used. Any failing case blocks full Files privacy acceptance.
"""
import copy
import json

import httpx
import openai
import pytest

from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent

PRIVATE = '/private/provider-echoed-files.txt'
SAFE = 'accepted prompt\n\n[Attached file: "a.txt"]'


@pytest.mark.parametrize('surface', ['error_hook', 'result', 'logs'])
def test_files_provider_error_never_exports_automatic_input(tmp_path, monkeypatch, record_property, caplog, surface):
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, sql = inert_agent(monkeypatch, db, 'error-owner')
        agent.logs_dir = tmp_path
        agent._api_max_retries = 1
        events = []
        def hook(name, **kwargs):
            if name == 'api_request_error':
                events.append(copy.deepcopy(kwargs))
            return []
        monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name == 'api_request_error')
        monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
        provider = agent.client.chat.completions.create.side_effect
        def fail(**kwargs):
            provider(**kwargs)
            raise openai.BadRequestError('invalid request: ' + PRIVATE,
                response=httpx.Response(400, request=httpx.Request('POST', 'https://inert.invalid/v1')),
                body={'error': {'message': 'invalid request: ' + PRIVATE}})
        agent.client.chat.completions.create.side_effect = fail
        with files_user_message_persistence(agent, SAFE) as transcript:
            result = agent.run_conversation(PRIVATE, persist_user_message=transcript)
        dumps = [json.loads(path.read_text()) for path in tmp_path.glob('request_dump_*.json')]
        assert len(sent) == 1
        assert db.get_messages('error-owner')[0]['content'] == SAFE
        assert db.get_messages('error-owner')[0]['api_content'] is None
        assert dumps and all(d.get('files_payload_omitted') is True for d in dumps)
        assert PRIVATE not in json.dumps(dumps)
        record_property('error_boundary', json.dumps({'hooks': events, 'result': result,
            'logs': caplog.text, 'dumps': dumps, 'sql': db.get_messages('error-owner')}, default=str))
        values = {'error_hook': events, 'result': result, 'logs': caplog.text}
        assert PRIVATE not in json.dumps(values[surface])
    finally:
        db.close()
