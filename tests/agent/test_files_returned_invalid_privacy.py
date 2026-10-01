"""Returned invalid SDK bodies traverse real validation/retry/export boundaries."""
import copy
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent
from tests.agent.test_files_error_export_privacy import PRIVATE, SAFE, DATA, _isolate_recovery


@pytest.mark.parametrize('files', [True, False], ids=['files', 'ordinary'])
@pytest.mark.parametrize('body', ['error', 'message', 'metadata', 'failed', 'cancelled', 'empty-output'])
def test_returned_invalid_response_omits_only_files_diagnostics(
        tmp_path, monkeypatch, caplog, capsys, record_property, files, body):
    import agent.turn_recovery as recovery
    import agent.turn_failure_copy as failure_copy
    caplog.set_level(logging.DEBUG)
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'invalid-owner')
        agent.verbose_logging = True
        agent._api_max_retries = 2
        agent.logs_dir = tmp_path
        _isolate_recovery(agent)
        events, buffers, waits, fallback_calls, validation_inputs, reason_inputs = [], [], [], [], [], []
        agent._try_activate_fallback = lambda: fallback_calls.append(True) or False
        original_buffer = agent._buffer_vprint
        def buffer(message):
            buffers.append(message)
            return original_buffer(message)
        agent._buffer_vprint = buffer
        def hook(name, **kw):
            if name == 'api_request_error':
                events.append(copy.deepcopy(kw))
            return []
        monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name == 'api_request_error')
        monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
        monkeypatch.setattr(recovery, 'interruptible_backoff_sleep',
                            lambda agent, wait, retry, **kw: waits.append(wait))
        # Deterministic wait input; the actual wait boundary remains inert.
        monkeypatch.setattr('agent.retry_utils.jittered_backoff', lambda *a, **kw: 5.0)
        echo = PRIVATE + ' ' + DATA
        response = SimpleNamespace(choices=[], usage=None)
        if body == 'message':
            response.message = echo
        else:
            response.error = SimpleNamespace(message=echo, code=429,
                metadata={'provider_name': echo} if body == 'metadata' else {})
        if body in ('metadata', 'empty-output'):
            response.model = echo
            response.usage = {'diagnostic': echo}
        responses = body in ('failed', 'cancelled', 'empty-output')
        if responses:
            agent.api_mode = 'codex_responses'
            response.output = []
            response.output_text = ''
            response.status = body if body != 'empty-output' else 'completed'
            response.incomplete_details = {'reason': echo}
            response.error = {'code': 429, 'message': echo}
        before = copy.deepcopy(vars(response))
        transport = agent._get_transport()
        validate = transport.validate_response
        def validation(value):
            validation_inputs.append(value)
            return validate(value)
        monkeypatch.setattr(transport, 'validate_response', validation)
        real_reason = failure_copy.invalid_response_failure_reason
        def classify(value):
            reason_inputs.append(value)
            return real_reason(value)
        monkeypatch.setattr('agent.turn_response_check.invalid_response_failure_reason', classify)
        provider = agent.client.chat.completions.create.side_effect
        def sdk(**kw):
            provider(**kw)
            return response
        if responses:
            # Only the transport request is inert; request assembly, actual Codex
            # validation, response-check and retry finalization are production.
            def returned_transport(kwargs):
                sent.append(copy.deepcopy(kwargs))
                return response
            agent._interruptible_api_call = returned_transport
        else:
            agent.client.chat.completions.create.side_effect = sdk
        content = [{'type': 'text', 'text': PRIVATE},
            {'type': 'image_url', 'image_url': {'url': DATA}}] if files else 'ordinary prompt'
        scope = files_user_message_persistence(agent, SAFE) if files else nullcontext(None)
        with scope as transcript:
            result = agent.run_conversation(content, persist_user_message=transcript)
        assert len(sent) == 2
        assert len(validation_inputs) == 2 and all(v is response for v in validation_inputs)
        assert reason_inputs == [response]
        assert vars(response) == before
        assert waits == [5.0] and len(fallback_calls) == 3
        assert result['failure_reason'] == ('invalid_response' if body == 'message' else 'rate_limit')
        assert result['failure_retryable'] is True
        assert len(events) == 2 and all(e['reason'] == 'invalid_response' for e in events)
        assert [e['retry_count'] for e in events] == [0, 1]
        assert all(e['retryable'] is True for e in events)
        expected_status = None if body == 'message' or responses else 429
        assert all(e['status_code'] == expected_status for e in events)
        observations = {'buffers': buffers, 'logs': caplog.text, 'stdio': capsys.readouterr().out,
                        'hooks': events, 'result': result}
        record_property('returned_invalid_diagnostics', json.dumps(observations, default=str))
        if files:
            assert PRIVATE in json.dumps(sent) and DATA in json.dumps(sent)
            for surface, value in observations.items():
                assert PRIVATE not in json.dumps(value), surface
                assert DATA not in json.dumps(value), surface
        else:
            assert PRIVATE in json.dumps(buffers) and DATA in json.dumps(buffers)
            assert PRIVATE in observations['stdio']
            assert 'files_payload_omitted' not in events[0]['request']
            if responses:
                assert PRIVATE in observations['logs']
    finally:
        db.close()
