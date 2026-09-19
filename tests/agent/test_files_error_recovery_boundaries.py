"""Real inert SDK/recovery: exports must not replace classification input."""
import copy
import json
from contextlib import nullcontext

import httpx
import openai
import pytest

from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent
from tests.agent.test_files_error_export_privacy import PRIVATE, SAFE, DATA, _isolate_recovery


@pytest.mark.parametrize('files', [True, False], ids=['files', 'ordinary'])
@pytest.mark.parametrize('outcome', ['success', 'second_error', 'interrupted', 'output_cap'])
def test_recovery_preserves_original_error_and_separates_exports(
        tmp_path, monkeypatch, caplog, capsys, record_property, files, outcome):
    import agent.turn_api_error as errors
    from agent.error_classifier import FailoverReason
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, at_provider = inert_agent(monkeypatch, db, 'recovery-owner')
        agent.logs_dir = tmp_path
        agent.verbose_logging = True
        agent._api_max_retries = 2
        _isolate_recovery(agent)
        events, classified_inputs, pool_inputs, waits = [], [], [], []
        real_classify = errors.classify_api_error
        def classify(error, **kw):
            verdict = real_classify(error, **kw)
            classified_inputs.append((error, verdict))
            return verdict
        monkeypatch.setattr(errors, 'classify_api_error', classify)
        def pool(**kw):
            pool_inputs.append(kw)
            if outcome == 'interrupted':
                agent._interrupt_requested = True
            return False, kw['has_retried_429']
        agent._recover_with_credential_pool = pool
        def sleep(agent, wait, retry, **kw):
            waits.append(wait)
            return None
        # Only the actual wait boundary is inert. The real backoff sees Retry-After.
        monkeypatch.setattr(errors, 'interruptible_backoff_sleep', sleep)
        def hook(name, **kw):
            if name == 'api_request_error':
                events.append(copy.deepcopy(kw))
            return []
        monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name == 'api_request_error')
        monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
        echo = PRIVATE + ' ' + DATA
        text = ('max_tokens exceeds model maximum output tokens: ' if outcome == 'output_cap'
                else 'inert provider failure: ') + echo
        status = 400 if outcome == 'output_cap' else 500
        exc_type = openai.BadRequestError if status == 400 else openai.InternalServerError
        original = exc_type(text, response=httpx.Response(status,
            headers={'retry-after': '7'}, request=httpx.Request('POST', 'https://inert.invalid/v1')),
            body={'error': {'message': text}})
        provider = agent.client.chat.completions.create.side_effect
        def sdk(**kw):
            response = provider(**kw)
            if len(sent) == 1 or outcome != 'success':
                raise original
            return response
        agent.client.chat.completions.create.side_effect = sdk
        content = [{'type': 'text', 'text': PRIVATE},
                   {'type': 'image_url', 'image_url': {'url': DATA}}] if files else 'ordinary prompt'
        scope = files_user_message_persistence(agent, SAFE) if files else nullcontext(None)
        with scope as transcript:
            result = agent.run_conversation(content, persist_user_message=transcript)
        assert classified_inputs and all(error is original for error, _ in classified_inputs)
        assert len(events) == len(classified_inputs)
        expected = FailoverReason.context_overflow if outcome == 'output_cap' else FailoverReason.server_error
        assert all(v.reason == expected and v.status_code == status for _, v in classified_inputs)
        assert all(p['classified_reason'] == expected and p['status_code'] == status for p in pool_inputs)
        assert all(PRIVATE in p['error_context']['message'] for p in pool_inputs)
        assert waits == ([7.0] if outcome in ('success', 'second_error') else [])
        assert len(sent) == (2 if outcome in ('success', 'second_error') else 1)
        if outcome == 'interrupted':
            assert result.get('interrupted') is True
        else:
            assert not result.get('interrupted')
        if outcome == 'success':
            assert result['final_response'] == 'inert answer'
            assert sent[0] == sent[1]
        elif outcome == 'second_error':
            assert result['failure_reason'] == 'server_error'
            assert result['failure_retryable'] is True
        elif outcome == 'output_cap':
            assert result['failure_reason'] == 'context_overflow'
            assert 'Lower model.max_tokens' in result['error']
        dumps = [json.loads(p.read_text()) for p in tmp_path.glob('request_dump_*.json')]
        observations = {'hooks': events, 'result': result, 'logs': caplog.text,
            'stdio': capsys.readouterr().out, 'sql': db.get_messages('recovery-owner'),
            'dumps': dumps, 'waits': waits}
        record_property('recovery_boundary', json.dumps(observations, default=str))
        if files:
            assert PRIVATE in json.dumps(sent) and DATA in json.dumps(sent)
            if outcome != 'output_cap':
                assert all(v.retryable is True for _, v in classified_inputs)
            encoded = json.dumps(observations)
            assert PRIVATE not in encoded and DATA not in encoded
            assert all(event['request'].get('files_payload_omitted') is True for event in events)
            assert db.get_messages('recovery-owner')[0]['api_content'] is None
        else:
            assert PRIVATE in json.dumps(events) and DATA in json.dumps(events)
            assert PRIVATE in observations['logs']
            assert 'files_payload_omitted' not in events[0]['request']
    finally:
        db.close()


def test_following_ordinary_request_does_not_inherit_error_omission(tmp_path, monkeypatch, caplog):
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'reset-owner')
        _isolate_recovery(agent)
        agent.logs_dir = tmp_path
        with files_user_message_persistence(agent, SAFE) as transcript:
            agent.run_conversation(PRIVATE, persist_user_message=transcript)
        assert agent._files_request_expanded is True
        assert agent._files_live_entries  # The previous success retained live context.
        provider = agent.client.chat.completions.create.side_effect
        def sdk(**kw):
            provider(**kw)
            raise openai.BadRequestError('ordinary diagnostic ' + PRIVATE,
                response=httpx.Response(400, request=httpx.Request('POST', 'https://inert.invalid/v1')),
                body={'error': {'message': 'ordinary diagnostic ' + PRIVATE}})
        agent.client.chat.completions.create.side_effect = sdk
        result = agent.run_conversation('ordinary replacement history', conversation_history=[])
        assert agent._files_request_expanded is False
        assert PRIVATE in result['error'] and PRIVATE in caplog.text
        assert PRIVATE not in json.dumps(sent[-1])
    finally:
        db.close()
