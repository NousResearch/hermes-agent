"""Provider refusals survive the worker wire and are not booked as crashes."""
import pytest
from agent.error_classifier import classify_api_error
from agent.managed_worker import accept_result, result_frame
from agent.turn_recovery import nonretryable_client_error_result
from hermes_cli.turn_exit import turn_exit_code
from hermes_cli.kanban_db_dispatch import _classify_dead_worker_exit


def test_terminal_provider_history_is_not_a_process_crash():
    from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    dead = _classify_dead_worker_exit(12345, 'synthetic', KANBAN_TERMINAL_PROVIDER_EXIT_CODE)
    assert dead.run_outcome == dead.event_kind == 'provider_rejected'
    assert dead.terminal_provider


class ProviderError(Exception):
    def __init__(self, status, code=None):
        super().__init__('synthetic provider refusal')
        self.status_code = status
        self.body = {'error': {'code': code, 'message': 'synthetic refusal'}} if code else {}


class Agent:
    log_prefix = ''
    verbose = False
    def _summarize_api_error(self, error):
        return 'synthetic refusal'
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.parametrize('status,code,expected', [
    (400, None, 'provider_rejected'), (401, None, 'provider_rejected'),
    (429, 'rate_limit_exceeded', 'rate_limited'),
    (429, 'insufficient_quota', 'rate_limited'),
    (503, None, 'rate_limited')])
def test_provider_result_is_distinct_from_process_loss(status, code, expected):
    error = ProviderError(status, code)
    classified = classify_api_error(error, provider='openai', model='synthetic')
    result = nonretryable_client_error_result(
        Agent(), error, classified, status_code=status, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=1, provider='openai',
        base_url='https://example.invalid/v1', model='synthetic')
    result, _ = accept_result(result_frame(result))
    assert result['failure_status_code'] == status
    assert result['failure_retryable'] == classified.retryable
    dead = _classify_dead_worker_exit(12345, 'synthetic', turn_exit_code(result, kanban_worker=True))
    assert dead.run_outcome == expected
    assert dead.event_kind == expected
    crash = _classify_dead_worker_exit(12345, 'synthetic', 1)
    assert crash.run_outcome == 'crashed'
    assert turn_exit_code(result | {'interrupted': True}, kanban_worker=True) == 130


@pytest.mark.parametrize('status', [400, 403])
def test_content_policy_exception_survives_worker_and_kanban(status):
    import io
    import json
    from unittest.mock import Mock

    import httpx
    import openai

    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.error_classifier import FailoverReason
    from agent.managed_worker import encode_frame, read_frame
    from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE

    body = {'error': {'code': 'content_policy_violation',
                      'message': 'Your request was rejected because it violates our usage policies.'}}
    response = httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid/v1/chat/completions'),
                              json=body)
    error_type = {400: openai.BadRequestError, 403: openai.PermissionDeniedError}[status]
    error = error_type(body['error']['message'], response=response, body=body)
    classified = classify_api_error(error, provider='openai', model='synthetic')
    assert classified.reason == FailoverReason.content_policy_blocked
    assert classified.status_code == status and not classified.retryable
    history = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'retained'}]
    messages = history + [
        {'role': 'user', 'content': 'refused request', '_row_id': 41, _DB_PERSISTED_MARKER: True},
        {'role': 'assistant', 'content': 'partial response'},
        {'role': 'user', 'content': 'later steer', '_row_id': 43, _DB_PERSISTED_MARKER: True},
    ]
    original = json.loads(json.dumps(messages))
    agent = Agent()
    agent._persist_session = Mock(side_effect=AssertionError('refused turn persisted'))
    result = nonretryable_client_error_result(
        agent, error, classified, status_code=status, api_kwargs=None, api_messages=messages,
        messages=messages, conversation_history=messages, current_turn_user_idx=len(history),
        api_call_count=1, approx_tokens=100, provider='openai', base_url='https://example.invalid/v1',
        model='synthetic')
    assert result['messages'] == history
    assert result['refused_turn_boundary_row_id'] == 41
    assert messages == original
    agent._persist_session.assert_not_called()
    wire, usage = accept_result(read_frame(io.BytesIO(encode_frame(result_frame(result)))))
    assert usage == {}
    assert wire['failed'] and wire['completed'] is False
    assert wire['failure_reason'] == 'content_policy_blocked'
    assert wire['failure_retryable'] is False
    assert wire['refused_turn_boundary_row_id'] == 41
    exit_code = turn_exit_code(wire, kanban_worker=True)
    assert exit_code == KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    dead = _classify_dead_worker_exit(12345, 'synthetic', exit_code)
    assert dead.run_outcome == dead.event_kind == 'provider_rejected'
    assert dead.terminal_provider
    assert wire['failure_status_code'] == status
    assert turn_exit_code(wire, kanban_worker=False) == 1
    assert turn_exit_code(wire | {'interrupted': True}, kanban_worker=True) == 130
