"""Provider refusals survive the worker wire and are not booked as crashes."""
import pytest
from agent.error_classifier import classify_api_error
from agent.managed_worker import accept_result, result_frame
from agent.turn_recovery import nonretryable_client_error_result
from hermes_cli.turn_exit import turn_exit_code
from hermes_cli.kanban_db_dispatch import _classify_dead_worker_exit


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
