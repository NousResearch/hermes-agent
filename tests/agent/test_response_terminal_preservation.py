"""A returned response must be observed before recovery can discard or mutate it."""
from types import SimpleNamespace

import pytest

from agent import turn_response_check
from agent.turn_response_check import check_api_response
from tests.agent.test_request_terminal_events import _Aggregate, _check_agent
from agent.turn_retry_state import TurnRetryState
from hermes_cli import lifecycle


def _check(agent, response):
    return check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None,
        messages=[{'role': 'user', 'content': 'go'}], api_messages=[{'role': 'user', 'content': 'go'}],
        api_kwargs={'model': 'fake-model'}, active_system_prompt='sys', conversation_history=[],
        finish_reason='stop', retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=1,
        api_request_id='turn:api:3', api_start_time=1.0, effective_task_id='task', turn_id='turn',
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )


@pytest.mark.parametrize('with_usage', [False, True])
def test_length_response_is_observed_before_recovery(monkeypatch, with_usage):
    response = _Aggregate('incomplete but returned')
    if with_usage:
        response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10,
            total_tokens=110, prompt_tokens_details=SimpleNamespace(cached_tokens=40))
    agent = _check_agent(response)
    events = []
    monkeypatch.setattr(lifecycle, 'has_hook', lambda event: event == 'post_api_request')
    monkeypatch.setattr(lifecycle, 'invoke_hook', lambda event, **data: events.append((event, data)))
    monkeypatch.setattr('agent.turn_recovery.validate_response_shape', lambda *args: (False, None))

    def recover(*args, **kwargs):
        assert len(events) == 1, 'terminal must precede recovery, even if recovery raises'
        event, data = events[0]
        assert event == 'post_api_request'
        assert data['finish_reason'] == 'length'
        assert data['api_request_id'] == 'turn:api:3'
        if with_usage:
            assert data['usage']['input_tokens'] == 60
            assert data['usage']['cache_read_tokens'] == 40
            assert data['usage']['output_tokens'] == 10
        else:
            assert data['usage'] is None
        raise LookupError('recovery failed after response observation')

    monkeypatch.setattr(turn_response_check, 'recover_from_truncation', recover)
    with pytest.raises(LookupError, match='recovery failed'):
        _check(agent, response)
    assert len(events) == 1


def test_terminal_is_not_repeated_after_recovery_error(monkeypatch):
    from agent.turn_response_intake import _fire_post_api_request_hook
    from agent.turn_api_request import _fire_pre_api_request_hook
    response = _Aggregate('returned text')
    agent = _check_agent(response)
    events = []
    monkeypatch.setattr(lifecycle, 'has_hook', lambda name: name != 'pre_api_request')
    monkeypatch.setattr(lifecycle, 'invoke_hook', lambda name, **data: events.append(name))
    kwargs = dict(api_messages=[], api_call_count=1, api_duration=1, api_start_time=1,
        api_request_id='same-request', effective_task_id='task', turn_id='turn')
    _fire_post_api_request_hook(agent, response, response.choices[0].message, 'length', **kwargs)
    _fire_post_api_request_hook(agent, response, response.choices[0].message, 'length', **kwargs)
    agent._invoke_api_request_error_hook(task_id='task', turn_id='turn',
        api_request_id='same-request', api_call_count=1, api_start_time=1, api_kwargs={},
        error_type='LookupError', error_message='recovery error')
    assert events == ['post_api_request']
    # A fresh begin must allow a new terminal even if the retry reuses identifiers.
    _fire_pre_api_request_hook(agent, {}, [], [], messages=[], original_user_message='go',
        approx_tokens=0, total_chars=0, retry_count=1, api_call_count=1,
        api_request_id='same-request', api_start_time=1, effective_task_id='task', turn_id='turn')
    _fire_post_api_request_hook(agent, response, response.choices[0].message, 'stop', **kwargs)
    assert events == ['post_api_request', 'post_api_request']


def test_length_hook_defers_payload_work_until_subscribed(monkeypatch):
    from agent import turn_response_intake
    from agent.turn_truncation import TruncationVerdict
    response = _Aggregate('returned text')
    agent = _check_agent(response)
    monkeypatch.setattr(lifecycle, 'has_hook', lambda name: False)
    def unexpected_normalization(*args):
        raise AssertionError('observer payload normalization without a subscriber')
    monkeypatch.setattr(turn_response_intake, 'normalize_response_for_agent', unexpected_normalization)
    original = turn_response_intake._fire_post_api_request_hook
    messages = []
    def observe(agent, response, message, *args, **kwargs):
        messages.append(message)
        return original(agent, response, message, *args, **kwargs)
    monkeypatch.setattr(turn_response_intake, '_fire_post_api_request_hook', observe)
    monkeypatch.setattr('agent.turn_recovery.validate_response_shape', lambda *args: (False, None))
    monkeypatch.setattr(turn_response_check, 'recover_from_truncation', lambda *args, **kwargs:
        TruncationVerdict(action='continue', result=None, length_continue_retries=1,
            truncated_response_parts=[], truncated_tool_call_retries=0,
            retry_count=0, messages=[], compression_attempts=0))
    assert _check(agent, response).action == 'continue'
    assert messages == [None]


@pytest.mark.parametrize('exit_kind', ['recovered', 'interpreter_shutdown'])
def test_error_exit_before_classification_has_a_terminal(monkeypatch, exit_kind):
    from agent import turn_api_error
    from tests.agent.test_request_terminal_events import _bare_interrupt_agent
    agent = _bare_interrupt_agent('')
    agent._extract_api_error_context = lambda error: {}
    events = []
    monkeypatch.setattr(lifecycle, 'has_hook', lambda name: name == 'api_request_error')
    monkeypatch.setattr(lifecycle, 'invoke_hook', lambda name, **data: events.append(data))
    def recovery(*args, **kwargs):
        if exit_kind == 'recovered':
            agent.api_mode = 'bedrock_converse'
            return True, 'sys'
        return False, 'sys'
    monkeypatch.setattr(turn_api_error, 'recover_before_classification', recovery)
    monkeypatch.setattr('tools.interpreter_shutdown.interpreter_shutting_down', lambda error: True)
    verdict = turn_api_error.handle_api_error(
        agent, api_error=RuntimeError('fixture error'), _retry=TurnRetryState(),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={'model':'fake-model'},
        active_system_prompt='sys', system_message='sys', conversation_history=[], approx_tokens=0,
        retry_count=0, max_retries=3, compression_attempts=0, max_compression_attempts=3,
        api_call_count=1, api_request_id='request-error', api_start_time=1,
        effective_task_id='task', turn_id='turn')
    assert verdict.action == ('continue' if exit_kind == 'recovered' else 'return')
    assert len(events) == 1
    assert events[0]['api_request_id'] == 'request-error'
    assert events[0]['api_mode'] == 'chat_completions'
    assert events[0].get('usage') is None
