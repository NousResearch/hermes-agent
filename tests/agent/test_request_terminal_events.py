"""A logical provider call must always get exactly one terminal event.

Gaps found by the loopback install acceptance run: (1) a user interrupt mid-call ends the
turn without firing ``api_request_error``; (2) a truncated/retried response leaves the same
hole for the attempt that gets retried. Both leave an observer holding a begin it can never
close. The hook payload is the existing one; only the missing dispatch is added.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent.turn_api_call import handle_api_interrupt
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


class _Aggregate:
    """Minimal OpenAI-shaped response: one text choice, no tool calls."""

    def __init__(self, content: str, finish_reason: str = "length"):
        self.model = "fake-model"
        self.status = "completed"
        self.incomplete_details = None
        self.usage = None
        self.choices = [SimpleNamespace(message=SimpleNamespace(
            role="assistant", content=content, tool_calls=None), finish_reason=finish_reason)]
        self._clean_eof = False
        self._overflow_terminal = False
        self._runaway_repetition = False


class _Transport:
    def normalize_response(self, response):
        message = response.choices[0].message
        return SimpleNamespace(content=message.content, tool_calls=None,
                               finish_reason=response.choices[0].finish_reason)

    def response_finish_reason(self, response):
        return response.choices[0].finish_reason


def _hook_recorder(agent: AIAgent) -> list[dict]:
    calls: list[dict] = []

    def _record(**kwargs):
        calls.append(kwargs)

    agent._invoke_api_request_error_hook = _record
    return calls


def _bare_interrupt_agent(streamed: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._current_streamed_assistant_text = streamed
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.thinking_callback = None
    agent._print_fn = lambda *args, **kwargs: None
    agent._persist_session = lambda *args, **kwargs: None
    agent.session_id = "session-a"
    agent.platform = "cli"
    agent.model = "fake-model"
    agent.provider = "fake"
    agent.base_url = "http://127.0.0.1:1/v1"
    agent.api_mode = "chat_completions"
    return agent


def test_interrupted_call_fires_one_terminal_error_event():
    agent = _bare_interrupt_agent("Visible draft.")
    calls = _hook_recorder(agent)

    verdict = handle_api_interrupt(
        agent, _retry=TurnRetryState(), thinking_spinner=None,
        messages=[{"role": "user", "content": "start"}], conversation_history=[],
        api_start_time=0.0, interrupted=False, final_response=None,
        api_request_id="turn:api:1", api_call_count=1, api_kwargs={"model": "fake-model"},
        effective_task_id="task", turn_id="turn",
    )

    assert verdict.action == "break" and verdict.interrupted is True
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["api_request_id"] == "turn:api:1"
    assert call["reason"] == "interrupted"
    assert call["error_type"] == "InterruptedError"
    assert call["retryable"] is False
    assert call["api_call_count"] == 1


def test_redirect_restart_fires_the_same_terminal_event():
    agent = _bare_interrupt_agent("stale answer")
    agent._has_pending_redirect = lambda: True
    agent.clear_interrupt = lambda preserve_redirect=False: True
    calls = _hook_recorder(agent)
    retry = TurnRetryState()

    verdict = handle_api_interrupt(
        agent, _retry=retry, thinking_spinner=None, messages=[], conversation_history=[],
        api_start_time=0.0, interrupted=False, final_response=None,
        api_request_id="turn:api:2", api_call_count=1, api_kwargs=None,
        effective_task_id="task", turn_id="turn",
    )

    assert verdict.action == "break" and retry.restart_with_redirected_messages is True
    assert [c["reason"] for c in calls] == ["redirected"]


def _check_agent(response) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.verbose_logging = False
    agent.thinking_callback = None
    agent._print_fn = lambda *args, **kwargs: None
    agent._vprint = lambda *args, **kwargs: None
    agent._flush_status_buffer = lambda: None
    agent._touch_activity = lambda *args, **kwargs: None
    agent._get_transport = lambda: _Transport()
    agent._should_treat_stop_as_truncated = lambda *args, **kwargs: False
    agent._turn_received_provider_response = False
    agent.session_id = "session-b"
    agent.platform = "cli"
    agent.model = "fake-model"
    agent.provider = "fake"
    agent.base_url = "http://127.0.0.1:1/v1"
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent.context_compressor = None
    agent._prepare_response_for_agent = getattr(agent, "_prepare_response_for_agent", None)
    return agent


def test_truncated_response_fires_terminal_before_retry(monkeypatch):
    from agent import turn_recovery, turn_truncation
    from agent import turn_response_check
    from agent.turn_response_check import check_api_response

    monkeypatch.setattr(turn_recovery, "validate_response_shape", lambda agent, response: (False, None))

    def _continue(*args, **kwargs):
        return turn_truncation.TruncationVerdict(
            action="continue", messages=kwargs.get("messages") or [],
            length_continue_retries=1, truncated_response_parts=[], truncated_tool_call_retries=0,
            retry_count=0, compression_attempts=0, result=None,
        )

    # The intake phase binds the helper at import; patch where it is read.
    monkeypatch.setattr(turn_response_check, "recover_from_truncation", _continue)
    agent = _check_agent_if_needed = _check_agent(_Aggregate("half an ans"))
    from hermes_cli import lifecycle
    calls = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda event: event == "post_api_request")
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **data: calls.append(data))
    retry = TurnRetryState()

    verdict = check_api_response(
        agent, response=_Aggregate("half an ans"), _retry=retry, thinking_spinner=None,
        messages=[{"role": "user", "content": "go"}], api_messages=[{"role": "user", "content": "go"}],
        api_kwargs={"model": "fake-model"}, active_system_prompt="sys", conversation_history=[],
        finish_reason="stop", retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=1,
        api_request_id="turn:api:3", api_start_time=0.0, effective_task_id="task", turn_id="turn",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )

    assert verdict.action == "continue"
    assert len(calls) == 1 and calls[0]["finish_reason"] == "length", calls
    assert calls[0]["api_request_id"] == "turn:api:3"
    assert calls[0]["usage"] is None
