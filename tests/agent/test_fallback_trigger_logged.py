"""A fallback whose trigger is only buffered leaves its cause in the log (#152101).

The "switching to fallback" status is only buffered, and the buffer is dropped once the fallback
answers, so without a log line nothing records why the session left its primary model.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_agent.agent.error_classifier import FailoverReason
from hermes_agent.agent.turn_response_check import retry_invalid_response
from hermes_agent.agent.turn_retry_state import TurnRetryState


def _agent(switches: bool) -> MagicMock:
    agent = MagicMock()
    agent.log_prefix, agent.quiet_mode, agent.api_mode = "", True, "chat_completions"
    agent.provider, agent.model = "nous", "anthropic/claude-opus-5.5"
    agent._fallback_chain, agent._fallback_index = [{"provider": "custom", "model": "fb"}], 0
    agent._try_activate_fallback.return_value = switches
    return agent


def _run(agent, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_agent.agent.conversation_loop"):
        return retry_invalid_response(
            agent, response=SimpleNamespace(choices=[], error=None), error_details=["response.choices is empty"],
            _retry=TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs=None,
            active_system_prompt=None, conversation_history=None, retry_count=0, max_retries=3,
            compression_attempts=0, api_call_count=1, api_request_id="r", api_start_time=0.0, api_duration=0.4,
            effective_task_id="t", turn_id="turn",
        )


def _switch_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "switched to fallback" in r.getMessage()]


def test_invalid_response_fallback_logs_its_trigger(caplog):
    verdict = _run(_agent(switches=True), caplog)
    assert verdict.action == "break"
    assert _switch_lines(caplog) == ["Invalid API response (response.choices is empty) — switched to fallback"]


def test_declined_fallback_logs_no_switch(caplog):
    _run(_agent(switches=False), caplog)
    assert _switch_lines(caplog) == []


def test_content_filter_stream_fallback_names_reason_and_logs(caplog):
    from hermes_agent.agent.turn_truncation import _Trunc, _content_filter_fallback

    agent = _agent(switches=True)
    st = _Trunc(
        agent=agent, response=SimpleNamespace(_content_filter_terminated=True), finish_reason="length",
        conversation_history=[], api_call_count=1, effective_task_id="t", current_turn_user_idx=None,
        result=None, messages=[], length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )
    with caplog.at_level(logging.WARNING, logger="hermes_agent.agent.conversation_loop"):
        verdict = _content_filter_fallback(st, TurnRetryState())
    assert verdict is not None and verdict.action == "break"
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.content_policy_blocked)
    assert _switch_lines(caplog) == ["Content filter terminated the stream — switched to fallback"]


def test_nous_rate_guard_fallback_logs(monkeypatch, caplog):
    from hermes_agent.agent import nous_rate_guard
    from hermes_agent.agent.turn_api_call import nous_rate_limit_guard

    monkeypatch.setattr(nous_rate_guard, "nous_rate_limit_remaining", lambda **kw: 600)
    monkeypatch.setattr("hermes_agent.hermes_cli.anon_auth.is_anonymous_agent", lambda agent: False)
    monkeypatch.setattr("hermes_agent.hermes_cli.anon_auth.apply_model_switch", lambda agent: None)
    monkeypatch.setattr("hermes_agent.agent.conversation_loop._arm_fallback_restart", lambda *a: "sys")
    agent = _agent(switches=True)
    with caplog.at_level(logging.WARNING, logger="hermes_agent.agent.conversation_loop"):
        verdict = nous_rate_limit_guard(
            agent, _retry=TurnRetryState(), api_messages=[], messages=[], conversation_history=[],
            active_system_prompt="system", retry_count=0, compression_attempts=0, api_call_count=0,
        )
    assert verdict.action == "break"
    assert len(_switch_lines(caplog)) == 1 and _switch_lines(caplog)[0].startswith("Nous rate limit recorded")


def test_codex_app_server_fallback_logs_its_trigger(caplog):
    from hermes_agent.agent.turn_recovery import activate_codex_app_server_fallback

    agent = _agent(switches=True)
    agent._has_pending_fallback.return_value = True
    with caplog.at_level(logging.WARNING, logger="hermes_agent.agent.conversation_loop"):
        assert activate_codex_app_server_fallback(agent, {"error": "429 Too Many Requests"}) is True
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.rate_limit)
    assert _switch_lines(caplog) == ["Codex app-server turn failed (rate_limit: 429 Too Many Requests) — switched to fallback"]
