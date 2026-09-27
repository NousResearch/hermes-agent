"""Regression tests for #125492: a length-stopped call that ends the turn from
``recover_from_truncation`` must still be accounted and logged.

The truncation verdicts (``return``/``break``/``continue``) exit ``check_api_response``
and the ``_Trunc.end_turn`` returns exit ``run_conversation`` without reaching
``finalize_turn``. Before the fix, those exits skipped ``record_response_usage``
(no ``API call #N`` line, ``session_api_calls`` left at 0), never logged a
``Turn ended:`` line, and never released the deferred title upgrade — so the most
expensive failures (a call that burned the whole output budget) were the invisible
ones.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

from hermes_constants import FINISH_REASON_LENGTH


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _length_response(prompt=100, completion=12288):
    """A valid chat_completions response stopped by the output cap with
    reasoning-only content (the thinking-budget-exhaustion abort path)."""
    message = SimpleNamespace(role="assistant", content="<think>reasoning that never ends</think>",
                              tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason=FINISH_REASON_LENGTH)
    return SimpleNamespace(
        choices=[choice], model="test/model", id=None, provider=None,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion),
    )


def _check(agent, response, messages):
    from agent.turn_response_check import check_api_response
    from agent.turn_retry_state import TurnRetryState

    return check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None,
        messages=messages, api_messages=list(messages), api_kwargs={},
        active_system_prompt=None, conversation_history=[], finish_reason=None,
        retry_count=0, max_retries=3, compression_attempts=0, max_compression_attempts=3,
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=1,
        api_request_id="req-1", api_start_time=__import__("time").time() - 185.0,
        effective_task_id=None, turn_id="t1",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )


class TestTruncationTurnAccounting:
    def test_length_stopped_end_turn_call_is_accounted_and_logged(self, tmp_path, monkeypatch, caplog):
        """The turn-ending truncation exit must still record usage (API call line +
        session counters) and log why the turn ended."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        agent = _make_agent()
        try:
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
                verdict = _check(agent, _length_response(), [{"role": "user", "content": "hi"}])

            assert verdict.action == "return"
            assert verdict.result.get("failure_reason") == "truncated"
            # The billable call is observable: usage was recorded BEFORE the
            # truncation branch returned (pre-fix: neither line existed).
            assert any(r.getMessage().startswith("API call #1:") for r in caplog.records)
            assert agent.session_api_calls == 1
            assert agent.session_completion_tokens == 12288
            # The turn exit is in agent.log with the failure verdict as the reason.
            assert any(
                r.getMessage().startswith("Turn ended: reason=truncated ")
                for r in caplog.records
            )
        finally:
            agent.close()

    def test_end_turn_releases_deferred_title_upgrade(self, tmp_path, monkeypatch):
        """``start_deferred_title_upgrade`` is finalize_turn's job on normal exits;
        the truncation end-turn must fire (and clear) it too (#117296)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        agent = _make_agent()
        try:
            agent._deferred_title_upgrade = {"session_id": agent.session_id}
            with patch("agent.title_generator.start_title_upgrade") as title_upgrade:
                verdict = _check(agent, _length_response(), [{"role": "user", "content": "hi"}])
            assert verdict.action == "return"
            title_upgrade.assert_called_once()
            # Cleared, so a later turn cannot fire it twice.
            assert agent._deferred_title_upgrade is None
        finally:
            agent.close()
