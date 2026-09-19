"""A provider stuck returning 5xx server errors never fell over to the configured
fallback chain — only ``timeout``/``overloaded`` were treated as transport failures
eligible for the eager-fallback-after-retries path, so ``server_error`` (plain 500/502/504,
or an unrecognised 503 that isn't the memory-ceiling/empty-response special case) retried
the same dead provider until ``max_retries`` exhausted instead of switching (issue #115898)."""

from unittest.mock import MagicMock, patch

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


def _route(agent, reason):
    return route_classified_error(
        agent, RuntimeError("boom"), ClassifiedError(reason=reason), TurnRetryState(),
        error_msg="boom", error_context={}, recovered_with_pool=False,
        base_url="https://example.test/v1", model="test/model",
        messages=[], api_messages=[], system_message=None, active_system_prompt=None,
        conversation_history=[], retry_count=2, max_retries=10,
        compression_attempts=0, max_compression_attempts=2, api_call_count=3,
        effective_task_id=None,
    )


def test_server_error_falls_back_after_retries():
    agent = MagicMock()
    agent._fallback_chain = [{"provider": "deepseek", "model": "deepseek-chat"}]
    agent._fallback_index = 0
    agent._credential_pool = None
    agent._try_activate_fallback.return_value = True
    agent.compression_enabled = True

    with patch("agent.conversation_loop._arm_fallback_restart", side_effect=lambda *a: a[2]):
        verdict = _route(agent, FailoverReason.server_error)

    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.server_error)
    assert verdict.action == "break"


def test_overloaded_already_falls_back_after_retries():
    """Control case: ``overloaded`` already gets eager fallback — proves the harness
    itself is sound, so the server_error failure above is the real gap, not a test bug."""
    agent = MagicMock()
    agent._fallback_chain = [{"provider": "deepseek", "model": "deepseek-chat"}]
    agent._fallback_index = 0
    agent._credential_pool = None
    agent._try_activate_fallback.return_value = True
    agent.compression_enabled = True

    with patch("agent.conversation_loop._arm_fallback_restart", side_effect=lambda *a: a[2]):
        verdict = _route(agent, FailoverReason.overloaded)

    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.overloaded)
    assert verdict.action == "break"
