"""Empty-response fallback hop refunds its API call (#77305)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import turn_empty_response as ter


def _agent():
    a = MagicMock()
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._last_content_with_tools = None
    a._thinking_prefill_retries = 2
    a._fallback_chain = ["fallback"]
    a._api_call_count = 5
    return a


def _recover(agent):
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    with patch.object(ter, "_retry_empty", return_value=(None, None, False)), \
            patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"):
        return ter.recover_empty_response(
            agent, msg, None, "stop", final_response="", messages=[{"role": "user", "content": "hi"}],
            api_messages=[], conversation_history=[], active_system_prompt="sys", api_call_count=5,
            turn_exit_reason=None, preflight_compression_blocked=False,
        )


def test_fallback_activation_refunds_call_and_budget():
    agent = _agent()
    agent._try_activate_fallback.return_value = True
    v = _recover(agent)
    assert v.action == "continue"
    assert v.api_call_count == 4
    assert agent._api_call_count == 4
    agent.iteration_budget.refund.assert_called_once()


def test_exhausted_without_fallback_keeps_count():
    agent = _agent()
    agent._try_activate_fallback.return_value = False
    with patch.object(ter, "_terminal_empty", return_value="(empty)"):
        v = _recover(agent)
    assert v.action == "break"
    assert v.api_call_count == 5
    agent.iteration_budget.refund.assert_not_called()
