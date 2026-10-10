"""Hard stops during overflow recovery must not become compression exhaustion."""
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_overflow import _Recovery


@pytest.mark.parametrize("hard_cancel", [True, False])
def test_cancelled_overflow_does_not_exhaust_compression(hard_cancel):
    messages = [{"role": "user", "content": "history"}]
    agent = SimpleNamespace(
        _interrupt_requested=False, _hard_interrupt_requested=Event(), _persist_session=Mock(), clear_interrupt=Mock(),
        _vprint=Mock(), log_prefix="", _drop_trailing_empty_response_scaffolding=Mock(),
    )

    def cancel(*args, **kwargs):
        agent._interrupt_requested = True
        if hard_cancel:
            agent._hard_interrupt_requested.set()
        return messages, "system"

    agent._compress_context = cancel
    recovery = _Recovery(
        agent=agent, messages=messages, active_system_prompt="system", conversation_history=messages,
        approx_tokens=1000, compression_attempts=1, api_messages=messages, system_message="system",
        effective_task_id=None, api_call_count=1, max_compression_attempts=3,
    )
    verdict = recovery.compress(1000, fail_on_timeout=True)
    if not hard_cancel:
        assert verdict is None
        agent.clear_interrupt.assert_not_called()
        return
    assert verdict.action == "return"
    assert verdict.result["interrupted"] is True
    assert "compression_exhausted" not in verdict.result
    assert "failed" not in verdict.result
    assert verdict.messages is messages
    agent.clear_interrupt.assert_called_once()
