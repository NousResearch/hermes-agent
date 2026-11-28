"""Hard stops during overflow recovery must not become compression exhaustion."""
import copy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from agent.turn_overflow import _Recovery
from tests.agent.test_compression_concurrent_fork import _build_agent_with_db
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


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


@pytest.mark.parametrize("reason", ["payload_too_large", "context_overflow"])
def test_hard_cancel_during_overflow_preserves_session(tmp_path: Path, reason: str) -> None:
    """Real compression rollback must not become an exhausted/resettable session."""
    from agent import auxiliary_client as aux
    from agent.error_classifier import FailoverReason
    from agent.turn_overflow import recover_from_overflow
    from agent.turn_retry_state import TurnRetryState

    db = storage_handle(tmp_path)
    sid = "OVERFLOW_CANCEL"
    seed_session(db, sid, "cli")
    agent = _build_agent_with_db(db, sid)
    agent.context_compressor.context_length = 100_000
    agent._try_strip_image_parts_from_tool_messages = MagicMock(return_value=False)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "history " * 1000} for i in range(20)]
    original = copy.deepcopy(messages)

    def cancel(current, **kwargs):
        agent.interrupt(hard_cancel=True)
        current[0]["content"] = "must be rolled back"
        raise aux.AuxiliaryExplicitCancellation()

    agent.context_compressor.compress.side_effect = cancel
    try:
        verdict = recover_from_overflow(
            agent, RuntimeError("context length exceeded: 120000 tokens"),
            SimpleNamespace(reason=FailoverReason(reason)), TurnRetryState(),
            status_code=413 if reason == "payload_too_large" else 400,
            error_msg="context length exceeded: 120000 tokens", wrapped_output_cap_budget=None,
            messages=messages, api_messages=messages, system_message="sys", active_system_prompt="sys",
            conversation_history=[], approx_tokens=120_000, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1, effective_task_id=None,
        )
        assert verdict.result.get("interrupted") is True
        assert not verdict.result.get("compression_exhausted")
        assert not verdict.result.get("failed")
        assert [(m["role"], m["content"]) for m in verdict.messages] == [(m["role"], m["content"]) for m in original]
        assert [(m["role"], m["content"]) for m in transcript_rows(db, sid)] == [
            (m["role"], m["content"]) for m in original
        ]
        assert db.coordination.get_compression_lock_holder(session_id=sid) is None
        seed_messages(db, sid, {"role": "user", "content": "next turn remains writable"})
    finally:
        db.close()
