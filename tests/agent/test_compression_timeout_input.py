"""A timed-out summary keeps accepted input without claiming permanent exhaustion."""

from unittest.mock import MagicMock, patch
import asyncio
from types import SimpleNamespace

import pytest

from agent.conversation_compression import mark_context_compression_timed_out
from hermes_state import SessionDB
from tests.agent.test_compression_timeout_terminal import _make_agent, _mock_response


@pytest.mark.parametrize("timeout", [True, False])
def test_turn_start_retains_input_only_for_transient_timeout(tmp_path, timeout):
    agent = _make_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    agent._session_db = db
    db.create_session(agent.session_id, source="cli")
    history = [{"role": "user", "content": "old request"},
               {"role": "assistant", "content": "old answer"}]
    agent._flush_messages_to_session_db(history, [])
    compressor = agent.context_compressor
    compressor.protect_first_n = compressor.protect_last_n = 0
    compressor.threshold_tokens = compressor._resolved_context_length = 1
    compressor.should_compress = MagicMock(return_value=True)
    compressor.should_defer_preflight_to_real_usage = lambda _: False
    compressor.get_active_compression_failure_cooldown = lambda: None

    def compress(messages, *_args, **_kwargs):
        if timeout:
            mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt

    agent._compress_context = compress
    with patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation("new input", conversation_history=history,
                                        persist_user_platform_id="input-123")
    agent.client.chat.completions.create.assert_not_called()
    assert result["failed"] is True
    if timeout:
        assert result.get("compression_retryable") is True
        assert result["failure_retryable"] is True
        rows = db.get_messages(agent.session_id)
        assert [r["content"] for r in rows].count("new input") == 1
        assert any(r.get("content") == "new input" for r in result["messages"])
        compressor.should_compress = lambda _: False
        compressor._resolved_context_length = 131072
        agent.client.chat.completions.create.return_value = _mock_response("Recovered")
        retried = agent.run_conversation("retry", conversation_history=result["messages"])
        assert retried["completed"] is True
        rows = db.get_messages(agent.session_id)
        assert [r["content"] for r in rows].count("new input") == 1
        assert any(r.get("content") == "Recovered" for r in rows)
    else:
        assert not result.get("compression_retryable")
        assert result["compression_exhausted"] is True
        assert all(r["content"] != "new input" for r in db.get_messages(agent.session_id))
    db.close()


def test_gateway_keeps_timed_out_session_and_accepts_input():
    from gateway.run_turn import is_context_overflow_failure_result
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    result = {"failed": True, "compression_exhausted": True, "compression_retryable": True,
              "error": "Context compression timed out", "turn_exit_reason": "context_compression_timeout"}
    assert is_context_overflow_failure_result(result, 200) is False
    runner = GatewayTurnPersistenceMixin()
    entry = SimpleNamespace(session_id="original-session")
    response, retained = asyncio.run(runner._hmwa_compression_exhaustion_reset(
        result, "Please retry compression.", entry, "session-key", SimpleNamespace(platform=None),
    ))
    assert retained is entry and retained.session_id == "original-session"
    assert response == "Please retry compression."


def test_provider_overflow_then_timeout_is_still_transient():
    agent = _make_agent()
    error = RuntimeError("This model's maximum context length is 8192 tokens. Your messages resulted in 95000 tokens.")
    error.status_code = 400
    agent.client.chat.completions.create.side_effect = error

    def compress(messages, *_args, **_kwargs):
        mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt

    agent._compress_context = compress
    with patch.object(agent, "_persist_session"), patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation("continue", conversation_history=[
            {"role": "user", "content": "previous"}, {"role": "assistant", "content": "answer"}])
    assert result.get("compression_retryable") is True
    assert result["failure_retryable"] is True
    assert result["failure_reason"] == "timeout"
    assert agent.client.chat.completions.create.call_count == 1
