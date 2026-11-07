"""A compressor recovery probe must also be executable by the turn's caller."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.conversation_compression import compress_context
from agent.turn_overflow import _Recovery, _recover_context_length
from agent.turn_retry_state import TurnRetryState
from agent.turn_usage import record_response_usage
from tests.agent.test_compression_ineffective_growth import _Agent, _compressor, _messages, _strike


@pytest.mark.parametrize("cap", [1, 3])
def test_provider_growth_restores_only_one_caller_attempt(cap):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url="https://example.invalid/v1", provider="custom",
        model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )
    cc = agent.context_compressor = _compressor()
    cc.threshold_tokens = 32_768

    def usage(tokens, attempts):
        response = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=tokens, completion_tokens=1, total_tokens=tokens + 1,
        ))
        return record_response_usage(agent, response, messages=_messages(), api_call_count=2,
                                     api_duration=0.1, compression_attempts=attempts, max_compression_attempts=cap)

    try:
        with patch("agent.context_compressor.time.time", return_value=1000):
            _strike(cc, 35_059)
            _strike(cc, 34_105)
            out = usage(35_000, cap)
            assert out.compression_attempts == cap and not out.rearmed
            out = usage(55_205, cap)
            assert out.compression_attempts == cap - 1 and out.rearmed
            assert cc.should_compress(55_205)
            # Spending the one slot and failing does not reopen it on every API call.
            _strike(cc, 54_000)
            out = usage(55_205, cap)
            assert out.compression_attempts == cap and not out.rearmed
            out = usage(61_000, cap)
            assert out.compression_attempts == cap - 1 and out.rearmed
            out = usage(61_000, cap)
            assert out.compression_attempts == cap and not out.rearmed
    finally:
        agent.close()


def test_exhausted_overflow_budget_reaches_outer_commit_once():
    cc = _compressor()
    cc.threshold_tokens = 32_768
    agent = _Agent(cc)
    agent.base_url, agent.max_tokens, agent.log_prefix = "https://example.invalid/v1", None, ""
    agent._buffer_vprint = agent._buffer_diagnostic_status = lambda *a, **k: None
    agent._flush_status_buffer = lambda: None
    agent._vprint = agent._persist_session = lambda *a, **k: None
    agent._compress_context = lambda messages, system, **kw: compress_context(agent, messages, system, **kw)
    messages = _messages()

    def recover(messages):
        st = _Recovery(agent=agent, api_messages=messages, system_message="sys", effective_task_id="test",
                       api_call_count=4, max_compression_attempts=3, messages=messages, active_system_prompt="sys",
                       conversation_history=None, approx_tokens=110_000, compression_attempts=3)
        return _recover_context_length(st, TurnRetryState(), "request (110000 tokens) exceeds context size")

    with patch("agent.context_compressor.time.time", return_value=1000), \
            patch("agent.turn_overflow.time.sleep", lambda _: None), \
            patch.object(cc, "_generate_summary", return_value="Recovered summary") as summary:
        _strike(cc, 35_059)
        _strike(cc, 34_105)
        result = recover(messages)
        assert result.action == "break"
        assert result.compression_attempts == 3
        summary.assert_called_once()
        assert cc._verify_compaction_cleared_threshold
        cc.update_from_response({"prompt_tokens": 54_000})
        repeated = recover(result.messages)
        assert repeated.action == "return"
        assert repeated.result["compression_exhausted"]
        summary.assert_called_once()
