"""Fresh structural eligibility is distinct from a failed summary or stale backoff."""

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.conversation_loop_compression_results import _compression_deferred_result


def test_provider_rejected_current_request_gets_an_actionable_preserving_result():
    compressor = ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True)
    messages = [{"role": "user", "content": "large current request\n" * 10_000}]
    compressor.compress(messages, current_tokens=65_536)
    agent = SimpleNamespace(context_compressor=compressor, session_id="S", _flush_status_buffer=lambda: None)
    result = _compression_deferred_result(agent, messages, 1, reason="transient_block",
                                         provider_rejected=True, request_tokens=65_536)
    assert result["failure_reason"] == "request_uncompressible"
    assert result["failure_retryable"] is False
    assert result["compression_deferred"] is True
    assert not result.get("compression_exhausted")
    assert result["messages"] is messages
    assert "Shorten" in result["final_response"]
    assert "failed attempt" not in result["final_response"]


def test_stale_structural_backoff_does_not_classify_a_new_eligible_history_as_impossible():
    compressor = ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True)
    earlier = {"role": "user", "content": "earlier historical text\n" * 10_000}
    compressor.compress([earlier], current_tokens=65_536)
    agent = SimpleNamespace(context_compressor=compressor, session_id="S", _flush_status_buffer=lambda: None)
    messages = [earlier, {"role": "assistant", "content": "SAVED"}, {"role": "user", "content": "Recall it."}]
    result = _compression_deferred_result(agent, messages, 1, reason="transient_block",
                                         provider_rejected=True, request_tokens=65_536)
    assert result.get("failure_reason") != "request_uncompressible"
    assert result["compression_deferred"] is True


def test_an_ordinary_preflight_or_manual_noop_is_not_a_provider_rejection():
    compressor = ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True)
    agent = SimpleNamespace(context_compressor=compressor, session_id="S", _flush_status_buffer=lambda: None)
    result = _compression_deferred_result(agent, [{"role": "user", "content": "small"}], 0,
                                         reason="transient_block")
    assert result.get("failure_reason") != "request_uncompressible"
    assert result["compression_deferred"] is True


def test_typed_definite_preflight_refusal_preserves_current_input_without_provider_proof():
    from agent.conversation_loop_compression_results import _preflight_refusal_result
    from agent.turn_context import PreflightCompressionInsufficientProgress

    compressor = ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True)
    messages = [{"role": "user", "content": "current request\n" * 30_000}]
    writes = []
    agent = SimpleNamespace(context_compressor=compressor, session_id="S",
                            _persist_session=lambda messages, history: writes.append((messages, history)))
    refusal = PreflightCompressionInsufficientProgress(
        "known over-window refusal", messages=messages, conversation_history=[], request_tokens=100_000,
    )
    result = _preflight_refusal_result(agent, refusal, [])
    assert result["failure_reason"] == "request_uncompressible"
    assert not result.get("compression_exhausted")
    assert result["messages"] is messages
    assert "configured model context window" in result["final_response"]
    assert "provider's" not in result["final_response"]
    assert writes == [(messages, [])]


def test_elapsed_wait_and_eligible_history_failure_keep_the_existing_failure_outcome():
    from agent.conversation_loop_compression_results import _preflight_refusal_result
    from agent.turn_context import PreflightCompressionInsufficientProgress, PreflightCompressionTimedOut

    compressor = ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True)
    history = [{"role": "user", "content": "historical input\n" * 30_000},
               {"role": "assistant", "content": "SAVED"}]
    messages = history + [{"role": "user", "content": "Recall it."}]
    agent = SimpleNamespace(context_compressor=compressor, session_id="S")
    insufficient = PreflightCompressionInsufficientProgress(
        "eligible summary failed", messages=messages, conversation_history=history, request_tokens=100_000,
    )
    for refusal in (insufficient, PreflightCompressionTimedOut("elapsed wait budget")):
        result = _preflight_refusal_result(agent, refusal, history)
        assert result["failure_reason"] == "context_overflow"
        assert result["failed"] is True
        assert result["compression_exhausted"] is True
        assert result["final_response"] == str(refusal)
        expected_reason = ("context_compression_no_progress" if refusal is insufficient
                           else "context_compression_timeout")
        assert result["turn_exit_reason"] == expected_reason
        assert result["messages"] == history
