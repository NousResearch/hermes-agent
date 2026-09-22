"""Deterministic loop outcomes (context overflow, empty reply, persistence failure) end as
failed turns whose chat copy names the slash command to run, and whose ``failure_reason``
lets the desktop card pick a code-specific action instead of a blind Retry.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_surface import build_error_surface_from_result
from agent.turn_explainers import EMPTY_RESPONSE_EXPLANATION, TurnExplainersMixin
from agent.turn_failure_copy import (
    SITE_FAILURE_CODES, exit_reason_failure, failure_cause_gloss, site_copy,
)
from agent.turn_overflow import _Recovery


def _recovery():
    agent = SimpleNamespace(
        model="gpt-5", log_prefix="", _flush_status_buffer=lambda: None, _vprint=lambda *a, **k: None,
        _persist_session=lambda *a, **k: None,
    )
    return _Recovery(
        agent=agent, api_messages=[], system_message=None, effective_task_id="t", api_call_count=2,
        max_compression_attempts=3, messages=[], active_system_prompt=None, conversation_history=None,
        approx_tokens=200_000, compression_attempts=3,
    )


def test_overflow_exhaustion_is_non_retryable_context_overflow_with_slash_commands():
    verdict = _recovery().count_attempt()
    result = verdict.result
    assert result["failed"] is True and result["compression_exhausted"] is True
    assert result["failure_reason"] == "context_overflow" and result["failure_retryable"] is False
    text = result["final_response"]
    assert "/new" in text and "/compress" in text and "gpt-5" in text
    for jargon in ("compression attempts", "Context length exceeded", "safe threshold"):
        assert jargon not in text
    assert build_error_surface_from_result(result)["code"] == "context_overflow"


def _context_rejection(request_tokens: int, window: int = 65_536, error="HTTP 500: Context size has been exceeded."):
    """Drive ``_recover_context_length`` with a provider "context exceeded" and a request the rough
    estimator prices at ``request_tokens`` against a ``window``-token model (no output cap)."""
    from unittest.mock import patch

    from agent.turn_overflow import _recover_context_length
    from agent.turn_retry_state import TurnRetryState

    st = _recovery()
    st.compression_attempts = 0
    st.agent.max_tokens = None
    st.agent.context_compressor = SimpleNamespace(context_length=window)
    st.agent.provider, st.agent.base_url, st.agent.tools = "lmstudio", "http://127.0.0.1:1234/v1", None
    st.agent._buffer_vprint = st.agent._buffer_diagnostic_status = lambda *a, **k: None
    compressed = []
    st.agent._compress_context = lambda msgs, *a, **k: (compressed.append(1) or [{"role": "user", "content": "x"}], None)
    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=request_tokens), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=[request_tokens, 10]), \
         patch("agent.turn_overflow.time.sleep", lambda _: None):
        verdict = _recover_context_length(st, TurnRetryState(), error)
    return verdict, compressed


def test_context_rejection_far_below_the_window_is_not_blamed_on_the_conversation():
    """A single-slot local server rejecting a ~3k-token request (another thread held its
    context) must not read as "this conversation has grown too long": no compression, transient
    + retryable, and the gateway's overflow verdict (drop message / auto-reset) stays off."""
    from gateway.run_turn import is_context_overflow_failure_result

    verdict, compressed = _context_rejection(3_000)
    result = verdict.result
    assert verdict.action == "return" and not compressed
    assert result["failed"] is True and not result.get("compression_exhausted")
    assert result["failure_reason"] == "server_error" and result["failure_retryable"] is True
    text = result["final_response"]
    assert "grown too long" not in text and "/new" not in text
    assert "3,000" in text and "65,536" in text and "background" in text and "/retry" in text
    assert is_context_overflow_failure_result(result, history_len=2) is False


def test_context_rejection_near_the_window_still_compresses():
    """Control: a request that plausibly overflows keeps the compress-and-retry path, and so does
    a small local estimate when the SERVER quoted its own count — its measurement wins."""
    verdict, compressed = _context_rejection(60_000)
    assert compressed and verdict.action == "break"
    verdict, compressed = _context_rejection(3_000, error="prompt is too long: 70000 tokens > 65536 maximum")
    assert compressed and verdict.action == "break"


def test_payload_and_context_overflow_share_one_next_step():
    """413 and context-length exhaustion differ in cause text but never in what to do."""
    a = _recovery().count_attempt(payload_too_large=True).result["final_response"]
    b = _recovery().count_attempt().result["final_response"]
    assert ("/new" in a) and ("/compress" in a) and ("/new" in b) and ("/compress" in b)


def test_empty_response_exhaustion_has_one_text_everywhere():
    """One constant feeds the CLI explainer and the gateway '(empty)' rewrite; no surface
    asserts 'after processing tool results' or 'inspect the tool output above'."""
    verdict = exit_reason_failure("empty_response_exhausted")
    assert (verdict.reason, verdict.retryable) == ("empty_response", True)
    text = TurnExplainersMixin._format_turn_completion_explanation("empty_response_exhausted", model="llama3")
    assert text.startswith("⚠️ No reply: ") and "llama3" in text
    assert "/model" in text and "continue" in text
    assert "tool" not in text
    assert EMPTY_RESPONSE_EXPLANATION.format(model="llama3") in text


def test_persistence_failure_default_copy_is_actionable_and_profile_aware(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/srv/hermes-profile")
    text = TurnExplainersMixin._format_turn_completion_explanation("session_persistence_failed", "replaced")
    assert "hermes gateway stop" in text and "hermes doctor" in text
    assert "~/.hermes" not in text and "/srv/hermes-profile" in text
    assert "manifest" not in text  # the runbook stays in logger.error at hermes_state


def test_reasoning_only_copy_gives_the_fix_before_the_scratchpad():
    text = site_copy("reasoning_only", model="r1", preview="the answer is 42")
    assert text.index("/reasoning low") < text.index("the answer is 42")
    assert "/model" in text


@pytest.mark.parametrize("exit_reason", ["empty_response_exhausted", "local_processing_error(TypeError)"])
def test_advisory_exit_reasons_stamp_a_code_but_never_flip_failed(exit_reason):
    """Cron silence, the kanban dispatcher breaker and gateway transcript persistence all key
    on ``failed``; these two exits must keep failed=False and only gain the descriptor code."""
    verdict = exit_reason_failure(exit_reason)
    assert verdict is not None and verdict.fails_turn is False
    assert verdict.reason in SITE_FAILURE_CODES


@pytest.mark.parametrize("exit_reason, code", [
    ("context_compression_timeout", "context_overflow"),
    ("ollama_runtime_context_too_small", "context_overflow"),
    ("redirect_restart_limit_exceeded", "loop_error"),
    ("rebuilt_restart_limit_exceeded", "loop_error"),
])
def test_previously_bare_exit_reasons_now_carry_a_code(exit_reason, code):
    verdict = exit_reason_failure(exit_reason)
    assert verdict is not None and verdict.reason == code
    assert build_error_surface_from_result(
        {"failed": True, "error": "x", "failure_reason": verdict.reason, "failure_retryable": verdict.retryable}
    )["code"] == code


def test_every_failure_code_copy_key_is_a_failure_code():
    """The copy table split: keys of the failure-code table are exactly codes the descriptor
    contract lists (one-off strings live in their own table)."""
    from agent.turn_failure_copy import _FAILURE_CODE_COPY, _ONE_OFF_COPY

    assert set(_FAILURE_CODE_COPY) <= SITE_FAILURE_CODES
    assert not (set(_ONE_OFF_COPY) & SITE_FAILURE_CODES)
    assert "loop_error" in _FAILURE_CODE_COPY and "local_processing_error" in _ONE_OFF_COPY


def test_cause_gloss_substitutes_the_subject_and_skips_unknown_reasons():
    assert "the job's" in failure_cause_gloss("context_overflow", subject="this job", possessive="the job's")
    assert failure_cause_gloss("model_not_found").startswith("the model it uses")
    assert failure_cause_gloss("unknown") is None and failure_cause_gloss(None) is None
