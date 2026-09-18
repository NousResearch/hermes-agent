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
