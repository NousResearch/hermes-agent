"""Tests for the main-agent bounded provider/model fallback gate."""

from agent.conversation_loop import _should_activate_bounded_fallback
from agent.error_classifier import ClassifiedError, FailoverReason


def test_classifier_fallback_contract_triggers_immediate_gate_for_dead_model():
    classified = ClassifiedError(
        reason=FailoverReason.model_not_found,
        retryable=False,
        should_fallback=True,
    )

    assert _should_activate_bounded_fallback(classified, retry_count=0) is True


def test_classifier_fallback_contract_triggers_immediate_gate_for_quota():
    classified = ClassifiedError(
        reason=FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )

    assert _should_activate_bounded_fallback(classified, retry_count=0) is True


def test_transport_failures_get_retry_window_before_fallback():
    classified = ClassifiedError(
        reason=FailoverReason.timeout,
        retryable=True,
        should_fallback=False,
    )

    assert _should_activate_bounded_fallback(classified, retry_count=0) is False
    assert _should_activate_bounded_fallback(classified, retry_count=1) is False
    assert _should_activate_bounded_fallback(classified, retry_count=2) is True


def test_compression_recovery_does_not_enter_provider_fallback_gate():
    classified = ClassifiedError(
        reason=FailoverReason.context_overflow,
        retryable=True,
        should_compress=True,
        should_fallback=True,
    )

    assert _should_activate_bounded_fallback(classified, retry_count=0) is False
