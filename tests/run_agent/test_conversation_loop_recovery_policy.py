from agent.conversation_loop import _is_client_error_from_policy, _should_try_eager_fallback
from agent.error_classifier import ClassifiedError, FailoverReason
from agent.failure_policy import RecoveryAction, decide_api_recovery


def test_rate_limit_uses_policy_for_eager_fallback() -> None:
    classified = ClassifiedError(
        reason=FailoverReason.rate_limit,
        retryable=True,
        should_fallback=True,
    )

    decision = decide_api_recovery(classified)

    assert decision.primary_action == RecoveryAction.fallback_provider
    assert _should_try_eager_fallback(decision, reason=classified.reason) is True


def test_billing_uses_secondary_fallback_action_for_eager_fallback() -> None:
    classified = ClassifiedError(
        reason=FailoverReason.billing,
        retryable=False,
        should_fallback=True,
        should_rotate_credential=True,
    )

    decision = decide_api_recovery(classified)

    assert decision.primary_action == RecoveryAction.rotate_credential
    assert RecoveryAction.fallback_provider in decision.secondary_actions
    assert _should_try_eager_fallback(decision, reason=classified.reason) is True


def test_context_overflow_stays_on_recovery_path_not_client_abort() -> None:
    classified = ClassifiedError(
        reason=FailoverReason.context_overflow,
        retryable=True,
        should_compress=True,
    )

    decision = decide_api_recovery(classified)

    assert _is_client_error_from_policy(classified, decision) is False


def test_non_retryable_auth_is_client_error_after_recovery_paths_exhaust() -> None:
    classified = ClassifiedError(
        reason=FailoverReason.auth,
        retryable=False,
        should_fallback=True,
        should_rotate_credential=True,
    )

    decision = decide_api_recovery(classified)

    assert _is_client_error_from_policy(classified, decision) is True


def test_local_validation_error_forces_client_error() -> None:
    classified = ClassifiedError(reason=FailoverReason.unknown, retryable=True)
    decision = decide_api_recovery(classified)

    assert _is_client_error_from_policy(
        classified,
        decision,
        is_local_validation_error=True,
    ) is True
