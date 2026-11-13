"""A Nous 403 ``data_policy_denied`` is deterministic for this context: no retry, no rotation, no fallback."""

from agent.error_classifier import FailoverReason, classify_api_error


class _ForbiddenError(Exception):
    def __init__(self, body):
        super().__init__("Forbidden")
        self.status_code = 403
        self.body = body


def test_403_data_policy_denied_never_retries_or_falls_back():
    body = {"error": {"code": "data_policy_denied", "type": "data_policy_denied", "message": "No endpoint qualifies.",
                      "requirement": ["no_training"], "sources": ["request"], "suggested_action": "switch_model"}}
    result = classify_api_error(_ForbiddenError(body), provider="nous")
    assert result.reason == FailoverReason.data_policy_denied
    assert (result.retryable, result.should_fallback, result.should_rotate_credential) == (False, False, False)


def test_data_policy_copy_gives_free_tier_users_a_way_out():
    from types import SimpleNamespace

    from agent.turn_failure_copy import nonretryable_copy

    copy = nonretryable_copy(SimpleNamespace(reason=FailoverReason.data_policy_denied), provider="nous",
                             model="some/model", summary="No endpoint qualifies.")
    free_tier = copy[copy.index("free tier"):]
    assert "'some/model'" in copy and "upgrade" in free_tier and "/compress" in free_tier and "/new" in free_tier


def test_limit_reset_line_only_explains_a_limit():
    import time
    from types import SimpleNamespace

    from agent.turn_recovery import _stamp_limit_reset

    agent = SimpleNamespace(_extract_api_error_context=lambda _e: {"reset_at": time.time() + 600})
    refused = {"final_response": "refused"}
    _stamp_limit_reset(refused, agent, Exception(), SimpleNamespace(reason=FailoverReason.data_policy_denied))
    limited = {"final_response": "slow down"}
    _stamp_limit_reset(limited, agent, Exception(), SimpleNamespace(reason=FailoverReason.rate_limit))
    assert refused == {"final_response": "refused"}
    assert "failure_resets_at" in limited and "Limit resets at" in limited["final_response"]
