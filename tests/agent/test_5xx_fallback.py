"""#68771: retryable 5xx joins the fallback chain after one primary retry.

The gate lives in ``route_classified_error``. ``retry_count`` is already
incremented when the gate runs, so ``>= 2`` is one completed primary retry.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.response = None


def _route(status_code: int, message: str, *, retry_count: int, provider: str = "openrouter"):
    err = _Err(status_code, message)
    classified = classify_api_error(err, provider=provider)
    activated = {}

    def try_fb(reason=None, reset_at=None):
        activated["reason"] = reason
        activated["reset_at"] = reset_at
        return True

    agent = MagicMock()
    agent.model = "primary-model"
    agent.provider = provider
    agent.log_prefix = ""
    agent.compression_enabled = True
    agent.tools = None
    agent._fallback_chain = [{"provider": "secondary", "model": "fb-model"}]
    agent._fallback_index = 0
    agent._credential_pool = None
    agent._try_activate_fallback = try_fb
    agent._cached_system_prompt = None

    verdict = route_classified_error(
        agent,
        err,
        classified,
        TurnRetryState(),
        error_msg=message,
        error_context={},
        recovered_with_pool=False,
        base_url="https://example.invalid/v1",
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        api_messages=[{"role": "user", "content": "hi"}],
        system_message=None,
        active_system_prompt="",
        conversation_history=[],
        retry_count=retry_count,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=2,
        api_call_count=1,
        effective_task_id=None,
    )
    return classified, verdict, activated


def test_502_is_server_error_and_waits_for_one_primary_retry():
    classified, verdict, activated = _route(502, "Bad Gateway", retry_count=1)
    assert classified.reason == FailoverReason.server_error
    assert verdict.action == "fallthrough"
    assert activated == {}


def test_502_falls_back_after_one_primary_retry():
    classified, verdict, activated = _route(502, "Bad Gateway", retry_count=2)
    assert classified.reason == FailoverReason.server_error
    assert verdict.action == "break"
    assert activated["reason"] == FailoverReason.server_error
    assert verdict.retry_count == 0


def test_500_falls_back_on_the_same_gate():
    classified, verdict, activated = _route(500, "Internal Server Error", retry_count=2, provider="anthropic")
    assert classified.reason == FailoverReason.server_error
    assert verdict.action == "break"
    assert activated["reason"] == FailoverReason.server_error


def test_503_overloaded_still_uses_the_transport_gate():
    classified, first, idle = _route(503, "Service Unavailable", retry_count=1, provider="nous")
    assert classified.reason == FailoverReason.overloaded
    assert first.action == "fallthrough"
    assert idle == {}
    _, second, activated = _route(503, "Service Unavailable", retry_count=2, provider="nous")
    assert second.action == "break"
    assert activated["reason"] == FailoverReason.overloaded


def test_server_error_status_is_not_a_rate_limit_line():
    from agent.turn_recovery import _eager_fallback_status

    classified = SimpleNamespace(reason=FailoverReason.server_error, error_context={}, billing_unverified=False)
    text = _eager_fallback_status(classified, False, False)
    assert "server error" in text
    assert "Rate limited" not in text
