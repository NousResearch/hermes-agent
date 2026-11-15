"""Unit tests for #CF-82: Vertex AI Dynamic Shared Quota (DSQ) 429 retry before fallback.

Verifies that when `agent.provider == "vertex"`, transient 429 rate-limit bursts
retry in-place with backoff on Attempts 1 and 2 (`retry_count < max(2, max_retries)`)
before escalating to the emergency OpenRouter fallback on the final retry attempt.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_recovery import TurnRetryState, route_classified_error


def _make_fake_agent(*, provider: str = "vertex", has_fallback: bool = True):
    activated = []
    statuses = []

    def _try_activate_fallback(reason=None, reset_at=None):
        activated.append((reason, reset_at))
        return True

    return SimpleNamespace(
        provider=provider,
        model="google/gemini-4-argon",
        tools=[],
        compression_enabled=True,
        _fallback_index=0,
        _fallback_chain=[{"provider": "openrouter", "model": "google/gemini-4-argon"}] if has_fallback else [],
        _credential_pool=None,
        _fallback_activated=False,
        _content_filter_fallback_sticky=False,
        _rebuilt_restarts_since_response=0,
        _rate_limited_until=0.0,
        _buffer_diagnostic_status=lambda msg: statuses.append(msg),
        _try_activate_fallback=_try_activate_fallback,
        _activated=activated,
        _statuses=statuses,
    )


def _rate_limit_classified(reason: FailoverReason = FailoverReason.rate_limit) -> ClassifiedError:
    return ClassifiedError(
        reason=reason,
        status_code=429,
        retryable=True,
        should_compress=False,
        should_rotate_credential=False,
        should_fallback=True,
        message="429 RESOURCE_EXHAUSTED: TPM quota exhausted",
    )


def _call_route(agent, classified, retry_state, *, retry_count: int, max_retries: int = 3):
    return route_classified_error(
        agent,
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        classified,
        retry_state,
        error_msg=classified.message,
        error_context={},
        recovered_with_pool=False,
        base_url="https://aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/global/endpoints/openapi",
        model="google/gemini-4-argon",
        messages=[{"role": "user", "content": "hi"}],
        api_messages=[{"role": "user", "content": "hi"}],
        system_message="sys",
        active_system_prompt="sys",
        conversation_history=[],
        retry_count=retry_count,
        max_retries=max_retries,
        compression_attempts=0,
        max_compression_attempts=2,
        api_call_count=1,
        effective_task_id="t1",
    )


def test_vertex_dsq_429_attempt_1_retries_in_place_without_fallback():
    agent = _make_fake_agent(provider="vertex")
    retry = TurnRetryState()
    verdict = _call_route(agent, _rate_limit_classified(), retry, retry_count=1, max_retries=3)
    assert verdict.action == "fallthrough"
    assert agent._activated == []


def test_vertex_dsq_429_attempt_2_retries_in_place_without_fallback():
    agent = _make_fake_agent(provider="vertex")
    retry = TurnRetryState()
    verdict = _call_route(agent, _rate_limit_classified(), retry, retry_count=2, max_retries=3)
    assert verdict.action == "fallthrough"
    assert agent._activated == []


def test_vertex_dsq_429_attempt_3_escalates_to_emergency_fallback():
    agent = _make_fake_agent(provider="vertex")
    retry = TurnRetryState()
    verdict = _call_route(agent, _rate_limit_classified(), retry, retry_count=3, max_retries=3)
    assert verdict.action == "break"
    assert verdict.retry_count == 0
    assert len(agent._activated) == 1
    assert agent._activated[0][0] == FailoverReason.rate_limit


def test_non_vertex_provider_without_pool_still_falls_back_eagerly_on_attempt_1():
    agent = _make_fake_agent(provider="openrouter")
    retry = TurnRetryState()
    verdict = _call_route(
        agent,
        _rate_limit_classified(FailoverReason.upstream_rate_limit),
        retry,
        retry_count=1,
        max_retries=3,
    )
    assert verdict.action == "break"
    assert verdict.retry_count == 0
    assert len(agent._activated) == 1


def test_vertex_auth_failure_escalates_immediately_on_attempt_1():
    agent = _make_fake_agent(provider="vertex")
    retry = TurnRetryState()
    auth_classified = ClassifiedError(
        reason=FailoverReason.auth,
        status_code=401,
        retryable=False,
        should_compress=False,
        should_rotate_credential=True,
        should_fallback=True,
        message="401 Unauthorized",
    )
    verdict = _call_route(agent, auth_classified, retry, retry_count=1, max_retries=3)
    assert verdict.action == "break"
    assert retry.auth_failover_attempted is True
    assert len(agent._activated) == 1
