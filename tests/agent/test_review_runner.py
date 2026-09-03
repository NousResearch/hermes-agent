"""Behavior contract for the proposed bounded review runner.

The runner is deliberately transport-agnostic.  A trusted backend resolver
owns credentials and returns only an opaque handle plus sanitized provenance;
the runner enforces route and payload invariants around exactly one no-tools
completion call.
"""

from __future__ import annotations

from dataclasses import replace

from agent.review_runner import (
    ReviewRequest,
    ResolvedReviewRoute,
    run_review,
)


def _request(**overrides):
    request = ReviewRequest(
        checkpoint_id="cp-1",
        session_id="session-1",
        phase="plan",
        objective="Ship the reviewed change",
        constraints=("Keep the core cache-safe",),
        candidate={"summary": "Run the focused tests, then edit one file"},
        provider="openai-codex",
        model="gpt-review",
        main_provider="openai-codex",
        main_model="gpt-economy",
    )
    return replace(request, **overrides)


def _route(**overrides):
    route = ResolvedReviewRoute(
        profile="default",
        provider="openai-codex",
        model="gpt-review",
        credential_kind="subscription_oauth",
        credential_handle=object(),
    )
    return replace(route, **overrides)


def _complete_ok(**payload_overrides):
    payload = {
        "verdict": "PASS",
        "summary": "Plan is bounded and testable.",
        "feedback": [],
        "usage": {"input_tokens": 24, "output_tokens": 9},
    }
    payload.update(payload_overrides)

    def complete(**kwargs):
        complete.calls.append(kwargs)
        return payload

    complete.calls = []
    return complete


def test_subscription_route_runs_once_without_tools_and_attests_actual_route():
    complete = _complete_ok()
    resolve_calls = []

    def resolve(**kwargs):
        resolve_calls.append(kwargs)
        return _route()

    result = run_review(_request(), resolve_route=resolve, complete=complete)

    assert result.status == "completed"
    assert result.verdict == "PASS"
    assert result.actual_route == {
        "profile": "default",
        "provider": "openai-codex",
        "model": "gpt-review",
        "credential_kind": "subscription_oauth",
    }
    assert resolve_calls == [{
        "provider": "openai-codex",
        "model": "gpt-review",
        "credential_policy": "subscription_oauth_only",
        "fallback_policy": "none",
    }]
    assert len(complete.calls) == 1
    assert complete.calls[0]["tools"] == []
    assert complete.calls[0]["tool_choice"] == "none"
    assert complete.calls[0]["idempotency_key"] == "cp-1"
    system_prompt = complete.calls[0]["messages"][0]["content"]
    assert "Return only one JSON object" in system_prompt
    assert '"verdict":"PASS|REVISE|ASK_USER|BLOCK"' in system_prompt


def test_api_key_route_is_rejected_before_any_model_call():
    complete = _complete_ok()

    result = run_review(
        _request(),
        resolve_route=lambda **_: _route(credential_kind="api_key"),
        complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "credential_policy_mismatch"
    assert complete.calls == []


def test_requested_and_actual_provider_model_must_match_exactly():
    complete = _complete_ok()

    result = run_review(
        _request(),
        resolve_route=lambda **_: _route(model="gpt-fallback"),
        complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "route_mismatch"
    assert complete.calls == []


def test_same_actual_route_as_main_is_rejected_when_distinct_is_required():
    complete = _complete_ok()

    result = run_review(
        _request(main_model="gpt-review"),
        resolve_route=lambda **_: _route(),
        complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "review_route_matches_main"
    assert complete.calls == []


def test_route_resolution_failure_is_truthful_and_does_not_fallback():
    complete = _complete_ok()

    def unavailable(**_):
        raise RuntimeError("subscription quota exhausted")

    result = run_review(
        _request(), resolve_route=unavailable, complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "route_unavailable"
    assert "quota exhausted" in result.summary
    assert complete.calls == []


def test_invalid_verdict_is_rejected_instead_of_treated_as_pass():
    complete = _complete_ok(verdict="ALLOW")

    result = run_review(
        _request(), resolve_route=lambda **_: _route(), complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "invalid_response"
    assert result.verdict is None


def test_completion_failure_does_not_retry_or_change_route():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("review timed out")

    result = run_review(
        _request(), resolve_route=lambda **_: _route(), complete=complete,
    )

    assert result.status == "timed_out"
    assert result.unavailable_reason == "timeout"
    assert len(calls) == 1


def test_packet_redacts_secret_shaped_fields_before_completion():
    complete = _complete_ok()
    request = _request(candidate={
        "summary": "Call the provider",
        "api_key": "secret-api-key",
        "nested": {"authorization": "Bearer secret-token"},
        "safe": "visible",
    })

    result = run_review(
        request, resolve_route=lambda **_: _route(), complete=complete,
    )

    assert result.status == "completed"
    serialized_messages = str(complete.calls[0]["messages"])
    assert "secret-api-key" not in serialized_messages
    assert "secret-token" not in serialized_messages
    assert "[REDACTED]" in serialized_messages
    assert "visible" in serialized_messages


def test_oversized_packet_is_rejected_before_resolution():
    complete = _complete_ok()
    resolve_calls = []

    def resolve(**kwargs):
        resolve_calls.append(kwargs)
        return _route()

    result = run_review(
        _request(candidate={"summary": "x" * 70_000}),
        resolve_route=resolve,
        complete=complete,
    )

    assert result.status == "unavailable"
    assert result.unavailable_reason == "packet_too_large"
    assert resolve_calls == []
    assert complete.calls == []


def test_request_requires_exact_route_and_supported_policies():
    complete = _complete_ok()

    missing_model = run_review(
        _request(model=""),
        resolve_route=lambda **_: _route(),
        complete=complete,
    )
    wrong_policy = run_review(
        _request(credential_policy="any"),
        resolve_route=lambda **_: _route(),
        complete=complete,
    )

    assert missing_model.unavailable_reason == "invalid_request"
    assert wrong_policy.unavailable_reason == "unsupported_policy"
    assert complete.calls == []
