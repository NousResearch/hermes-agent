"""Tests for typed repeated-provider-stall failures."""

from __future__ import annotations

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.provider_health_probe import ProbeOutcome
from agent.provider_stall import ProviderStalledError


@pytest.mark.parametrize(
    ("probe_status", "http_status"),
    [
        ("reachable", 404),
        ("unreachable", None),
        ("unavailable", None),
        ("disabled", None),
    ],
)
def test_provider_stalled_error_carries_probe_without_leaking_probe_detail(
    probe_status: str,
    http_status: int | None,
):
    secret_url = "https://user:password@example.test/v1?api_key=secret-token"
    secret_prompt = "private prompt contents"
    probe = ProbeOutcome(
        status=probe_status,
        http_status=http_status,
        detail=f"{secret_url}; prompt={secret_prompt}",
    )

    error = ProviderStalledError(
        provider="xiaomi",
        model="mimo-v2.5-pro",
        silent_seconds=360.2,
        attempt=2,
        probe=probe,
    )

    assert error.provider == "xiaomi"
    assert error.model == "mimo-v2.5-pro"
    assert error.silent_seconds == 360.2
    assert error.attempt == 2
    assert error.probe is probe
    assert f"probe={probe_status}" in str(error)
    assert secret_url not in str(error)
    assert "user:password" not in str(error)
    assert "api_key" not in str(error)
    assert "secret-token" not in str(error)
    assert secret_prompt not in str(error)


@pytest.mark.parametrize(
    ("probe_status", "http_status"),
    [
        ("reachable", 404),
        ("unreachable", None),
        ("unavailable", None),
        ("disabled", None),
    ],
)
def test_repeated_provider_stall_is_nonretryable_and_fallbackable(
    probe_status: str,
    http_status: int | None,
):
    error = ProviderStalledError(
        provider="xiaomi",
        model="mimo-v2.5-pro",
        silent_seconds=360.2,
        attempt=2,
        probe=ProbeOutcome(
            status=probe_status,
            http_status=http_status,
            detail="diagnostic detail must not enter classifier output",
        ),
    )

    classified = classify_api_error(
        error,
        provider="xiaomi",
        model="mimo-v2.5-pro",
    )

    assert classified.reason is FailoverReason.provider_stalled
    assert classified.retryable is False
    assert classified.should_fallback is True
    assert classified.error_context == {
        "silent_seconds": 360.2,
        "attempt": 2,
        "probe_status": probe_status,
        "probe_http_status": http_status,
    }
