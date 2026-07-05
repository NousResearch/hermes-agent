"""Tests for ProviderHealthMonitor mapping of Phase 4c error codes."""

from __future__ import annotations

from agent.provider_adapter import ProviderHealth
from agent.provider_errors import ProviderErrorCode
from agent.provider_health_monitor import (
    ProviderHealthMonitor,
    ProviderHealthStatus,
)
from agent.providers import registry as provider_registry
from agent.providers.fake import FakeConfig, FakeProviderAdapter


def _check_with_last_error(last_error: str, *, available: bool = False, failures: int = 3) -> ProviderHealthStatus:
    """Inject a health snapshot for 'minimax' with given last_error and ask the monitor.

    Returns the resulting ProviderHealthSnapshot.status.
    """
    health = ProviderHealth(
        is_available=available,
        consecutive_failures=failures,
        last_error=last_error,
        last_checked_at_utc="2026-06-27T00:00:00Z",
    )
    config = FakeConfig(configured_health=health)
    original_get = provider_registry.get_provider

    def patched_get(name):
        if name == "minimax":
            return FakeProviderAdapter(provider_name="minimax", config=config)
        return original_get(name)

    provider_registry.get_provider = patched_get
    try:
        monitor = ProviderHealthMonitor()
        return monitor.check("minimax").status
    finally:
        provider_registry.get_provider = original_get


# ---------------------------------------------------------------------------
# Mapping coverage
# ---------------------------------------------------------------------------


class TestMonitorErrorCodeMapping:
    def test_auth_error_to_auth_blocked(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.AUTH_ERROR.value)
            is ProviderHealthStatus.AUTH_BLOCKED
        )

    def test_rate_limit_exceeded_to_rate_limited(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.RATE_LIMIT_EXCEEDED.value)
            is ProviderHealthStatus.RATE_LIMITED
        )

    def test_quota_exceeded_to_quota_blocked(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.QUOTA_EXCEEDED.value)
            is ProviderHealthStatus.QUOTA_BLOCKED
        )

    def test_context_length_exceeded_to_context_limit(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED.value)
            is ProviderHealthStatus.CONTEXT_LIMIT
        )

    def test_transient_error_to_transient(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.TRANSIENT_ERROR.value)
            is ProviderHealthStatus.TRANSIENT
        )

    def test_timeout_to_transient(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.TIMEOUT.value)
            is ProviderHealthStatus.TRANSIENT
        )

    def test_invalid_response_to_unknown(self) -> None:
        assert (
            _check_with_last_error(ProviderErrorCode.INVALID_RESPONSE.value)
            is ProviderHealthStatus.UNKNOWN
        )

    def test_missing_credentials_to_auth_blocked(self) -> None:
        # Policy preference: missing credentials map to AUTH_BLOCKED.
        assert (
            _check_with_last_error(ProviderErrorCode.MISSING_CREDENTIALS.value)
            is ProviderHealthStatus.AUTH_BLOCKED
        )


# ---------------------------------------------------------------------------
# Available + no errors → AVAILABLE
# ---------------------------------------------------------------------------


class TestAvailableBaseline:
    def test_no_failures_no_error_is_available(self) -> None:
        assert (
            _check_with_last_error("", available=True, failures=0)
            is ProviderHealthStatus.AVAILABLE
        )