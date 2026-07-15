"""Test for #65176 — platform lock conflict should auto-terminate live gateway holder.

When a new gateway detects a platform lock conflict (e.g. bot token already in use)
and the holder is a live gateway process, it should terminate the holder and retry the lock
acquisition instead of retrying indefinitely with a fatal error.

This fix applies to all platform adapters (Telegram, Discord, WhatsApp, Weixin, Signal, etc.)
since it's implemented in BasePlatformAdapter._acquire_platform_lock().
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete subclass for testing _acquire_platform_lock."""

    platform = MagicMock(value="test-platform")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}


@pytest.fixture()
def adapter():
    """Create a stub adapter with __init__ bypassed."""
    obj = _StubAdapter.__new__(_StubAdapter)
    obj._running = True
    obj._fatal_error_code = None
    obj._fatal_error_message = None
    obj._fatal_error_retryable = True
    obj._fatal_error_handler = None
    obj._platform_lock_scope = None
    obj._platform_lock_identity = None
    obj._status_write_logged = None
    return obj


def test_platform_lock_conflict_with_live_gateway_terminates_holder(adapter):
    """When lock conflict holder is a live gateway, terminate it and retry (#65176)."""
    call_count = {"n": 0}
    holder_pid = 12345
    test_scope = "test-platform-token"
    test_identity = "test-identity"
    test_desc = "Test platform token"

    def mock_acquire_lock(scope, identity, metadata=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: conflict with live gateway
            return (False, {"pid": holder_pid, "start_time": "2026-01-01T00:00:00Z"})
        else:
            # Second call (after termination): success
            return (True, None)

    with patch(
        "gateway.status.acquire_scoped_lock",
        side_effect=mock_acquire_lock,
    ), patch(
        "gateway.status._looks_like_gateway_process",
        return_value=True,  # Holder is a live gateway
    ), patch(
        "gateway.status.terminate_pid",
    ) as mock_terminate, patch(
        "gateway.status.write_takeover_marker",
    ), patch(
        "gateway.status._pid_exists",
        return_value=False,  # Process already exited
    ), patch.object(adapter, "_write_runtime_status_safe"):
        result = adapter._acquire_platform_lock(
            test_scope, test_identity, test_desc
        )

    # Should have terminated the holder
    mock_terminate.assert_called_once_with(holder_pid, force=True)
    # Should have succeeded after retry
    assert result is True
    assert adapter._fatal_error_code is None, "Should not set fatal error on successful retry"
    assert call_count["n"] == 2, "Should have called acquire_scoped_lock twice (initial + retry)"


def test_platform_lock_conflict_with_non_gateway_does_not_terminate(adapter):
    """When lock conflict holder is NOT a gateway, do NOT terminate it."""
    holder_pid = 12345
    test_scope = "test-platform-token"
    test_identity = "test-identity"
    test_desc = "Test platform token"

    with patch(
        "gateway.status.acquire_scoped_lock",
        return_value=(False, {"pid": holder_pid, "start_time": "2026-01-01T00:00:00Z"}),
    ), patch(
        "gateway.status._looks_like_gateway_process",
        return_value=False,  # Holder is NOT a gateway
    ), patch(
        "gateway.status.terminate_pid",
    ) as mock_terminate, patch.object(adapter, "_write_runtime_status_safe"):
        result = adapter._acquire_platform_lock(
            test_scope, test_identity, test_desc
        )

    # Should NOT have terminated the holder
    mock_terminate.assert_not_called()
    # Should fail with fatal error
    assert result is False
    assert adapter._fatal_error_code == f"{test_scope}_lock"
    assert adapter._fatal_error_retryable is True


def test_platform_lock_conflict_terminate_fails_still_sets_fatal_error(adapter):
    """If termination fails, still set fatal error as fallback."""
    holder_pid = 12345
    test_scope = "test-platform-token"
    test_identity = "test-identity"
    test_desc = "Test platform token"

    with patch(
        "gateway.status.acquire_scoped_lock",
        return_value=(False, {"pid": holder_pid, "start_time": "2026-01-01T00:00:00Z"}),
    ), patch(
        "gateway.status._looks_like_gateway_process",
        return_value=True,
    ), patch(
        "gateway.status.terminate_pid",
        side_effect=Exception("Termination failed"),
    ), patch(
        "gateway.status.write_takeover_marker",
    ), patch.object(adapter, "_write_runtime_status_safe"):
        result = adapter._acquire_platform_lock(
            test_scope, test_identity, test_desc
        )

    # Should fail with fatal error when termination fails
    assert result is False
    assert adapter._fatal_error_code == f"{test_scope}_lock"
    assert adapter._fatal_error_retryable is True
