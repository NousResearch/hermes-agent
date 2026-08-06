"""Regression test for #54167 — stale platform lock must be retryable.

When a gateway process is killed (SIGKILL, crash) during Telegram
initialization, the scoped lock file survives. On next startup,
``acquire_scoped_lock()`` detects the stale lock and deletes it, but may
still return ``(False, existing_dict)`` to the caller (e.g. if the
unlink fails due to permissions, or a race condition lets another
process grab the lock first).

``_acquire_platform_lock()`` must mark such failures as **retryable**
so the reconnect watcher can retry after a delay — not permanently kill
the platform.

Contract asserted here
----------------------
``_set_fatal_error`` is called with ``retryable=True`` when lock
acquisition fails, regardless of the reason.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.base import BasePlatformAdapter


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete subclass for testing _acquire_platform_lock."""

    platform = MagicMock(value="telegram")

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


def test_stale_lock_failure_is_retryable(adapter):
    """Lock failure must be retryable, not permanently fatal (#54167)."""
    with patch(
        "gateway.status.acquire_scoped_lock",
        return_value=(False, {"pid": 99999, "start_time": "2026-01-01T00:00:00Z"}),
    ), patch.object(adapter, "_write_runtime_status_safe"):
        result = adapter._acquire_platform_lock(
            "telegram-bot-token", "test-token", "Telegram bot token"
        )

    assert result is False
    assert adapter._fatal_error_retryable is True
    assert adapter._fatal_error_code == "telegram-bot-token_lock"
