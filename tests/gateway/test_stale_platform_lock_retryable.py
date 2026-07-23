"""Regression tests for platform-lock acquire behavior.

#54167 — stale platform lock must be retryable.
When a gateway process is killed (SIGKILL, crash) during Telegram
initialization, the scoped lock file survives. On next startup,
``acquire_scoped_lock()`` detects the stale lock and deletes it, but may
still return ``(False, existing_dict)`` to the caller (e.g. if the
unlink fails due to permissions, or a race condition lets another
process grab the lock first).

``_acquire_platform_lock()`` must mark such failures as **retryable**
so the reconnect watcher can retry after a delay — not permanently kill
the platform.

#65176 — a live gateway token conflict may attempt one-shot takeover only
during the initial connect of an explicit ``gateway run --replace`` startup.
``gateway run --replace`` only kills same-HERMES_HOME PID-file holders.
A normal start or reconnect must retain the retryable conflict behavior and
must never evict the active holder.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner


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
    obj._platform_lock_takeover_allowed = False
    obj._platform_lock_takeover_attempted = False
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


def test_explicit_replace_takeover_reacquires_lock_once(adapter):
    """Initial explicit --replace may hand off and re-acquire once (#65176)."""
    existing = {
        "pid": 4242,
        "kind": "hermes-gateway",
        "argv": ["hermes", "gateway", "run"],
        "start_time": 123,
    }
    acquire = MagicMock(side_effect=[(False, existing), (True, None)])
    adapter._platform_lock_takeover_allowed = True

    with patch("gateway.status.acquire_scoped_lock", acquire), patch(
        "gateway.status.take_over_scoped_lock_holder",
        return_value=4242,
    ) as takeover, patch.object(
        adapter, "_write_runtime_status_safe"
    ):
        result = adapter._acquire_platform_lock(
            "telegram-bot-token", "test-token", "Telegram bot token"
        )

    assert result is True
    assert adapter._platform_lock_takeover_allowed is False
    assert adapter._platform_lock_takeover_attempted is True
    takeover.assert_called_once_with(existing)
    assert acquire.call_count == 2


def test_normal_connect_conflict_never_attempts_takeover(adapter):
    """A normal start/reconnect cannot evict the current token holder."""
    existing = {
        "pid": 5555,
        "kind": "hermes-gateway",
        "argv": ["hermes", "gateway", "run"],
        "start_time": 123,
    }
    with patch(
        "gateway.status.acquire_scoped_lock",
        return_value=(False, existing),
    ), patch(
        "gateway.status.take_over_scoped_lock_holder",
    ) as takeover, patch.object(
        adapter, "_write_runtime_status_safe"
    ):
        result = adapter._acquire_platform_lock(
            "telegram-bot-token", "test-token", "Telegram bot token"
        )

    assert result is False
    takeover.assert_not_called()
    assert adapter._platform_lock_takeover_attempted is False
    assert adapter._fatal_error_retryable is True


def test_failed_explicit_takeover_consumes_authority(adapter):
    """A failed handoff is not retried by a later acquire on the same adapter."""
    existing = {
        "pid": 7777,
        "kind": "hermes-gateway",
        "argv": ["hermes", "gateway", "run"],
        "start_time": 456,
    }
    adapter._platform_lock_takeover_allowed = True

    with patch(
        "gateway.status.acquire_scoped_lock",
        return_value=(False, existing),
    ), patch(
        "gateway.status.take_over_scoped_lock_holder",
        return_value=None,
    ) as takeover, patch.object(
        adapter, "_write_runtime_status_safe"
    ):
        first = adapter._acquire_platform_lock(
            "telegram-bot-token", "test-token", "Telegram bot token"
        )
        second = adapter._acquire_platform_lock(
            "telegram-bot-token", "test-token", "Telegram bot token"
        )

    assert first is False
    assert second is False
    assert adapter._platform_lock_takeover_allowed is False
    assert adapter._platform_lock_takeover_attempted is True
    takeover.assert_called_once_with(existing)


@pytest.mark.asyncio
async def test_runner_scopes_replace_intent_to_initial_connect():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._platform_lock_takeover_on_start = True
    adapter = MagicMock()
    adapter._platform_lock_takeover_allowed = False
    seen = []

    async def connect(current_adapter, _platform):
        seen.append(current_adapter._platform_lock_takeover_allowed)
        return True

    runner._connect_adapter_with_timeout = connect

    assert await runner._connect_initial_adapter_with_timeout(
        adapter, MagicMock(value="telegram")
    ) is True
    assert seen == [True]
    assert adapter._platform_lock_takeover_allowed is False


@pytest.mark.asyncio
async def test_runner_clears_replace_intent_when_initial_connect_raises():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._platform_lock_takeover_on_start = True
    adapter = MagicMock()
    adapter._platform_lock_takeover_allowed = False

    async def connect(_adapter, _platform):
        raise RuntimeError("connect failed")

    runner._connect_adapter_with_timeout = connect

    with pytest.raises(RuntimeError, match="connect failed"):
        await runner._connect_initial_adapter_with_timeout(
            adapter, MagicMock(value="telegram")
        )
    assert adapter._platform_lock_takeover_allowed is False
