"""Regression tests for #64512 — WeCom adapter _cleanup_ws() is exception-unsafe.

Background: ``_cleanup_ws()`` at plugins/platforms/wecom/adapter.py:272 calls
``await self._ws.close()`` and then ``self._ws = None``. When ``close()``
raises ``RuntimeError("Cannot write to closing transport")`` (a Python
asyncio behavior on a transport in "closing" state), the ``self._ws = None``
assignment is never reached. The broken transport persists, blocking all
future reconnection attempts.

The same defect applies to the ``_session.close()`` block.

Fix: wrap each close in try/except, always set the attribute to None
in a ``finally`` block.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    """Build a WeComAdapter with mocked transport/session."""
    from plugins.platforms.wecom.adapter import WeComAdapter
    from unittest.mock import MagicMock

    adapter = WeComAdapter.__new__(WeComAdapter)
    # 'name' is a @property that reads self.platform.value. Mock the platform.
    platform_mock = MagicMock()
    platform_mock.value = "wecom"
    adapter.platform = platform_mock
    # The actual fields under test
    adapter._ws = MagicMock()
    adapter._ws.closed = False
    adapter._ws.close = AsyncMock()  # default: succeeds
    adapter._session = MagicMock()
    adapter._session.closed = False
    adapter._session.close = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_cleanup_ws_sets_none_on_successful_close():
    """Sanity: when close() succeeds, self._ws must be set to None."""
    adapter = _make_adapter()

    await adapter._cleanup_ws()

    assert adapter._ws is None, "_ws must be None after successful close"
    assert adapter._session is None, "_session must be None after successful close"


@pytest.mark.asyncio
async def test_cleanup_ws_sets_none_when_ws_close_raises():
    """The bug: when self._ws.close() raises RuntimeError, the cleanup
    function must STILL set self._ws = None. Without the fix, the
    broken transport persists and blocks reconnection.
    """
    adapter = _make_adapter()
    adapter._ws.close = AsyncMock(
        side_effect=RuntimeError("Cannot write to closing transport")
    )

    # Before the fix, this raises RuntimeError AND leaves self._ws set
    await adapter._cleanup_ws()

    # After the fix, self._ws MUST be None even when close() raised
    assert adapter._ws is None, (
        "#64512 regression: self._ws must be reset to None even when "
        "close() raises RuntimeError, otherwise the broken transport "
        "blocks all future reconnection attempts."
    )


@pytest.mark.asyncio
async def test_cleanup_ws_sets_none_when_session_close_raises():
    """The same defect on _session.close(). After the fix, _session
    must be None even if close() raises.
    """
    adapter = _make_adapter()
    adapter._session.close = AsyncMock(
        side_effect=RuntimeError("Cannot write to closing transport")
    )

    await adapter._cleanup_ws()

    assert adapter._session is None, (
        "#64512 regression: self._session must be reset to None even when "
        "close() raises RuntimeError."
    )


@pytest.mark.asyncio
async def test_cleanup_ws_handles_already_closed_ws():
    """When self._ws is already closed (truthy but closed), the function
    should still set it to None without calling close() again. Pre-fix
    code only handled this if close() succeeded.
    """
    adapter = _make_adapter()
    adapter._ws.closed = True
    # Capture the close() mock's call_count BEFORE _cleanup_ws resets _ws to None
    pre_close_call_count = adapter._ws.close.call_count

    await adapter._cleanup_ws()

    # The closed-already path should NOT call close()
    assert pre_close_call_count == 0, (
        f"#64512 regression: when self._ws is already closed, close() must NOT be "
        f"called. Pre-cleanup call_count was {pre_close_call_count} (expected 0)."
    )


@pytest.mark.asyncio
async def test_cleanup_ws_handles_none_ws():
    """Edge case: self._ws is None (never connected). Should not raise,
    and should leave it as None.
    """
    adapter = _make_adapter()
    adapter._ws = None

    # Should not raise
    await adapter._cleanup_ws()

    assert adapter._ws is None