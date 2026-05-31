"""Tests for text batch task cleanup on Matrix and WhatsApp disconnect.

Both adapters use _pending_text_batch_tasks / _pending_text_batches to
debounce rapid message bursts.  disconnect() must cancel and clear those
tasks so sleeping flush coroutines do not wake up and call handle_message()
on a dead adapter.  Parity with Telegram/Discord (PR #8109).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Minimal adapter factories (avoid full __init__ which needs live credentials)
# ---------------------------------------------------------------------------


def _make_matrix_adapter():
    from gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(MatrixAdapter)
    adapter._platform = Platform.MATRIX
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.1
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter.handle_message = AsyncMock()
    # Stubs for disconnect()
    adapter._closing = False
    adapter._sync_task = None
    adapter._reaction_redaction_tasks = set()
    adapter._pending_reactions = {}
    adapter._crypto_db = None
    adapter._client = None
    adapter._mark_disconnected = MagicMock()
    return adapter


def _make_whatsapp_adapter():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    config = PlatformConfig(enabled=True, extra={"session_name": "test"})
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.05
    adapter._text_batch_split_delay_seconds = 0.1
    adapter.handle_message = AsyncMock()
    # Stubs for disconnect()
    adapter._shutting_down = False
    adapter._bridge_process = None
    adapter._poll_task = None
    adapter._http_session = None
    adapter._release_platform_lock = MagicMock()
    adapter._mark_disconnected = MagicMock()
    adapter._close_bridge_log = MagicMock()
    adapter._fatal_error_message = None
    return adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sleeping_task(delay: float = 10.0) -> asyncio.Task:
    """Return a running task that sleeps for `delay` seconds."""
    return asyncio.ensure_future(asyncio.sleep(delay))


# ---------------------------------------------------------------------------
# Matrix tests
# ---------------------------------------------------------------------------


class TestMatrixTextBatchDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_batch_tasks(self):
        """disconnect() must cancel sleeping flush tasks."""
        adapter = _make_matrix_adapter()

        task = _make_sleeping_task()
        adapter._pending_text_batch_tasks["key1"] = task

        await adapter.disconnect()
        await asyncio.sleep(0)  # let the event loop process the cancellation

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_disconnect_clears_batch_dicts(self):
        """disconnect() must empty both batch dicts."""
        adapter = _make_matrix_adapter()

        task = _make_sleeping_task()
        adapter._pending_text_batch_tasks["key1"] = task
        adapter._pending_text_batches["key1"] = MagicMock()

        await adapter.disconnect()

        assert adapter._pending_text_batch_tasks == {}
        assert adapter._pending_text_batches == {}

    @pytest.mark.asyncio
    async def test_disconnect_with_no_pending_tasks_is_safe(self):
        """disconnect() with empty dicts must not raise."""
        adapter = _make_matrix_adapter()
        await adapter.disconnect()  # should not raise


# ---------------------------------------------------------------------------
# WhatsApp tests
# ---------------------------------------------------------------------------


class TestWhatsAppTextBatchDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_batch_tasks(self):
        """disconnect() must cancel sleeping flush tasks."""
        adapter = _make_whatsapp_adapter()

        task = _make_sleeping_task()
        adapter._pending_text_batch_tasks["key1"] = task

        await adapter.disconnect()
        await asyncio.sleep(0)  # let the event loop process the cancellation

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_disconnect_clears_batch_dicts(self):
        """disconnect() must empty both batch dicts."""
        adapter = _make_whatsapp_adapter()

        task = _make_sleeping_task()
        adapter._pending_text_batch_tasks["key1"] = task
        adapter._pending_text_batches["key1"] = MagicMock()

        await adapter.disconnect()

        assert adapter._pending_text_batch_tasks == {}
        assert adapter._pending_text_batches == {}

    @pytest.mark.asyncio
    async def test_disconnect_with_no_pending_tasks_is_safe(self):
        """disconnect() with empty dicts must not raise."""
        adapter = _make_whatsapp_adapter()
        await adapter.disconnect()  # should not raise
