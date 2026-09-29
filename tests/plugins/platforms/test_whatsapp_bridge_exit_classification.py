"""Regression tests for the SIGTERM race in ``_check_managed_bridge_exit`` (#127047).

A supervisor SIGTERMs the gateway on shutdown; the gateway SIGTERMs the Node
bridge; the message-poll loop can observe ``returncode == -15`` in the window
before ``disconnect()`` flips ``_shutting_down``. ``-15`` must classify as an
intentional exit in that window, or every supervised restart crash-loops the
whole gateway (all platforms down, not just WhatsApp).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class _FakeBridgeProcess:
    def __init__(self, returncode):
        self._returncode = returncode

    def poll(self):
        return self._returncode


def _race_window_adapter(returncode: int) -> WhatsAppAdapter:
    """Adapter built via ``__new__`` with no ``_shutting_down`` attribute.

    Mirrors the race window: the poll loop observes the bridge exit before
    ``disconnect()`` has run, so the flag is not merely ``False`` — it does
    not exist yet (the code reads it with ``getattr(..., False)``).
    """
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._bridge_process = _FakeBridgeProcess(returncode)
    adapter._fatal_error_message = None
    adapter._set_fatal_error = MagicMock()
    adapter._notify_fatal_error = AsyncMock()
    adapter._close_bridge_log = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_sigterm_exit_is_nonfatal_without_shutdown_flag():
    """-15 observed before disconnect() flips the flag must not be fatal."""
    adapter = _race_window_adapter(-15)
    assert await adapter._check_managed_bridge_exit() is None
    adapter._set_fatal_error.assert_not_called()
    adapter._notify_fatal_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_sigterm_exit_is_nonfatal_during_shutdown():
    """The original flag-gated path keeps working."""
    adapter = _race_window_adapter(-15)
    adapter._shutting_down = True
    assert await adapter._check_managed_bridge_exit() is None


@pytest.mark.asyncio
async def test_zero_and_sigint_exits_still_require_the_shutdown_flag():
    """Only disconnect() produces 0/-2 deliberately, so without the flag they stay fatal."""
    for returncode in (0, -2):
        adapter = _race_window_adapter(returncode)
        result = await adapter._check_managed_bridge_exit()
        assert result is not None
        adapter._set_fatal_error.assert_called_once()


@pytest.mark.asyncio
async def test_real_crash_still_fatal():
    """A genuine bridge crash (rc=1) keeps the retryable fatal path."""
    adapter = _race_window_adapter(1)
    message = await adapter._check_managed_bridge_exit()
    assert message is not None and "exited unexpectedly" in message
    adapter._set_fatal_error.assert_called_once_with(
        "whatsapp_bridge_exited", message, retryable=True
    )
    adapter._notify_fatal_error.assert_awaited_once()
