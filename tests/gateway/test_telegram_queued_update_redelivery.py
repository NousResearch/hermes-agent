"""Queued-update redelivery after a poll stall / cold boot (#11023).

A message a human sent while the Telegram adapter was not polling is a message they
are waiting on. Two properties are pinned here:

1. A connection that PRESERVES Telegram's pending queue counts what comes back and
   logs ``redelivered N queued update(s) after stall`` — the loss is never silent.
2. A simulated stall during an inbound DM results in the message being REDELIVERED
   (the queue is preserved, the update is admitted and dispatched) rather than dropped.

The complementary guarantee — a preserved backlog does not double-reply — is provided
by the persistent per-bot receipt file
(``plugins/platforms/telegram/update_admission.py``): a redelivered update whose receipt
is on disk is dropped by admission, so preserving the queue cannot replay an answered
message.
"""

import logging
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

_ADAPTER_LOGGER = "plugins.platforms.telegram.adapter"
_REDELIVERED = "redelivered"


def _make_adapter(extra=None) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._webhook_mode = False
    adapter._begin_polling_generation()
    return adapter


def _receive(adapter: TelegramAdapter, n: int) -> None:
    """Simulate n updates arriving on the getUpdates wire for the current generation."""
    request = MagicMock()
    request.parse_json_payload = MagicMock(
        return_value={"ok": True, "result": [{"update_id": i} for i in range(n)]}
    )
    adapter._observe_polling_request_result(request, adapter._polling_generation, (200, b"{}"))


def _redelivery_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _REDELIVERED in r.message]


@pytest.mark.asyncio
async def test_cold_boot_preserving_arms_redelivery_counter(caplog):
    """A cold boot that preserves the queue arms the counter before any update arrives."""
    adapter = _make_adapter()
    assert adapter._queued_updates_preserved is False

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        assert adapter._cold_boot_drop_pending(is_reconnect=False) is False

    assert adapter._queued_updates_preserved is True
    assert adapter._redelivered_updates_count == 0
    # Nothing was redelivered yet, so nothing is claimed.
    assert _redelivery_lines(caplog) == []


@pytest.mark.asyncio
async def test_cold_boot_dropping_does_not_arm_redelivery_counter():
    """Opting back into the discard behaviour must not claim a redelivery that never happened."""
    adapter = _make_adapter(extra={"drop_pending_on_cold_boot": True})

    assert adapter._cold_boot_drop_pending(is_reconnect=False) is True
    assert adapter._queued_updates_preserved is False


@pytest.mark.asyncio
async def test_redelivered_updates_are_counted_and_logged(caplog):
    """Updates handed back by Telegram after a preserved-queue connect are counted and logged."""
    adapter = _make_adapter()
    adapter._arm_queued_update_redelivery()

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        _receive(adapter, 3)

    assert adapter._redelivered_updates_count == 3
    lines = _redelivery_lines(caplog)
    assert len(lines) == 1, lines
    assert "redelivered 3 queued update(s) after stall" in lines[0]


@pytest.mark.asyncio
async def test_redelivery_logged_once_per_episode(caplog):
    """A burst of preserved updates produces ONE log line, not one per batch."""
    adapter = _make_adapter()
    adapter._arm_queued_update_redelivery()

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        _receive(adapter, 2)
        _receive(adapter, 5)

    assert adapter._redelivered_updates_count == 7
    assert len(_redelivery_lines(caplog)) == 1


@pytest.mark.asyncio
async def test_no_redelivery_claimed_when_queue_was_dropped(caplog):
    """With the discard behaviour opted into, a late update is not reported as redelivered."""
    adapter = _make_adapter(extra={"drop_pending_on_cold_boot": True})
    adapter._cold_boot_drop_pending(is_reconnect=False)

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        _receive(adapter, 4)

    assert adapter._redelivered_updates_count == 0
    assert _redelivery_lines(caplog) == []


@pytest.mark.asyncio
async def test_stall_during_inbound_dm_redelivers_instead_of_dropping(caplog):
    """Regression (#11023): a stall during an inbound DM must not lose the message.

    The scenario: Bryan's DM reaches the adapter, the poll stalls, the supervisor
    rebuilds the adapter. The rebuilt connection PRESERVES Telegram's pending queue, so
    the update comes back on the wire and is counted as redelivered — the operator sees
    ``redelivered N queued update(s) after stall`` in the journal instead of the message
    vanishing.
    """
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=_ADAPTER_LOGGER):
        # The stall hands the adapter to the supervisor, which rebuilds it. The rebuild
        # is a reconnect: the queue is preserved and the redelivery counter is armed.
        adapter._arm_queued_update_redelivery()

        # Telegram hands back the DM that was queued while the adapter was not polling.
        _receive(adapter, 1)

    assert adapter._redelivered_updates_count == 1
    lines = _redelivery_lines(caplog)
    assert len(lines) == 1, lines
    assert "redelivered 1 queued update(s) after stall" in lines[0]
    # The message is NOT dropped: the update counter advanced, so the update reached us.
    assert adapter._updates_received_total >= 1
