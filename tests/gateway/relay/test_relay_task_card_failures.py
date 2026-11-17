"""Regression: task-card transport failures degrade, never raise
(PR 85796 review, B7).

send_native_task_card_progress / stop_native_task_card_progress let
transport exceptions escape. The stop runs in the progress loop's
finally block on the turn-cleanup path, and the cleanup awaits caught
only CancelledError — a socket drop during card publish/stop therefore
aborted cleanup BEFORE the final-delivery bookkeeping ran.
"""

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class ExplodingTransport:
    async def send_outbound(self, payload, platform=None):
        raise ConnectionError("socket dropped")


def _adapter():
    adapter, _ = _connected_adapter()
    adapter._transport = ExplodingTransport()
    return adapter


TASKS = [{"id": "t1", "title": "terminal", "status": "in_progress"}]


class TestTaskCardTransportFailure:
    @pytest.mark.asyncio
    async def test_progress_returns_failed_result(self):
        adapter = _adapter()
        result = await adapter.send_native_task_card_progress(
            "C1", TASKS, reply_to="1700.1"
        )
        assert result.success is False
        assert "transport error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_stop_returns_failed_result(self):
        adapter = _adapter()
        result = await adapter.stop_native_task_card_progress(
            "C1", reply_to="1700.1"
        )
        assert result.success is False
        assert "transport error" in (result.error or "")
