"""Tests for the Matrix startup-grace backfill drop report (issue #133265).

Messages that arrive while the gateway is down are replayed by the initial sync and dropped
by the startup grace filter — previously with no log line at all. The adapter now counts
those drops and reports them after the initial sync: WARNING when the newest drop is close
to startup (likely downtime messages that got no reply), INFO for older already-handled
replays.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    """Create a MatrixAdapter with mocked config and a fresh startup timestamp."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time()
    return adapter


def _make_message_event(
    adapter, seconds_before_startup: float, event_id: str = "$evt1"
):
    """A room message whose origin_server_ts lands the given seconds before startup (seconds —
    _matrix_event_timestamp_seconds keeps sub-10^13 values un-scaled)."""
    return SimpleNamespace(
        room_id="!room:example.org",
        sender="@alice:example.org",
        event_id=event_id,
        timestamp=adapter._startup_ts - seconds_before_startup,
        content=SimpleNamespace(msgtype="m.text", body="hello"),
    )


class TestBackfillDropAccounting:
    @pytest.mark.asyncio
    async def test_grace_drop_is_counted_not_dispatched(self):
        """A message older than the startup grace is dropped by _on_room_message AND counted
        for the post-sync report — the drop itself stays silent until the report fires."""
        adapter = _make_adapter()
        await adapter._on_room_message(_make_message_event(adapter, 21.0))
        assert adapter._backfill_drops == 1
        assert adapter._backfill_newest_ts == pytest.approx(adapter._startup_ts - 21.0)
        assert not adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_fresh_message_is_neither_dropped_nor_counted(self):
        """A message inside the grace window flows through to dispatch untouched."""
        adapter = _make_adapter()
        await adapter._on_room_message(_make_message_event(adapter, 1.0))
        assert adapter._backfill_drops == 0

    def test_counter_tracks_oldest_and_newest(self):
        adapter = _make_adapter()
        for age in (1200.0, 30.0, 600.0):
            adapter._note_backfill_drop(adapter._startup_ts - age)
        assert adapter._backfill_drops == 3
        assert adapter._backfill_oldest_ts == pytest.approx(
            adapter._startup_ts - 1200.0
        )
        assert adapter._backfill_newest_ts == pytest.approx(adapter._startup_ts - 30.0)


class TestBackfillReport:
    def test_downtime_message_drop_warns(self, caplog):
        """A newest drop within the loss window escalates to WARNING and tells the operator
        the events were not processed — the exact silent loss reported in #133265."""
        adapter = _make_adapter()
        adapter._note_backfill_drop(adapter._startup_ts - 1200.0)
        adapter._note_backfill_drop(adapter._startup_ts - 21.0)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.matrix.adapter"):
            adapter._report_backfill_drops()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "arrived while the gateway was offline" in r.message for r in warnings
        )
        assert any("2 replayed event" in r.message for r in warnings)

    def test_old_replay_only_logs_info(self, caplog):
        """Drops whose newest event is far older than startup are ordinary already-handled
        replays: a single INFO summary, no downtime WARNING."""
        adapter = _make_adapter()
        adapter._note_backfill_drop(adapter._startup_ts - 7200.0)
        adapter._note_backfill_drop(adapter._startup_ts - 3600.0)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.matrix.adapter"):
            adapter._report_backfill_drops()
        assert not [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "startup grace" in r.message
        ]
        assert any(
            r.levelno == logging.INFO and "2 replayed event" in r.message
            for r in caplog.records
        )

    def test_no_drops_no_log(self, caplog):
        """A clean initial sync with no grace drops stays silent."""
        adapter = _make_adapter()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.matrix.adapter"):
            adapter._report_backfill_drops()
        assert not [r for r in caplog.records if "startup grace" in r.message]

    def test_report_resets_accounting(self, caplog):
        """Reporting consumes the counters, so a later (non-initial) backfill drop from a
        freshly joined room does not re-report the initial sync's numbers."""
        adapter = _make_adapter()
        adapter._note_backfill_drop(adapter._startup_ts - 21.0)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.matrix.adapter"):
            adapter._report_backfill_drops()
        assert adapter._backfill_drops == 0
        assert adapter._backfill_oldest_ts == 0.0
        assert adapter._backfill_newest_ts == 0.0


class TestAbsorbSyncReportsInitialOnly:
    @pytest.mark.asyncio
    async def test_initial_sync_dispatches_then_reports(self):
        """_absorb_sync(initial=True) fires the report after the replayed timeline has been
        dispatched; an incremental sync never reports."""
        adapter = _make_adapter()
        calls = []
        adapter._report_backfill_drops = lambda: calls.append(1)
        adapter._dispatch_sync = AsyncMock()
        adapter._refresh_dm_cache = AsyncMock()
        adapter._schedule_pending_invite_joins = lambda sync: None
        client = SimpleNamespace(sync_store=SimpleNamespace(put_next_batch=AsyncMock()))
        await adapter._absorb_sync(
            client, {"rooms": {"join": {"!r:x": {}}}, "next_batch": "nb1"}, initial=True
        )
        assert calls == [1]
        await adapter._absorb_sync(client, {"next_batch": "nb2"}, initial=False)
        assert calls == [1]
