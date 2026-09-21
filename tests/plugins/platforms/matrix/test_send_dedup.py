"""Tests for MatrixAdapter.send's duplicate-send guard.

The Matrix adapter can emit the same message twice when two extractors
resolve the same content (observed with image + text paths), producing
visible double-posts. send() suppresses an identical body re-sent to the
same room within a short window — only for substantial bodies (short ACK/
status lines may legitimately repeat), and only after a successful send (a
retry of a failed send must never be mistaken for a duplicate).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"},
    )
    adapter = MatrixAdapter(config)
    adapter._client = MagicMock()  # truthy: a client is "connected"
    adapter._encryption = False
    adapter._send_room_message = AsyncMock(return_value="$evt1")
    return adapter


_LONG_BODY = "x" * 40  # >= the 40-char dedup threshold


class TestSendDedup:
    def test_identical_body_within_window_is_suppressed(self):
        adapter = _make_adapter()
        room = "!room1:example.org"

        first = asyncio.run(adapter.send(room, _LONG_BODY))
        second = asyncio.run(adapter.send(room, _LONG_BODY))

        assert first.success is True
        assert second.success is True
        # The real send only happened once — the second call was suppressed.
        assert adapter._send_room_message.await_count == 1

    def test_different_body_is_not_suppressed(self):
        adapter = _make_adapter()
        room = "!room1:example.org"

        asyncio.run(adapter.send(room, _LONG_BODY))
        asyncio.run(adapter.send(room, _LONG_BODY + " different"))

        assert adapter._send_room_message.await_count == 2

    def test_same_body_different_room_is_not_suppressed(self):
        adapter = _make_adapter()

        asyncio.run(adapter.send("!room1:example.org", _LONG_BODY))
        asyncio.run(adapter.send("!room2:example.org", _LONG_BODY))

        assert adapter._send_room_message.await_count == 2

    def test_short_body_is_never_deduped(self):
        """ACK/status lines may legitimately repeat -- only substantial bodies (>= 40
        chars) are deduped."""
        adapter = _make_adapter()
        room = "!room1:example.org"

        asyncio.run(adapter.send(room, "ok"))
        asyncio.run(adapter.send(room, "ok"))

        assert adapter._send_room_message.await_count == 2

    def test_duplicate_outside_window_is_not_suppressed(self, monkeypatch):
        adapter = _make_adapter()
        room = "!room1:example.org"

        _now = [1_000_000.0]
        monkeypatch.setattr(time, "time", lambda: _now[0])

        asyncio.run(adapter.send(room, _LONG_BODY))
        _now[0] += 13  # just past the 12s window
        asyncio.run(adapter.send(room, _LONG_BODY))

        assert adapter._send_room_message.await_count == 2

    def test_failed_send_is_never_recorded_so_retry_is_not_suppressed(self):
        """A retry of a failed send must go through — the cache only records after
        a successful send."""
        adapter = _make_adapter()
        adapter._send_room_message = AsyncMock(side_effect=RuntimeError("network error"))
        room = "!room1:example.org"

        first = asyncio.run(adapter.send(room, _LONG_BODY))
        assert first.success is False

        adapter._send_room_message = AsyncMock(return_value="$evt1")
        second = asyncio.run(adapter.send(room, _LONG_BODY))

        assert second.success is True
        adapter._send_room_message.assert_awaited_once()

    def test_stale_cache_entries_are_pruned(self, monkeypatch):
        """The cache must not grow unboundedly: entries older than the window are
        dropped on the next send, not just skipped."""
        adapter = _make_adapter()

        _now = [1_000_000.0]
        monkeypatch.setattr(time, "time", lambda: _now[0])

        asyncio.run(adapter.send("!roomA:example.org", _LONG_BODY))
        assert len(adapter._recent_sends) == 1

        _now[0] += 13  # past the window
        asyncio.run(adapter.send("!roomB:example.org", _LONG_BODY + " other"))

        # The stale !roomA entry was pruned; only the fresh !roomB one remains.
        assert len(adapter._recent_sends) == 1
        assert ("!roomB:example.org", _LONG_BODY + " other") in adapter._recent_sends
