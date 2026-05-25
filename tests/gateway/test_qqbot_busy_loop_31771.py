"""Regression tests for #31771 — QQBot busy-loop on closed WebSocket.

When the QQ Bot WebSocket closed and the next reconnect attempt failed
(e.g. ``Failed to get QQ Bot gateway URL: ...`` from a transient DNS
glitch), ``self._ws`` retained the *old, already-closed* reference
because the failure happened in ``_ensure_token`` / ``_get_gateway_url``
before ``_open_ws`` could clear it.  The next ``_listen_loop`` iteration
re-entered ``_read_events``, where the ``while not self._ws.closed:``
loop body never ran, the function returned ``None`` immediately, and
``_listen_loop`` reset the backoff and looped again — pegging the
process at 99-100 % CPU with no further reconnect logs.

Two layers of defense in this PR:

1. ``_read_events`` raises when entering with a closed socket and again
   if the read loop exits silently while the listener still wants to
   run, so a closed-socket return is never mistaken for a successful
   read cycle.
2. ``_reconnect`` clears ``self._ws`` on its exception path so the next
   iteration sees a missing socket rather than a closed one.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**extra):
    from gateway.platforms.qqbot import QQAdapter
    return QQAdapter(PlatformConfig(enabled=True, extra=extra))


class _ClosedWS:
    """Stand-in for an already-closed ``aiohttp.ClientWebSocketResponse``.

    The real closed object is truthy and reports ``closed=True``; calling
    ``receive()`` on it would either return a CLOSED frame or raise.  The
    test guards never need ``receive()`` because the read loop body must
    not run for a closed socket.
    """
    closed = True

    async def receive(self):  # pragma: no cover — never reached by the fix
        raise AssertionError(
            "receive() must not be called on an already-closed WebSocket"
        )


class _OpenWS:
    closed = False

    async def receive(self):  # pragma: no cover — not exercised here
        raise AssertionError("test must not actually receive frames")


# ---------------------------------------------------------------------------
# _read_events: explicit raises on closed / missing / disappearing sockets
# ---------------------------------------------------------------------------


class TestReadEventsRaisesOnClosedSocket:
    """The read loop must never return silently when the socket is closed."""

    @pytest.mark.asyncio
    async def test_raises_when_ws_is_already_closed_on_entry(self):
        """The exact #31771 scenario: stale closed socket from a failed reconnect."""
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        adapter._ws = _ClosedWS()

        with pytest.raises(RuntimeError, match="closed before read"):
            await adapter._read_events()

    @pytest.mark.asyncio
    async def test_raises_when_ws_is_missing(self):
        """``self._ws is None`` must surface as a connection error, not a silent return."""
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        adapter._ws = None

        with pytest.raises(RuntimeError, match="not connected"):
            await adapter._read_events()

    @pytest.mark.asyncio
    async def test_raises_when_running_but_ws_disappears_mid_read(self):
        """Open socket → adapter still running → loop exits → must raise.

        Mimics another task setting ``self._ws = None`` while the read
        loop's predicate is being re-evaluated, or the socket transitioning
        to closed without a CLOSE / CLOSED / ERROR frame ever surfacing.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = True

        # Start with an "open" socket whose closed flag flips to True on
        # the *next* read loop predicate evaluation.  Returning
        # ``closed=False`` once and then ``True`` causes the body to skip
        # and the post-loop guard to fire.
        flips = [False, True]

        class _FlipWS:
            @property
            def closed(self):
                # First evaluation: False (loop entered);
                # subsequent: True (loop body skipped).
                return flips.pop(0) if flips else True

            async def receive(self):  # pragma: no cover — not reached
                raise AssertionError("receive() must not be called")

        adapter._ws = _FlipWS()

        with pytest.raises(RuntimeError, match="closed during read"):
            await adapter._read_events()

    @pytest.mark.asyncio
    async def test_does_not_raise_when_listener_was_stopped(self):
        """If ``_running`` flipped to False, returning silently is correct.

        The post-loop guard must NOT raise during a clean shutdown — that
        would surface a spurious "WebSocket closed during read" up to
        ``_listen_loop`` after the user already requested disconnect.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = False
        adapter._ws = _OpenWS()

        # No exception expected; clean return.
        result = await adapter._read_events()
        assert result is None


# ---------------------------------------------------------------------------
# _reconnect: stale closed _ws is cleared so the next read raises cleanly
# ---------------------------------------------------------------------------


class TestReconnectClearsStaleClosedSocket:
    """A failed reconnect must drop the stale closed ``self._ws`` reference."""

    @pytest.mark.asyncio
    async def test_clears_ws_when_get_gateway_url_raises(self):
        """The exact path from the #31771 trace.

        ``Reconnect failed: Failed to get QQ Bot gateway URL`` means
        ``_get_gateway_url`` raised before ``_open_ws`` ran.  The closed
        ``self._ws`` from the previous session must be discarded so the
        next iteration's ``_read_events`` does not silently short-circuit.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        stale_ws = _ClosedWS()
        adapter._ws = stale_ws

        adapter._ensure_token = mock.AsyncMock()
        adapter._get_gateway_url = mock.AsyncMock(
            side_effect=RuntimeError("Failed to get QQ Bot gateway URL: dns")
        )

        # Patch out the backoff sleep so the test runs instantly.
        with mock.patch("gateway.platforms.qqbot.adapter.asyncio.sleep",
                        new=mock.AsyncMock()):
            ok = await adapter._reconnect(backoff_idx=0)

        assert ok is False
        assert adapter._ws is None, (
            "stale closed _ws must be dropped on failed reconnect"
        )

    @pytest.mark.asyncio
    async def test_does_not_clear_ws_when_open_ws_assigned_a_new_one(self):
        """A successful reconnect leaves ``self._ws`` set to the new open socket."""
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._ws = _ClosedWS()  # stale leftover from previous session
        new_ws = _OpenWS()

        adapter._ensure_token = mock.AsyncMock()
        adapter._get_gateway_url = mock.AsyncMock(return_value="wss://example")

        async def fake_open_ws(_url):
            adapter._ws = new_ws

        adapter._open_ws = fake_open_ws
        adapter._mark_connected = mock.MagicMock()

        with mock.patch("gateway.platforms.qqbot.adapter.asyncio.sleep",
                        new=mock.AsyncMock()):
            ok = await adapter._reconnect(backoff_idx=0)

        assert ok is True
        assert adapter._ws is new_ws

    @pytest.mark.asyncio
    async def test_does_not_clear_open_ws_on_unrelated_failure(self):
        """If the surviving ``self._ws`` is somehow open, leave it alone.

        Defensive: only the ``ws.closed`` branch should null out the
        reference.  This guards against the cleanup accidentally killing
        a still-live socket.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        live_ws = _OpenWS()
        adapter._ws = live_ws

        adapter._ensure_token = mock.AsyncMock(side_effect=RuntimeError("boom"))

        with mock.patch("gateway.platforms.qqbot.adapter.asyncio.sleep",
                        new=mock.AsyncMock()):
            ok = await adapter._reconnect(backoff_idx=0)

        assert ok is False
        assert adapter._ws is live_ws


# ---------------------------------------------------------------------------
# Integration: the listener no longer busy-loops when reconnect fails
# ---------------------------------------------------------------------------


class TestListenLoopNoBusyLoopAfterReconnectFailure:
    """End-to-end: the #31771 traceback no longer produces a hot loop."""

    @pytest.mark.asyncio
    async def test_failed_reconnect_breaks_out_after_max_attempts(self):
        """``_listen_loop`` increments backoff on each failure and gives up.

        Before the fix: ``_read_events`` returned silently → backoff_idx
        was reset to 0 each iteration → ``MAX_RECONNECT_ATTEMPTS`` was
        never reached → infinite hot loop.  After the fix:
        ``_read_events`` raises, the ``except Exception`` branch ticks
        ``backoff_idx`` until it hits the cap.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        # Stale closed ws — the #31771 starting state.
        adapter._ws = _ClosedWS()

        # Every reconnect fails.  Use a shallow attempt cap so the test
        # finishes quickly.
        with mock.patch.object(adapter, "_reconnect",
                               new=mock.AsyncMock(return_value=False)) as recon, \
             mock.patch("gateway.platforms.qqbot.adapter.MAX_RECONNECT_ATTEMPTS", 3), \
             mock.patch.object(adapter, "_mark_disconnected") as md, \
             mock.patch.object(adapter, "_mark_transport_disconnected"):

            await asyncio.wait_for(adapter._listen_loop(), timeout=2.0)

        # Reconnect was called exactly MAX attempts before the loop gave up
        # — proving _read_events raises (not returns) so backoff progresses.
        assert recon.await_count == 3
        md.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_busy_loop_with_stale_closed_ws(self):
        """Direct repro of #31771: ``_read_events`` must not return silently.

        Without the fix, this test would loop forever; the
        ``asyncio.wait_for`` timeout protects the test runner.
        """
        adapter = _make_adapter(app_id="a", client_secret="b")
        adapter._running = True
        adapter._ws = _ClosedWS()

        # First _read_events() invocation must raise (not return None).
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(adapter._read_events(), timeout=1.0)


# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------


class TestReadEventsSourceGuards:
    """Pin the defensive checks in source so an accidental revert is loud."""

    def test_read_events_has_closed_pre_check(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        src = inspect.getsource(QQAdapter._read_events)
        # Pre-check: raise on closed socket BEFORE the while loop.
        assert "closed before read" in src
        # Post-loop guard: raise if running but loop exited silently.
        assert "closed during read" in src
        # Issue reference so the guard isn't refactored away.
        assert "31771" in src

    def test_reconnect_clears_stale_closed_ws_in_except(self):
        from gateway.platforms.qqbot.adapter import QQAdapter
        src = inspect.getsource(QQAdapter._reconnect)
        assert "Reconnect failed" in src
        # The stale-clear branch is gated on closed=True so it never kills
        # a still-live socket.
        assert "self._ws.closed" in src
        assert "self._ws = None" in src
        assert "31771" in src
