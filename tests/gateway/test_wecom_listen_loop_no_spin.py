"""WeCom `_listen_loop` must back off, not spin, when the websocket is already closed.

Regression test for the failure mode where a reconnect fails during the subscribe
handshake and leaves a closed socket behind.  `_read_events` used to return
immediately for a closed socket (no exception, no suspending ``await``), so
`_listen_loop` re-entered it in a tight loop and starved the gateway's event loop
(observed in production: 11 days at ~99 % CPU with WeCom never reconnecting).
"""

from __future__ import annotations

import asyncio
import types

from plugins.platforms.wecom.adapter import WeComAdapter

# The buggy loop never yields, so the test stops it itself after this many re-entries
# rather than relying on a timeout that a starved event loop could never fire.
MAX_REENTRIES = 25


class _ClosedWS:
    closed = True

    async def close(self):
        return None

    async def receive(self):
        raise AssertionError("receive() must not be called on a closed socket")


class _Adapter(WeComAdapter):
    name = "wecom-test"  # BasePlatformAdapter exposes ``name`` as a read-only property


def _bare_adapter() -> _Adapter:
    adapter = _Adapter.__new__(_Adapter)
    adapter._running = True
    adapter._ws = _ClosedWS()
    adapter._session = None
    adapter._fail_pending_responses = lambda exc: None
    adapter._fail_all = lambda exc: None
    adapter._mark_connected = lambda: None
    return adapter


def test_listen_loop_backs_off_instead_of_spinning_on_closed_socket():
    adapter = _bare_adapter()
    counts = {"read_events": 0, "reconnect": 0}
    orig_read = WeComAdapter._read_events

    async def counted_read(self):
        counts["read_events"] += 1
        if counts["read_events"] > MAX_REENTRIES:
            self._running = False  # stop the runaway loop so the test cannot hang
            return None
        return await orig_read(self)

    async def failing_open():
        counts["reconnect"] += 1
        raise RuntimeError("WeCom websocket closed during authentication")

    adapter._read_events = types.MethodType(counted_read, adapter)
    adapter._open_connection = failing_open

    async def scenario():
        task = asyncio.create_task(adapter._listen_loop())
        # A healthy loop raises once, logs, and parks in the first backoff sleep (2 s),
        # which lets this coroutine run again.  A spinning loop only gets here after
        # the MAX_REENTRIES guard has forced it to exit.
        await asyncio.sleep(0.3)
        parked = not task.done()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return parked

    parked = asyncio.run(scenario())

    assert parked, "listen loop exited instead of waiting in the reconnect backoff"
    assert counts["read_events"] == 1, (
        f"_read_events re-entered {counts['read_events']} times without yielding — "
        "a closed socket must raise so the loop goes through backoff"
    )
    assert counts["reconnect"] == 0, "reconnect must wait for the backoff delay first"
