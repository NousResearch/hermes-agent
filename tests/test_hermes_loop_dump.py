"""Helpers that name the event-loop thread in a dump.

``faulthandler.dump_traceback(all_threads=True)`` caps its output at 100 threads, newest first, so
the loop thread — created before every worker, hence the oldest — is dropped from any dump taken
in a process running a few hundred threads. ``hermes_loop_dump`` locates that thread so dump sites
can emit it explicitly. The end-to-end dump tests live beside their owners:
``tests/gateway/test_loop_liveness_watchdog.py`` and ``tests/agent/test_deadline.py``.
"""

from __future__ import annotations

import asyncio

from hermes_loop_dump import LOOP_THREAD_HEADER, capture_loop_thread_ident, format_loop_thread_block


def test_capture_on_the_loop_is_that_thread():
    async def capture() -> int | None:
        return capture_loop_thread_ident(asyncio.get_running_loop())

    ident = asyncio.run(capture())
    assert ident is not None


def test_block_never_omits_the_section():
    """An unresolvable loop thread is stated, never silently dropped from a dump."""
    block = format_loop_thread_block(None, frames={})
    assert block.startswith(LOOP_THREAD_HEADER)
    assert "not located" in block
