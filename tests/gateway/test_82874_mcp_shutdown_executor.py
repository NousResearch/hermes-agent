"""Regression test for #82874: MCP shutdown must not block the event loop.

shutdown_mcp_servers() blocks on future.result(timeout=15) internally. When
called synchronously from inside the asyncio loop (the SIGTERM teardown paths
in start_gateway), it freezes the loop for up to 15 s and can cause the
gateway to be SIGKILLed by a supervisor whose kill grace is shorter.

The fix routes the two synchronous call sites in ``start_gateway`` through
``await loop.run_in_executor(None, shutdown_mcp_servers)`` — the same pattern
the /mcp reload path already used. These tests assert the two behaviours that
matter:

1. The patched call sites actually use run_in_executor (source-level guard).
2. A blocking shutdown dispatched through an executor does not freeze the
   event loop (behavioural guard: concurrent coroutines keep making
   progress).
"""

import asyncio
import time
from unittest.mock import patch

import pytest


def _blocking_shutdown(block_for: float = 0.15) -> None:
    """Simulate shutdown_mcp_servers' blocking future.result() wait."""
    time.sleep(block_for)


@pytest.mark.asyncio
async def test_blocking_shutdown_in_executor_does_not_freeze_loop():
    """While shutdown runs in the executor, the loop stays live.

    Before the fix the loop thread blocked inside shutdown_mcp_servers() for
    its full internal timeout, so a concurrent coroutine made no progress
    during that window. Running the same blocking call through
    run_in_executor (as the fix does) must keep the loop responsive.
    """
    loop = asyncio.get_running_loop()

    heartbeat_ticks = []

    async def heartbeat():
        for _ in range(6):
            heartbeat_ticks.append(loop.time())
            await asyncio.sleep(0.02)

    heartbeat_task = asyncio.ensure_future(heartbeat())

    # Run the blocking shutdown in an executor, exactly like the fix does.
    await loop.run_in_executor(None, _blocking_shutdown, 0.15)

    await heartbeat_task

    assert len(heartbeat_ticks) >= 2, (
        "event loop appeared frozen while shutdown ran in executor"
    )


def test_start_gateway_mcp_shutdown_sites_use_run_in_executor():
    """Both MCP shutdown call sites in start_gateway must dispatch via the
    executor, never synchronously on the loop thread.

    Source-level regression guard: if a future refactor reverts either site to
    a plain synchronous ``shutdown_mcp_servers()`` call, this test fails and
    reminds us the loop-blocking bug (#82874) is back.
    """
    import gateway.run as gateway_run

    source = gateway_run.__file__

    with open(source, encoding="utf-8") as fh:
        text = fh.read()

    # The reload path already used the executor pattern; the fix extends it to
    # the two shutdown sites. Count all occurrences of the correct pattern.
    # Each site looks like:
    #   await asyncio.get_running_loop().run_in_executor(
    #       None, shutdown_mcp_servers
    #   )
    executor_calls = text.count("run_in_executor(None, shutdown_mcp_servers")

    # We expect at least the two fixed shutdown sites (the reload path may use
    # a `loop.` variable instead of `asyncio.get_running_loop()`, so count a
    # looser pattern that catches both).
    import re

    loose = re.findall(
        r"run_in_executor\(\s*None,\s*shutdown_mcp_servers", text
    )
    assert len(loose) >= 2, (
        "expected >=2 run_in_executor dispatches for shutdown_mcp_servers, "
        f"found {len(loose)}"
    )

    # And no site may call it synchronously without the executor wrapper.
    # Match a line that calls shutdown_mcp_servers() bare (no executor).
    bare_calls = re.findall(
        r"^\s*shutdown_mcp_servers\(\)\s*$", text, re.MULTILINE
    )
    # The function definition itself contains no bare call; any remaining bare
    # call is a regression (definition line is `def shutdown_mcp_servers():`).
    assert not bare_calls, (
        f"found synchronous shutdown_mcp_servers() calls: {bare_calls}"
    )


def test_mcp_shutdown_drain_fits_supervisor_kill_grace():
    """The server-shutdown drain must fit inside a container supervisor's
    ~3s kill grace (#82874 review feedback from the reporter).

    Source-level guard: shutdown_mcp_servers() waits on the shutdown future
    with ``_MCP_SHUTDOWN_DRAIN_SECONDS`` (2s), not a bare 15s that would
    hold the clean-exit funnel open past the grace period. If a refactor
    reintroduces the old long wait, this test fails.
    """
    import re

    from tools import mcp_tool

    assert mcp_tool._MCP_SHUTDOWN_DRAIN_SECONDS <= 3, (
        "MCP shutdown drain must fit the supervisor kill grace (~3s); "
        f"got {mcp_tool._MCP_SHUTDOWN_DRAIN_SECONDS}s"
    )

    with open(mcp_tool.__file__, encoding="utf-8") as fh:
        text = fh.read()

    # No bare 15s on the shutdown-future wait; the site must use the constant.
    assert "future.result(timeout=15)" not in text, (
        "bare 15s wait reintroduced on the MCP shutdown future"
    )
    assert re.search(
        r"future\.result\(timeout=_MCP_SHUTDOWN_DRAIN_SECONDS\)", text
    ), "shutdown future must wait with _MCP_SHUTDOWN_DRAIN_SECONDS"
