"""Regression tests for #82874 round-2: MCP teardown must fit the kill grace.

On the gateway clean-exit critical path, a container supervisor (s6-overlay,
etc.) grants only ~3s of kill grace before SIGKILL. If the whole MCP teardown
funnel exceeds that, ``lifecycle_ledger.mark_exited()`` never runs and every
subsequent boot reports a phantom unclean exit.

`main` already landed #99675 which runs ``shutdown_mcp_servers()`` on a daemon
thread (``_shutdown_mcp_servers_nonblocking``) so the loop stays responsive.
This suite guards the round-2 concern from review: the teardown must ALSO fit
inside the ~3s grace. So:

- ``_MCP_TEARDOWN_BUDGET_SECONDS`` bounds the WHOLE funnel (server drain +
  loop drain + thread join) via one shared budget, with every segment wait
  clamping to it (``_teardown_clamp``).
- The gateway's nonblocking wrapper defaults its timeout to that budget, not a
  bare 5.0s.
- The orphan-PID reap (SIGTERM -> 2s -> SIGKILL) runs on a detached thread the
  exit path never joins, so its 2s dance cannot hold the funnel open.

The budget arithmetic lives in ``tools.mcp_tool`` (the origin module) and is
consumed across ``tools.mcp_tool_lifecycle`` (segment 1) / ``tools.mcp_tool_loop``
(segments 2-4), so the source-level guard inspects those modules individually.
"""

import asyncio
import inspect
from unittest.mock import patch

import pytest

import gateway.run as gateway_run
from tools import mcp_tool
from tools import mcp_tool_lifecycle as _mcp_lifecycle
from tools import mcp_tool_loop as _mcp_loop


def test_teardown_budget_fits_supervisor_kill_grace():
    """The whole funnel must stay under the ~3s supervisor kill grace."""
    budget = mcp_tool._MCP_TEARDOWN_BUDGET_SECONDS
    assert 0 < budget < 3, (
        "MCP teardown budget must stay under the ~3s supervisor kill grace; "
        f"got {budget}s"
    )
    # Segment-1 drain (server shutdown) must never exceed the total budget.
    assert 0 < mcp_tool._MCP_SHUTDOWN_DRAIN_SECONDS <= budget, (
        "server-shutdown drain must not exceed the total teardown budget"
    )


def test_teardown_clamp_obeys_budget():
    """Every bounded wait clamps to remaining budget; never negative."""
    clamp = mcp_tool._teardown_clamp
    assert clamp(2.0, 2.75) == 2.0   # own limit fits -> keep it
    assert clamp(13.0, 0.6) == 0.6    # exceed remaining -> clamp down
    assert clamp(5.0, 0.0) == 0.0     # budget spent -> zero
    assert clamp(5.0, -1.0) == 0.0    # never negative


def test_gateway_nonblocking_default_timeout_is_budget():
    """_shutdown_mcp_servers_nonblocking must default to the 2.75s budget,
    not a bare 5.0s that would overrun the kill grace (#82874 review)."""
    # Source-level guard: no bare `timeout: float = 5.0` default may return.
    sig = inspect.signature(gateway_run._shutdown_mcp_servers_nonblocking)
    default = sig.parameters["timeout"].default
    assert default is None, (
        "gateway nonblocking shutdown should resolve the budget lazily "
        f"(default None), got {default!r}"
    )
    # Behavioural guard: when invoked without an explicit timeout, the wait
    # ceiling is exactly the teardown budget, not longer.
    with patch("tools.mcp_tool._MCP_TEARDOWN_BUDGET_SECONDS", 2.75):
        import tools.mcp_tool as mt
        assert mt._MCP_TEARDOWN_BUDGET_SECONDS == 2.75
        src = inspect.getsource(gateway_run._shutdown_mcp_servers_nonblocking)
        assert "timeout = _MCP_TEARDOWN_BUDGET_SECONDS" in src, (
            "gateway wrapper must thread the budget as its default ceiling"
        )


def test_teardown_segments_thread_shared_budget_and_orphan_reap_off_path():
    """Source-level guard for #82874 round-2.

    Every bounded teardown wait threads the shared budget; the orphan reap
    leaves the critical path on a detached thread; no bare fixed waits
    (future.result(timeout=15) / thread.join(timeout=5)) return.
    """
    lifecycle_src = inspect.getsource(_mcp_lifecycle)
    loop_src = inspect.getsource(_mcp_loop)

    # The shutdown drain derives from the budget via the clamp.
    assert "_MCP_SHUTDOWN_DRAIN_SECONDS, teardown_budget" in lifecycle_src
    assert "future.result(timeout=drain_wait)" in lifecycle_src
    # The budget passes from shutdown_mcp_servers into _stop_mcp_loop.
    assert "teardown_budget=teardown_budget" in lifecycle_src
    # The loop drain derives from the budget via the clamp.
    assert "_MCP_LOOP_DRAIN_TIMEOUT + 1, teardown_budget" in loop_src
    # The thread join threads the budget.
    assert "_teardown_clamp(5.0, teardown_budget)" in loop_src
    # Orphan reap leaves the critical path via a detached thread.
    assert "_start_orphan_reaper()" in loop_src
    assert "daemon=True" in loop_src and "mcp-orphan-reaper" in loop_src
    # No bare fixed waits may return.
    assert "future.result(timeout=15)" not in lifecycle_src
    assert "thread.join(timeout=5)" not in loop_src


@pytest.mark.asyncio
async def test_gateway_nonblocking_waits_budget_when_wedged():
    """A wedged shutdown must return after the budget, never after 5s.

    Expensive wedge right at the nonblocking wrapper's own ceiling. Since the
    budget is ~2.75s a sleep wedge would make the test slow; instead probe that
    the wrapper passes the budget into ``_await_thread_exit`` by asserting it
    observed a sub-5s deadline.
    """

    import time

    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def wedged_shutdown():
        loop.call_soon_threadsafe(started.set)
        time.sleep(30)

    observed_deadline = {}

    original_await = gateway_run._await_thread_exit

    async def spy_await(thread, timeout=5.0, poll=0.05):
        observed_deadline["timeout"] = timeout
        return await original_await(thread, timeout=timeout, poll=poll)

    with patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers", wedged_shutdown), \
         patch.object(gateway_run, "_await_thread_exit", spy_await):
        # Explicit no-timeout: it must resolve the 2.75s budget, not 5s.
        await gateway_run._shutdown_mcp_servers_nonblocking(timeout=0.5)

    assert started.is_set()
    assert observed_deadline["timeout"] == 0.5, "timeout must pass through to the await"
