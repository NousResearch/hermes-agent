"""Regression tests for #82874 round-2: MCP teardown must fit the kill grace.

On the gateway clean-exit critical path a container supervisor (s6-overlay and
friends) grants only ~3s of kill grace before SIGKILL. If the whole MCP teardown
funnel exceeds that, ``lifecycle_ledger.mark_exited()`` never runs and every
subsequent boot reports a phantom unclean exit.

Main already runs ``shutdown_mcp_servers()`` on a daemon thread
(``_shutdown_mcp_servers_nonblocking``) so the event loop stays responsive. These
tests guard the round-2 concern: the teardown must ALSO fit inside the ~3s grace.

- ``_MCP_TEARDOWN_BUDGET_SECONDS`` bounds the whole funnel (server drain + loop
  drain + thread join) as ONE shared budget, and every segment wait clamps to it
  through ``_teardown_clamp``.
- The gateway's nonblocking wrapper defaults its wait to that budget, not a bare
  5.0s — verified behaviourally by timing it against a wedged teardown.
- The orphan-PID reap (SIGTERM -> 2s -> SIGKILL) runs on a detached thread the
  exit path never joins, so its dance cannot hold the funnel open.

The budget arithmetic lives in ``tools.mcp_tool`` (the origin module) and is
consumed across ``tools.mcp_tool_lifecycle`` (segment 1) and ``tools.mcp_tool_loop``
(segments 2-4), so the source-level guards inspect those modules individually.
"""

import asyncio
import inspect
import threading
import time
from unittest.mock import patch

from gateway.run import _shutdown_mcp_servers_nonblocking
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
    # Segment 1 (server shutdown) must never exceed the total budget.
    assert 0 < mcp_tool._MCP_SHUTDOWN_DRAIN_SECONDS <= budget, (
        "server-shutdown drain must not exceed the total teardown budget"
    )


def test_teardown_clamp_obeys_budget():
    """Every bounded wait clamps to the remaining budget; never negative."""
    assert mcp_tool._teardown_clamp(2.0, 2.75) == 2.0   # own limit fits -> keep it
    assert mcp_tool._teardown_clamp(13.0, 0.6) == 0.6   # exceeds remaining -> clamp down
    assert mcp_tool._teardown_clamp(5.0, 0.0) == 0.0    # budget spent -> zero
    assert mcp_tool._teardown_clamp(5.0, -1.0) == 0.0   # never negative


def test_gateway_nonblocking_wait_is_bounded_by_the_budget(monkeypatch):
    """Behavioural: a wedged teardown must not hold the exit path past the budget.

    Pre-fix the wrapper defaulted to a bare 5.0s wait, so this returns only after
    ~5s — longer than the supervisor's whole kill grace.
    """
    release = threading.Event()

    def _wedged_shutdown():
        # Stands in for a server whose shutdown() never returns.
        release.wait(timeout=10)

    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", _wedged_shutdown)
    monkeypatch.setattr(mcp_tool, "_MCP_TEARDOWN_BUDGET_SECONDS", 0.2)
    try:
        start = time.monotonic()
        completed = asyncio.run(_shutdown_mcp_servers_nonblocking())
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert completed is False, "a wedged teardown must report as not-completed"
    assert elapsed < 1.5, (
        f"exit funnel waited {elapsed:.2f}s; the budget (0.2s) must bound it, "
        "not a bare 5.0s default"
    )


def test_gateway_nonblocking_signature_resolves_budget_lazily():
    """The default must be resolved from the budget, not frozen at import time.

    A bare ``timeout: float = 5.0`` default is exactly what overran the grace.
    """
    sig = inspect.signature(_shutdown_mcp_servers_nonblocking)
    assert sig.parameters["timeout"].default is None, (
        "gateway nonblocking shutdown should resolve the budget lazily "
        f"(default None), got {sig.parameters['timeout'].default!r}"
    )
    src = inspect.getsource(_shutdown_mcp_servers_nonblocking)
    assert "timeout = _MCP_TEARDOWN_BUDGET_SECONDS" in src


def test_orphan_reaper_never_blocks_the_caller(monkeypatch):
    """The reap's SIGTERM -> 2s -> SIGKILL dance must not sit on the exit path."""
    started = threading.Event()
    release = threading.Event()

    def _slow_reap(include_active):
        started.set()
        release.wait(timeout=10)

    monkeypatch.setattr(_mcp_lifecycle, "_kill_orphaned_mcp_children", _slow_reap)
    try:
        start = time.monotonic()
        _mcp_loop._start_orphan_reaper()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"_start_orphan_reaper blocked for {elapsed:.2f}s"
        assert started.wait(timeout=2), "the reaper thread never ran"
    finally:
        release.set()


def test_teardown_segments_thread_the_shared_budget():
    """Source-level guard: no teardown segment may escape the one budget.

    The funnel spans two modules (lifecycle = segment 1, loop = segments 2-4), so
    a caller-visible signature change is the cheapest place to catch a regression
    of the arithmetic itself.
    """
    lifecycle_src = inspect.getsource(_mcp_lifecycle)
    loop_src = inspect.getsource(_mcp_loop)

    # Segment 1 clamps its wait and hands the remainder on.
    assert "_MCP_SHUTDOWN_DRAIN_SECONDS, teardown_budget" in lifecycle_src
    assert "future.result(timeout=drain_wait)" in lifecycle_src
    assert "teardown_budget = max(0.0, teardown_budget - drain_wait)" in lifecycle_src
    assert "future.result(timeout=15)" not in lifecycle_src, (
        "a bare 15s server-drain wait would outrun the kill grace"
    )
    assert "_stop_mcp_loop(only_if_idle=scope is not None, teardown_budget=teardown_budget)" in lifecycle_src

    # Segments 2-4 derive from what the budget still leaves.
    assert "_MCP_LOOP_DRAIN_TIMEOUT + 1, teardown_budget" in loop_src
    assert "_teardown_clamp(5.0, teardown_budget)" in loop_src
    assert "thread.join(timeout=join_wait)" in loop_src
    assert "_start_orphan_reaper()" in loop_src
    stop_src = inspect.getsource(_mcp_loop._stop_mcp_loop)
    assert "_kill_orphaned_mcp_children" not in stop_src, (
        "the orphan reap must not run inline on the exit path"
    )
