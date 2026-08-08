"""Tests for the bounded synchronous Relay execution (agent.relay_await).

The sequential tool path has no batch deadline (unlike the concurrent path's
HERMES_CONCURRENT_TOOL_TIMEOUT_S), so a tool whose awaitable never resolves
used to wedge the conversation turn forever (#79568). ``relay_await`` now
bounds every synchronous Relay execution with a configurable cooperative
ceiling plus a hard thread-abandon deadline, and both Relay adapters
(``relay_tools`` for tools, ``relay_llm`` for managed LLM calls) route their
``_run_awaitable`` through it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent import relay_await, relay_llm, relay_tools


# ---------------------------------------------------------------------------
# tool_execution_ceiling_seconds
# ---------------------------------------------------------------------------


def test_default_ceiling(monkeypatch):
    monkeypatch.delenv("HERMES_TOOL_EXECUTION_CEILING_S", raising=False)
    assert (
        relay_await.tool_execution_ceiling_seconds()
        == relay_await._DEFAULT_TOOL_EXECUTION_CEILING_S
    )


def test_invalid_ceiling_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "not-a-number")
    assert (
        relay_await.tool_execution_ceiling_seconds()
        == relay_await._DEFAULT_TOOL_EXECUTION_CEILING_S
    )


@pytest.mark.parametrize("raw", ["0", "-1", "-0.5", "nan"])
def test_non_positive_and_nan_disable(monkeypatch, raw):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", raw)
    assert relay_await.tool_execution_ceiling_seconds() is None


# ---------------------------------------------------------------------------
# run_awaitable — cooperative ceiling (layer 1)
# ---------------------------------------------------------------------------


def test_completed_awaitable_returns_value(monkeypatch):
    monkeypatch.delenv("HERMES_TOOL_EXECUTION_CEILING_S", raising=False)

    async def _quick():
        return "done"

    assert relay_tools._run_awaitable(_quick()) == "done"


def test_non_awaitable_passes_through():
    assert relay_tools._run_awaitable("plain") == "plain"


def test_wedged_awaitable_raises_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")

    async def _never():
        await asyncio.Event().wait()

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        relay_tools._run_awaitable(_never())
    # Cooperative wedge is cancelled at the ceiling, well before the 3x
    # hard-abandon deadline.
    assert time.monotonic() - start < 0.5


def test_ceiling_zero_disables_bound(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0")

    def _no_wait_for(*args, **kwargs):  # pragma: no cover - failure path
        pytest.fail("wait_for must not be used when the ceiling is disabled")

    monkeypatch.setattr(relay_await.asyncio, "wait_for", _no_wait_for)

    async def _quick():
        return 42

    assert relay_tools._run_awaitable(_quick()) == 42


def test_raises_on_event_loop_thread():
    async def _driver():
        async def _payload():
            return "unused"

        payload = _payload()
        try:
            with pytest.raises(RuntimeError, match="event-loop thread"):
                relay_tools._run_awaitable(payload)
        finally:
            payload.close()

    asyncio.run(_driver())


def test_slow_sync_block_past_ceiling_still_returns_result(monkeypatch):
    """A tool that synchronously blocks past the ceiling must NOT be killed.

    The cooperative timer is starved while the loop is blocked; the worker is
    only abandoned at the hard deadline (3x ceiling). A legitimately slow
    synchronous tool that finishes within that window keeps its result.
    """
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")

    async def _sync_block():
        time.sleep(0.4)  # blocks the worker's loop well past the 0.2s ceiling
        return "slow-but-done"

    assert relay_tools._run_awaitable(_sync_block()) == "slow-but-done"


# ---------------------------------------------------------------------------
# run_awaitable — hard abandon (layer 2)
# ---------------------------------------------------------------------------


def test_cancel_swallowing_wedge_is_abandoned(monkeypatch):
    """A wedge that swallows CancelledError defeats wait_for (it joins the
    cancellation); the hard deadline must abandon the worker thread and raise.
    """
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")

    async def _swallows_cancel():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                continue  # refuses to die

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="worker thread abandoned"):
        relay_tools._run_awaitable(_swallows_cancel())
    elapsed = time.monotonic() - start
    # Fired at the hard deadline (0.6s = 0.2 * 3), not never.
    assert 0.5 < elapsed < 2.0


def test_exception_from_awaitable_propagates(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "1")

    async def _boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        relay_tools._run_awaitable(_boom())


# ---------------------------------------------------------------------------
# relay_tools.execute — TimeoutError interaction with the dispatch fallback
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal managed-execution runtime driving execute()'s real code path."""

    def __init__(self, coro_factory):
        self._coro_factory = coro_factory
        # execute() dereferences runtime.relay.tools.execute before dispatch.
        tools_ns = type("_ToolsNS", (), {"execute": staticmethod(lambda *a, **k: None)})
        self.relay = type("_RelayNS", (), {"tools": tools_ns})

    def managed_execution_enabled(self):
        return True

    def run_in_session_async(self, session, relay_execute, tool_name, args, invoke,
                             **kwargs):
        return self._coro_factory(invoke, args)


class _FakeSession:
    handle = object()


def _patch_context(monkeypatch, runtime):
    monkeypatch.setattr(
        relay_tools.relay_runtime,
        "resolve_execution_context",
        lambda session_id: (runtime, _FakeSession(), None),
    )


def test_execute_post_dispatch_timeout_returns_tool_result(monkeypatch):
    """The safety contract for slow-but-successful tools: once the callback
    has produced a result, a late TimeoutError from relay post-processing is
    absorbed and the real tool result is returned (never discarded).
    """
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")

    async def _dispatch_then_wedge(invoke, args):
        invoke(args)  # tool runs and populates raw_result
        await asyncio.Event().wait()  # relay post-processing wedges

    _patch_context(monkeypatch, _FakeRuntime(_dispatch_then_wedge))

    result, final_args = relay_tools.execute(
        "mytool", {"a": 1}, lambda args: {"ok": True, "args": args},
        session_id="s1",
    )
    assert result == {"ok": True, "args": {"a": 1}}
    assert final_args == {"a": 1}


def test_execute_pre_dispatch_timeout_propagates(monkeypatch):
    """A wedge before the tool ever ran has no result to salvage — the
    TimeoutError must surface instead of hanging the turn forever."""
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")

    async def _wedge_before_dispatch(invoke, args):
        await asyncio.Event().wait()

    _patch_context(monkeypatch, _FakeRuntime(_wedge_before_dispatch))

    with pytest.raises(TimeoutError):
        relay_tools.execute(
            "mytool", {"a": 1}, lambda args: {"ok": True}, session_id="s1"
        )


# ---------------------------------------------------------------------------
# relay_llm sibling — same funnel, same bound
# ---------------------------------------------------------------------------


def test_relay_llm_run_awaitable_is_bounded(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0.2")
    # Neutralize the LLM floor so the test doesn't wait 1800s.
    monkeypatch.setattr(relay_await, "LLM_HARD_DEADLINE_FLOOR_S", 0.0)

    async def _never():
        await asyncio.Event().wait()

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        relay_llm._run_awaitable(_never())
    assert time.monotonic() - start < 0.5


def test_relay_llm_hard_deadline_floor_applied(monkeypatch):
    """The LLM twin's hard deadline honors the floor for legitimately long
    synchronous provider callbacks (codex hard timeout is 1500s)."""
    captured = {}

    def _fake_run_bounded(value, ceiling, hard_deadline):
        captured["hard_deadline"] = hard_deadline
        value.close()
        return "ok"

    monkeypatch.delenv("HERMES_TOOL_EXECUTION_CEILING_S", raising=False)
    monkeypatch.setattr(relay_await, "_run_bounded", _fake_run_bounded)

    async def _quick():
        return "unused"

    assert relay_llm._run_awaitable(_quick()) == "ok"
    assert captured["hard_deadline"] == relay_await.LLM_HARD_DEADLINE_FLOOR_S
