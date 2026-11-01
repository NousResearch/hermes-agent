"""Regression tests for bounded synchronous Relay awaitables."""

import asyncio

import pytest

from agent import relay_llm


def test_run_awaitable_enforces_timeout():
    async def wedged():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        relay_llm._run_awaitable(wedged(), timeout=0.01)


def test_run_awaitable_keeps_unlimited_mode():
    async def completed():
        return "ok"

    assert relay_llm._run_awaitable(completed(), timeout=0) == "ok"


def test_tool_funnel_cooperative_wait_uses_configured_deadline(monkeypatch):
    from agent import relay_tools, tool_executor
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.01)
    cancelled = []

    async def wedged():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    with pytest.raises(TimeoutError):
        relay_tools._run_awaitable(wedged())
    assert cancelled == [True]


def test_tool_funnel_does_not_preempt_sync_callback(monkeypatch):
    import time
    from agent import relay_tools, tool_executor
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.01)
    finished = []

    def callback():
        time.sleep(0.05)
        finished.append(True)
        return "completed"

    async def invoke():
        return callback()

    started = time.monotonic()
    assert relay_tools._run_awaitable(invoke()) == "completed"
    assert time.monotonic() - started >= 0.05
    assert finished == [True]
