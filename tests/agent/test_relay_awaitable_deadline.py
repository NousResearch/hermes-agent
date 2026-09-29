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
