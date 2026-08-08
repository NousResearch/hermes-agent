"""Tests for the synchronous tool-execution ceiling in relay_tools._run_awaitable.

The sequential tool path has no batch deadline (unlike the concurrent path's
HERMES_CONCURRENT_TOOL_TIMEOUT_S), so a tool whose awaitable never resolves
used to wedge the conversation turn forever. _run_awaitable now bounds every
synchronous execution with a configurable ceiling.
"""

from __future__ import annotations

import asyncio

import pytest

from agent import relay_tools


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

    with pytest.raises(TimeoutError):
        relay_tools._run_awaitable(_never())


def test_ceiling_zero_disables_bound(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "0")

    async def _quick():
        return 42

    # With the ceiling disabled the awaitable still runs to completion.
    assert relay_tools._run_awaitable(_quick()) == 42


def test_invalid_ceiling_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HERMES_TOOL_EXECUTION_CEILING_S", "not-a-number")
    assert relay_tools._tool_execution_ceiling_seconds() == 420.0
