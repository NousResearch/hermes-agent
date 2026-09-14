"""Tests for the fail-closed pre-agent gateway decision boundary."""

import pytest

from gateway.hooks import HookRegistry
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_agent_before_gate_allows_when_no_hook_returns_decision():
    runner = object.__new__(GatewayRunner)
    runner.hooks = HookRegistry()
    assert await runner._evaluate_agent_before_gate({"message": "hello"}) is None


@pytest.mark.asyncio
async def test_agent_before_gate_denies_with_hook_message():
    runner = object.__new__(GatewayRunner)
    runner.hooks = HookRegistry()
    runner.hooks._handlers["agent:before"] = [
        lambda _event, _context: {
            "decision": "deny",
            "message": "state conflict",
        }
    ]

    assert await runner._evaluate_agent_before_gate({"message": "hello"}) == "state conflict"


@pytest.mark.asyncio
async def test_agent_before_gate_fails_closed_on_hook_exception():
    runner = object.__new__(GatewayRunner)
    runner.hooks = HookRegistry()

    def broken(_event, _context):
        raise RuntimeError("runtime unavailable")

    runner.hooks._handlers["agent:before"] = [broken]

    response = await runner._evaluate_agent_before_gate({"message": "hello"})
    assert response is not None
    assert "状態検証に失敗" in response


@pytest.mark.asyncio
async def test_strict_collection_preserves_handler_order():
    registry = HookRegistry()
    registry._handlers["agent:before"] = [
        lambda _event, _context: {"decision": "allow"},
        lambda _event, _context: {"decision": "deny"},
    ]

    assert await registry.emit_collect_strict("agent:before", {}) == [
        {"decision": "allow"},
        {"decision": "deny"},
    ]

