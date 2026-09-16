"""Tests that the ``agent_loop_stopped`` plugin hook fires when the gateway
interrupts a running agent turn — both via ``/stop`` and via the running-agent
fast-path inside ``/new``.

Mirrors ``test_session_boundary_hooks.py``: we bypass ``GatewayRunner.__init__``
and stub just enough attributes to drive ``_interrupt_and_clear_session``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key
from hermes_cli.plugins import VALID_HOOKS


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    # SimpleNamespace, not MagicMock — _interrupt_and_clear_session calls
    # adapter.interrupt_session_activity() only when hasattr(...) is true,
    # and MagicMock fakes every attribute, which would push us into an
    # await on a non-awaitable.
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {}
    runner._pending_messages = {}

    # _invalidate_session_run_generation + _release_running_agent_state are
    # called downstream; stub to no-op so we exercise the hook emit only.
    runner._invalidate_session_run_generation = lambda *a, **kw: None
    runner._release_running_agent_state = lambda *a, **kw: None
    return runner


def test_agent_loop_stopped_in_valid_hooks():
    """Plugins must be able to subscribe to the new hook without warnings."""
    assert "agent_loop_stopped" in VALID_HOOKS


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_interrupt_fires_agent_loop_stopped_hook(mock_invoke_hook):
    """Calling ``_interrupt_and_clear_session`` with a real running agent
    must interrupt it AND dispatch the hook with the session, platform,
    and reasons attached."""
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)

    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    await runner._interrupt_and_clear_session(
        session_key,
        source,
        interrupt_reason="user_stop",
        invalidation_reason="stop_command",
    )

    running_agent.interrupt.assert_called_once_with("user_stop")
    mock_invoke_hook.assert_any_call(
        "agent_loop_stopped",
        session_key=session_key,
        platform="telegram",
        reason="user_stop",
        invalidation_reason="stop_command",
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_hook_not_fired_for_pending_sentinel_stop(mock_invoke_hook):
    """The pending-sentinel /stop path (slash_commands.py) has no in-flight
    work to cancel — the agent loop never started. The hook is gated on a
    real running agent and must not fire in this case."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)

    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

    await runner._interrupt_and_clear_session(
        session_key,
        source,
        interrupt_reason="user_stop",
        invalidation_reason="stop_command_pending",
    )

    agent_loop_stopped_calls = [
        call for call in mock_invoke_hook.call_args_list
        if call.args and call.args[0] == "agent_loop_stopped"
    ]
    assert agent_loop_stopped_calls == [], (
        f"agent_loop_stopped must not fire on the pending-sentinel path "
        f"(saw {len(agent_loop_stopped_calls)} call(s))"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_hook_not_fired_when_no_running_agent(mock_invoke_hook):
    """If there is no agent at all for the session key (already cleared),
    there is nothing to interrupt and the hook must not fire."""
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)

    # _running_agents stays empty — no entry for this session_key.
    await runner._interrupt_and_clear_session(
        session_key,
        source,
        interrupt_reason="user_stop",
        invalidation_reason="stop_command_no_agent",
    )

    agent_loop_stopped_calls = [
        call for call in mock_invoke_hook.call_args_list
        if call.args and call.args[0] == "agent_loop_stopped"
    ]
    assert agent_loop_stopped_calls == []


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_hook_dispatch_failure_does_not_break_interrupt(mock_invoke_hook):
    """A misbehaving plugin must not prevent the interrupt from completing."""
    mock_invoke_hook.side_effect = RuntimeError("plugin exploded")

    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)

    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    # Should not raise — the hook block has try/except for exactly this.
    await runner._interrupt_and_clear_session(
        session_key,
        source,
        interrupt_reason="user_stop",
        invalidation_reason="stop_command",
    )

    # The interrupt itself still happened despite the hook blowing up.
    running_agent.interrupt.assert_called_once_with("user_stop")
