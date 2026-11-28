"""A progress surface that claims a running turn must not silence it forever.

``long_running_heartbeat_claim`` lets a surface that renders the turn itself
(a streaming card) suppress the duplicate text heartbeat. That claim is a
freshness promise, not a mute: once the agent has been inactive for a full
notify interval the heartbeat must go out anyway (with copy that names the
stale card), or the chat shows no sign of life at all.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _runner(seconds_since_activity):
    """A partially constructed GatewayRunner, as the neighbouring heartbeat tests build it."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False
    runner._agent_activity_summary = staticmethod(lambda agent: {
        "api_call_count": 2, "max_iterations": 10, "current_tool": "terminal",
        "seconds_since_activity": seconds_since_activity,
    })
    return runner


def _display():
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = True
    return disp


def _ctx(agent):
    from gateway.turn_context import TurnContext

    ctx = TurnContext(source=SimpleNamespace(chat_id="c", platform="telegram"), session_key="sess")
    ctx.agent_holder[0] = agent
    return ctx


@pytest.mark.asyncio
async def test_silent_claimed_turn_still_sends_a_heartbeat(monkeypatch):
    """Claim + a full interval of silence -> the text heartbeat overrides the claim."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    monkeypatch.setattr("gateway.run_turn.t", lambda key, **kw: f"{key}|{kw.get('idle_minutes')}")

    runner = _runner(seconds_since_activity=900.0)  # 15 min with no tool/API activity
    adapter = MagicMock()
    sent: list[str] = []

    async def _send(chat_id, text, **_kw):
        sent.append(text)
        runner._restart_requested = True  # end the loop after the first heartbeat
        return SimpleNamespace(success=True, message_id="hb-1")

    adapter.send = AsyncMock(side_effect=_send)
    adapter.edit_message = AsyncMock()
    runner._delivery_adapter_for = lambda source: adapter
    runner._progress_surface_claims_heartbeat = AsyncMock(return_value=True)
    agent = MagicMock()
    runner._running_agents["sess"] = agent

    await asyncio.wait_for(
        runner._run_agent_notify_long_running(_display(), _ctx(agent), [None]), 5
    )

    adapter.send.assert_awaited_once()
    assert sent == ["gateway.progress.working_heartbeat_stale_surface|15"]


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds_since_activity", [0.0, None])
async def test_claimed_turn_with_fresh_or_unknown_activity_stays_suppressed(
    monkeypatch, seconds_since_activity,
):
    """Fresh activity (and unknown activity) keep the claim: only *observed* silence overrides.

    Unknown idle time is an observation gap, not a stall -- the same rule
    ``gateway/session_stall.py`` applies.
    """
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")

    runner = _runner(seconds_since_activity)
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    adapter.edit_message = AsyncMock()
    runner._delivery_adapter_for = lambda source: adapter
    agent = MagicMock()
    runner._running_agents["sess"] = agent

    calls = {"n": 0}

    async def _claim(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] >= 2:  # let the loop run a full tick, then end it
            runner._restart_requested = True
        return True

    runner._progress_surface_claims_heartbeat = _claim

    await asyncio.wait_for(
        runner._run_agent_notify_long_running(_display(), _ctx(agent), [None]), 5
    )

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_unclaimed_turn_is_never_asked_to_override(monkeypatch):
    """No claim -> the hook result is False -> the ordinary heartbeat goes out unchanged."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    seen: list[str] = []
    monkeypatch.setattr("gateway.run_turn.t", lambda key, **kw: seen.append(key) or key)

    runner = _runner(seconds_since_activity=900.0)
    adapter = MagicMock()

    async def _send(chat_id, text, **_kw):
        runner._restart_requested = True
        return SimpleNamespace(success=True, message_id="hb-1")

    adapter.send = AsyncMock(side_effect=_send)
    adapter.edit_message = AsyncMock()
    runner._delivery_adapter_for = lambda source: adapter
    runner._progress_surface_claims_heartbeat = AsyncMock(return_value=False)
    agent = MagicMock()
    runner._running_agents["sess"] = agent

    await asyncio.wait_for(
        runner._run_agent_notify_long_running(_display(), _ctx(agent), [None]), 5
    )

    adapter.send.assert_awaited_once()
    assert seen == ["gateway.progress.working_heartbeat"]


@pytest.mark.asyncio
async def test_claim_hook_returns_true_only_for_an_affirmative_result(monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli import plugins

    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(plugins, "has_hook", lambda name: name == "long_running_heartbeat_claim")

    monkeypatch.setattr(plugins, "ainvoke_hook", AsyncMock(return_value=[None, False, True]))
    assert await runner._progress_surface_claims_heartbeat(
        SimpleNamespace(chat_id="c"), "sess", None, 10.0, 180.0
    ) is True

    monkeypatch.setattr(plugins, "ainvoke_hook", AsyncMock(return_value=[None, False]))
    assert await runner._progress_surface_claims_heartbeat(
        SimpleNamespace(chat_id="c"), "sess", None, 10.0, 180.0
    ) is False


@pytest.mark.asyncio
async def test_claim_hook_failure_fails_open(monkeypatch):
    """A crashing claim must never suppress the heartbeat."""
    from gateway.run import GatewayRunner
    from hermes_cli import plugins

    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)
    monkeypatch.setattr(plugins, "ainvoke_hook", AsyncMock(side_effect=RuntimeError("boom")))

    assert await runner._progress_surface_claims_heartbeat(
        SimpleNamespace(chat_id="c"), "sess", None, 10.0, 180.0
    ) is False
