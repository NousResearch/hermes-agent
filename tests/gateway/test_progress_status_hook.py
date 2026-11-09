"""gateway_progress_status: a plugin phrases the long-running heartbeat, phase changes are edited into the same
bubble (only when they change), and the bubble is settled to the turn's outcome when the turn ends instead of being
left reading "Working" after a failure, an interrupt or a restart."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False
    runner._delivery_adapter_for = lambda source: adapter
    runner._agent_activity_summary = staticmethod(lambda agent: None)
    return runner


def _ctx(agent):
    ctx = TurnContext(source=SimpleNamespace(chat_id="c", platform="telegram"), session_key="sess", session_id="s-1")
    ctx.agent_holder[0] = agent
    return ctx


def _disp(mode="on"):
    disp = MagicMock()
    disp._display_surface_mode.return_value = mode
    disp.resolve_display_setting.return_value = False
    return disp


def _adapter():
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    return adapter


async def _run_heartbeats(runner, ctx, agent, disp, ticks, monkeypatch, phases=None):
    """Drive the heartbeat for ``ticks`` sleeps, then end the run (agent leaves the slot)."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "100")
    monkeypatch.setattr("gateway.run_turn._PHASE_RECHECK", 0.0)
    runner._running_agents["sess"] = agent
    n = {"i": 0}
    real_sleep = asyncio.sleep

    async def fake_sleep(_s):
        n["i"] += 1
        if phases:
            phases["i"] = n["i"]
        if n["i"] > ticks:
            runner._running_agents["sess"] = None
        await real_sleep(0)

    monkeypatch.setattr("gateway.run_turn.asyncio.sleep", fake_sleep)
    await asyncio.wait_for(runner._run_agent_notify_long_running(disp, ctx, [None]), 5)


@pytest.mark.asyncio
async def test_plugin_text_is_sent_once_then_edited_only_on_change(monkeypatch):
    adapter, agent = _adapter(), MagicMock(session_id="s-live")
    runner, ctx = _runner(adapter), _ctx(agent)
    seen, state = [], {"i": 0}
    keys = {1: "a", 2: "a", 3: "a", 4: "b", 5: "b"}

    def hook(name, **kw):
        assert name == "gateway_progress_status"
        seen.append(kw)
        k = keys.get(state["i"], "b")
        return [{"text": f"⏳ Working\nNow: {k}", "change_key": k}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    await _run_heartbeats(runner, ctx, agent, _disp(), 5, monkeypatch, phases=state)

    assert adapter.send.await_count == 1                       # one bubble
    edits = [c.args[2] for c in adapter.edit_message.await_args_list]
    assert edits == ["⏳ Working\nNow: b"]                      # unchanged phases cost no edit
    assert ctx._heartbeat_msg_id == "hb-1"
    assert seen[0]["session_id"] == "s-live" and seen[0]["state"] == "running" and seen[0]["message_id"] is None
    assert any(k["message_id"] == "hb-1" for k in seen)        # the plugin learns its bubble


@pytest.mark.asyncio
async def test_without_a_plugin_the_heartbeat_is_unchanged(monkeypatch):
    adapter, agent = _adapter(), MagicMock()
    runner, ctx = _runner(adapter), _ctx(agent)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: [])
    await _run_heartbeats(runner, ctx, agent, _disp(), 1, monkeypatch)
    assert adapter.send.await_args.args[1].startswith("⏳ Working — ")


@pytest.mark.asyncio
async def test_generic_mode_never_asks_plugins(monkeypatch):
    adapter, agent = _adapter(), MagicMock()
    runner, ctx = _runner(adapter), _ctx(agent)
    called = []
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: called.append(name) or ["x"])
    disp = _disp("generic")
    disp._generic_status_phrase.return_value = "still on it"
    await _run_heartbeats(runner, ctx, agent, disp, 1, monkeypatch)
    assert not called and adapter.send.await_args.args[1] == "still on it"


@pytest.mark.asyncio
async def test_a_failing_plugin_falls_back(monkeypatch):
    adapter, agent = _adapter(), MagicMock()
    runner, ctx = _runner(adapter), _ctx(agent)

    def boom(name, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", boom)
    await _run_heartbeats(runner, ctx, agent, _disp(), 1, monkeypatch)
    assert adapter.send.await_args.args[1].startswith("⏳ Working — ")


@pytest.mark.parametrize("result,flags,want", [
    ({"final_response": "hi", "completed": True}, {}, "completed"),
    ({"final_response": "", "interrupted": True}, {}, "interrupted"),
    ({"failed": True, "error": "x"}, {}, "failed"),
    (None, {}, "failed"),
    ({"final_response": "hi"}, {"_restart_requested": True}, "restarted"),
    ({"final_response": "hi"}, {"_draining": True}, "restarted"),
])
@pytest.mark.asyncio
async def test_settle_edits_the_bubble_to_the_outcome(monkeypatch, result, flags, want):
    adapter, agent = _adapter(), MagicMock(session_id="s")
    runner, ctx = _runner(adapter), _ctx(agent)
    for k, v in flags.items():
        setattr(runner, k, v)
    ctx._heartbeat_msg_id, ctx._heartbeat_started_at = "hb-9", 0.0
    ctx.result_holder[0] = result
    got = []
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: got.append(kw) or [])
    await runner._run_agent_settle_heartbeat(ctx)
    assert got[0]["state"] == "ended" and got[0]["outcome"] == want and got[0]["message_id"] == "hb-9"
    chat, mid, text = adapter.edit_message.await_args.args
    assert (chat, mid) == ("c", "hb-9") and "Working" not in text and "⏳" not in text


@pytest.mark.asyncio
async def test_settle_uses_the_plugin_text_and_survives_edit_failure(monkeypatch):
    adapter, agent = _adapter(), MagicMock()
    adapter.edit_message = AsyncMock(side_effect=RuntimeError("message to edit not found"))
    runner, ctx = _runner(adapter), _ctx(agent)
    ctx._heartbeat_msg_id = "hb-2"
    ctx.result_holder[0] = {"interrupted": True}
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: ["⏹ Interrupted after 4 min."])
    await runner._run_agent_settle_heartbeat(ctx)                  # no raise
    assert adapter.edit_message.await_args.args[2] == "⏹ Interrupted after 4 min."


@pytest.mark.asyncio
async def test_no_bubble_no_settle(monkeypatch):
    adapter = _adapter()
    runner, ctx = _runner(adapter), _ctx(MagicMock())
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: pytest.fail("not called"))
    await runner._run_agent_settle_heartbeat(ctx)
    adapter.edit_message.assert_not_awaited()


def test_hook_is_registered():
    from hermes_cli.plugins import VALID_HOOKS
    assert "gateway_progress_status" in VALID_HOOKS
