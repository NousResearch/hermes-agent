"""A prompt raised after the platform adapter was rebuilt goes through the live replacement.

The reconnect watcher replaces a platform's adapter instance while a turn (or an update watch)
is still running. The retired instance has no bot: an approval card, clarify question or update
prompt sent through the adapter captured when the turn started fails "Not connected", and the
gated command waits out its whole approval timeout with no card in the chat (#138133).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

# Module level on purpose: gateway.run's import side effects stat the checkout's git dir, which the
# fixture-scoped home-I/O guard would refuse inside a test.
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

SESSION = "agent:main:telegram:dm:111"


class _Adapter:
    """Records every prompt; a retired instance answers like a bot-less Telegram adapter."""

    typed_command_prefix = "/"

    def __init__(self, name: str, connected: bool = True) -> None:
        self.name, self.connected, self.calls = name, connected, []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    def _result(self, what: str) -> SendResult:
        self.calls.append(what)
        return SendResult(success=True, message_id=f"{self.name}-1") if self.connected else SendResult(
            success=False, error="Not connected")

    async def send_exec_approval(self, **_kw: Any) -> SendResult:
        return self._result("card")

    async def send_update_prompt(self, **_kw: Any) -> SendResult:
        return self._result("update_prompt")

    async def send(self, chat_id: str, text: str, **_kw: Any) -> SendResult:
        return self._result("text")


def _bare_runner(adapter) -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = None
    runner.hooks = None
    return runner


def test_turn_approval_card_goes_through_the_replacement_adapter(tmp_path, monkeypatch):
    """Wire the turn the way ``_run_agent_inner`` does, rebuild the adapter, then raise an approval."""
    (tmp_path / "config.yaml").write_text("display: {}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    retired, live = _Adapter("old"), _Adapter("new")
    runner = _bare_runner(retired)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="111")
    ctx = TurnContext(source=source, session_key=SESSION, user_config={}, _run_still_current=lambda: True)
    turn = object.__new__(TurnRunner)
    turn._runner, turn._ctx = runner, ctx

    async def wire_then_rebuild():
        runner._run_agent_bind_turn_wiring(ctx, turn, source, None, False)
        retired.connected = False  # the reconnect watcher retires it and publishes a new instance
        runner.adapters[Platform.TELEGRAM] = live

    asyncio.run(wire_then_rebuild())
    turn._schedule = lambda coro, _label: SimpleNamespace(result=lambda timeout=None, _r=asyncio.run(coro): _r)
    turn._close_native_stream_boundary = lambda _why: None

    turn._approval_notify_sync({"command": "rm -rf /x", "description": "delete", "request_id": "req-1"})

    assert live.calls == ["card"], "the approval card must reach the live adapter"
    assert retired.calls == [], "nothing may go through the retired, bot-less adapter"


def test_update_prompt_after_reconnect_reaches_the_replacement(tmp_path):
    """An update watch outlives a reconnect: its prompt must be sent by the adapter registered NOW."""
    retired, live = _Adapter("old"), _Adapter("new")
    runner = _bare_runner(retired)
    runner._session_state = lambda key: SimpleNamespace(persistent=SimpleNamespace())
    (tmp_path / ".update_pending.json").write_text(json.dumps(
        {"platform": "telegram", "chat_id": "111", "user_id": "111", "session_key": SESSION}))

    async def scenario():
        with patch("gateway.run._hermes_home", tmp_path):
            target = runner._resolve_update_target(runner._update_paths())
            retired.connected = False
            runner.adapters[Platform.TELEGRAM] = live
            await runner._forward_update_prompt(target, "Restore local changes?", "y")
            await target.send("update finished")

    asyncio.run(scenario())
    assert live.calls == ["update_prompt", "text"]
    assert retired.calls == []


def test_long_running_heartbeat_after_reconnect_reaches_the_replacement(monkeypatch):
    """The "still working" heartbeat loop outlives a reconnect: each tick sends through the adapter
    registered NOW, not the bot-less instance the turn started with."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    retired, live = _Adapter("old"), _Adapter("new")
    runner = _bare_runner(retired)
    runner._draining = runner._restart_requested = False
    runner._agent_activity_summary = staticmethod(lambda agent: None)
    agent = object()
    runner._running_agents = {SESSION: agent}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="111")
    ctx = TurnContext(source=source, session_key=SESSION, user_config={})
    ctx.agent_holder[0] = agent
    disp = SimpleNamespace(_display_surface_mode=lambda *a, **k: "on",
                           resolve_display_setting=lambda *a, **k: False, user_config={}, platform_key="telegram")
    for adapter in (retired, live):
        async def send_then_stop(chat_id, text, _send=adapter.send, **kw):
            runner._restart_requested = True  # one heartbeat is enough, whichever adapter sent it
            return await _send(chat_id, text, **kw)

        adapter.send = send_then_stop

    async def scenario():
        task = asyncio.ensure_future(runner._run_agent_notify_long_running(disp, ctx, [None]))
        await asyncio.sleep(0)  # the loop resolved its adapter and is sleeping out the first interval
        retired.connected = False
        runner.adapters[Platform.TELEGRAM] = live
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert live.calls == ["text"], "the heartbeat must reach the live adapter"
    assert retired.calls == []
