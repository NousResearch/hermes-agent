"""``gateway_internal_admission``: plugins decide whether an automatic event may reach the model.

* user (non-internal) events never consult the hook;
* no ``block`` result admits the event; a blocked event's claimed ``/loop`` tick is released;
* an automatic event dequeued behind a busy turn is re-checked before its follow-up runs;
* ``block`` stops the turn, sends ``response`` when given, and fires ``phase="delivered"``
  with the receipt only after a successful send;
* a raising callback or a failing hook call blocks (fail closed);
* slash-command events are not gated;
* the hook is registered, timeout-bounded and fail-closed.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

import hermes_cli.lifecycle as lifecycle  # noqa: E402
from gateway.platforms.base import SessionSource  # noqa: E402
from gateway.platforms.event import MessageEvent, MessageType  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
from hermes_cli import plugins_dispatch  # noqa: E402
from hermes_cli.plugins import VALID_HOOKS, PluginManager  # noqa: E402


def _event(internal: bool = True) -> MessageEvent:
    source = SessionSource(platform=MagicMock(value="telegram"), chat_id="123",
                           chat_type="private", user_id="user1")
    return MessageEvent(text="[background process finished]", message_type=MessageType.TEXT,
                        source=source, message_id="m1", internal=internal)


def _runner(send_ok: bool = True):
    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=send_ok))
    runner._delivery_adapter_for = lambda _source: adapter
    runner._thread_metadata_for_target = lambda *a, **k: None
    return runner, adapter


def _hook(monkeypatch, admit_result=None, raises=None):
    calls = []

    async def fake_invoke(name, **kwargs):
        assert name == "gateway_internal_admission"
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return [admit_result] if kwargs["phase"] == "admit" and admit_result is not None else []

    monkeypatch.setattr(lifecycle, "ainvoke_hook", fake_invoke)
    return calls


@pytest.mark.asyncio
async def test_user_events_never_consult_the_hook(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, _ = _runner()
    assert await runner._admit_internal_event(_event(internal=False), "sk") is True
    assert calls == []


@pytest.mark.asyncio
async def test_no_block_admits(monkeypatch):
    calls = _hook(monkeypatch, {"action": "allow"})
    runner, adapter = _runner()
    assert await runner._admit_internal_event(_event(), "sk") is True
    assert [c["phase"] for c in calls] == ["admit"]
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_block_sends_notice_and_acknowledges_delivery(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block", "response": "nothing new", "receipt": {"id": 7}})
    runner, adapter = _runner()
    assert await runner._admit_internal_event(_event(), "sk") is False
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("123", "nothing new")
    assert adapter.send.await_args.kwargs["metadata"]["_interim_send"] is True  # never seals a live stream
    assert [c["phase"] for c in calls] == ["admit", "delivered"]
    assert calls[1]["receipt"] == {"id": 7} and calls[1]["session_key"] == "sk"


@pytest.mark.asyncio
async def test_failed_send_is_not_acknowledged(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block", "response": "nothing new"})
    runner, _ = _runner(send_ok=False)
    assert await runner._admit_internal_event(_event(), "sk") is False
    assert [c["phase"] for c in calls] == ["admit"]


@pytest.mark.asyncio
async def test_silent_block_sends_nothing(monkeypatch):
    _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    assert await runner._admit_internal_event(_event(), "sk") is False
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_async_callback_runs_on_the_gateway_loop():
    import asyncio
    loop = asyncio.get_running_loop()
    seen = []

    async def admit(phase, **kwargs):
        seen.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return {"action": "block"}

    manager = PluginManager()
    manager._hooks["gateway_internal_admission"] = [admit]
    [result] = await manager.ainvoke_hook("gateway_internal_admission", phase="admit", event=None, session_key="sk")
    assert result == {"action": "block"} and seen == [loop]



@pytest.mark.asyncio
async def test_blocked_loop_tick_is_released(monkeypatch):
    _hook(monkeypatch, {"action": "block"})
    runner, _ = _runner()
    released = []

    async def run_in_executor(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = run_in_executor
    event = _event()
    event._release_on_admission_block = lambda: released.append(True)
    assert await runner._admit_internal_event(event, "sk") is False
    assert released == [True]


@pytest.mark.asyncio
async def test_slash_commands_are_not_gated(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, _ = _runner()
    event = _event()
    event.text = "/recap"
    assert await runner._admit_internal_event(event, "sk") is True
    assert calls == []
    # Command dispatch may rewrite the text into a prompt (/plan, skills); the exemption holds.
    import dataclasses
    rewritten = dataclasses.replace(event, text="Plan the following task: ...")
    assert await runner._admit_internal_event(rewritten, "sk") is True
    assert calls == []



@pytest.mark.asyncio
async def test_busy_queue_keeps_the_command_exemption(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    queued = []
    runner._enqueue_fifo = lambda _key, ev, _adapter: queued.append(ev)
    runner._queue_depth = lambda *_a, **_k: 1
    event = _event()
    event.text = "/queue summarize the run"
    assert await runner._admit_internal_event(event, "sk") is True  # command: exempt on arrival
    await runner._busy_queue_command(event, "sk", event.source)
    [followup] = queued
    assert followup.internal and followup.text == "summarize the run"
    assert await runner._admit_internal_event(followup, "sk") is True
    assert calls == []

@pytest.mark.asyncio
async def test_failed_hook_still_rolls_back_loop_tick(monkeypatch):
    _hook(monkeypatch, raises=RuntimeError("boom"))
    runner, _ = _runner()
    rolled_back = []

    async def run_in_executor(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = run_in_executor
    event = _event()
    event._release_on_admission_block = lambda: rolled_back.append(True)
    assert await runner._admit_internal_event(event, "sk") is False
    assert rolled_back == [True]

@pytest.mark.asyncio
async def test_hook_failure_blocks(monkeypatch):
    _hook(monkeypatch, raises=RuntimeError("boom"))
    runner, _ = _runner()
    assert await runner._admit_internal_event(_event(), "sk") is False



@pytest.mark.asyncio
async def test_queued_followup_is_rechecked_before_it_runs(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._session_key_for_source = lambda _source: "sk"
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock()
    event = _event()
    turn_ctx = SimpleNamespace(source=event.source, session_id="s1", session_key="sk", run_generation=1,
                               _interrupt_depth=0, history=[], _status_thread_metadata=None)
    result = {"messages": []}
    assert await runner._run_agent_queued_followup(
        turn_ctx, adapter, event.text, event, "done", result, None) is result
    assert [c["phase"] for c in calls] == ["admit"]
    runner._prepare_profile_scoped_inbound_message_text.assert_not_called()

@pytest.mark.asyncio
async def test_raising_callback_is_a_block_directive():
    manager = PluginManager()
    manager._hooks["gateway_internal_admission"] = [MagicMock(side_effect=RuntimeError("boom"))]
    [result] = await manager.ainvoke_hook("gateway_internal_admission", phase="admit", event=None, session_key="sk")
    assert result["action"] == "block"


def test_hook_contract():
    assert "gateway_internal_admission" in VALID_HOOKS
    assert "gateway_internal_admission" in plugins_dispatch._HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
    # Not caller-thread: an async callback stays bounded by plugins.hook_callback_timeout.
    assert "gateway_internal_admission" not in plugins_dispatch._HOOK_CALLER_THREAD_HOOKS
