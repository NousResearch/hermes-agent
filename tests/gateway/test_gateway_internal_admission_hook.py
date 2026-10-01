"""``gateway_internal_admission``: plugins decide whether an automatic event may reach the model.

* user (non-internal) events never consult the hook;
* no ``block`` result admits the event;
* ``/loop`` wakeups (marked exempt at injection) are not gated;
* an automatic event dequeued behind a busy turn is re-checked before its follow-up runs;
* ``block`` stops the turn without sending anything (plugins notify via ``gateway`` themselves);
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
from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS, PluginManager  # noqa: E402


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
        return [admit_result] if admit_result is not None else []

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
    assert len(calls) == 1
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_silent_block_sends_nothing(monkeypatch):
    _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    event = _event()
    calls = _hook(monkeypatch, {"action": "block"})
    assert await runner._admit_internal_event(event, "sk") is False
    adapter.send.assert_not_called()
    assert calls[0]["gateway"] is runner and calls[0]["session_key"] == "sk"
    # Consumed, and flagged so _handle_message skips post-turn (/goal, /loop) settling.
    assert event._gateway_accepted is True and event.metadata["internal_admission_blocked"] is True


@pytest.mark.asyncio
async def test_async_callback_runs_on_the_gateway_loop():
    import asyncio
    loop = asyncio.get_running_loop()
    seen = []

    async def admit(**kwargs):
        seen.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return {"action": "block"}

    manager = PluginManager()
    manager._hooks["gateway_internal_admission"] = [admit]
    [result] = await manager.ainvoke_hook("gateway_internal_admission", event=None, session_key="sk", gateway=None)
    assert result == {"action": "block"} and seen == [loop]



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
async def test_loop_wakeups_are_not_gated(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, _ = _runner()
    event = _event()
    event.metadata["internal_admission_exempt"] = True  # set by the /loop watcher at injection
    assert await runner._admit_internal_event(event, "sk") is True
    assert calls == []


@pytest.mark.asyncio
async def test_blocked_wake_never_enters_the_busy_queue(monkeypatch):
    calls = _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    adapter._pending_messages = {"sk": object()}
    queued = []
    runner._queue_or_replace_pending_event = lambda key, ev: queued.append(ev)
    event = _event()
    assert await runner._handle_active_session_busy_message(event, "sk") is True
    assert queued == [] and len(calls) == 1
    assert event._gateway_accepted is True  # consumed: wake producers must not retry it


@pytest.mark.asyncio
async def test_block_promotes_the_next_queued_item(monkeypatch):
    _hook(monkeypatch, {"action": "block"})
    runner, adapter = _runner()
    adapter._pending_messages = {}
    human = MagicMock(name="queued human message")
    overflow = {"sk": [human]}
    runner._overflow_queue = lambda key: overflow.get(key)
    assert await runner._admit_internal_event(_event(), "sk") is False
    assert adapter._pending_messages == {"sk": human} and overflow["sk"] == []


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
    assert len(calls) == 1
    runner._prepare_profile_scoped_inbound_message_text.assert_not_called()

@pytest.mark.asyncio
async def test_raising_callback_is_a_block_directive():
    manager = PluginManager()
    manager._hooks["gateway_internal_admission"] = [MagicMock(side_effect=RuntimeError("boom"))]
    [result] = await manager.ainvoke_hook("gateway_internal_admission", event=None, session_key="sk", gateway=None)
    assert result["action"] == "block"


def test_hook_contract():
    assert "gateway_internal_admission" in VALID_HOOKS
    assert "gateway_internal_admission" in plugins_dispatch._HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
    assert "gateway_internal_admission" in SHELL_UNSUPPORTED_HOOKS  # shell hooks cannot express a block
    # Not caller-thread: an async callback stays bounded by plugins.hook_callback_timeout.
    assert "gateway_internal_admission" not in plugins_dispatch._HOOK_CALLER_THREAD_HOOKS


class _NonPushAdapter:
    """api_server shape: no handle_message lane, the wake self-posts or runs in-process."""
    supports_async_delivery = False

    def __init__(self, runner=None):
        self.gateway_runner = runner
        self.turns = []

    async def handle_message(self, event):  # pragma: no cover - must not be reached
        raise AssertionError("non-push wakes never reach handle_message")

    async def run_internal_session_turn(self, **kwargs):
        self.turns.append(kwargs)


@pytest.mark.asyncio
async def test_non_push_wake_block_runs_no_turn(monkeypatch):
    from gateway.config import Platform
    from gateway.wake import deliver_wake
    calls = _hook(monkeypatch, {"action": "block"})
    adapter = _NonPushAdapter()
    await deliver_wake(adapter, text="[background process finished]", session_id="raw-1", profile="p1")
    assert adapter.turns == []
    [call] = calls
    assert call["session_key"] == "raw-1"
    assert call["event"].internal is True
    assert call["event"].source.platform == Platform.API_SERVER


@pytest.mark.asyncio
async def test_non_push_wake_blocked_before_http_self_post(monkeypatch):
    import gateway.wake as wake
    _hook(monkeypatch, {"action": "block"})
    posted = AsyncMock()
    monkeypatch.setattr(wake, "_self_post_chat_completion", posted)
    await wake.deliver_wake(_NonPushAdapter(), text="done", session_id="raw-1")
    posted.assert_not_called()


@pytest.mark.asyncio
async def test_non_push_wake_gates_slash_text(monkeypatch):
    # This lane never dispatches commands: "/recap" would reach the model as plain text.
    from gateway.wake import deliver_wake
    calls = _hook(monkeypatch, {"action": "block"})
    adapter = _NonPushAdapter()
    await deliver_wake(adapter, text="/recap", session_id="raw-1", profile="p1")
    assert adapter.turns == [] and len(calls) == 1


@pytest.mark.asyncio
async def test_non_push_wake_hook_runs_in_the_routed_profile_scope(monkeypatch):
    import contextlib
    from gateway.wake import deliver_wake
    seen, active = [], []

    @contextlib.asynccontextmanager
    async def scope(source):
        seen.append(source.profile)
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    async def fake_invoke(name, **kwargs):
        assert active, "hook ran outside the routed profile scope"
        return []

    monkeypatch.setattr(lifecycle, "ainvoke_hook", fake_invoke)
    runner, _ = _runner()
    runner._async_profile_scope_for_source = scope
    adapter = _NonPushAdapter(runner)
    await deliver_wake(adapter, text="done", session_id="raw-1", profile="p1")
    assert seen == ["p1"] and len(adapter.turns) == 1


@pytest.mark.asyncio
async def test_non_push_wake_admitted_runs_its_turn(monkeypatch):
    from gateway.wake import deliver_wake
    _hook(monkeypatch)
    adapter = _NonPushAdapter()
    await deliver_wake(adapter, text="done", session_id="raw-1", profile="p1")
    assert [t["session_id"] for t in adapter.turns] == ["raw-1"]


@pytest.mark.asyncio
async def test_non_push_wake_hook_failure_blocks(monkeypatch):
    from gateway.wake import deliver_wake
    _hook(monkeypatch, raises=RuntimeError("boom"))
    adapter = _NonPushAdapter()
    await deliver_wake(adapter, text="done", session_id="raw-1", profile="p1")
    assert adapter.turns == []


@pytest.mark.asyncio
async def test_non_push_wake_uses_the_gateway_admission_lock(monkeypatch):
    import asyncio
    from gateway.wake import deliver_wake
    _hook(monkeypatch)
    runner, _ = _runner()
    lock = runner._internal_admission_guard()
    adapter = _NonPushAdapter(runner)
    async with lock:
        task = asyncio.ensure_future(deliver_wake(adapter, text="done", session_id="raw-1", profile="p1"))
        await asyncio.sleep(0)
        assert not task.done() and adapter.turns == []
    await task
    assert len(adapter.turns) == 1
