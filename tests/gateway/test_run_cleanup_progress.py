"""Tests for opt-in cleanup of temporary progress bubbles.

When ``display.platforms.<plat>.cleanup_progress: true`` is set for a
platform whose adapter supports message deletion (e.g. Telegram), the
tool-progress bubble, "⏳ Working — N min" heartbeats, and status-callback
messages sent during a run are deleted after the final response is
delivered.

Failed runs skip cleanup so the bubbles remain as breadcrumbs.
Adapters without ``delete_message`` silently no-op.
"""

import asyncio
import importlib
import inspect as _inspect
import sys
import time
import types
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig


async def _fire_post_delivery_cb(cb):
    """Invoke a popped post-delivery callback, awaiting if it's async.

    Chained registrations return an async wrapper; single registrations
    return the raw sync callable. Either way, await any awaitable result.
    """
    result = cb()
    if _inspect.isawaitable(result):
        await result
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_agent.gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Test fakes — mirror those in test_run_progress_topics.py but add a
# delete_message implementation that records ids instead of hitting a bot.
# ---------------------------------------------------------------------------


class CleanupCaptureAdapter(BasePlatformAdapter):
    """Adapter that records every delete_message call for inspection."""

    _next_mid = 100

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.deleted = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        CleanupCaptureAdapter._next_mid += 1
        return str(CleanupCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata}
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class NoDeleteAdapter(CleanupCaptureAdapter):
    """Adapter that inherits the base no-op delete_message (used to prove
    the cleanup path skips adapters without deletion support)."""

    async def delete_message(self, chat_id, message_id) -> bool:  # type: ignore[override]
        # Pretend to be an adapter whose platform doesn't support deletion:
        # match the base class behavior exactly. hermes_agent/gateway/run.py checks
        # ``type(adapter).delete_message is BasePlatformAdapter.delete_message``
        # to detect this, so we re-assign at class body level below.
        raise AssertionError("should not be called — cleanup must skip this adapter")


# Re-bind so the class's delete_message identity equals the base's.
NoDeleteAdapter.delete_message = BasePlatformAdapter.delete_message


class ProgressAgent:
    """Emits two tool-progress events and returns a normal final response."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.2)
            cb("tool.started", "terminal", "ls", {})
            time.sleep(0.2)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FailingAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.2)
        # Empty final_response + failed=True is the shape the gateway
        # actually returns on provider errors (see hermes_agent/gateway/run.py where
        # failed keys are only propagated when final_response is empty).
        return {
            "final_response": "",
            "messages": [],
            "api_calls": 1,
            "failed": True,
            "error": "simulated provider failure",
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("hermes_agent.gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(
    monkeypatch,
    agent_cls,
    *,
    cleanup_on: bool,
    cleanup_platform: Platform = Platform.TELEGRAM,
):
    """Wire up the module stubs every _run_agent test needs."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import hermes_agent.tools.terminal_tool

    gateway_run = importlib.import_module("hermes_agent.gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    # Wire the per-platform cleanup_progress flag via the config loader the
    # gateway actually reads (``_load_gateway_config`` returns user config).
    cfg = {
        "display": {
            "platforms": {
                cleanup_platform.value: {"cleanup_progress": True},
            }
        }
    } if cleanup_on else {}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    return gateway_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messaging_agent_forwards_checkpoint_config(monkeypatch, tmp_path):
    """Writable gateway agents must receive the configured checkpoint limits."""
    captured = {}

    class CheckpointCaptureAgent(ProgressAgent):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch, CheckpointCaptureAgent, cleanup_on=False,
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 9,
                "max_total_size_mb": 444,
                "max_file_size_mb": 6,
            }
        },
    )

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-checkpoints",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done"
    assert captured["checkpoints_enabled"] is True
    assert captured["checkpoint_max_snapshots"] == 9
    assert captured["checkpoint_max_total_size_mb"] == 444
    assert captured["checkpoint_max_file_size_mb"] == 6


@pytest.mark.asyncio
async def test_cleanup_chains_with_existing_callback(monkeypatch, tmp_path):
    """When a bg-review-style callback is already registered, the cleanup
    callback chains with it — both fire, neither clobbers the other."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    pre_existing_fired = []

    def _preexisting_callback() -> None:
        pre_existing_fired.append(True)

    # Pre-register a callback with the same generation the run will use
    # (run_generation=None in this test path — matches the default slot).
    adapter.register_post_delivery_callback(session_key, _preexisting_callback)

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    cb = adapter.pop_post_delivery_callback(session_key)
    assert callable(cb)
    # Deletion is complete when the callback completes: no polling (a queued follow-up starts
    # right after this await).
    await _fire_post_delivery_cb(cb)

    # Both effects land: the pre-existing callback fires AND the cleanup
    # deletes at least one progress bubble.
    assert pre_existing_fired == [True]
    assert len(adapter.deleted) >= 1


class QueuedFollowupAgent:
    """Turn 1 emits progress, then a follow-up is queued; turn 2 records what the chat still showed."""

    adapter = None
    session_key = ""
    deleted_at_turn2_start = None

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        from hermes_agent.gateway.platforms.base import MessageEvent, MessageType
        cls = type(self)
        if message == "follow-up":
            cls.deleted_at_turn2_start = list(cls.adapter.deleted)
            return {"final_response": "second", "messages": [], "api_calls": 1}
        cb = self.tool_progress_callback
        cb("tool.started", "terminal", "pwd", {})
        from tests.waiting import wait_until
        wait_until(lambda: cls.adapter.sent, timeout=5, what="progress bubble delivered")
        cls.adapter._pending_messages[cls.session_key] = MessageEvent(
            text="follow-up", message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"))
        return {"final_response": "first", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_queued_followup_starts_after_prior_turn_bubbles_are_deleted(monkeypatch, tmp_path):
    """The queued follow-up path returns before the normal-path cleanup registration; turn 1's
    progress bubbles must still be deleted, and deleted BEFORE turn 2 begins."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, QueuedFollowupAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:telegram:group:-1001"
    QueuedFollowupAgent.adapter, QueuedFollowupAgent.session_key = adapter, session_key
    QueuedFollowupAgent.deleted_at_turn2_start = None

    result = await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"),
        session_id="sess-1", session_key=session_key,
    )

    assert result["final_response"] == "second"
    progress_ids = {m["message_id"] for m in adapter.sent if m["content"] != "first"}
    assert progress_ids
    assert QueuedFollowupAgent.deleted_at_turn2_start is not None
    assert {d["message_id"] for d in QueuedFollowupAgent.deleted_at_turn2_start} >= progress_ids


class HeldSendAdapter(CleanupCaptureAdapter):
    """Holds every progress send open until the queued follow-up's turn starts."""

    def __init__(self):
        super().__init__()
        self.loop = self.release = None
        self.send_started = __import__("threading").Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if content not in ("first", "second"):
            self.send_started.set()
            await self.release.wait()
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


class HeldSendAgent(QueuedFollowupAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        from hermes_agent.gateway.platforms.base import MessageEvent, MessageType
        from tests.waiting import wait_until
        cls, adapter = type(self), type(self).adapter
        if message == "follow-up":
            adapter.loop.call_soon_threadsafe(adapter.release.set)
            return {"final_response": "second", "messages": [], "api_calls": 1}
        self.tool_progress_callback("tool.started", "terminal", "pwd", {})
        wait_until(adapter.send_started.is_set, timeout=5, what="progress send in flight")
        adapter._pending_messages[cls.session_key] = MessageEvent(
            text="follow-up", message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"))
        return {"final_response": "first", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_progress_send_in_flight_at_handoff_never_leaves_a_bubble(monkeypatch, tmp_path):
    """A progress send still open when a queued follow-up takes over either lands before the
    cleanup snapshot or never lands: no bubble from the finished turn outlives it."""
    adapter = HeldSendAdapter()
    adapter.loop, adapter.release = asyncio.get_running_loop(), asyncio.Event()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, HeldSendAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:telegram:group:-1001"
    HeldSendAgent.adapter, HeldSendAgent.session_key = adapter, session_key

    await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001"),
        session_id="sess-1", session_key=session_key,
    )
    for _ in range(20):
        await asyncio.sleep(0)
    progress_ids = {m["message_id"] for m in adapter.sent if m["content"] not in ("first", "second")}
    assert progress_ids <= {d["message_id"] for d in adapter.deleted}


@pytest.mark.asyncio
async def test_one_slow_delete_does_not_strand_the_other_bubbles(monkeypatch):
    """Cleanup runs under the post-delivery timeout; a hung delete must not use it all."""
    from hermes_agent.gateway import run_turn_bubble_cleanup as cleanup
    from hermes_agent.gateway.platforms.base_post_delivery import run_post_delivery_callback

    class HungFirstDelete(CleanupCaptureAdapter):
        async def delete_message(self, chat_id, message_id) -> bool:
            if message_id == "1":
                await asyncio.Event().wait()
            return await super().delete_message(chat_id, message_id)

    monkeypatch.setattr(cleanup, "DELETE_TIMEOUT_SECONDS", 0.05, raising=False)
    adapter = HungFirstDelete()
    turn_ctx = SimpleNamespace(
        _cleanup_adapter=adapter, _cleanup_msg_ids=["1", "2"], session_key="k", _cleanup_progress=True,
        source=SimpleNamespace(chat_id="-1001"), run_generation=None)
    cleanup.register_bubble_cleanup({"final_response": "done"}, turn_ctx)
    await run_post_delivery_callback(adapter.pop_post_delivery_callback("k"), timeout=5)
    assert [d["message_id"] for d in adapter.deleted] == ["2"]


@pytest.mark.asyncio
async def test_hung_sender_delays_the_followup_by_at_most_the_cap(monkeypatch):
    """A sender whose cancel-time flush never returns holds the handoff for the cap, not forever."""
    from hermes_agent.gateway import run_turn_bubble_cleanup as cleanup

    async def _hung_sender():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()  # cancel-time edit that never returns

    monkeypatch.setattr(cleanup, "DELETE_TIMEOUT_SECONDS", 0.1)
    sender = asyncio.ensure_future(_hung_sender())
    await asyncio.sleep(0)
    # The 0.1 s cap is what returns; without it the outer bound expires and the test fails.
    await asyncio.wait_for(cleanup.settle_turn_senders(sender, None), timeout=5)
    await asyncio.sleep(0)
    assert sender.done()


@pytest.mark.asyncio
async def test_cancelling_the_turn_during_the_sender_wait_starts_no_followup():
    """A cancel aimed at the turn (shutdown, /stop) while it waits on its senders must propagate."""
    from hermes_agent.gateway import run_turn_bubble_cleanup as cleanup

    release = asyncio.Event()

    async def _slow_sender():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    events = []

    async def _turn(sender):
        await cleanup.settle_turn_senders(sender)
        events.append("follow-up started")

    sender = asyncio.ensure_future(_slow_sender())
    await asyncio.sleep(0)
    turn = asyncio.ensure_future(_turn(sender))
    for _ in range(5):
        await asyncio.sleep(0)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert events == []
    release.set()
    await asyncio.sleep(0)
