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

from gateway.config import Platform, PlatformConfig


async def _fire_post_delivery_cb(cb):
    """Invoke a popped post-delivery callback, awaiting if it's async.

    Chained registrations return an async wrapper; single registrations
    return the raw sync callable. Either way, await any awaitable result.
    """
    result = cb()
    if _inspect.isawaitable(result):
        await result
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


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
        self.delete_success = True
        self.fail_final_response = False

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
        return SendResult(
            success=not (self.fail_final_response and content == "done-1"),
            message_id=mid,
        )

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return self.delete_success

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
        # match the base class behavior exactly. gateway/run.py checks
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
        # actually returns on provider errors (see gateway/run.py where
        # failed keys are only propagated when final_response is empty).
        return {
            "final_response": "",
            "messages": [],
            "api_calls": 1,
            "failed": True,
            "error": "simulated provider failure",
        }


class QueuedProgressAgent:
    """Records whether the first turn's cleanup ran before turn two starts."""

    adapter: CleanupCaptureAdapter | None = None
    runner = None
    replacement: CleanupCaptureAdapter | None = None
    run_count = 0
    deleted_before_followup: int | None = None

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).run_count += 1
        if type(self).run_count == 2:
            adapter = type(self).replacement or type(self).adapter
            assert adapter is not None
            type(self).deleted_before_followup = len(adapter.deleted)
        if self.tool_progress_callback is not None:
            self.tool_progress_callback("tool.started", "terminal", "pwd", {})
            time.sleep(0.5)
        if type(self).run_count == 1 and type(self).replacement is not None:
            runner = type(self).runner
            replacement = type(self).replacement
            adapter = type(self).adapter
            assert runner is not None
            assert replacement is not None
            assert adapter is not None
            replacement._pending_messages.update(adapter._pending_messages)
            adapter._pending_messages.clear()
            runner.adapters[Platform.TELEGRAM] = replacement
        return {
            "final_response": f"done-{type(self).run_count}",
            "messages": [],
            "api_calls": 1,
        }


class PendingSteerAgent(QueuedProgressAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None):
        result = super().run_conversation(message, conversation_history, task_id)
        if type(self).run_count == 1:
            result["pending_steer"] = "steered follow-up"
        return result


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
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
    runner._session_run_generation = {}
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
    import tools.terminal_tool  # noqa: F401 — register tool emoji

    gateway_run = importlib.import_module("gateway.run")
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
@pytest.mark.parametrize("delete_success", [True, False])
async def test_cleanup_chains_with_existing_callback(
    monkeypatch, tmp_path, caplog, delete_success
):
    """When a bg-review-style callback is already registered, the cleanup
    callback chains with it — both fire, neither clobbers the other."""
    caplog.set_level("INFO", logger="gateway.run")
    adapter = CleanupCaptureAdapter()
    adapter.delete_success = delete_success
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
    await _fire_post_delivery_cb(cb)

    # The callback is the completion boundary used before an in-band queued
    # follow-up starts. It must not escape into an unobserved background future.
    assert pre_existing_fired == [True]
    assert len(adapter.deleted) >= 1
    expected_summary = (
        "requested=1 deleted=1 failed=0"
        if delete_success
        else "requested=1 deleted=0 failed=1"
    )
    assert expected_summary in caplog.text
    if not delete_success:
        assert "returned_false" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_adapter", [False, True])
async def test_queued_followup_waits_for_prior_progress_cleanup(
    monkeypatch, tmp_path, caplog, replace_adapter
):
    caplog.set_level("INFO", logger="gateway.run")
    adapter = CleanupCaptureAdapter()
    replacement = CleanupCaptureAdapter() if replace_adapter else None
    QueuedProgressAgent.adapter = adapter
    QueuedProgressAgent.runner = None
    QueuedProgressAgent.replacement = replacement
    QueuedProgressAgent.run_count = 0
    QueuedProgressAgent.deleted_before_followup = None
    runner = _make_runner(adapter)
    QueuedProgressAgent.runner = runner
    gateway_run = _install_fakes(monkeypatch, QueuedProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:dm:-1001"
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-queued-cleanup",
        session_key=session_key,
    )

    assert result["final_response"] == "done-2"
    assert QueuedProgressAgent.run_count == 2
    assert QueuedProgressAgent.deleted_before_followup == 1, caplog.text

    callback = adapter.pop_post_delivery_callback(session_key)
    assert callable(callback)
    await _fire_post_delivery_cb(callback)
    live_adapter = replacement or adapter
    assert len(live_adapter.deleted) >= 2


@pytest.mark.asyncio
async def test_failed_first_delivery_defers_cleanup_and_queued_followup(
    monkeypatch, tmp_path
):
    adapter = CleanupCaptureAdapter()
    adapter.fail_final_response = True
    QueuedProgressAgent.adapter = adapter
    QueuedProgressAgent.runner = None
    QueuedProgressAgent.replacement = None
    QueuedProgressAgent.run_count = 0
    QueuedProgressAgent.deleted_before_followup = None
    runner = _make_runner(adapter)
    QueuedProgressAgent.runner = runner
    gateway_run = _install_fakes(monkeypatch, QueuedProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:dm:-1001"
    queued_event = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )
    adapter._pending_messages[session_key] = queued_event

    result = await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delivery-failure",
        session_key=session_key,
    )

    assert result["final_response"] == "done-1"
    assert QueuedProgressAgent.run_count == 1
    assert adapter.deleted == []
    assert adapter._pending_messages[session_key] is queued_event


@pytest.mark.asyncio
async def test_failed_first_delivery_preserves_pending_steer(monkeypatch, tmp_path):
    adapter = CleanupCaptureAdapter()
    adapter.fail_final_response = True
    PendingSteerAgent.adapter = adapter
    PendingSteerAgent.runner = None
    PendingSteerAgent.replacement = None
    PendingSteerAgent.run_count = 0
    PendingSteerAgent.deleted_before_followup = None
    runner = _make_runner(adapter)
    PendingSteerAgent.runner = runner
    gateway_run = _install_fakes(monkeypatch, PendingSteerAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:dm:-1001"
    result = await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-steer-delivery-failure",
        session_key=session_key,
    )

    assert result["final_response"] == "done-1"
    assert PendingSteerAgent.run_count == 1
    assert adapter.deleted == []
    assert adapter._pending_messages[session_key].text == "steered follow-up"
