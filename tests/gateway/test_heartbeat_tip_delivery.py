"""End-to-end tests for the ``heartbeat_tip`` display setting.

Where ``test_display_config.py::TestHeartbeatTip`` covers config resolution
only, these tests exercise the real heartbeat send/edit flow in
``_notify_long_running`` — proving that:

* the tip is appended to the heartbeat text when ``heartbeat_tip`` is on,
* the tip is absent when the setting is off (default),
* the tip is absent in ``generic`` long-running mode (the guard excludes it).

Uses the same capture-adapter pattern as ``test_run_cleanup_progress.py``:
a fake adapter records every ``send`` and ``edit_message`` call so the
heartbeat text can be inspected without hitting a real bot.

``hermes_cli.tips.get_random_tip`` is stubbed to return a fixed sentinel
string so assertions are deterministic (the real function uses
``random.choice``).
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
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


# Deterministic tip sentinel — if this shows up in heartbeat text, the tip
# append path fired. If it doesn't, the path was correctly suppressed.
_TIP_SENTINEL = "TIP_SENTINEL_FOR_TEST_USE_ONLY"


async def _fire_post_delivery_cb(cb):
    """Invoke a popped post-delivery callback, awaiting if it's async."""
    result = cb()
    if _inspect.isawaitable(result):
        await result


# ---------------------------------------------------------------------------
# Test fakes — mirror those in test_run_cleanup_progress.py.
# ---------------------------------------------------------------------------


class HeartbeatCaptureAdapter(BasePlatformAdapter):
    """Adapter that records every send/edit for heartbeat inspection."""

    _next_mid = 200

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        HeartbeatCaptureAdapter._next_mid += 1
        return str(HeartbeatCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata}
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SlowAgent:
    """Sleeps long enough for at least one heartbeat tick to fire, then returns.

    The heartbeat task fires every ``HERMES_AGENT_NOTIFY_INTERVAL`` seconds.
    With the interval set to 0.1s in these tests, sleeping 0.5s guarantees
    3-4 heartbeat cycles — enough for both the initial send and a follow-up
    edit-in-place.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        time.sleep(0.5)
        return {"final_response": "done", "messages": [], "api_calls": 1}


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


def _install_fakes(monkeypatch, *, display_config: dict):
    """Wire up the module stubs every _run_agent test needs.

    ``display_config`` is merged into the ``display`` key of the gateway
    config the runner reads, e.g. ``{"heartbeat_tip": True}`` or
    ``{"long_running_notifications": "generic"}``.
    """
    # Force a fast heartbeat cycle so the test doesn't wait 180s.
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.1")
    # Disable the inactivity timeout so the slow agent isn't killed.
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = SlowAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 — register tool emoji

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    cfg = {"display": dict(display_config)} if display_config else {}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)

    # Stub get_random_tip so assertions are deterministic — the real function
    # uses random.choice, which would make these tests flaky.
    fake_tips = types.ModuleType("hermes_cli.tips")
    fake_tips.get_random_tip = lambda *a, **k: _TIP_SENTINEL
    monkeypatch.setitem(sys.modules, "hermes_cli.tips", fake_tips)

    return gateway_run


def _heartbeat_texts(adapter):
    """All heartbeat texts the adapter saw, in send-then-edit order."""
    texts = [entry["content"] for entry in adapter.sent]
    texts.extend(entry["content"] for entry in adapter.edits)
    return texts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_tip_off_by_default(monkeypatch, tmp_path):
    """No ``heartbeat_tip`` config → heartbeats fire but carry no tip.

    Proves the default-off path: the heartbeat text is the plain
    ``⏳ Working — N min`` form, never the ``✦ Tip:`` append.
    """
    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, display_config={})
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    # At least one heartbeat should have fired during the 0.5s sleep.
    texts = _heartbeat_texts(adapter)
    assert texts, f"expected at least one heartbeat, got none. sent={adapter.sent} edits={adapter.edits}"
    # No heartbeat text should contain the tip sentinel.
    for text in texts:
        assert _TIP_SENTINEL not in text, f"tip leaked into default-off heartbeat: {text!r}"
        assert "✦ Tip:" not in text, f"tip prefix leaked into default-off heartbeat: {text!r}"


@pytest.mark.asyncio
async def test_heartbeat_tip_appended_when_enabled(monkeypatch, tmp_path):
    """``display.heartbeat_tip: true`` → each heartbeat carries the tip.

    Proves the enabled append path: the heartbeat text contains both the
    ``⏳ Working`` prefix and the ``✦ Tip:`` append with the stubbed tip.
    """
    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        display_config={"heartbeat_tip": True},
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    texts = _heartbeat_texts(adapter)
    assert texts, f"expected at least one heartbeat, got none. sent={adapter.sent} edits={adapter.edits}"
    # At least one heartbeat (typically the first send; subsequent ones are
    # edits of the same message id) should carry the tip sentinel.
    tipped = [t for t in texts if _TIP_SENTINEL in t]
    assert tipped, (
        f"expected tip sentinel in at least one heartbeat, but none had it. "
        f"texts={texts!r}"
    )
    # Every tipped heartbeat should also have the working prefix and the
    # ``✦ Tip:`` separator — proves the append format, not just the string.
    for text in tipped:
        assert "⏳ Working" in text, f"missing working prefix in tipped heartbeat: {text!r}"
        assert "✦ Tip:" in text, f"missing tip prefix in tipped heartbeat: {text!r}"


@pytest.mark.asyncio
async def test_heartbeat_tip_suppressed_in_generic_mode(monkeypatch, tmp_path):
    """``long_running_notifications: "generic"`` → no tip, even if enabled.

    Proves the generic-mode exclusion guard: the heartbeat text is the
    generic status phrase (``_generic_status_phrase("status")``), never the
    ``⏳ Working`` form, and the tip append is skipped.
    """
    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        display_config={
            "heartbeat_tip": True,                 # enabled, but...
            "long_running_notifications": "generic",  # ...generic mode excludes tips
        },
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    texts = _heartbeat_texts(adapter)
    assert texts, f"expected at least one heartbeat, got none. sent={adapter.sent} edits={adapter.edits}"
    # Generic mode emits a phrase like "still on it" — never the working
    # prefix, never the tip.
    for text in texts:
        assert "⏳ Working" not in text, (
            f"generic mode should not emit the working prefix: {text!r}"
        )
        assert _TIP_SENTINEL not in text, (
            f"tip leaked into generic-mode heartbeat: {text!r}"
        )
        assert "✦ Tip:" not in text, (
            f"tip prefix leaked into generic-mode heartbeat: {text!r}"
        )
