"""Invariant: the long-running heartbeat says WHAT is running and FOR HOW LONG.

The heartbeat used to render ``⏳ Working — N min — iteration I/M, terminal``: a bare tool name,
no argument, no step timer. That reads as activity without information — it cannot tell a user
whether the agent is fetching a page or running a 20-minute build. The line now carries the
running tool's short argument preview plus how long that step has taken, and falls back to the
last activity description with its age when no tool is running.

Rendered through the real ``_run_agent_notify_long_running`` heartbeat path with a capturing
adapter, so this asserts what a user actually receives.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.session_activity import format_current_step, format_step_duration
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class HeartbeatCaptureAdapter(BasePlatformAdapter):
    """Adapter that records heartbeat sends instead of hitting a platform API."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"hb{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _activity(**overrides):
    activity = {
        "current_tool": None, "current_tool_detail": None, "current_tool_started_at": None,
        "api_call_count": 19, "max_iterations": 150,
        "last_activity_desc": "executing tool: terminal", "seconds_since_activity": 1.0,
    }
    activity.update(overrides)
    return activity


async def _heartbeat_text(activity, monkeypatch) -> str:
    """Drive the real long-running heartbeat once and return the text it sent."""
    from gateway.run_turn import GatewayTurnMixin
    from gateway.turn_context import TurnContext

    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    mixin = GatewayTurnMixin()
    adapter = HeartbeatCaptureAdapter()
    mixin._adapter_for_source = lambda source: adapter
    # Called once per loop turn, once again before the pre-send guard, then the loop must exit.
    answers = iter([True, True, False])
    mixin._should_emit_long_running_notification = lambda *a, **k: next(answers, False)
    disp = SimpleNamespace(
        user_config={}, platform_key="telegram",
        _display_surface_mode=lambda setting, **kwargs: "raw",
        resolve_display_setting=lambda *a, **k: True,
    )
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="c1", platform="telegram"), session_key="s1",
        agent_holder=[SimpleNamespace(get_activity_summary=lambda: activity)],
    )
    await mixin._run_agent_notify_long_running(disp, ctx, [None])
    assert adapter.sent, "heartbeat never sent"
    return adapter.sent[0]


@pytest.mark.asyncio
async def test_heartbeat_reports_running_tool_detail_and_step_timer(monkeypatch):
    activity = _activity(
        current_tool="terminal", current_tool_detail="git status --short",
        current_tool_started_at=time.time() - 72,
    )
    text = await _heartbeat_text(activity, monkeypatch)
    assert "iteration 19/150" in text
    assert "terminal: git status --short" in text
    assert "(1m 12s)" in text


@pytest.mark.asyncio
async def test_heartbeat_falls_back_to_activity_age_without_a_running_tool(monkeypatch):
    activity = _activity(
        last_activity_desc="waiting for non-streaming API response", seconds_since_activity=42.0,
    )
    text = await _heartbeat_text(activity, monkeypatch)
    assert "waiting for non-streaming API response (42s ago)" in text


@pytest.mark.asyncio
async def test_heartbeat_stays_terse_for_a_bare_tool_name(monkeypatch):
    """No detail and no step timer of at least the minimum: just the tool, never a fake ``(0s)``."""
    activity = _activity(current_tool="terminal", current_tool_started_at=time.time() - 1)
    text = await _heartbeat_text(activity, monkeypatch)
    assert text.endswith("terminal")
    assert "terminal (" not in text


def test_format_current_step_prefers_the_running_tool():
    step = format_current_step(_activity(
        current_tool="read_file", current_tool_detail="~/.hermes/config.yaml",
        current_tool_started_at=1000.0,
    ), now=1205.0)
    assert step == "read_file: ~/.hermes/config.yaml (3m 25s)"


def test_format_current_step_skips_unset_and_initializing_labels():
    assert format_current_step(_activity()) == "executing tool: terminal"
    assert format_current_step(_activity(last_activity_desc="initializing")) == ""
    assert format_current_step(_activity(last_activity_desc="")) == ""


def test_format_step_duration_units():
    assert format_step_duration(9) == "9s"
    assert format_step_duration(72) == "1m 12s"
    assert format_step_duration(3660) == "1h 01m"
    assert format_step_duration("nonsense") == ""
