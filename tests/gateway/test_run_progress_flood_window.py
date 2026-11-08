"""A flood-refused progress edit never answers with a fresh send into the same chat (#76494).

Telegram bills every call against the chat's open penalty and lengthens it while hammered: the
progress loop used to answer a refused edit of its bubble with ``adapter.send()``, one extra message
per tick for as long as the penalty lasted. Drives the real ``GatewayRunner._run_agent`` progress
loop; only the agent and the platform transport are fakes.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class FloodedEditAdapter(BasePlatformAdapter):
    """Every edit is refused with the canonical Telegram fail-closed flood result."""

    def __init__(self, retry_after: float):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.retry_after = retry_after
        self.sent, self.edits = [], []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(content)
        return SendResult(success=False, error=f"flood_control:{self.retry_after}", retry_after=self.retry_after)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class FourToolAgent:
    """Opens the bubble, then three more tool events, each after the 1.5 s edit throttle."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    @property
    def is_interrupted(self) -> bool:
        return False

    def run_conversation(self, message, conversation_history=None, task_id=None):
        for i in range(4):
            self.tool_progress_callback("tool.started", "web_search", f"search {i}", {})
            time.sleep(1.8)  # health: allow HX013 -- the gap must outlast the loop's real 1.5s edit throttle
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _run(monkeypatch, tmp_path, retry_after, adapter=None, agent_cls=FourToolAgent):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    adapter = adapter or FloodedEditAdapter(retry_after)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode, runner._prefill_messages, runner._running_agents = {}, [], {}
    runner._ephemeral_system_prompt, runner._reasoning_config, runner._fallback_model = "", None, None
    runner._provider_routing, runner._session_db = {}, None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")
    result = await runner._run_agent(
        message="hi", context_prompt="", history=[], source=source, session_id="s-flood",
        session_key="agent:main:telegram:group:-1001")
    return adapter, result


@pytest.mark.asyncio
async def test_flood_refused_progress_edit_sends_nothing_inside_the_window(monkeypatch, tmp_path):
    adapter, result = await _run(monkeypatch, tmp_path, retry_after=30.0)

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1, adapter.sent  # the bubble itself; no fallback sends into the penalty
    assert len(adapter.edits) == 1, adapter.edits  # the refusal opened a window: no further edits in it


@pytest.mark.asyncio
async def test_flood_refused_finalize_edit_is_not_retried_as_plain_text():
    """The MarkdownV2 -> plain fallback is for parse errors; on a flood it was a second refused call."""
    from unittest.mock import AsyncMock

    from plugins.platforms.telegram.adapter import TelegramAdapter

    class RetryAfter(Exception):
        retry_after = 30

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(edit_message_text=AsyncMock(side_effect=RetryAfter("Flood control exceeded. Retry in 30 seconds")))

    result = await adapter.edit_message("-1001", "7", "**progress** line", finalize=True)

    assert result.error == "flood_control:30"
    assert adapter._bot.edit_message_text.await_count == 1


class FloodOnceAdapter(FloodedEditAdapter):
    """The first edit is refused with ``error`` (and ``retry_after``); later edits land."""

    def __init__(self, error: str, retry_after=None):
        super().__init__(0.0)
        self.error, self.refused_retry_after, self.landed = error, retry_after, []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(content)
        if len(self.edits) == 1:
            return SendResult(success=False, error=self.error, retry_after=self.refused_retry_after)
        self.landed.append((time.monotonic(), content))
        return SendResult(success=True, message_id=message_id)


class LongLastToolAgent:
    """Two tool events, then a long-running last tool that emits no further progress."""

    finished_at = 0.0

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    @property
    def is_interrupted(self) -> bool:
        return False

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "web_search", "search 0", {})
        time.sleep(1.8)  # health: allow HX013 -- the gap must outlast the loop's real 1.5s edit throttle
        self.tool_progress_callback("tool.started", "terminal", "long build", {})
        time.sleep(4.0)  # health: allow HX013 -- the busy tool outlives the 1 s flood window
        LongLastToolAgent.finished_at = time.monotonic()
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_progress_held_by_a_flood_window_lands_when_it_closes_while_the_agent_is_busy(monkeypatch, tmp_path):
    adapter = FloodOnceAdapter("flood_control:1.0", retry_after=1.0)

    await _run(monkeypatch, tmp_path, 1.0, adapter=adapter, agent_cls=LongLastToolAgent)

    in_turn = [content for at, content in adapter.landed if at < LongLastToolAgent.finished_at]
    assert in_turn and "long build" in in_turn[0], adapter.landed  # not left stale until the turn ends
    assert len(adapter.sent) == 1  # still no fallback send into the penalty


@pytest.mark.asyncio
async def test_flood_window_uses_the_wait_stated_in_raw_platform_wording(monkeypatch, tmp_path):
    """The overflow-split path forwards PTB's text without ``retry_after``: the stated wait counts."""
    adapter = FloodOnceAdapter("Flood control exceeded. Retry in 2 seconds")

    await _run(monkeypatch, tmp_path, 0.0, adapter=adapter)

    first_tick = adapter.landed[0][0] if adapter.landed else None
    assert first_tick is not None, adapter.edits  # a 2 s penalty is not stretched into the 30 s default
    assert len(adapter.sent) == 1
