"""Tests for interrupt-aware tool-progress suppression in gateway.

When a user sends `stop` while the agent is executing a batch of parallel
tool calls, the gateway's progress_callback should stop queuing 🔍 bubbles
and the drain loop should drop any already-queued events.  Without this
guard, the stop acknowledgement appears first but is followed by a trail
of tool-progress bubbles for calls that were already parsed from the LLM
response — making the interrupt feel ignored.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class PreInterruptAgent:
    """Fires tool-progress events BEFORE the interrupt lands.

    These should render normally.  Baseline for comparison with the
    interrupted case — proves the harness renders events when no
    interrupt is active.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "web_search", "first search", {})
        time.sleep(0.35)  # let the drain loop process
        return {"final_response": "done", "messages": [], "api_calls": 1}


class InterruptedAgent:
    """Fires tool.started events AFTER interrupt — all should be suppressed.

    Mirrors the failure mode in the bug report: LLM returned N parallel
    web_search calls, interrupt flag flipped, remaining events still
    rendered as bubbles.  With the fix, none of these should appear.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        # Start already interrupted — simulates stop having already landed
        # by the time the agent batch starts firing tool.started events.
        self._interrupt_requested = True

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        # Parallel tool batch — in production these come from one LLM
        # response with 5 tool_calls.  All are post-interrupt.
        self.tool_progress_callback("tool.started", "web_search", "cognee hermes", {})
        self.tool_progress_callback("tool.started", "web_search", "McBee deer hunting", {})
        self.tool_progress_callback("tool.started", "web_search", "kuzu graph db", {})
        self.tool_progress_callback("tool.started", "web_search", "moonshot kimi api", {})
        self.tool_progress_callback("tool.started", "web_search", "platform.moonshot.cn", {})
        time.sleep(0.35)  # let the drain loop attempt to process the queue
        return {"final_response": "interrupted", "messages": [], "api_calls": 1}


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


async def _run_once(monkeypatch, tmp_path, agent_cls, session_id):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key="agent:main:telegram:group:-1001:17585",
    )
    return adapter, result


@pytest.mark.asyncio
async def test_baseline_non_interrupted_agent_renders_progress(monkeypatch, tmp_path):
    """Sanity check: when is_interrupted is False, tool-progress renders normally."""
    adapter, result = await _run_once(monkeypatch, tmp_path, PreInterruptAgent, "sess-baseline")
    assert result["final_response"] == "done"
    rendered = " ".join(c["content"] for c in adapter.sent) + " " + " ".join(
        c["content"] for c in adapter.edits
    )
    assert "first search" in rendered, (
        "baseline agent should render its tool-progress event — "
        "if this fails the test harness is broken, not the fix"
    )


@pytest.mark.asyncio
async def test_progress_suppressed_when_agent_is_interrupted(monkeypatch, tmp_path):
    """Post-interrupt tool.started events must not render as bubbles.

    This is Bug B from the screenshot: user sends `stop`, agent acks with
    ⚡ Interrupting, but 5 more 🔍 web_search bubbles still render because
    their tool.started events were already parsed from the LLM response.
    With the fix, progress_callback and the drain loop both check
    is_interrupted and skip these events.
    """
    adapter, result = await _run_once(
        monkeypatch, tmp_path, InterruptedAgent, "sess-interrupted"
    )
    assert result["final_response"] == "interrupted"

    rendered = " ".join(c["content"] for c in adapter.sent) + " " + " ".join(
        c["content"] for c in adapter.edits
    )

    # None of the post-interrupt queries should appear.
    for leaked_query in (
        "cognee hermes",
        "McBee deer hunting",
        "kuzu graph db",
        "moonshot kimi api",
        "platform.moonshot.cn",
    ):
        assert leaked_query not in rendered, (
            f"event '{leaked_query}' leaked into the UI after interrupt — "
            f"progress_callback / drain loop is not checking is_interrupted"
        )
