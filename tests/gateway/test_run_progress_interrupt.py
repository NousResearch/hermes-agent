"""Tests for interrupt-aware tool-progress suppression in gateway.

When a user sends `stop` while the agent is executing a batch of parallel
tool calls, the gateway's progress_callback should stop queuing 🔍 bubbles
and the drain loop should drop any already-queued events.  Without this
guard, the stop acknowledgement appears first but is followed by a trail
of tool-progress bubbles for calls that were already parsed from the LLM
response — making the interrupt feel ignored.
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


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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


class PartialTruncationAgent:
    """Returns an incomplete turn with no visible assistant text."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": None,
            "messages": [],
            "api_calls": 2,
            "completed": False,
            "partial": True,
            "error": "Response truncated due to output length limit",
        }


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
async def test_partial_empty_agent_response_is_normalized(monkeypatch, tmp_path):
    """Messaging gateways should not echo raw truncation errors as final text."""
    adapter, result = await _run_once(
        monkeypatch, tmp_path, PartialTruncationAgent, "sess-partial-empty"
    )

    assert result["final_response"].startswith("⚠️ Processing stopped:")
    assert "Response truncated due to output length limit" in result["final_response"]
    assert result["final_response"] != "⚠️ Response truncated due to output length limit"
    assert result["partial"] is True
    assert adapter.sent == []


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

    rendered = " ".join(c["content"] for c in adapter.sent) + " ".join(
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


# ============================================================
# Flood-control fallback regression
# ============================================================
#
# Bug: when the progress bubble's edit call fails with a Telegram
# flood-control error, the gateway's send_progress_messages()
# handler used to "fall back" to a fresh adapter.send() in the
# same chat. That fallback is exactly the burst pattern that
# triggers the Telegram penalty in the first place — re-firing
# it during the penalty window keeps the timer pinned at
# "Retry in 7000+ seconds" and escalates the penalty by minutes
# each tick.
#
# Fix: when the edit fails with a flood-control error, drop the
# tick and let the per-chat send cooldown (added in
# plugins/platforms/telegram/adapter.py) release the chat before
# the next tick retries the edit. No fresh send() during the
# penalty window.


class FloodControlEditAdapter(ProgressCaptureAdapter):
    """Adapter whose edit_message() fails with a Telegram flood-control
    error for the first ``edit_failures_remaining`` calls, then
    succeeds. Used to verify the progress handler does NOT call
    send() as a fallback when the edit fails with flood-control.
    """

    def __init__(self, *args, edit_failures_remaining=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_failures_remaining = edit_failures_remaining

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        if self.edit_failures_remaining > 0:
            self.edit_failures_remaining -= 1
            return SendResult(
                success=False,
                error="flood control exceeded. Retry in 7000 seconds",
                retryable=False,
            )
        return SendResult(success=True, message_id=message_id)


class FloodControlProgressAgent:
    """Fires a handful of tool.started events so the progress consumer
    has enough work to drain past the throttle interval and observe
    the flood-edit failure."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        for _ in range(3):
            self.tool_progress_callback("tool.started", "web_search", "first hit", {})
            time.sleep(0.05)
        # Let the drain loop observe the flood-control edit failure
        # and (under the fix) skip the fallback send. The 1.5s
        # progress-edit throttle means we need at least that long.
        time.sleep(2.0)
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_progress_flood_control_does_not_trigger_fallback_send(
    monkeypatch, tmp_path
):
    """When edit_message fails with a Telegram flood-control error, the
    progress handler must NOT issue a fresh adapter.send() in the
    same chat. Issuing one re-triggers the Telegram penalty and
    extends it from minutes to hours. The fix is to drop the tick
    and let the per-chat send cooldown release the chat.
    """
    adapter = FloodControlEditAdapter(edit_failures_remaining=3)
    # Tiny cooldown so the test runs fast — the actual cooldown
    # doesn't matter for this test, only that one is set.
    adapter._send_cooldown_seconds = 0.05
    adapter._send_cooldown_max_wait = 1.0
    runner = _make_runner(adapter)

    import importlib
    gateway_run = importlib.import_module("gateway.run")

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FloodControlProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
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
        session_id="sess-flood",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"

    # The handler should have tried (and failed) at least one edit —
    # the flood-control-failing edit_message is the signal we need
    # to react to.
    assert any("first hit" in e["content"] for e in adapter.edits), (
        "expected at least one edit attempt against the progress bubble; "
        "without it the flood-control path is not exercised"
    )

    # The fix: no fresh send() fallback on flood-control. Filter out
    # any final-response send (which is unrelated to the progress
    # path) — we only care about progress-driven fallback sends,
    # which would appear with content matching the progress line.
    #
    # Note: a single progress send is EXPECTED here — the first
    # progress bubble has to be sent (not edited) because no
    # previous progress message exists yet. The bug we are
    # regressing on is the SECOND and onward sends that the legacy
    # code issued as a fallback when the edit hit flood control.
    # With the fix, the edit failures drop the tick instead of
    # firing a fresh send, so the count should stay at 1.
    progress_sends = [s for s in adapter.sent if "first hit" in s["content"]]
    assert len(progress_sends) <= 1, (
        f"progress handler issued {len(progress_sends)} progress-bubble "
        f"send(s) — at most 1 is expected (the initial bubble). "
        f"Multiple sends here means the flood-control fallback bug "
        f"is back: every extra send re-triggers Telegram's penalty "
        f"and extends it from minutes to hours. "
        f"Sent: {progress_sends!r}"
    )


