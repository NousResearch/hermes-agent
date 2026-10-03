"""Native task cards must write the turn's TERMINAL state back before the card is sealed.

#115177: with ``platforms.slack.extra.native_task_cards: true`` a tool-call turn left the
card stuck on "Hermes is working" with the tool still listed as running. The card lane only
published a frame when a lifecycle event was drained from the progress queue, and the events
for the turn's last tool are exactly the ones that can go missing:

* ``agent/tool_executor.py`` skips ``tool_complete_callback`` for a BLOCKED call, so no
  completion event is ever queued for that tool;
* an interrupted turn (/stop, a superseding message) is torn down with its tail events still
  queued, and the pre-write-back drain was gated on the run being current and uninterrupted.

Either way the card was sealed (``chat.stopStream``) with its last published frame — a task
spinning on ``in_progress`` that nothing will ever update again. The terminal write-back is
shared (``TurnRunner._send_native_task_card_progress``), so it covers every platform and card
mode; the text-only turn must be untouched.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class NativeCardCaptureAdapter(BasePlatformAdapter):
    """Slack-shaped adapter capturing the card lane and the text lane separately."""

    def __init__(self, platform=Platform.SLACK):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.native_updates = []
        self.native_stops = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"content": content, "reply_to": reply_to})
        return SendResult(success=True, message_id="text-1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content, "finalize": finalize})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def stop_typing(self, chat_id) -> None:
        pass

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    def native_task_cards_enabled(self):
        return True

    async def send_native_task_card_progress(
        self, chat_id, tasks, *, title="Hermes is working", reply_to=None, metadata=None, fallback_text=None
    ) -> SendResult:
        self.native_updates.append({"title": title, "tasks": [dict(task) for task in tasks]})
        return SendResult(success=True, message_id="native-stream-1")

    async def stop_native_task_card_progress(self, chat_id, *, reply_to=None, metadata=None):
        self.native_stops += 1

    def last_card_statuses(self) -> dict:
        assert self.native_updates, "the card lane never published a frame"
        return {task["id"]: task["status"] for task in self.native_updates[-1]["tasks"]}


class ToolTurnAgent:
    """One tool call. ``completes`` mirrors the engine: a blocked call never completes."""

    FINAL = "done with the tool"
    completes = False
    interrupted = False

    def __init__(self, **kwargs):
        self.tools = []
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.is_interrupted = False

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        start = self.tool_start_callback or (lambda *a, **k: None)
        complete = self.tool_complete_callback or (lambda *a, **k: None)
        start("call-1", "terminal", {"command": "sleep 300"})
        time.sleep(0.25)
        if self.completes:
            complete("call-1", "terminal", {"command": "sleep 300"}, '{"success": true}')
        if self.interrupted:
            # /stop lands while the tool result is still in flight.
            self.is_interrupted = True
        if self.stream_delta_callback:
            self.stream_delta_callback(self.FINAL)
        time.sleep(0.25)
        return {
            "final_response": self.FINAL, "messages": [], "api_calls": 2,
            "interrupted": self.interrupted, "completed": not self.interrupted,
        }


class BlockedToolTurnAgent(ToolTurnAgent):
    """The tool is blocked: the engine never fires ``tool_complete_callback``."""

    completes = False
    interrupted = False


class InterruptedToolTurnAgent(ToolTurnAgent):
    """The tool DID complete, but the turn is torn down before its event is rendered."""

    completes = True
    interrupted = True


class TextOnlyTurnAgent:
    FINAL = "plain answer, no tools"

    def __init__(self, **kwargs):
        self.tools = []
        self.stream_delta_callback = kwargs.get("stream_delta_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        if self.stream_delta_callback:
            self.stream_delta_callback(self.FINAL)
        time.sleep(0.2)
        return {"final_response": self.FINAL, "messages": [], "api_calls": 1}


def _make_runner(adapter):
    GatewayRunner = importlib.import_module("gateway.run").GatewayRunner
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
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False
    )
    return runner


async def _run_turn(monkeypatch, tmp_path, agent_cls, session_id):
    import yaml

    streaming = {"enabled": True, "transport": "edit", "edit_interval": 0.05, "buffer_threshold": 8}
    (tmp_path / "config.yaml").write_text(yaml.dump({"streaming": streaming}), encoding="utf-8")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = NativeCardCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    runner.config.streaming = StreamingConfig.from_dict(streaming)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    source = SessionSource(
        platform=Platform.SLACK, chat_id="C1", chat_type="channel", thread_id="thread-1"
    )
    result = await runner._run_agent(
        message="hello", context_prompt="", history=[], source=source,
        session_id=session_id, session_key=f"agent:main:slack:channel:C1:thread-1:{session_id}",
    )
    return adapter, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_cls, expected_status",
    [
        (BlockedToolTurnAgent, "error"),      # no completion event ever fires
        (InterruptedToolTurnAgent, "complete"),  # the event fires but the turn ends first
    ],
    ids=["blocked-tool", "interrupted-turn"],
)
async def test_native_task_card_turn_writes_back_a_terminal_state(
    monkeypatch, tmp_path, agent_cls, expected_status
):
    """A tool-call turn must not seal the card with a row stuck on "running"."""
    adapter, result = await _run_turn(monkeypatch, tmp_path, agent_cls, f"sess-{expected_status}-{agent_cls.__name__}")

    assert result["final_response"] == agent_cls.FINAL
    # The card lane's LAST word before the stop is a terminal state for every row.
    assert adapter.last_card_statuses() == {"call-1": expected_status}, adapter.native_updates
    # The card is still finalized exactly once (and not opened late by the write-back).
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_native_task_card_text_only_turn_is_unchanged(monkeypatch, tmp_path):
    """No tool calls: the card lane publishes nothing and the answer is delivered as before."""
    adapter, result = await _run_turn(monkeypatch, tmp_path, TextOnlyTurnAgent, "sess-text-only")

    assert result["final_response"] == TextOnlyTurnAgent.FINAL
    assert adapter.native_updates == [], "a text-only turn must not open a task card"
    assert adapter.native_stops == 1
    visible = [entry["content"] for entry in adapter.sent] + [entry["content"] for entry in adapter.edits]
    assert any(TextOnlyTurnAgent.FINAL in content for content in visible), visible
