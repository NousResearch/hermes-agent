"""Gateway integration tests for the persistent cross-turn todo plan card.

Drives the real GatewayRunner._run_agent path with a fake AIAgent that emits
todo tool.started / tool.completed events, asserting ONE living card is seeded
once then edited in place (no stacking) within a turn AND across turns, the
terminal all-completed card survives the throttle on turn-end (B1), and the
feature is fully dormant when todo_progress is off (back-compat).

Harness mirrors tests/gateway/test_subagent_roster_integration.py.
"""

import importlib
import json
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
        self._send_seq = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._send_seq += 1
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=f"card-{self._send_seq}")

    async def edit_message(self, chat_id, message_id, content, **kwargs) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _result(items):
    return json.dumps({"todos": items, "summary": {"total": len(items)}}, ensure_ascii=False)


def _emit_todo(cb, items, *, merge, result_items=None):
    """Emit a todo started+completed pair through the progress callback."""
    if cb is None:
        return
    cb("tool.started", "todo", None, {"todos": items, "merge": merge})
    cb("tool.completed", "todo", None, None, result=_result(result_items or items))


# Stamps are anchored to real now() because _plan_wall_clock_seconds substitutes
# time.time() for an in_progress item's missing ended_at; fake tiny epochs would
# produce a garbage multi-thousand-hour header. _T0 is "plan start ~90s ago".
_T0 = time.time() - 90.0


class _BaseTodoAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested


class TwoCallOneTurnAgent(_BaseTodoAgent):
    """Two todo calls in ONE turn -> seed once, edit once (no stacking)."""

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_todo(cb, [
            {"id": "a", "content": "first", "status": "in_progress",
             "started_at": _T0, "elapsed_seconds": 1.0},
            {"id": "b", "content": "second", "status": "pending"},
        ], merge=False)
        time.sleep(1.7)  # clear the 1.5s throttle so the 2nd edit flushes live
        _emit_todo(cb, [
            {"id": "a", "content": "first", "status": "completed",
             "started_at": _T0, "ended_at": _T0 + 90.0, "elapsed_seconds": 90.0},
            {"id": "b", "content": "second", "status": "completed",
             "started_at": _T0 + 90.0, "ended_at": _T0 + 100.0, "elapsed_seconds": 10.0},
        ], merge=True)
        time.sleep(0.4)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class TerminalThrottledAgent(_BaseTodoAgent):
    """B1: the final all-completed card lands within the throttle window and
    the turn ends immediately -> it must still flush on drain."""

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_todo(cb, [
            {"id": "a", "content": "task", "status": "in_progress",
             "started_at": _T0, "elapsed_seconds": 1.0},
        ], merge=False)
        # immediately complete (within 1.5s throttle) then return -> drain race
        _emit_todo(cb, [
            {"id": "a", "content": "task", "status": "completed",
             "started_at": _T0, "ended_at": _T0 + 5.0, "elapsed_seconds": 5.0},
        ], merge=True, result_items=[
            {"id": "a", "content": "task", "status": "completed",
             "started_at": _T0, "ended_at": _T0 + 5.0, "elapsed_seconds": 5.0},
        ])
        return {"final_response": "done", "messages": [], "api_calls": 1}


class TodoOffAgent(_BaseTodoAgent):
    """todo_progress off + tool_progress all -> legacy shared-bubble path."""

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_todo(cb, [
            {"id": "a", "content": "x", "status": "in_progress",
             "started_at": _T0, "elapsed_seconds": 1.0},
        ], merge=False)
        time.sleep(0.3)
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
    runner._todo_card_state = __import__("collections").OrderedDict()
    runner._todo_card_state_max = 512
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run(monkeypatch, tmp_path, agent_cls, session_id, *,
               adapter=None, todo_progress="on", tool_progress="off"):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    if adapter is None:
        adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    cfg = {
        "display": {
            "platforms": {
                "telegram": {
                    "todo_progress": todo_progress,
                    "tool_progress": tool_progress,
                }
            }
        }
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    result = await runner._run_agent(
        message="hi", context_prompt="", history=[], source=source,
        session_id=session_id, session_key="agent:main:telegram:group:-1001:17585",
    )
    return adapter, runner, result


def _all_text(adapter):
    return [c["content"] for c in adapter.sent] + [c["content"] for c in adapter.edits]


@pytest.mark.asyncio
async def test_two_calls_one_turn_seed_once_edit_in_place(monkeypatch, tmp_path):
    adapter, runner, result = await _run(
        monkeypatch, tmp_path, TwoCallOneTurnAgent, "sess-2call"
    )
    assert result["final_response"] == "done"
    # THE core contract: seeded ONCE, never stacked.
    assert len(adapter.sent) == 1, f"expected 1 seed, got {len(adapter.sent)} (stacking!)"
    assert adapter.sent[0]["content"].startswith("📋 Plan")
    # later state edited the SAME message id
    assert adapter.edits, "expected at least one in-place edit"
    assert {e["message_id"] for e in adapter.edits} == {"card-1"}
    blob = "\n".join(_all_text(adapter))
    assert "✅ completed (1m 30s) - first" in blob  # per-task time
    # seed carried the forum-topic metadata
    assert adapter.sent[0]["metadata"]["thread_id"] == "17585"
    # store persists for the session
    assert runner._todo_card_state["agent:main:telegram:group:-1001:17585"]["message_id"] == "card-1"


@pytest.mark.asyncio
async def test_terminal_card_survives_throttle_on_turn_end(monkeypatch, tmp_path):
    # B1: completed card within the throttle window + immediate return.
    adapter, runner, result = await _run(
        monkeypatch, tmp_path, TerminalThrottledAgent, "sess-b1"
    )
    assert result["final_response"] == "done"
    blob = "\n".join(_all_text(adapter))
    # The final all-completed state MUST land (not frozen on in_progress).
    assert "✅ completed" in blob, "terminal card lost to throttle (B1 regression)"
    assert "🔄 in progress" not in _all_text(adapter)[-1], "card stuck on in_progress"


@pytest.mark.asyncio
async def test_todo_progress_off_no_persistent_card(monkeypatch, tmp_path):
    # Back-compat: off -> no persistent card store entry; legacy path renders.
    adapter, runner, result = await _run(
        monkeypatch, tmp_path, TodoOffAgent, "sess-off",
        todo_progress="off", tool_progress="all",
    )
    assert result["final_response"] == "done"
    # No persistent card state was created.
    assert len(runner._todo_card_state) == 0
    # The legacy shared-bubble path still rendered the plan somewhere.
    blob = "\n".join(_all_text(adapter))
    assert "📋 Plan" in blob


@pytest.mark.asyncio
async def test_card_persists_across_two_turns(monkeypatch, tmp_path):
    # True cross-turn proof: drive TWO _run_agent turns sharing ONE adapter AND
    # ONE _todo_card_state store. Turn 1 leaves the plan UNFINISHED (pending
    # item); turn 2 CONTINUES it with merge=true. Because the turn-1 plan was
    # not finished, turn 2's update edits the SAME card (no fresh seed). (A
    # merge=false new plan after a FINISHED turn-1 would correctly seed fresh by
    # D4 — that is a different, intended scenario.)
    import collections

    class _Turn1Agent(_BaseTodoAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None):
            _emit_todo(self.tool_progress_callback, [
                {"id": "a", "content": "first", "status": "in_progress",
                 "started_at": _T0, "elapsed_seconds": 1.0},
                {"id": "b", "content": "second", "status": "pending"},
            ], merge=False)  # NOT finished -> card stays live for turn 2
            time.sleep(0.3)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    class _Turn2Agent(_BaseTodoAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None):
            _emit_todo(self.tool_progress_callback, [
                {"id": "a", "content": "first", "status": "completed",
                 "started_at": _T0, "ended_at": _T0 + 30.0, "elapsed_seconds": 30.0},
                {"id": "b", "content": "second", "status": "in_progress",
                 "started_at": _T0 + 30.0, "elapsed_seconds": 5.0},
            ], merge=True)  # continuation of the SAME plan -> edit in place
            time.sleep(0.3)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    shared_store = collections.OrderedDict()
    adapter = ProgressCaptureAdapter()

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    cfg = {"display": {"platforms": {"telegram": {
        "todo_progress": "on", "tool_progress": "off"}}}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    skey = "agent:main:telegram:group:-1001:17585"

    # ONE runner + store reused; SAME session_id both turns (one conversation).
    runner = _make_runner(adapter)
    runner._todo_card_state = shared_store

    for agent_cls in (_Turn1Agent, _Turn2Agent):
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = agent_cls
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
        await runner._run_agent(
            message="hi", context_prompt="", history=[], source=source,
            session_id="conv-1", session_key=skey,
        )

    # Exactly ONE seed across BOTH turns; turn 2 edited the same card in place.
    assert len(adapter.sent) == 1, (
        f"cross-turn stacking: {len(adapter.sent)} seeds (expected 1)"
    )
    assert adapter.edits, "turn 2 should have edited the turn-1 card"
    assert {e["message_id"] for e in adapter.edits} == {"card-1"}
    assert shared_store[skey]["message_id"] == "card-1"
    # turn 2's continuation state is reflected in the edited card
    assert "✅ completed" in adapter.edits[-1]["content"]
