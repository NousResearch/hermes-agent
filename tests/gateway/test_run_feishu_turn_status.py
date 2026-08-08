"""Feishu keeps one editable turn-status message for all tool progress."""

import asyncio
import importlib
import queue
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class FeishuStatusAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.FEISHU)
        self.sent = []
        self.edits = []
        self.typing = []
        self.registered = []
        self.reaction_starts = []
        self.status_updates = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="unexpected-new-message")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ) -> SendResult:
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    def register_turn_status_message(self, **kwargs) -> None:
        self.registered.append(kwargs)

    async def start_processing_reaction(self, message_id: str) -> None:
        self.reaction_starts.append(message_id)

    def update_turn_status_progress(
        self, trigger_message_id: str, *, tool_count: int, current_stage: str
    ) -> None:
        self.status_updates.append({
            "trigger_message_id": trigger_message_id,
            "tool_count": tool_count,
            "current_stage": current_stage,
        })


class IntegratedProgressAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        for index in range(10):
            self.tool_progress_callback(
                "tool.started", "terminal", f"command-{index}", {}
            )
            self.tool_progress_callback(
                "tool.completed", "terminal", None, None
            )
        time.sleep(0.5)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class PermanentEditFailureAdapter(FeishuStatusAdapter):
    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=False, error="message cannot be edited")


class InitialStatusFailureAdapter(FeishuStatusAdapter):
    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=False, error="status send failed")


def _make_gateway_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
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
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.asyncio
async def test_gateway_precreates_one_threaded_feishu_status_message(
    monkeypatch, tmp_path
):
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.dump({}), encoding="utf-8")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = IntegratedProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = FeishuStatusAdapter()
    runner = _make_gateway_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="omt_thread",
    )

    result = await runner._run_agent(
        message="run tools",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-status",
        session_key="agent:main:feishu:thread:oc_chat:omt_thread",
        event_message_id="om_trigger",
    )

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    status_send = adapter.sent[0]
    assert status_send["content"] == "處理中"
    assert status_send["reply_to"] == "om_trigger"
    assert status_send["metadata"] == {"thread_id": "omt_thread"}
    assert len(adapter.registered) == 1
    registration = adapter.registered[0]
    assert registration["trigger_message_id"] == "om_trigger"
    assert registration["chat_id"] == "oc_chat"
    assert registration["status_message_id"] == "unexpected-new-message"
    assert registration["lifecycle_id"].startswith("om_trigger:")
    assert adapter.reaction_starts == ["unexpected-new-message"]
    assert getattr(source, "_feishu_turn_status_adapter") is adapter
    assert adapter.edits
    assert {edit["message_id"] for edit in adapter.edits} == {
        "unexpected-new-message"
    }
    assert adapter.status_updates[-1]["tool_count"] == 10


@pytest.mark.asyncio
async def test_initial_feishu_status_failure_never_falls_back_to_progress_bubbles(
    monkeypatch, tmp_path
):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({
            "display": {
                "platforms": {
                    "feishu": {
                        "tool_progress": "all",
                        "turn_status_message": True,
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = IntegratedProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = InitialStatusFailureAdapter()
    runner = _make_gateway_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="omt_thread",
    )

    result = await runner._run_agent(
        message="run tools",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-status-failure",
        session_key="agent:main:feishu:thread:oc_chat:omt_thread",
        event_message_id="om_trigger",
    )

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"] == "處理中"
    assert adapter.edits == []
    assert adapter.registered == []


@pytest.mark.asyncio
async def test_ten_tool_calls_only_edit_registered_feishu_status_message():
    adapter = FeishuStatusAdapter()
    replacement_adapter = FeishuStatusAdapter()
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="omt_thread",
    )
    progress_queue = queue.Queue()
    for index in range(10):
        progress_queue.put(f"tool-{index}")
        progress_queue.put(("__status_completed__",))

    current = True
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: current,
        progress_queue=progress_queue,
        progress_grouping="separate",
        tool_progress_enabled=True,
        status_message_id="om_status",
        turn_status_enabled=True,
        _turn_status_adapter=adapter,
        _progress_metadata={
            "thread_id": "omt_thread",
            "reply_to_message_id": "om_trigger",
        },
        _progress_reply_to="om_trigger",
    )
    # A reconnect may replace the registry entry mid-turn. Progress must stay
    # bound to the adapter that created and registered this status message.
    gateway = SimpleNamespace(
        _adapter_for_source=lambda _source: replacement_adapter
    )
    runner = TurnRunner(gateway, ctx)

    task = asyncio.create_task(runner.send_progress_messages())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.wait_for(task, timeout=2.0)

    assert adapter.sent == []
    assert adapter.edits
    assert replacement_adapter.edits == []
    assert {call["message_id"] for call in adapter.edits} == {"om_status"}
    final = adapter.edits[-1]["content"]
    assert final.startswith("處理中\n目前：tool-9\n已完成：10 個工具操作")
    assert "tool-4" not in final
    for index in range(5, 10):
        assert f"• tool-{index}" in final


@pytest.mark.asyncio
async def test_permanent_feishu_status_edit_failure_never_falls_back_to_new_messages():
    adapter = PermanentEditFailureAdapter()
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat")
    progress_queue = queue.Queue()
    for index in range(3):
        progress_queue.put(f"tool-{index}")
        progress_queue.put(("__status_completed__",))

    current = True
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: current,
        progress_queue=progress_queue,
        tool_progress_enabled=True,
        status_message_id="om_status",
        turn_status_enabled=True,
    )
    gateway = SimpleNamespace(_adapter_for_source=lambda _source: adapter)
    task = asyncio.create_task(TurnRunner(gateway, ctx).send_progress_messages())
    await asyncio.sleep(0.8)
    current = False
    await asyncio.wait_for(task, timeout=2.0)

    assert adapter.edits
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_feishu_status_reset_marker_does_not_create_second_message():
    adapter = FeishuStatusAdapter()
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat")
    progress_queue = queue.Queue()
    progress_queue.put("first tool")
    progress_queue.put(("__reset__",))
    progress_queue.put("second tool")

    current = True
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: current,
        progress_queue=progress_queue,
        progress_grouping="grouped",
        tool_progress_enabled=True,
        status_message_id="om_status",
        turn_status_enabled=True,
    )
    gateway = SimpleNamespace(_adapter_for_source=lambda _source: adapter)
    runner = TurnRunner(gateway, ctx)

    task = asyncio.create_task(runner.send_progress_messages())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.wait_for(task, timeout=2.0)

    assert adapter.sent == []
    assert {call["message_id"] for call in adapter.edits} == {"om_status"}
    assert "first tool" in adapter.edits[-1]["content"]
    assert "second tool" in adapter.edits[-1]["content"]
