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


class DiscordProgressAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 80
    _next_mid = 200

    def __init__(self, *, fail_edit: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
        self.fail_edit = fail_edit
        self.sent = []
        self.edits = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        DiscordProgressAdapter._next_mid += 1
        return str(DiscordProgressAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "message_id": mid,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, *, metadata=None) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": str(message_id),
                "content": content,
                "metadata": metadata,
            }
        )
        if self.fail_edit:
            return SendResult(success=False, error="edit failed")
        return SendResult(success=True, message_id=str(message_id))

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class ProgressAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = None
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "skill_view", "load skill", {})
            time.sleep(0.25)
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.25)
            cb("tool.started", "search_files", "find gateway", {})
            time.sleep(0.25)
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


def _install_fakes(monkeypatch):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "display": {
                "platforms": {
                    "discord": {
                        "tool_progress": "all",
                        "final_response_edits_progress": True,
                    }
                }
            }
        },
    )
    return gateway_run


@pytest.mark.asyncio
async def test_discord_progress_updates_one_message_and_returns_edit_target(monkeypatch, tmp_path):
    adapter = DiscordProgressAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="42")

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:discord:dm:42",
    )

    assert result["final_response"] == "done"
    assert result["final_response_edits_progress"] is True
    assert result["progress_message_id"] == adapter.sent[0]["message_id"]
    assert len(adapter.sent) == 1
    assert adapter.edits
    assert adapter.edits[-1]["message_id"] == adapter.sent[0]["message_id"]
    assert "\n" not in adapter.edits[-1]["content"]
    assert "ファイル検索" in adapter.edits[-1]["content"]
    assert "スキル確認" not in adapter.edits[-1]["content"]
    assert "端末確認" not in adapter.edits[-1]["content"]


@pytest.mark.asyncio
async def test_final_response_edits_progress_message_without_new_send():
    adapter = DiscordProgressAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.DISCORD, chat_id="42")
    event = SimpleNamespace(source=source, message_id="9")

    ok = await runner._edit_progress_message_to_final_response(
        adapter=adapter,
        source=source,
        event=event,
        message_id="201",
        content="final answer",
    )

    assert ok is True
    assert adapter.edits[-1]["message_id"] == "201"
    assert adapter.edits[-1]["content"] == "final answer"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_final_response_edit_failure_returns_false_for_send_fallback():
    adapter = DiscordProgressAdapter(fail_edit=True)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.DISCORD, chat_id="42")
    event = SimpleNamespace(source=source, message_id="9")

    ok = await runner._edit_progress_message_to_final_response(
        adapter=adapter,
        source=source,
        event=event,
        message_id="201",
        content="final answer",
    )

    assert ok is False
    assert adapter.edits[-1]["message_id"] == "201"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_final_response_splits_only_when_discord_limit_requires_it():
    adapter = DiscordProgressAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.DISCORD, chat_id="42")
    event = SimpleNamespace(source=source, message_id="9")

    ok = await runner._edit_progress_message_to_final_response(
        adapter=adapter,
        source=source,
        event=event,
        message_id="201",
        content="x" * 180,
    )

    assert ok is True
    assert adapter.edits[-1]["message_id"] == "201"
    assert adapter.sent
    assert all(len(entry["content"]) <= adapter.MAX_MESSAGE_LENGTH for entry in adapter.sent)
