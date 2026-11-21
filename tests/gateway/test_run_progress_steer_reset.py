"""A steer that closes the tool-progress bubble (``busy_continue_below``) must not drop the lines the
edit throttle is still holding: they land in the bubble above the steer, and later tool lines open a
new bubble below it. Drives the real ``GatewayRunner._run_agent`` progress loop; only the agent and
the platform transport are fakes."""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

KEY = "agent:main:telegram:group:-1001"
_RUNNER: list = []


class EditingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.bubbles: dict[str, str] = {}

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        message_id = f"m{len(self.bubbles) + 1}"
        self.bubbles[message_id] = content
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.bubbles[message_id] = content
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SteeredMidThrottleAgent:
    """Two tool lines inside one edit-throttle window, then a steer closes the bubble, then a third."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    @property
    def is_interrupted(self) -> bool:
        return False

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "web_search", "search alpha", {})
        time.sleep(0.4)  # health: allow HX013 -- the first line must be posted before the second
        self.tool_progress_callback("tool.started", "web_search", "search beta", {})
        turn = _RUNNER[0]._session_state(KEY).turn
        turn.ctx.progress_queue.put(("__reset__",))  # what a folded-in steer queues
        time.sleep(2.0)  # health: allow HX013 -- outlast the loop's real 1.5s edit throttle
        self.tool_progress_callback("tool.started", "web_search", "search gamma", {})
        time.sleep(2.0)  # health: allow HX013 -- let the loop post the post-steer bubble
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_lines_held_by_the_edit_throttle_survive_a_steer_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = SteeredMidThrottleAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    adapter = EditingAdapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    _RUNNER[:] = [runner]
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode, runner._prefill_messages, runner._running_agents = {}, [], {}
    runner._ephemeral_system_prompt, runner._reasoning_config, runner._fallback_model = "", None, None
    runner._provider_routing, runner._session_db = {}, None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")

    result = await runner._run_agent(
        message="hi", context_prompt="", history=[], source=source, session_id="s-steer", session_key=KEY)

    assert result["final_response"] == "done"
    progress = [text for text in adapter.bubbles.values() if "search" in text]
    assert len(progress) == 2, adapter.bubbles
    above, below = progress
    assert "search alpha" in above and "search beta" in above, above
    assert "search gamma" in below and "search beta" not in below, below
