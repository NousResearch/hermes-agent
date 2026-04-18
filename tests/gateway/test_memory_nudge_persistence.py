"""Tests for gateway memory/skill nudge counter persistence across fresh agent instances."""

import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class SilentAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.FEISHU):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CounterStore:
    def __init__(self):
        self.memory_turns = 0
        self.skill_iters = 0
        self.updated = []

    def get_memory_nudge_turns(self, session_key: str) -> int:
        return self.memory_turns

    def get_skill_nudge_iters(self, session_key: str) -> int:
        return self.skill_iters

    def update_session(
        self,
        session_key: str,
        last_prompt_tokens=None,
        memory_turns_since_review=None,
        skill_iters_since_review=None,
    ):
        if memory_turns_since_review is not None:
            self.memory_turns = memory_turns_since_review
        if skill_iters_since_review is not None:
            self.skill_iters = skill_iters_since_review
        self.updated.append(
            {
                "session_key": session_key,
                "last_prompt_tokens": last_prompt_tokens,
                "memory_turns_since_review": memory_turns_since_review,
                "skill_iters_since_review": skill_iters_since_review,
            }
        )


class FakeAgent:
    starts = []

    def __init__(self, **kwargs):
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.model = kwargs.get("model", "fake-model")
        self._turns_since_memory = 0
        self._iters_since_skill = 0

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).starts.append(
            {
                "memory": self._turns_since_memory,
                "skill": self._iters_since_skill,
            }
        )
        self._turns_since_memory += 4
        self._iters_since_skill += 3
        return {
            "final_response": f"ok:{message}",
            "messages": [],
            "api_calls": 1,
        }


class ThresholdReviewAgent(FakeAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).starts.append(
            {
                "memory": self._turns_since_memory,
                "skill": self._iters_since_skill,
            }
        )
        if self._turns_since_memory >= 9:
            self._turns_since_memory = 0
            cb = getattr(self, "background_review_callback", None)
            if cb:
                cb("💾 Memory updated.")
        else:
            self._turns_since_memory += 1

        if self._iters_since_skill >= 14:
            self._iters_since_skill = 0
            cb = getattr(self, "background_review_callback", None)
            if cb:
                cb("🛠️ Skill reviewed.")
        else:
            self._iters_since_skill += 1

        return {
            "final_response": f"ok:{message}",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter, session_store):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner.session_store = session_store
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None  # force fresh agent each turn
    runner._running_agents_ts = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=None,
    )
    runner._resolve_session_agent_runtime = lambda **kwargs: (
        "fake-model",
        {"api_key": "***", "provider": "custom", "base_url": "http://127.0.0.1:8080/v1"},
    )
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": None,
    }
    runner._load_reasoning_config = lambda: None
    runner._load_service_tier = lambda: None
    runner._agent_config_signature = lambda *args, **kwargs: "sig"
    runner._cleanup_agent_resources = lambda agent: None
    runner._evict_cached_agent = lambda session_key: None
    return runner


@pytest.mark.asyncio
async def test_run_agent_persists_memory_and_skill_nudge_counters_across_fresh_agents(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    session_store = CounterStore()
    adapter = SilentAdapter()
    runner = _make_runner(adapter, session_store)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test_chat",
        chat_type="dm",
        user_id="ou_test_user",
    )

    FakeAgent.starts = []

    first = await runner._run_agent(
        message="first",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:feishu:dm:oc_test_chat",
    )
    second = await runner._run_agent(
        message="second",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:feishu:dm:oc_test_chat",
    )

    assert first["final_response"] == "ok:first"
    assert second["final_response"] == "ok:second"
    assert FakeAgent.starts == [
        {"memory": 0, "skill": 0},
        {"memory": 4, "skill": 3},
    ]
    assert session_store.memory_turns == 8
    assert session_store.skill_iters == 6
    assert [u["memory_turns_since_review"] for u in session_store.updated] == [4, 8]
    assert [u["skill_iters_since_review"] for u in session_store.updated] == [3, 6]


@pytest.mark.asyncio
async def test_run_agent_releases_background_review_when_persisted_counters_hit_threshold(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ThresholdReviewAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    session_store = CounterStore()
    session_store.memory_turns = 9
    session_store.skill_iters = 14
    adapter = SilentAdapter()
    runner = _make_runner(adapter, session_store)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test_chat",
        chat_type="dm",
        user_id="ou_test_user",
    )
    session_key = "agent:main:feishu:dm:oc_test_chat"

    ThresholdReviewAgent.starts = []

    result = await runner._run_agent(
        message="threshold",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "ok:threshold"
    assert ThresholdReviewAgent.starts == [{"memory": 9, "skill": 14}]
    assert adapter.sent == []

    release_cb = adapter._post_delivery_callbacks[session_key]
    await __import__("asyncio").to_thread(release_cb)
    await __import__("asyncio").sleep(0)

    assert [msg["content"] for msg in adapter.sent] == [
        "💾 Memory updated.",
        "🛠️ Skill reviewed.",
    ]
    assert session_store.memory_turns == 0
    assert session_store.skill_iters == 0
