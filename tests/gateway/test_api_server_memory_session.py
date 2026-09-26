"""api_server must reuse the session memory provider across requests (#120116).

Every POST /v1/chat/completions builds a fresh agent; external providers whose
recall is the previous turn's queued async prefetch (hindsight default mode,
honcho, retaindb) therefore inject nothing on turn 2+ of a continued session,
even with X-Hermes-Session-Id continuity. The adapter must hand the same
session's idle memory manager to the next turn's fresh agent.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

PROVIDER_NAME = "fake-async-120116"
SESSION = "gwsess120116"


class FakeAsyncProvider(MemoryProvider):
    """Hindsight-like async recall: prefetch() consumes the PREVIOUS turn's
    queued result, so a fresh instance per request always injects nothing."""

    instances = []

    @property
    def name(self):
        return PROVIDER_NAME

    def __init__(self):
        FakeAsyncProvider.instances.append(self)
        self._pending = None

    def is_available(self):
        return True

    def initialize(self, session_id="", **kwargs):
        self.session_id = session_id

    def get_tool_schemas(self):
        return []

    def prefetch(self, query, *, session_id=""):
        result, self._pending = self._pending, None
        return result or ""

    def queue_prefetch(self, query, *, session_id=""):
        if query:
            self._pending = "recall-for:%s" % query


class FakeAgent:
    """Mirrors the real turn lifecycle: prefetch at turn start, sync + queue
    at turn end (drained, like the issue's 45s inter-turn gap)."""

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self._last_compaction_in_place = False
        mgr = MemoryManager()
        mgr.add_provider(FakeAsyncProvider())
        mgr.initialize_all(
            session_id=self.session_id or "", hermes_home="/tmp/fake-home-120116",
            platform="api_server", agent_context="primary")
        self._memory_manager = mgr

    def run_conversation(self, user_message, conversation_history, task_id=None):
        mgr = self._memory_manager
        injected = mgr.prefetch_all(user_message, session_id=self.session_id or "") if mgr else ""
        final = "INJECTED[%s]" % injected if injected else "INJECTED[EMPTY]"
        if mgr:
            mgr.sync_all(user_message, final, session_id=self.session_id or "")
            mgr.queue_prefetch_all(user_message, session_id=self.session_id or "")
            assert mgr.flush_pending(timeout=15)
        return {"final_response": final, "session_id": self.session_id}


def _make_adapter(monkeypatch):
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "stub", "base_url": "http://localhost:9/v1", "api_mode": "openai"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "stub-model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": None),
    )
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    return adapter


@pytest.mark.asyncio
async def test_continued_session_second_turn_gets_prefetch(monkeypatch):
    FakeAsyncProvider.instances.clear()
    adapter = _make_adapter(monkeypatch)
    turn1, _ = await adapter._run_agent(
        "Remember that Anjali loves giraffes.", [], session_id=SESSION)
    assert turn1["final_response"] == "INJECTED[EMPTY]"
    turn2, _ = await adapter._run_agent(
        "What does Anjali love?", [], session_id=SESSION)
    assert "recall-for:Remember that Anjali loves giraffes." in turn2["final_response"]


@pytest.mark.asyncio
async def test_memory_managers_are_not_shared_across_sessions(monkeypatch):
    FakeAsyncProvider.instances.clear()
    adapter = _make_adapter(monkeypatch)
    await adapter._run_agent("Remember that Mochi loves trains.", [], session_id="sess-A")
    turn_b, _ = await adapter._run_agent("What does Mochi love?", [], session_id="sess-B")
    assert turn_b["final_response"] == "INJECTED[EMPTY]"
