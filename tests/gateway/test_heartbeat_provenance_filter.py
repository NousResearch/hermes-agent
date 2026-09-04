"""Tests for heartbeat activity provenance filtering and unified status filtering.

Ensures that internal maintenance activities (such as context compression)
do not leak internal diagnostic descriptions into user-facing heartbeat
notifications across messaging gateway surfaces (e.g. Telegram), while
preserving legitimate heartbeats and visible tool activity.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.conversation_compression import COMPACTION_STATUS
from agent.session_activity import ActivityProvenance
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import (
    _INTERNAL_ACTIVITY_PROVENANCES,
    _is_internal_activity_provenance,
    _prepare_gateway_status_message,
)
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Test Adapter
# ---------------------------------------------------------------------------

class HeartbeatCaptureAdapter(BasePlatformAdapter):
    """Adapter capturing sends and edits for heartbeat assertions."""

    _next_mid = 200

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        HeartbeatCaptureAdapter._next_mid += 1
        return str(HeartbeatCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata}
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


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
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        multiplex_profiles=False,
    )
    return runner


def _install_fakes(monkeypatch, agent_cls, *, extra_display_cfg=None):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    real_float_env = gateway_run._float_env
    monkeypatch.setattr(
        gateway_run,
        "_float_env",
        lambda name, default: 0.05 if name == "HERMES_AGENT_NOTIFY_INTERVAL" else real_float_env(name, default),
    )

    cfg = {"display": extra_display_cfg or {}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    return gateway_run


# ---------------------------------------------------------------------------
# Unit Tests for Helper Function
# ---------------------------------------------------------------------------

def test_is_internal_activity_provenance_constants():
    """Verify internal activity provenance classification."""
    # Enum instances
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_TIMEOUT) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_COOLDOWN) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_TURNHOLD) is True

    # String values
    assert _is_internal_activity_provenance("agent.compression") is True
    assert _is_internal_activity_provenance("agent.compression_timeout") is True
    assert _is_internal_activity_provenance("AGENT.COMPRESSION") is True
    assert _is_internal_activity_provenance("agent.compression.future_variant") is True

    # Non-internal / user / tool provenances
    assert _is_internal_activity_provenance(ActivityProvenance.UNKNOWN) is False
    assert _is_internal_activity_provenance("unknown") is False
    assert _is_internal_activity_provenance(None) is False
    assert _is_internal_activity_provenance("") is False
    assert _is_internal_activity_provenance("user") is False
    assert _is_internal_activity_provenance("tool.call") is False


def test_prepare_gateway_status_message_sanitizes_heartbeat_content():
    """Test that heartbeat messages passing through _prepare_gateway_status_message
    preserve legitimate status while redacting secrets and stripping noise."""
    # Legitimate heartbeats pass through untouched
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min") == "⏳ Working — 3 min"
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min — web_search") == "⏳ Working — 3 min — web_search"
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "still on it") == "still on it"

    # Routine compression noise is suppressed
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM, "heartbeat", f"⏳ Working — 3 min — {COMPACTION_STATUS}"
    ) is None
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min — preflight compression"
    ) is None

    # Redacts secrets in heartbeat
    msg_with_secret = "⏳ Working — 1 min — tool_call(api_key=sk-ant-api03-abcdef1234567890abcdef1234567890abcdef1234567890)"
    sanitized = _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", msg_with_secret)
    assert sanitized is not None
    assert "sk-ant-api03" not in sanitized


# ---------------------------------------------------------------------------
# Integration Tests for Long-Running Heartbeat in _run_agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heartbeat_suppresses_compression_provenance(monkeypatch, tmp_path):
    """When agent has internal compression provenance, last_activity_desc must
    NOT be included in heartbeat; only terse '⏳ Working — N min' is sent."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.04")

    class CompressingAgent:
        def __init__(self, **kwargs):
            self._current_tool = None
            self._last_activity_desc = "context compression in progress"
            self._last_activity_provenance = ActivityProvenance.AGENT_COMPRESSION
            self._api_call_count = 1
            self.max_iterations = 10

        def get_activity_summary(self):
            return {
                "current_tool": self._current_tool,
                "last_activity_desc": self._last_activity_desc,
                "last_activity_provenance": self._last_activity_provenance,
                "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations,
            }

        def run_conversation(self, message, conversation_history=None, task_id=None):
            time.sleep(0.18)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, CompressingAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")
    result = await runner._run_agent(
        message="summarize discussion",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-comp-hb",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done"
    all_messages = [s["content"] for s in adapter.sent] + [e["content"] for e in adapter.edits]
    assert len(all_messages) >= 1

    for msg in all_messages:
        # Must NOT leak internal compression description
        assert "context compression in progress" not in msg
        assert "compression" not in msg
        # Must retain legitimate heartbeat prefix
        assert "⏳ Working" in msg


@pytest.mark.asyncio
async def test_heartbeat_preserves_active_tool(monkeypatch, tmp_path):
    """When agent has an active current_tool, it takes precedence and is included."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.04")

    class ToolRunningAgent:
        def __init__(self, **kwargs):
            self._current_tool = "web_search"
            # Even if last_activity_desc had a compression stamp from earlier
            self._last_activity_desc = "context compression in progress"
            self._last_activity_provenance = ActivityProvenance.AGENT_COMPRESSION
            self._api_call_count = 1
            self.max_iterations = 10

        def get_activity_summary(self):
            return {
                "current_tool": self._current_tool,
                "last_activity_desc": self._last_activity_desc,
                "last_activity_provenance": self._last_activity_provenance,
                "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations,
            }

        def run_conversation(self, message, conversation_history=None, task_id=None):
            time.sleep(0.18)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ToolRunningAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")
    result = await runner._run_agent(
        message="search topic",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-tool-hb",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done"
    all_messages = [s["content"] for s in adapter.sent] + [e["content"] for e in adapter.edits]
    assert len(all_messages) >= 1

    # Active tool is preserved in heartbeat
    assert any("web_search" in msg for msg in all_messages)
    # Internal compression is NOT leaked
    assert not any("context compression in progress" in msg for msg in all_messages)


@pytest.mark.asyncio
async def test_heartbeat_preserves_user_facing_activity_desc(monkeypatch, tmp_path):
    """When last_activity_provenance is UNKNOWN (non-internal), the activity
    description is preserved in the heartbeat."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.04")

    class NormalWorkAgent:
        def __init__(self, **kwargs):
            self._current_tool = None
            self._last_activity_desc = "synthesizing search results"
            self._last_activity_provenance = ActivityProvenance.UNKNOWN
            self._api_call_count = 2
            self.max_iterations = 10

        def get_activity_summary(self):
            return {
                "current_tool": self._current_tool,
                "last_activity_desc": self._last_activity_desc,
                "last_activity_provenance": self._last_activity_provenance,
                "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations,
            }

        def run_conversation(self, message, conversation_history=None, task_id=None):
            time.sleep(0.18)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, NormalWorkAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")
    result = await runner._run_agent(
        message="work task",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-normal-hb",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done"
    all_messages = [s["content"] for s in adapter.sent] + [e["content"] for e in adapter.edits]
    assert len(all_messages) >= 1
    assert any("synthesizing search results" in msg for msg in all_messages)


@pytest.mark.asyncio
async def test_heartbeat_fallback_when_noisy_detail_filtered(monkeypatch, tmp_path):
    """If an activity description happens to trigger the noise regex (e.g. unknown
    provenance with routine compression phrase), _prepare_gateway_status_message
    filters it and the heartbeat safely falls back to '⏳ Working — N min'."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.04")

    class NoisyDescAgent:
        def __init__(self, **kwargs):
            self._current_tool = None
            # Unknown provenance so provenance filter doesn't catch it,
            # but description matches _TELEGRAM_NOISY_STATUS_RE
            self._last_activity_desc = COMPACTION_STATUS
            self._last_activity_provenance = ActivityProvenance.UNKNOWN
            self._api_call_count = 1
            self.max_iterations = 10

        def get_activity_summary(self):
            return {
                "current_tool": self._current_tool,
                "last_activity_desc": self._last_activity_desc,
                "last_activity_provenance": self._last_activity_provenance,
                "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations,
            }

        def run_conversation(self, message, conversation_history=None, task_id=None):
            time.sleep(0.18)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter = HeartbeatCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, NoisyDescAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group")
    result = await runner._run_agent(
        message="work task",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-noisy-hb",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done"
    all_messages = [s["content"] for s in adapter.sent] + [e["content"] for e in adapter.edits]
    assert len(all_messages) >= 1

    # Noisy phrase was stripped by status filter fallback
    for msg in all_messages:
        assert COMPACTION_STATUS not in msg
        assert "⏳ Working" in msg
