"""Tests for the diagnostic_status display config gating in the gateway.

``diagnostic_status`` defaults to ``"all"`` (forward internal recovery messages).
Operators who do not want implementation-detail messages in customer-facing chats
set ``display.diagnostic_status: off`` (or per-platform).  When ``"off"``, the
agent's ``status_callback`` is set to ``None`` so messages like
"⚠️ Model returned empty after tool calls" never reach the adapter.
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


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StatusCaptureAdapter(BasePlatformAdapter):
    """Records every send() call so tests can assert on diagnostic messages."""

    def __init__(self, platform=Platform.WHATSAPP):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="m1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class DiagnosticEmittingAgent:
    """Agent that fires one status event (simulating a recovery nudge) then
    returns a normal response."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        # Simulate the "empty after tool calls" recovery nudge
        if self.status_callback is not None:
            self.status_callback(
                "agent.status",
                "⚠️ Model returned empty after tool calls — nudging to continue",
            )
        return {"final_response": "Done", "messages": [], "api_calls": 2}


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


def _install_fakes(monkeypatch, *, diagnostic_off: bool, platform=Platform.WHATSAPP):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = DiagnosticEmittingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    platform_key = platform.value if hasattr(platform, "value") else str(platform).lower()
    # Suppress by explicitly setting "off"; omitting the key leaves the default "all" in effect.
    cfg = {
        "display": {
            "platforms": {
                platform_key: {"diagnostic_status": "off"},
            }
        }
    } if diagnostic_off else {}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    return gateway_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostic_status_off_suppresses_status_callback(monkeypatch, tmp_path):
    """diagnostic_status: off — recovery messages must not reach adapter."""
    adapter = StatusCaptureAdapter(platform=Platform.WHATSAPP)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, diagnostic_off=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.WHATSAPP, chat_id="+1234567890")
    session_key = "agent:main:whatsapp:dm:+1234567890"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "Done"
    diagnostic_sends = [
        m for m in adapter.sent
        if "nudging to continue" in (m.get("content") or "")
        or "Model returned empty" in (m.get("content") or "")
    ]
    assert diagnostic_sends == [], (
        f"Diagnostic messages leaked to adapter: {diagnostic_sends}"
    )


@pytest.mark.asyncio
async def test_diagnostic_status_all_forwards_status_callback(monkeypatch, tmp_path):
    """Default (diagnostic_status=all) — recovery messages are forwarded to adapter."""
    adapter = StatusCaptureAdapter(platform=Platform.WHATSAPP)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, diagnostic_off=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.WHATSAPP, chat_id="+1234567890")
    session_key = "agent:main:whatsapp:dm:+1234567890"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "Done"
    diagnostic_sends = [
        m for m in adapter.sent
        if "nudging to continue" in (m.get("content") or "")
        or "Model returned empty" in (m.get("content") or "")
    ]
    assert len(diagnostic_sends) == 1, (
        f"Expected exactly one diagnostic message, got: {diagnostic_sends}"
    )
