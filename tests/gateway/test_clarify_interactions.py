"""Tests for gateway-native clarify interactions."""

import asyncio
import concurrent.futures
import importlib
import json
import sys
import time
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import GATEWAY_NO_RESPONSE, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource
from tools.clarify_tool import CLARIFY_TIMED_OUT_RESPONSE


class ClarifyCaptureAdapter:
    def __init__(self):
        self.sent = []
        self.typing = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None):
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def edit_message(self, chat_id, message_id, content):
        return SendResult(success=True, message_id=message_id)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        user_name="Alice",
        chat_id="c1",
        chat_type="dm",
    )


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._session_db = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_clarifications = {}
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._show_reasoning = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    return runner


class FakeClarifyAgent:
    def __init__(self, **kwargs):
        self.clarify_callback = kwargs["clarify_callback"]
        self.tools = []

    def run_conversation(self, user_message, system_message=None, conversation_history=None, task_id=None):
        answer = self.clarify_callback("Pick a mode", ["Fast", "Thorough"])
        return {
            "final_response": f"picked {answer}",
            "messages": [],
            "api_calls": 1,
        }


class FakeClarifyTimeoutAgent:
    def __init__(self, **kwargs):
        self.clarify_callback = kwargs["clarify_callback"]
        self.tools = []

    def run_conversation(self, user_message, system_message=None, conversation_history=None, task_id=None):
        answer = self.clarify_callback("Pick a mode", ["Fast", "Thorough"])
        return {
            "final_response": "This should be suppressed",
            "messages": [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "question": "Pick a mode",
                            "choices_offered": ["Fast", "Thorough"],
                            "user_response": "" if answer == CLARIFY_TIMED_OUT_RESPONSE else answer,
                            "timed_out": answer == CLARIFY_TIMED_OUT_RESPONSE,
                        }
                    ),
                }
            ],
            "api_calls": 1,
        }


@pytest.mark.asyncio
async def test_run_agent_clarify_uses_telegram_controls(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeClarifyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    adapter = ClarifyCaptureAdapter()
    runner = _make_runner(adapter)
    source = _make_source()

    run_task = asyncio.create_task(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-1",
            session_key="agent:main:telegram:dm:c1",
        )
    )

    deadline = time.monotonic() + 5
    while not runner._pending_clarifications and time.monotonic() < deadline:
        await asyncio.sleep(0.05)

    assert runner._pending_clarifications
    pending = runner._pending_clarifications["agent:main:telegram:dm:c1"]
    sent = adapter.sent[0]
    assert sent["content"].startswith("**Hermes needs your input**")
    controls = sent["metadata"]["controls"]
    assert controls["buttons"][0][0]["action"] == f"clarify-{pending['request_id']}-choice-0"
    assert controls["buttons"][1][0]["action"] == f"clarify-{pending['request_id']}-choice-1"
    assert controls["buttons"][2][0]["action"] == f"clarify-{pending['request_id']}-other"

    event = MessageEvent(
        text="Thorough",
        source=source,
        message_id="cb-1",
        metadata={"interaction": {"kind": "button", "action": f"clarify-{pending['request_id']}-choice-1"}},
    )
    handled = await runner._handle_message(event)

    assert handled == GATEWAY_NO_RESPONSE
    result = await run_task
    assert result["final_response"] == "picked Thorough"
    assert runner._pending_clarifications == {}


@pytest.mark.asyncio
async def test_stale_clarify_selection_does_not_fall_through_to_agent():
    runner = _make_runner(ClarifyCaptureAdapter())
    event = MessageEvent(
        text="Fast",
        source=_make_source(),
        message_id="cb-stale",
        metadata={"interaction": {"kind": "button", "action": "clarify-deadbeef-choice-0"}},
    )

    result = await runner._handle_message(event)

    assert "no longer active" in result


@pytest.mark.asyncio
async def test_stop_command_cancels_pending_clarify():
    runner = _make_runner(ClarifyCaptureAdapter())
    source = _make_source()
    future = concurrent.futures.Future()
    runner._pending_clarifications["agent:main:telegram:dm:c1"] = {
        "request_id": "abc123",
        "future": future,
        "choices": ["Fast", "Thorough"],
    }
    runner._running_agents["agent:main:telegram:dm:c1"] = MagicMock()

    result = await runner._handle_stop_command(
        MessageEvent(text="/stop", source=source, message_id="m-stop")
    )

    assert "Stopping the current task" in result
    assert future.done() is True
    assert "interrupted" in future.result().lower()


@pytest.mark.asyncio
async def test_run_agent_suppresses_followup_after_clarify_timeout(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeClarifyTimeoutAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    runner = _make_runner(ClarifyCaptureAdapter())
    runner._CLARIFY_TIMEOUT_SECONDS = 0.01

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="sess-1",
        session_key="agent:main:telegram:dm:c1",
    )

    assert result["final_response"] == GATEWAY_NO_RESPONSE
    assert runner._pending_clarifications == {}


# ---------------------------------------------------------------------------
# render_controls_as_text unit tests
# ---------------------------------------------------------------------------

from gateway.platforms.base import BasePlatformAdapter


class TestRenderControlsAsText:
    def test_empty_controls(self):
        assert BasePlatformAdapter.render_controls_as_text(None) == ""
        assert BasePlatformAdapter.render_controls_as_text({}) == ""
        assert BasePlatformAdapter.render_controls_as_text({"buttons": []}) == ""

    def test_single_row(self):
        controls = {"buttons": [[{"label": "Alpha", "action": "a"}]]}
        assert BasePlatformAdapter.render_controls_as_text(controls) == "1. Alpha"

    def test_multi_row(self):
        controls = {"buttons": [
            [{"label": "Alpha", "action": "a"}],
            [{"label": "Beta", "action": "b"}],
            [{"label": "Gamma", "action": "c"}],
        ]}
        result = BasePlatformAdapter.render_controls_as_text(controls)
        assert result == "1. Alpha\n2. Beta\n3. Gamma"

    def test_url_only_buttons_skipped(self):
        controls = {"buttons": [
            [{"label": "Action", "action": "do-it"}],
            [{"label": "Docs", "url": "https://example.com"}],
            [{"label": "Both", "action": "act", "url": "https://x.com"}],
        ]}
        result = BasePlatformAdapter.render_controls_as_text(controls)
        # "Docs" is url-only so skipped; "Both" has action so kept
        assert "1. Action" in result
        assert "2. Both" in result
        assert "Docs" not in result

    def test_multiple_buttons_per_row(self):
        controls = {"buttons": [
            [{"label": "A", "action": "a"}, {"label": "B", "action": "b"}],
        ]}
        result = BasePlatformAdapter.render_controls_as_text(controls)
        assert result == "1. A\n2. B"


# ---------------------------------------------------------------------------
# WhatsApp button overflow test
# ---------------------------------------------------------------------------


class TestTruncateControlsForPlatform:
    def test_whatsapp_truncation_with_overflow(self):
        """5 buttons with max=3 → 2 + Other as buttons, 2 overflow labels."""
        from gateway.platforms.whatsapp import WhatsAppAdapter

        adapter = MagicMock(spec=WhatsAppAdapter)
        adapter.MAX_INTERACTIVE_BUTTONS = 3

        runner = _make_runner(ClarifyCaptureAdapter())
        runner.adapters[Platform.TELEGRAM] = adapter  # reuse slot

        source = _make_source()
        controls = {"buttons": [
            [{"label": "Fast", "action": "clarify-abc-choice-0"}],
            [{"label": "Thorough", "action": "clarify-abc-choice-1"}],
            [{"label": "Balanced", "action": "clarify-abc-choice-2"}],
            [{"label": "Custom", "action": "clarify-abc-choice-3"}],
            [{"label": "Other", "action": "clarify-abc-other"}],
        ]}

        truncated, overflow, numeric_reply_choices = runner._truncate_controls_for_platform(controls, source)
        # Should keep first 2 + Other (last), overflow = [Balanced, Custom]
        flat = [b for row in truncated["buttons"] for b in row]
        assert len(flat) == 3
        assert [b["label"] for b in flat] == ["Fast", "Thorough", "Other"]
        assert overflow == ["Balanced", "Custom"]
        assert numeric_reply_choices == ["Balanced", "Custom"]

    def test_no_truncation_when_under_limit(self):
        """When buttons fit, return unchanged."""
        adapter = MagicMock()
        adapter.MAX_INTERACTIVE_BUTTONS = 10

        runner = _make_runner(ClarifyCaptureAdapter())
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_source()
        controls = {"buttons": [
            [{"label": "A", "action": "a"}],
            [{"label": "B", "action": "b"}],
        ]}

        truncated, overflow, numeric_reply_choices = runner._truncate_controls_for_platform(controls, source)
        assert truncated is controls
        assert overflow == []
        assert numeric_reply_choices == []

    def test_no_limit_returns_unchanged(self):
        """Platforms with no limit return controls unchanged."""
        runner = _make_runner(ClarifyCaptureAdapter())
        source = _make_source()
        controls = {"buttons": [[{"label": "X", "action": "x"}]] * 20}

        truncated, overflow, numeric_reply_choices = runner._truncate_controls_for_platform(controls, source)
        assert truncated is controls
        assert overflow == []
        assert numeric_reply_choices == []


@pytest.mark.asyncio
async def test_clarify_numeric_reply_maps_to_overflow_choices():
    adapter = ClarifyCaptureAdapter()
    adapter.MAX_INTERACTIVE_BUTTONS = 3

    runner = _make_runner(adapter)

    source = _make_source()
    future = concurrent.futures.Future()

    await runner._start_clarify_interaction(
        source=source,
        session_key="agent:main:telegram:dm:c1",
        question="Pick a mode",
        choices=["Fast", "Thorough", "Balanced", "Custom"],
        response_future=future,
        request_id="abc123",
    )

    pending = runner._pending_clarifications["agent:main:telegram:dm:c1"]
    assert pending["numeric_reply_choices"] == ["Balanced", "Custom"]

    result = await runner._handle_pending_clarification(
        MessageEvent(
            text="1",
            source=source,
            message_id="m1",
        ),
        "agent:main:telegram:dm:c1",
    )

    assert result == GATEWAY_NO_RESPONSE
    assert future.done() is True
    assert future.result() == "Balanced"
