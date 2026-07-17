"""Regression test for #66074 — a queued follow-up must not deliver the
first turn's intentional-silence marker to the chat.

When a gateway turn's successful agent result is an exact intentional-silence
marker (``[SILENT]`` / ``NO_REPLY``) AND another message is queued for the
same session, the queued-follow-up fallback in ``gateway.run`` used to deliver
the marker literally before processing the queued message.

Root cause: the stream consumer deliberately leaves final delivery
*unconfirmed* when it *suppresses* an intentional-silence marker.  The
queued-follow-up branch read ``not _already_streamed`` as "undelivered" and
sent the raw marker.

These tests drive the real ``GatewayRunner._run_agent`` queued-follow-up path
with a fake two-turn agent (the same harness style as
``test_queued_native_image_session_key``), plus a focused contract test for
the guard predicate.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.response_filters import is_intentional_silence_agent_result
from gateway.session import SessionSource


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

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
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SilenceThenReplyAgent:
    """First turn returns an intentional-silence marker; the queued second
    turn returns an ordinary reply."""

    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        if len(type(self).calls) == 1:
            return {
                "final_response": "[SILENT]",
                "messages": [],
                "api_calls": 1,
            }
        return {
            "final_response": "here is the real answer",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
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
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kw: "native"
    return runner


@pytest.mark.asyncio
async def test_queued_followup_suppresses_first_turn_silence_marker(monkeypatch, tmp_path):
    """Issue #66074: with a queued follow-up, a first-turn ``[SILENT]`` marker
    must NOT be delivered to the chat, and the queued turn must still run."""
    SilenceThenReplyAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = SilenceThenReplyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002",
        chat_type="group",
    )

    session_key = "agent:main:telegram:group:-1002"
    adapter._pending_messages[session_key] = MessageEvent(
        text="follow-up question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-silence-followup",
        session_key=session_key,
    )

    # The queued turn still ran (agent called twice) and its reply is final.
    assert len(SilenceThenReplyAgent.calls) == 2
    assert result["final_response"] == "here is the real answer"

    # The raw silence marker must never have been delivered to the chat.
    sent_contents = [s["content"] for s in adapter.sent]
    assert "[SILENT]" not in sent_contents, (
        f"intentional-silence marker was delivered to the chat: {sent_contents!r}"
    )
    assert not any(
        isinstance(c, str) and c.strip() == "[SILENT]" for c in sent_contents
    ), f"a silence marker leaked into a send: {sent_contents!r}"


@pytest.mark.asyncio
async def test_queued_followup_still_sends_normal_first_response(monkeypatch, tmp_path):
    """Guard is additive: a normal (non-silence) first response with a queued
    follow-up is still delivered before the queued turn runs."""

    class NormalThenReplyAgent(SilenceThenReplyAgent):
        calls = []

        def run_conversation(self, message, conversation_history=None, task_id=None):
            type(self).calls.append(message)
            if len(type(self).calls) == 1:
                return {"final_response": "first answer", "messages": [], "api_calls": 1}
            return {"final_response": "second answer", "messages": [], "api_calls": 1}

    NormalThenReplyAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = NormalThenReplyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003",
        chat_type="group",
    )
    session_key = "agent:main:telegram:group:-1003"
    adapter._pending_messages[session_key] = MessageEvent(
        text="follow-up question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-normal-followup",
        session_key=session_key,
    )

    assert len(NormalThenReplyAgent.calls) == 2
    assert result["final_response"] == "second answer"
    sent_contents = [s["content"] for s in adapter.sent]
    assert "first answer" in sent_contents, (
        f"normal first response was not delivered before the queued turn: {sent_contents!r}"
    )


def test_silence_predicate_contract():
    """Documents the exact contract the run.py guard relies on.

    ``is_intentional_silence_agent_result(result, first_response)`` must be
    True for a successful agent turn whose final response is an exact marker,
    and False for ordinary prose or a failed turn.
    """
    assert is_intentional_silence_agent_result({"final_response": "[SILENT]"}, "[SILENT]") is True
    assert is_intentional_silence_agent_result({"final_response": "NO_REPLY"}, "NO_REPLY") is True
    assert is_intentional_silence_agent_result({"final_response": "hello there"}, "hello there") is False
    # A failed turn is never treated as intentional silence.
    assert is_intentional_silence_agent_result({"final_response": "[SILENT]", "failed": True}, "[SILENT]") is False
    # Substantive prose that merely mentions the marker is delivered normally.
    assert is_intentional_silence_agent_result(
        {"final_response": "I will reply with [SILENT] when idle"},
        "I will reply with [SILENT] when idle",
    ) is False
