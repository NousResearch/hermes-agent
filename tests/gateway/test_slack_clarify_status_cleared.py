"""Regression test: the clarify prompt must clear a durable status indicator.

Pausing the typing refresh loop is sufficient where the indicator is
ephemeral — Telegram's ``sendChatAction`` and Discord's typing trigger both
expire within a few seconds once refreshes stop.  Slack's
``assistant.threads.setStatus`` is durable: it stays on screen until it is
explicitly cleared.  Without a clear, the last written "still working…"
freezes above the clarify question and the user reads the prompt as
work-in-progress rather than as a question waiting on them, so nobody answers
and the turn hangs until ``clarify_timeout``.
"""

import importlib
import inspect
import sys
import types

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class StatusCaptureAdapter(BasePlatformAdapter):
    """Records the typing/status lifecycle in call order."""

    def __init__(self, platform=Platform.SLACK, accepts_metadata=True):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.calls = []
        self._accepts_metadata = accepts_metadata
        if not accepts_metadata:
            # Mimic an adapter that kept the historical stop_typing(chat_id)
            # signature, so the metadata-forwarding chokepoint is exercised.
            async def _legacy_stop_typing(chat_id):
                self.calls.append(("stop_typing", chat_id))

            self.stop_typing = _legacy_stop_typing

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="m-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.calls.append(("send_typing", chat_id))

    async def stop_typing(self, chat_id, metadata=None) -> None:
        self.calls.append(("stop_typing", chat_id, metadata))

    def pause_typing_for_chat(self, chat_id) -> None:
        self.calls.append(("pause_typing_for_chat", chat_id))
        return super().pause_typing_for_chat(chat_id)

    def resume_typing_for_chat(self, chat_id) -> None:
        self.calls.append(("resume_typing_for_chat", chat_id))
        return super().resume_typing_for_chat(chat_id)

    async def send_clarify(self, **kwargs) -> SendResult:
        self.calls.append(("send_clarify", kwargs.get("question")))
        return SendResult(success=True, message_id="clarify-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class ClarifyingAgent:
    """Calls the gateway's clarify callback once, mid-turn."""

    def __init__(self, **kwargs):
        self.clarify_callback = None
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        got = self.clarify_callback("Which environment?", ["staging", "production"])
        return {"final_response": f"got={got}", "messages": [], "api_calls": 1}


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
    runner.hooks = types.SimpleNamespace(loaded_hooks=False)
    runner.config = types.SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run_clarify_turn(monkeypatch, tmp_path, adapter, answer="staging"):
    runner = _make_runner(adapter)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ClarifyingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # Resolve the wait immediately instead of blocking on a real user reply.
    monkeypatch.setattr(gateway_run, "_clarify_send_then_wait", lambda fut, **kw: answer)

    source = SessionSource(platform=adapter.platform, chat_id="C1", chat_type="channel")
    return await runner._run_agent(
        message="deploy it",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-clarify-status",
        session_key=f"agent:main:{adapter.platform.value}:channel:C1",
    )


@pytest.mark.asyncio
async def test_clarify_clears_status_before_posting_the_prompt(monkeypatch, tmp_path):
    """The status is cleared, and cleared before the prompt is dispatched.

    Ordering matters: the clear and the prompt are both handed to the event
    loop, and a clear issued *after* the prompt can land behind the
    answer-time resume and blank an indicator that is legitimately active
    again (same hazard as #92201).
    """
    adapter = StatusCaptureAdapter()
    result = await _run_clarify_turn(monkeypatch, tmp_path, adapter)

    names = [c[0] for c in adapter.calls]
    assert "stop_typing" in names, f"status never cleared; calls={adapter.calls}"
    assert names.index("pause_typing_for_chat") < names.index("stop_typing")
    assert names.index("stop_typing") < names.index("send_clarify")
    assert result["final_response"] == "got=staging"


@pytest.mark.asyncio
async def test_clear_carries_thread_metadata(monkeypatch, tmp_path):
    """The clear must be thread-scoped, not channel-wide.

    A metadata-less clear falls back to "the only status in this channel",
    which would blank the indicator of a concurrent turn running in a sibling
    thread of the same channel.
    """
    adapter = StatusCaptureAdapter()
    await _run_clarify_turn(monkeypatch, tmp_path, adapter)

    clears = [c for c in adapter.calls if c[0] == "stop_typing"]
    assert clears, f"status never cleared; calls={adapter.calls}"
    # The call goes through _stop_typing_with_metadata, so an adapter whose
    # stop_typing accepts metadata is called with the keyword form.
    assert len(clears[0]) == 3, f"metadata not forwarded: {clears[0]}"


@pytest.mark.asyncio
async def test_legacy_adapter_signature_still_works(monkeypatch, tmp_path):
    """Adapters that kept stop_typing(chat_id) are unaffected.

    _stop_typing_with_metadata introspects the signature, so a metadata-less
    adapter is called in its historical form rather than raising TypeError.
    """
    adapter = StatusCaptureAdapter(accepts_metadata=False)
    await _run_clarify_turn(monkeypatch, tmp_path, adapter)

    clears = [c for c in adapter.calls if c[0] == "stop_typing"]
    assert clears, f"status never cleared; calls={adapter.calls}"
    assert len(clears[0]) == 2, f"legacy adapter got metadata: {clears[0]}"


def test_metadata_forwarding_chokepoint_exists():
    """Guards the assumption this fix relies on."""
    assert hasattr(BasePlatformAdapter, "_stop_typing_with_metadata")
    sig = inspect.signature(BasePlatformAdapter._stop_typing_with_metadata)
    assert "metadata" in sig.parameters
