"""Regression tests for #52374 — raw clarify tool-call JSON must never leak
into the chat as a tool-progress bubble.

The adapter's ``send_clarify`` is the user-facing rendering of a clarify
prompt (interactive buttons, or the numbered-text fallback).  The gateway's
tool-progress callback used to also render a progress bubble for the
``clarify`` tool.started event — in verbose mode that bubble contains the raw
tool-call args JSON (``{"question": ..., "choices": [...]}``), and because the
progress queue drains on a background task the JSON landed right underneath
the rendered interactive prompt on Slack.
"""

import importlib
import sys
import time
import types

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    """Records every send so the test can assert nothing leaked."""

    def __init__(self, platform=Platform.SLACK):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="m-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class NativeClarifyCaptureAdapter(ProgressCaptureAdapter):
    def __init__(self, platform=Platform.SLACK):
        super().__init__(platform=platform)
        self.clarifies = []

    async def send_clarify(
        self, chat_id, question, choices, clarify_id, session_key, metadata=None
    ) -> SendResult:
        self.clarifies.append(
            {
                "chat_id": chat_id,
                "question": question,
                "choices": choices,
                "clarify_id": clarify_id,
                "session_key": session_key,
            }
        )
        return SendResult(success=True, message_id="clarify-native-1")


class ClarifyThenToolAgent:
    """Emits a clarify tool.started (with raw args) then a normal tool."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb(
                "tool.started",
                "clarify",
                "Which environment?",
                {"question": "Which environment?", "choices": ["staging", "production"]},
            )
            time.sleep(0.35)
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FencedClarifyAgent:
    PRIVATE_QUESTION = "PRIVATE_SENTINEL_81312_CLARIFY_QUESTION"
    PRIVATE_CHOICE = "PRIVATE_SENTINEL_81312_CLARIFY_CHOICE"
    RAW_QUESTION = (
        "Visible question?\n"
        f"<memory-context>\n{PRIVATE_QUESTION}\n</memory-context>"
    )
    RAW_CHOICES = [
        "Visible choice",
        f"<memory-context>\n{PRIVATE_CHOICE}\n</memory-context>",
    ]

    def __init__(self, **kwargs):
        self.tools = []
        self.clarify_callback = None

    def run_conversation(self, message, conversation_history=None, task_id=None):
        assert self.clarify_callback is not None
        response = self.clarify_callback(self.RAW_QUESTION, self.RAW_CHOICES)
        return {"final_response": response, "messages": [], "api_calls": 1}


class EmptyFencedClarifyAgent(FencedClarifyAgent):
    RAW_QUESTION = "<memory-context>\nPRIVATE_ONLY_QUESTION\n</memory-context>"
    RAW_CHOICES = ["visible choice"]


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
    runner.hooks = types.SimpleNamespace(loaded_hooks=False)
    runner.config = types.SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(monkeypatch, mode, agent_cls=ClarifyThenToolAgent):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", mode)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 — register terminal emoji

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    return gateway_run


@pytest.mark.parametrize("mode", ["verbose", "all"])
@pytest.mark.asyncio
async def test_clarify_tool_never_renders_progress_bubble(monkeypatch, tmp_path, mode):
    """No progress bubble for clarify — in any mode, especially verbose.

    Verbose mode used to dump the raw args JSON
    (``{"question": ..., "choices": [...]}``) into the chat right under the
    interactive prompt (#52374).
    """
    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, mode)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm")

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-clarify-leak",
        session_key="agent:main:slack:dm:C1",
    )

    assert result["final_response"] == "done"
    all_content = "\n".join(
        [m["content"] for m in adapter.sent] + [e["content"] for e in adapter.edits]
    )
    # Raw clarify args JSON must not leak anywhere.
    assert '"question"' not in all_content
    assert '"choices"' not in all_content
    assert "Which environment?" not in all_content
    # No clarify progress line at all (verb "Asking" / tool name).
    assert "clarify" not in all_content
    assert "Asking" not in all_content
    # The unrelated terminal tool still renders progress normally.
    assert "pwd" in all_content


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.asyncio
async def test_clarify_fences_question_and_choices_before_native_or_fallback_delivery(
    monkeypatch, tmp_path, native, request
):
    adapter = NativeClarifyCaptureAdapter() if native else ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, "off", FencedClarifyAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    from tools import clarify_gateway

    monkeypatch.setattr(clarify_gateway, "wait_for_response", lambda *_a, **_k: "chosen")
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm")
    session_key = f"agent:main:slack:dm:C1:{native}"
    request.addfinalizer(lambda: clarify_gateway.clear_session(session_key))
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=f"sess-clarify-context-{native}",
        session_key=session_key,
    )

    assert result["final_response"] == "chosen"
    if native:
        assert len(adapter.clarifies) == 1
        delivered = adapter.clarifies[0]
        assert delivered["question"] == "Visible question?"
        assert delivered["choices"] == ["Visible choice", "[details withheld]"]
        visible = delivered["question"] + " " + " ".join(delivered["choices"])
    else:
        visible = " ".join(call["content"] for call in adapter.sent)
        assert "Visible question?" in visible
        assert "Visible choice" in visible
    assert FencedClarifyAgent.PRIVATE_QUESTION not in visible
    assert FencedClarifyAgent.PRIVATE_CHOICE not in visible
    assert "memory-context" not in visible


@pytest.mark.asyncio
async def test_clarify_fails_closed_when_sanitized_question_is_empty(monkeypatch, tmp_path):
    adapter = NativeClarifyCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, "off", EmptyFencedClarifyAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm")

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-clarify-empty-context",
        session_key="agent:main:slack:dm:C1:empty",
    )

    assert result["final_response"] == "[clarify prompt could not be delivered]"
    assert adapter.clarifies == []


def test_programmatic_clarify_text_keeps_raw_contract():
    from gateway.run import _sanitize_gateway_agent_text

    raw = FencedClarifyAgent.RAW_QUESTION
    assert _sanitize_gateway_agent_text(Platform.API_SERVER, raw) == raw
