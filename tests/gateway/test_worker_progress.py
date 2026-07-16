import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.worker_progress import (
    WORKER_PROGRESS_EVENT,
    notice_from_worker_progress_event,
    render_worker_progress_notice,
)
from tools.delegate_tool import _build_child_progress_callback


class SlackWorkerProgressAdapter(BasePlatformAdapter):
    def __init__(self, *, fail_first_edit: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.SLACK)
        self.sent = []
        self.edits = []
        self.typing = []
        self._next_id = 1
        self._fail_first_edit = fail_first_edit

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        message_id = f"msg-{self._next_id}"
        self._next_id += 1
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                "message_id": message_id,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self,
        chat_id,
        message_id,
        content,
        *,
        finalize=False,
        metadata=None,
    ) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": metadata,
            }
        )
        if self._fail_first_edit:
            self._fail_first_edit = False
            return SendResult(success=False, error="message not found")
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class StructuredWorkerAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb(
            WORKER_PROGRESS_EVENT,
            args={
                "phase": "inspecting",
                "status": "active",
                "text": "raw task text API_KEY=should-not-render",
                "log_path": "/tmp/worker.log",
            },
        )
        time.sleep(0.35)
        cb(
            WORKER_PROGRESS_EVENT,
            args={
                "phase": "validating",
                "status": "active",
                "command": "pytest --token=should-not-render",
                "preview": "terminal output should-not-render",
            },
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class DelegatedWorkerAgent(StructuredWorkerAgent):
    relayed_events = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.__class__.relayed_events = []

        def parent_progress(event_type, tool_name=None, preview=None, args=None, **kwargs):
            self.__class__.relayed_events.append((event_type, tool_name, preview, args))
            self.tool_progress_callback(event_type, tool_name, preview, args, **kwargs)

        child_progress = _build_child_progress_callback(
            0,
            "Rotate API_KEY=should-not-render and inspect /tmp/worker.log",
            SimpleNamespace(tool_progress_callback=parent_progress),
            task_count=1,
            subagent_id="child-1",
            session_ref={"session_id": "child-session-1"},
        )
        child_progress("subagent.start")
        time.sleep(0.35)
        child_progress(
            "tool.started",
            "terminal",
            "cat /tmp/worker.log",
            {"command": "cat /tmp/worker.log && echo API_KEY=should-not-render"},
        )
        time.sleep(0.35)
        child_progress("subagent.complete", status="completed")
        time.sleep(0.1)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class PreviewCompleteWorkerAgent(StructuredWorkerAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            WORKER_PROGRESS_EVENT,
            args={"phase": "editing", "status": "active"},
        )
        time.sleep(0.35)
        self.tool_progress_callback(
            WORKER_PROGRESS_EVENT,
            args={
                "phase": "complete",
                "status": "complete",
                "preview_url": "https://example.com/preview/123",
                "text": "API_KEY=should-not-render",
            },
        )
        time.sleep(0.1)
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
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run_once(
    monkeypatch,
    tmp_path,
    agent_cls,
    adapter,
    *,
    source=None,
    event_message_id="1700000000.000100",
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    source = source or SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        thread_id="1700000000.000100",
        scope_id="T123",
    )
    return await runner._run_agent(
        message="please run worker",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-worker-progress",
        session_key="agent:main:slack:channel:C123:1700000000.000100",
        event_message_id=event_message_id,
    )


def _rendered(adapter):
    return "\n".join([*(c["content"] for c in adapter.sent), *(c["content"] for c in adapter.edits)])


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"phase": "editing", "status": "active"}, "Editing files"),
        ({"phase": "complete", "preview_url": "https://example.com/preview/123"}, "<https://example.com/preview/123|Open preview>"),
        ({"phase": "complete", "preview_url": "http://example.com/preview/123"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/?token=secret"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/?authToken=secret"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/?next=sk-abcdefghijklmnopqrstuvwxyz"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/preview/API_KEY=secret"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/preview/token/123"}, "Open preview"),
        ({"phase": "complete", "preview_url": "https://example.com/preview#token=secret"}, "Open preview"),
        ({"phase": "rm -rf", "status": "cat /tmp/log"}, "Preparing the workspace"),
    ],
)
def test_worker_progress_contract_renders_only_allowlisted_fields(payload, expected):
    notice = notice_from_worker_progress_event(WORKER_PROGRESS_EVENT, args=payload)
    rendered = render_worker_progress_notice(notice)
    if expected.startswith("<https://"):
        assert expected in rendered
    elif expected == "Open preview":
        assert expected not in rendered
    else:
        assert expected in rendered


def test_worker_progress_contract_ignores_raw_text_and_secret_shapes():
    notice = notice_from_worker_progress_event(
        WORKER_PROGRESS_EVENT,
        args={
            "phase": "validating",
            "status": "active",
            "text": "run pytest with API_KEY=should-not-render",
            "command": "cat /tmp/worker.log",
            "preview_url": "https://example.com/?api_key=secret",
        },
    )
    rendered = render_worker_progress_notice(notice)
    assert "Validating the result" in rendered
    assert "pytest" not in rendered
    assert "/tmp/worker.log" not in rendered
    assert "API_KEY" not in rendered
    assert "should-not-render" not in rendered
    assert "Open preview" not in rendered


@pytest.mark.parametrize(
    "delegated_status, expected, unexpected",
    [
        ("completed", "Work complete", "Blocked"),
        ("timeout", "Blocked", "Work complete"),
        ("error", "Blocked", "Work complete"),
        ("failed", "Blocked", "Work complete"),
    ],
)
def test_delegated_subagent_complete_statuses_render_only_canned_terminal_states(
    delegated_status,
    expected,
    unexpected,
):
    notice = notice_from_worker_progress_event(
        "subagent.complete",
        kwargs={
            "status": delegated_status,
            "summary": "API_KEY=should-not-render",
            "preview": "cat /tmp/worker.log",
            "error": "timeout with secret token=should-not-render",
        },
    )
    rendered = render_worker_progress_notice(notice)

    assert expected in rendered
    assert unexpected not in rendered
    assert "API_KEY" not in rendered
    assert "/tmp/worker.log" not in rendered
    assert "token=should-not-render" not in rendered


@pytest.mark.asyncio
async def test_structured_worker_progress_uses_one_slack_thread_card_without_raw_leakage(monkeypatch, tmp_path):
    adapter = SlackWorkerProgressAdapter()
    result = await _run_once(monkeypatch, tmp_path, StructuredWorkerAgent, adapter)

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {
        "thread_id": "1700000000.000100",
        "slack_team_id": "T123",
    }
    assert adapter.edits
    assert all(edit["message_id"] == adapter.sent[0]["message_id"] for edit in adapter.edits)
    rendered = _rendered(adapter)
    assert "Inspecting the request" in rendered
    assert "Validating the result" in rendered
    assert "Work complete" in rendered
    assert "pytest" not in rendered
    assert "/tmp/worker.log" not in rendered
    assert "terminal output" not in rendered
    assert "API_KEY" not in rendered
    assert "should-not-render" not in rendered


@pytest.mark.asyncio
async def test_delegated_subagent_relay_opens_safe_worker_card(monkeypatch, tmp_path):
    adapter = SlackWorkerProgressAdapter()
    result = await _run_once(monkeypatch, tmp_path, DelegatedWorkerAgent, adapter)

    assert result["final_response"] == "done"
    assert [event[0] for event in DelegatedWorkerAgent.relayed_events[:3]] == [
        "subagent.start",
        "subagent.tool",
        "subagent.complete",
    ]
    assert len(adapter.sent) == 1
    assert adapter.edits
    assert adapter.edits[-1]["message_id"] == adapter.sent[0]["message_id"]
    assert "Work complete" in adapter.edits[-1]["content"]
    rendered = _rendered(adapter)
    assert "Preparing the workspace" in rendered
    assert "Work complete" in rendered
    assert "Rotate API_KEY" not in rendered
    assert "cat /tmp/worker.log" not in rendered
    assert "should-not-render" not in rendered


@pytest.mark.asyncio
async def test_worker_progress_complete_notice_can_render_safe_preview_url(monkeypatch, tmp_path):
    adapter = SlackWorkerProgressAdapter()
    await _run_once(monkeypatch, tmp_path, PreviewCompleteWorkerAgent, adapter)

    rendered = _rendered(adapter)
    assert "<https://example.com/preview/123|Open preview>" in rendered
    assert "API_KEY" not in rendered
    assert "should-not-render" not in rendered


@pytest.mark.asyncio
async def test_worker_progress_dm_fallback_keeps_event_thread_and_slack_team(monkeypatch, tmp_path):
    adapter = SlackWorkerProgressAdapter()
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        thread_id=None,
        scope_id="T999",
    )
    await _run_once(
        monkeypatch,
        tmp_path,
        StructuredWorkerAgent,
        adapter,
        source=source,
        event_message_id="1711111111.000200",
    )

    assert adapter.sent[0]["metadata"] == {
        "thread_id": "1711111111.000200",
        "slack_team_id": "T999",
    }
    assert all(
        edit["metadata"] == {
            "thread_id": "1711111111.000200",
            "slack_team_id": "T999",
        }
        for edit in adapter.edits
    )


@pytest.mark.asyncio
async def test_worker_progress_edit_failure_replaces_card_and_continues_editing_replacement(monkeypatch, tmp_path):
    adapter = SlackWorkerProgressAdapter(fail_first_edit=True)
    await _run_once(monkeypatch, tmp_path, StructuredWorkerAgent, adapter)

    assert [sent["message_id"] for sent in adapter.sent] == ["msg-1", "msg-2"]
    assert adapter.edits[0]["message_id"] == "msg-1"
    assert adapter.edits[-1]["message_id"] == "msg-2"
    assert adapter.sent[1]["metadata"] == {
        "thread_id": "1700000000.000100",
        "slack_team_id": "T123",
    }
    assert adapter.edits[-1]["metadata"] == {
        "thread_id": "1700000000.000100",
        "slack_team_id": "T123",
    }
