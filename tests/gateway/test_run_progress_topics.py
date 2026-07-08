"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
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
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SmallLimitProgressAdapter(ProgressCaptureAdapter):
    """Adapter with a tiny platform limit to exercise progress rollover."""

    MAX_MESSAGE_LENGTH = 180

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self._next_id = 0
        self.oversized_edits = []
        self.oversized_sends = []

    def _mint_id(self):
        self._next_id += 1
        return f"progress-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if len(content) > self.MAX_MESSAGE_LENGTH:
            self.oversized_sends.append(content)
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=self._mint_id())

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if len(content) > self.MAX_MESSAGE_LENGTH:
            self.oversized_edits.append(content)
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)


class MetadataEditProgressCaptureAdapter(ProgressCaptureAdapter):
    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)


class NonEditingProgressCaptureAdapter(ProgressCaptureAdapter):
    SUPPORTS_MESSAGE_EDITING = False

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        raise AssertionError("non-editable adapters should not receive edit_message calls")


class FakeAgent:
    def __init__(self, **kwargs):
        # Capture anything passed via kwargs (older code path) but don't
        # freeze it — production now assigns tool_progress_callback after
        # construction (see gateway/run.py around the agent-cache hit),
        # so we must read it at call time, not at init.
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
            cb("tool.started", "browser_navigate", "https://example.com", {})
            time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class LongPreviewAgent:
    """Agent that emits a tool call with a very long preview string."""
    LONG_CMD = "cd /home/teknium/.hermes/hermes-agent/.worktrees/hermes-d8860339 && source .venv/bin/activate && python -m pytest tests/gateway/test_run_progress_topics.py -n0 -q"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", self.LONG_CMD, {})
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class DelayedProgressAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", "first command", {})
        time.sleep(0.45)
        self.tool_progress_callback("tool.started", "terminal", "second command", {})
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class ManyProgressLinesAgent:
    """Emits enough tool-progress lines to exceed a single platform bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", "first-short", {})
        # Let the progress task create the first editable bubble, then enqueue
        # the rest quickly.  The cancellation drain must roll them into fresh
        # editable bubbles instead of trying to edit the first one past limit.
        time.sleep(0.35)
        for idx in range(1, 8):
            cb("tool.started", "terminal", f"overflow-line-{idx}-" + "x" * 45, {})
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class DelayedInterimAgent:
    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.interim_assistant_callback("first interim")
        time.sleep(0.45)
        self.interim_assistant_callback("second interim")
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


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


@pytest.mark.asyncio
async def test_run_agent_progress_stays_in_originating_topic(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji for this fake-agent test

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": '💻 terminal: "pwd"',
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]
    assert adapter.edits
    assert all(call["metadata"] == {"thread_id": "17585"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_edits_keep_originating_topic_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = MetadataEditProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-progress-edit-topic",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"
    assert adapter.edits
    assert all(call["metadata"] == {"thread_id": "17585"} for call in adapter.edits)


@pytest.mark.asyncio
async def test_run_agent_progress_does_not_use_event_message_id_for_telegram_dm(monkeypatch, tmp_path):
    """Telegram DM progress must not reuse event message id as thread metadata."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-2",
        session_key="agent:main:telegram:dm:12345",
        event_message_id="777",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] is None
    assert all(call["metadata"] is None for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_uses_event_message_id_for_slack_dm(monkeypatch, tmp_path):
    """Slack DM progress should keep event ts fallback threading."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    # Since PR #8006, Slack's built-in display tier sets tool_progress="off"
    # by default. Override via config so this test still exercises the
    # progress-callback path the Slack DM event_message_id threading depends on.
    import yaml
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"platforms": {"slack": {"tool_progress": "all"}}}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-3",
        session_key="agent:main:slack:dm:D123",
        event_message_id="1234567890.000001",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] == {"thread_id": "1234567890.000001"}
    assert all(call["metadata"] == {"thread_id": "1234567890.000001"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_feishu_progress_replies_inside_existing_thread(monkeypatch, tmp_path):
    """Feishu needs reply_to plus reply_in_thread metadata for topic-scoped progress."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="topic_17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-progress",
        session_key="agent:main:feishu:group:oc_chat:topic_17585",
        event_message_id="om_triggering_user_message",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["reply_to"] == "om_triggering_user_message"
    assert adapter.sent[0]["metadata"] == {"thread_id": "topic_17585"}
    assert adapter.edits
    assert adapter.edits[0]["message_id"] == "progress-1"


# ---------------------------------------------------------------------------
# Preview truncation tests (all/new mode respects tool_preview_length)
# ---------------------------------------------------------------------------


def _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0):
    """Shared setup for long-preview truncation tests.

    Returns (adapter, result) after running the agent with LongPreviewAgent.
    ``preview_length`` controls display.tool_preview_length in the config file
    that _run_agent reads — so the gateway picks it up the same way production does.
    """
    import asyncio
    import yaml

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = LongPreviewAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Write config.yaml so _run_agent picks up tool_preview_length
    config = {"display": {"tool_preview_length": preview_length}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = asyncio.get_event_loop().run_until_complete(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-trunc",
            session_key="agent:main:telegram:dm:12345",
        )
    )
    return adapter, result


def test_all_mode_default_truncation_40_chars(monkeypatch, tmp_path):
    """When tool_preview_length is 0 (default), all/new mode truncates to 40 chars."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # The long command should be truncated — total preview <= 40 chars
    assert "..." in content
    # Extract the preview part between quotes
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    assert len(preview_text) <= 40, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_respects_custom_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is explicitly set (e.g. 120), all/new mode uses that."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=120)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With 120-char cap, the command (165 chars) should still be truncated but longer
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    # Should be longer than the 40-char default
    assert len(preview_text) > 40, f"Preview suspiciously short ({len(preview_text)}): {preview_text}"
    # But still capped at 120
    assert len(preview_text) <= 120, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_no_truncation_when_preview_fits(monkeypatch, tmp_path):
    """Short previews (under the cap) are not truncated."""
    # Set a generous cap — the LongPreviewAgent's command is ~165 chars
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=200)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With a 200-char cap, the 165-char command should NOT be truncated
    assert "..." not in content, f"Preview was truncated when it shouldn't be: {content}"


class CommentaryAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        time.sleep(0.1)
        if self.stream_delta_callback:
            self.stream_delta_callback("done")
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class PreviewedResponseAgent:
    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("You're welcome.", already_streamed=False)
        return {
            "final_response": "You're welcome.",
            "response_previewed": True,
            "messages": [],
            "api_calls": 1,
        }


class StreamingRefineAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Continuing to refine:")
        time.sleep(0.1)
        if self.stream_delta_callback:
            self.stream_delta_callback(" Final answer.")
        return {
            "final_response": "Continuing to refine: Final answer.",
            "response_previewed": True,
            "messages": [],
            "api_calls": 1,
        }


class QueuedCommentaryAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1 and self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        return {
            "final_response": f"final response {type(self).calls}",
            "messages": [],
            "api_calls": 1,
        }


class BackgroundReviewAgent:
    def __init__(self, **kwargs):
        self.background_review_callback = kwargs.get("background_review_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.background_review_callback:
            self.background_review_callback("💾 Skill 'prospect-scanner' created.")
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class VerboseAgent:
    """Agent that emits a tool call with args whose JSON exceeds 200 chars."""
    LONG_CODE = "x" * 300

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "execute_code", None,
            {"code": self.LONG_CODE},
        )
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


async def _run_with_agent(
    monkeypatch,
    tmp_path,
    agent_cls,
    *,
    session_id,
    pending_text=None,
    config_data=None,
    platform=Platform.TELEGRAM,
    chat_id="-1001",
    chat_type="group",
    thread_id="17585",
    adapter_cls=ProgressCaptureAdapter,
):
    if config_data:
        import yaml

        (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = adapter_cls(platform=platform)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    if config_data and "streaming" in config_data:
        runner.config.streaming = StreamingConfig.from_dict(config_data["streaming"])
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )
    session_key = f"agent:main:{platform.value}:{chat_type}:{chat_id}"
    if thread_id:
        session_key = f"{session_key}:{thread_id}"
    if pending_text is not None:
        adapter._pending_messages[session_key] = MessageEvent(
            text=pending_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id="queued-1",
        )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=session_key,
    )
    return adapter, result


@pytest.mark.asyncio
async def test_run_agent_rolls_progress_bubble_before_platform_limit(monkeypatch, tmp_path):
    """Tool progress should start a second editable bubble before Telegram's limit.

    Regression: once the first progress bubble grew past the platform limit,
    the gateway kept trying to edit that same oversized full transcript.  The
    Telegram adapter then split-and-sent a fresh continuation on every update,
    causing a noisy trail of one-line messages instead of a new editable bubble.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-overflow-rollover",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
                "tool_preview_length": 60,
            }
        },
        adapter_cls=SmallLimitProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, SmallLimitProgressAdapter)
    assert len(adapter.sent) >= 2, "expected a fresh progress bubble after the first filled"
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []
    all_bubbles = [call["content"] for call in adapter.sent + adapter.edits]
    assert all(len(text) <= adapter.MAX_MESSAGE_LENGTH for text in all_bubbles)


@pytest.mark.asyncio
async def test_run_agent_surfaces_real_interim_commentary(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result.get("already_sent") is not True
    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_surfaces_interim_commentary_by_default(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-default-on",
    )

    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_suppresses_interim_commentary_when_disabled(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-disabled",
        config_data={"display": {"interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_tool_progress_does_not_control_interim_commentary(monkeypatch, tmp_path):
    """tool_progress=all with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-tool-progress",
        config_data={"display": {"tool_progress": "all", "interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_streaming_does_not_enable_completed_interim_commentary(
    monkeypatch, tmp_path
):
    """Streaming alone with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-streaming",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True},
        },
    )

    assert result.get("already_sent") is True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_display_streaming_does_not_enable_gateway_streaming(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-display-streaming-cli-only",
        config_data={
            "display": {
                "streaming": True,
                "interim_assistant_messages": True,
            },
            "streaming": {"enabled": False},
        },
    )

    assert result.get("already_sent") is not True
    assert adapter.edits == []
    assert [call["content"] for call in adapter.sent] == ["I'll inspect the repo first."]


@pytest.mark.asyncio
async def test_run_agent_interim_commentary_works_with_tool_progress_off(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-explicit-on",
        config_data={
            "display": {
                "tool_progress": "off",
                "interim_assistant_messages": True,
            },
        },
    )

    assert result.get("already_sent") is not True
    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_bluebubbles_uses_commentary_send_path_for_quick_replies(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-bluebubbles-commentary",
        config_data={"display": {"interim_assistant_messages": True}},
        platform=Platform.BLUEBUBBLES,
        chat_id="iMessage;-;user@example.com",
        chat_type="dm",
        thread_id=None,
        adapter_cls=NonEditingProgressCaptureAdapter,
    )

    assert result.get("already_sent") is not True
    assert [call["content"] for call in adapter.sent] == ["I'll inspect the repo first."]
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_run_agent_previewed_final_marks_already_sent(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PreviewedResponseAgent,
        session_id="sess-previewed",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result.get("already_sent") is True
    assert [call["content"] for call in adapter.sent] == ["You're welcome."]


@pytest.mark.asyncio
async def test_run_agent_matrix_streaming_omits_cursor(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        StreamingRefineAgent,
        session_id="sess-matrix-streaming",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1},
        },
        platform=Platform.MATRIX,
        chat_id="!room:matrix.example.org",
        chat_type="group",
        thread_id="$thread",
    )

    assert result.get("already_sent") is True
    all_text = [call["content"] for call in adapter.sent] + [call["content"] for call in adapter.edits]
    assert all_text, "expected streamed Matrix content to be sent or edited"
    assert all("▉" not in text for text in all_text)
    assert any("Continuing to refine:" in text for text in all_text)


class TransformedStreamAgent:
    """Streams a response, then signals the gateway that a plugin hook
    (``transform_llm_output``) modified the final text after streaming
    finished. ``run_conversation`` returns ``response_transformed=True``
    plus a ``final_response`` that diverges from what was streamed.
    """

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("original answer")
        return {
            "final_response": "original answer\n\n[plugin appended this]",
            "response_previewed": True,
            "response_transformed": True,
            "messages": [],
            "api_calls": 1,
        }


@pytest.mark.asyncio
async def test_transformed_response_edits_streamed_message_in_place(monkeypatch, tmp_path):
    """When a transform_llm_output hook modifies the response after streaming,
    the gateway must edit the existing streamed message in place with the full
    transformed content (so plugins like content filters / appenders reach the
    user) and still mark already_sent=True (no duplicate send).
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        TransformedStreamAgent,
        session_id="sess-transformed-stream",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1},
        },
        platform=Platform.MATRIX,
        chat_id="!room:matrix.example.org",
        chat_type="group",
        thread_id="$thread",
        adapter_cls=MetadataEditProgressCaptureAdapter,
    )

    # Final delivery happened (no duplicate send fallback).
    assert result.get("already_sent") is True
    # The transformed final text reached the user — appended portion is present
    # in an edit_message call (not just in the streamed sends).
    edited_texts = [e["content"] for e in adapter.edits]
    assert any("[plugin appended this]" in text for text in edited_texts), (
        f"expected transformed text in adapter.edits, got: {edited_texts!r}"
    )


@pytest.mark.asyncio
async def test_run_agent_queued_message_does_not_treat_commentary_as_final(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-queued-commentary",
        pending_text="queued follow-up",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert result["final_response"] == "final response 2"
    assert "I'll inspect the repo first." in sent_texts
    assert "final response 1" in sent_texts


@pytest.mark.asyncio
async def test_run_agent_defers_background_review_notification_until_release(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        BackgroundReviewAgent,
        session_id="sess-bg-review-order",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_base_processing_releases_post_delivery_callback_after_main_send():
    """Post-delivery callbacks on the adapter fire after the main response."""
    adapter = ProgressCaptureAdapter()

    async def _handler(event):
        return "done"

    adapter.set_message_handler(_handler)

    released = []

    def _post_delivery_cb():
        released.append(True)
        adapter.sent.append(
            {
                "chat_id": "bg-review",
                "content": "💾 Skill 'prospect-scanner' created.",
                "reply_to": None,
                "metadata": None,
            }
        )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._post_delivery_callbacks[session_key] = _post_delivery_cb

    await adapter._process_message_background(event, session_key)

    sent_texts = [call["content"] for call in adapter.sent]
    assert sent_texts == ["done", "💾 Skill 'prospect-scanner' created."]
    assert released == [True]


@pytest.mark.asyncio
async def test_base_processing_stops_typing_before_hung_post_delivery_callback(
    monkeypatch,
):
    """A stuck post-delivery callback must not keep the typing task alive."""
    monkeypatch.setattr(base_platform, "_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS", 0.01)
    adapter = ProgressCaptureAdapter()
    events = []

    async def _handler(event):
        return "done"

    async def _post_delivery_cb():
        events.append("callback-start")
        await asyncio.Event().wait()

    async def _stop_typing(chat_id):
        events.append("typing-stopped")
        await ProgressCaptureAdapter.stop_typing(adapter, chat_id)

    adapter.set_message_handler(_handler)
    adapter.stop_typing = _stop_typing

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._post_delivery_callbacks[session_key] = _post_delivery_cb

    await asyncio.wait_for(
        adapter._process_message_background(event, session_key), timeout=1.0
    )

    assert [call["content"] for call in adapter.sent] == ["done"]
    # Invariant: typing must stop before the (hung) post-delivery callback
    # starts.  Don't pin the exact stop_typing call count — the shared
    # cleanup path may make more than one bounded stop attempt.
    assert "typing-stopped" in events
    assert "callback-start" in events
    assert events.index("typing-stopped") < events.index("callback-start")
    assert events[: events.index("callback-start")] == (
        ["typing-stopped"] * events.index("callback-start")
    )
    assert any(call["metadata"] == {"stopped": True} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_drops_tool_progress_after_generation_invalidation(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"tool_progress": "all"}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = DelayedProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal tool metadata

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm-1",
        chat_type="dm",
        thread_id=None,
    )
    session_key = "agent:main:discord:dm:dm-1"
    runner._session_run_generation[session_key] = 1

    original_send = adapter.send
    invalidated = {"done": False}

    async def send_and_invalidate(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if "first command" in content and not invalidated["done"]:
            invalidated["done"] = True
            runner._invalidate_session_run_generation(session_key, reason="test_stop")
        return result

    adapter.send = send_and_invalidate

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-progress-stop",
        session_key=session_key,
        run_generation=1,
    )

    all_progress_text = " ".join(call["content"] for call in adapter.sent)
    all_progress_text += " ".join(call["content"] for call in adapter.edits)
    assert result["final_response"] == "done"
    assert 'first command' in all_progress_text
    assert 'second command' not in all_progress_text


@pytest.mark.asyncio
async def test_run_agent_drops_interim_commentary_after_generation_invalidation(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"tool_progress": "off", "interim_assistant_messages": True}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = DelayedInterimAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm-2",
        chat_type="dm",
        thread_id=None,
    )
    session_key = "agent:main:discord:dm:dm-2"
    runner._session_run_generation[session_key] = 1

    original_send = adapter.send
    invalidated = {"done": False}

    async def send_and_invalidate(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if content == "first interim" and not invalidated["done"]:
            invalidated["done"] = True
            runner._invalidate_session_run_generation(session_key, reason="test_stop")
        return result

    adapter.send = send_and_invalidate

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-commentary-stop",
        session_key=session_key,
        run_generation=1,
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert result["final_response"] == "done"
    assert "first interim" in sent_texts
    assert "second interim" not in sent_texts


@pytest.mark.asyncio
async def test_keep_typing_stops_immediately_when_interrupt_event_is_set():
    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    stop_event = asyncio.Event()

    task = asyncio.create_task(
        adapter._keep_typing(
            "dm-typing-stop",
            interval=30.0,
            stop_event=stop_event,
        )
    )
    await asyncio.sleep(0.05)
    stop_event.set()
    await asyncio.wait_for(task, timeout=0.5)

    normal_typing_calls = [
        call for call in adapter.typing if call.get("metadata") != {"stopped": True}
    ]
    stopped_calls = [
        call for call in adapter.typing if call.get("metadata") == {"stopped": True}
    ]
    assert len(normal_typing_calls) == 1
    assert len(stopped_calls) == 1


@pytest.mark.asyncio
async def test_verbose_mode_does_not_truncate_args_by_default(monkeypatch, tmp_path):
    """Verbose mode with default tool_preview_length (0) should NOT truncate args.

    Previously, verbose mode capped args at 200 chars when tool_preview_length
    was 0 (default).  The user explicitly opted into verbose — show full detail.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VerboseAgent,
        session_id="sess-verbose-no-truncate",
        config_data={"display": {"tool_progress": "verbose", "tool_preview_length": 0}},
    )

    assert result["final_response"] == "done"
    # The full 300-char 'x' string should be present, not truncated to 200
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    assert VerboseAgent.LONG_CODE in all_content


@pytest.mark.asyncio
async def test_verbose_mode_respects_explicit_tool_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is set to a positive value, verbose truncates to that."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VerboseAgent,
        session_id="sess-verbose-explicit-cap",
        config_data={"display": {"tool_progress": "verbose", "tool_preview_length": 50}},
    )

    assert result["final_response"] == "done"
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    # Should be truncated — full 300-char string NOT present
    assert VerboseAgent.LONG_CODE not in all_content
    # But should still contain the truncated portion with "..."
    assert "..." in all_content


class CodeBlockProgressAdapter(ProgressCaptureAdapter):
    """A markdown-capable progress adapter (declares supports_code_blocks)."""

    supports_code_blocks = True


class TerminalCommandAgent:
    """Emits a terminal tool.started with a real, multi-line command arg."""

    CMD = (
        "set -euo pipefail\n"
        "printf 'node: '; node --version\n"
        "npm install -g hyperframes@latest"
    )

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "terminal", self.CMD, {"command": self.CMD}
        )
        # Let the async progress task drain the queue and send before returning.
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_terminal_progress_renders_fenced_code_block(monkeypatch, tmp_path):
    """Terminal progress on a markdown-capable (supports_code_blocks) gateway
    renders a bare fenced code block — no language tag (Slack mrkdwn would print
    'bash' as a literal first code line).  In non-verbose ("all"/"new") mode the
    command is collapsed to a single line capped at tool_preview_length so a long
    or multi-line command doesn't render as a huge block (#42634)."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = TerminalCommandAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji

    adapter = CodeBlockProgressAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-terminal-code-block",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "done"
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    # Bare fenced block, no language tag (no '```bash').
    assert "```" in all_content
    assert "```bash" not in all_content
    # Non-verbose collapses to the first line + truncation marker — the later
    # command lines must NOT appear (this was the "huge block" regression).
    assert "set -euo pipefail" in all_content
    assert "npm install -g hyperframes@latest" not in all_content
    assert "node --version" not in all_content
    # No truncated quoted preview for the terminal command.
    assert 'terminal: "' not in all_content


@pytest.mark.asyncio
async def test_terminal_progress_verbose_shows_full_command(monkeypatch, tmp_path):
    """Verbose mode on a markdown-capable gateway renders the FULL multi-line
    command in a bare fenced block (no truncation, no 'bash' tag).  This is the
    parity guarantee for #42634: verbose keeps full detail, non-verbose caps."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "verbose")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = TerminalCommandAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji

    adapter = CodeBlockProgressAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-terminal-code-block-verbose",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "done"
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    assert "```" in all_content
    assert "```bash" not in all_content
    # Full command body present — verbose is uncapped.
    assert "npm install -g hyperframes@latest" in all_content
    assert "node --version" in all_content


@pytest.mark.asyncio
async def test_terminal_progress_no_bash_block_in_verbose_mode(monkeypatch, tmp_path):
    """#41215 also rendered the bash block in verbose mode. The revert removed it
    from both branches, so verbose progress must not emit a fenced ```bash block
    either (verbose still shows args by opt-in, just not as a code block)."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "verbose")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = TerminalCommandAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji

    adapter = CodeBlockProgressAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-terminal-verbose-no-bash",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "done"
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    assert "```bash" not in all_content

class MultiTerminalCommandAgent:
    """Emits several consecutive terminal tool.started events, then a
    different tool, then terminal again — to exercise header collapsing."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "terminal", "echo one", {"command": "echo one"})
        cb("tool.started", "terminal", "echo two", {"command": "echo two"})
        cb("tool.started", "terminal", "echo three", {"command": "echo three"})
        cb("tool.started", "web_search", "query stuff", {"query": "query stuff"})
        cb("tool.started", "terminal", "echo four", {"command": "echo four"})
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_consecutive_terminal_progress_collapses_headers(monkeypatch, tmp_path):
    """Back-to-back terminal calls render ONE "terminal" header followed by
    adjacent code blocks; a different tool in between resets the header so the
    next terminal call gets a fresh one."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = MultiTerminalCommandAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji

    adapter = CodeBlockProgressAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-terminal-consecutive",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "done"
    contents = [call["content"] for call in adapter.sent] + [
        call["content"] for call in adapter.edits
    ]
    final = max(contents, key=len) if contents else ""
    # All four commands present as code blocks.
    for cmd in ("echo one", "echo two", "echo three", "echo four"):
        assert cmd in final
    # Exactly TWO terminal headers: one for the first run of three calls,
    # one for the terminal call after web_search broke the streak.
    assert final.count("terminal\n```") == 2
