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
        self.voice_events = []
        self.held_spoken_turns = []
        self.released_spoken_turns = []
        self.cleaned_voice_streams = []
        self._voice_turn_id = None
        self._voice_seq = 0
        self._voice_turn_counter = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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

    async def start_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        metadata = metadata if isinstance(metadata, dict) else {}
        if not metadata.get("turn_id") and not metadata.get("_voice_server_turn_id"):
            self._voice_turn_counter += 1
            metadata["_voice_server_turn_id"] = f"voice-turn-{self._voice_turn_counter}"
        self._voice_turn_id = str(metadata.get("turn_id") or metadata.get("_voice_server_turn_id"))
        self._voice_seq = 0
        self.voice_events.append(
            {
                "type": "assistant_llm_start",
                "chat_id": chat_id,
                "turn_id": self._voice_turn_id,
                "seq": self._voice_seq,
            }
        )
        return SendResult(success=True)

    async def push_assistant_delta(self, chat_id, text, metadata=None) -> SendResult:
        self._voice_seq += 1
        self.voice_events.append(
            {
                "type": "assistant_llm_text",
                "chat_id": chat_id,
                "turn_id": self._voice_turn_id or "voice-turn-1",
                "seq": self._voice_seq,
                "text": text,
            }
        )
        return SendResult(success=True)

    async def end_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        self._voice_seq += 1
        self.voice_events.append(
            {
                "type": "assistant_llm_end",
                "chat_id": chat_id,
                "turn_id": self._voice_turn_id or "voice-turn-1",
                "seq": self._voice_seq,
            }
        )
        return SendResult(success=True)

    async def abort_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        self._voice_seq += 1
        self.voice_events.append(
            {
                "type": "assistant_llm_abort",
                "chat_id": chat_id,
                "turn_id": self._voice_turn_id or "voice-turn-1",
                "seq": self._voice_seq,
            }
        )
        return SendResult(success=True)

    def hold_pending_spoken_turn(self, turn_id) -> None:
        self.held_spoken_turns.append(str(turn_id))

    def release_pending_spoken_turn(self, turn_id) -> None:
        self.released_spoken_turns.append(str(turn_id))

    def cleanup_assistant_stream(self, metadata=None) -> None:
        metadata = metadata if isinstance(metadata, dict) else {}
        self.cleaned_voice_streams.append(str(metadata.get("turn_id") or metadata.get("_voice_server_turn_id")))


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


def test_voice_final_matcher_requires_visible_content_match():
    from gateway.run import _assistant_message_matches_voice_final, _stamp_voice_turn_id_on_final_assistant

    assert _assistant_message_matches_voice_final(
        {"role": "assistant", "content": "final answer"},
        "final answer",
    )
    assert not _assistant_message_matches_voice_final(
        {"role": "assistant", "content": "intermediate note"},
        "final answer",
    )
    assert not _assistant_message_matches_voice_final(
        {"role": "assistant", "content": ""},
        "final answer",
    )
    assert _assistant_message_matches_voice_final(
        {
            "role": "assistant",
            "content": "final answer",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        },
        "final answer",
    )
    messages = [
        {"role": "assistant", "content": "raw final answer"},
    ]
    assert _stamp_voice_turn_id_on_final_assistant(
        messages,
        visible_final="decorated final answer",
        voice_turn_id="turn-1",
    )
    assert messages[-1]["voice_turn_id"] == "turn-1"
    assert _stamp_voice_turn_id_on_final_assistant(
        messages,
        visible_final="raw final answer",
        voice_turn_id="turn-2",
    )
    assert messages[-1]["voice_turn_id"] == "turn-2"


def test_voice_server_partial_streaming_defaults_on():
    from gateway.voice_stream import partial_llm_streaming_enabled

    adapter = SimpleNamespace(config=SimpleNamespace(extra={}))

    assert partial_llm_streaming_enabled({}, adapter) is True
    assert partial_llm_streaming_enabled(
        {
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": False},
                    }
                }
            }
        },
        adapter,
    ) is False
    assert partial_llm_streaming_enabled(
        {},
        SimpleNamespace(config=SimpleNamespace(extra={"partial_llm_streaming": False})),
    ) is False


class VoiceStreamTextFailureAdapter(ProgressCaptureAdapter):
    async def push_assistant_delta(self, chat_id, text, metadata=None) -> SendResult:
        if self.voice_events:
            text_count = sum(1 for event in self.voice_events if event["type"] == "assistant_llm_text")
            if text_count >= 1:
                return SendResult(success=False, error="text stream failed", retryable=True)
        return await super().push_assistant_delta(chat_id, text, metadata=metadata)


class VoiceStreamEndFailureAdapter(ProgressCaptureAdapter):
    async def end_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        return SendResult(success=False, error="end stream failed", retryable=True)


class VoiceStreamSecondEndFailureAdapter(ProgressCaptureAdapter):
    """Fail end_assistant_stream only on the FINAL streamed segment.

    Lets the preamble segment complete cleanly, then fails the final
    segment's end. Used to verify that a multi-segment streamed turn
    keeps the completed preamble's voice_turn_id even when the later
    segment fails.
    """

    async def end_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        end_count = sum(
            1 for event in self.voice_events if event["type"] == "assistant_llm_end"
        )
        if end_count >= 1:
            return SendResult(success=False, error="final segment end failed", retryable=True)
        return await super().end_assistant_stream(chat_id, metadata=metadata)


class SlowVoiceStreamEndAdapter(ProgressCaptureAdapter):
    async def end_assistant_stream(self, chat_id, metadata=None) -> SendResult:
        await asyncio.sleep(10)
        return await super().end_assistant_stream(chat_id, metadata=metadata)


class SlowVoiceStreamTextAdapter(ProgressCaptureAdapter):
    async def push_assistant_delta(self, chat_id, text, metadata=None) -> SendResult:
        await asyncio.sleep(10)
        return await super().push_assistant_delta(chat_id, text, metadata=metadata)


class RecordingVoiceStreamAdapter(ProgressCaptureAdapter):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.instances.append(self)


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


class ThinkingAgent:
    """Agent that emits _thinking scratch text (no tool calls).

    Used to prove the progress callback relays _thinking bubbles when
    thinking_progress is enabled but tool_progress is off.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("_thinking", "weighing the options here")
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
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
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
            "content": '💻 Running pwd',
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


def _extract_progress_preview(content: str) -> str | None:
    """Extract the argument-preview portion from a tool-progress message.

    Handles both render styles:
    - Legacy / custom tools:  ``🔧 tool_name: "<preview>"`` (quoted)
    - Friendly built-in verb: ``💻 Running <preview>`` (verb prefix, no quotes)
    """
    import re

    # Legacy quoted form takes precedence when present.
    match = re.search(r'"(.+)"', content)
    if match:
        return match.group(1)
    # Friendly form: "<emoji> <verb> <preview>". The terminal verb is "Running".
    marker = " Running "
    idx = content.find(marker)
    if idx != -1:
        return content[idx + len(marker):].strip()
    return None


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
    # The long command should be truncated — the preview portion <= 40 chars.
    assert "..." in content
    preview_text = _extract_progress_preview(content)
    assert preview_text is not None, f"No preview found in: {content}"
    assert len(preview_text) <= 40, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_respects_custom_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is explicitly set (e.g. 120), all/new mode uses that."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=120)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With 120-char cap, the command (165 chars) should still be truncated but longer.
    preview_text = _extract_progress_preview(content)
    assert preview_text is not None, f"No preview found in: {content}"
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


class PreviewedSplitAfterCommentaryAgent:
    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.session_id = kwargs.get("session_id")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        self.session_id = f"{self.session_id}-child"
        return {
            "final_response": "Final answer after compression.",
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


class VoicePartialAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
            self.stream_delta_callback(", voice.")
        return {
            "final_response": "Hello, voice.",
            "messages": [],
            "api_calls": 1,
        }


class VoicePreviewedPartialFailureAgent(VoicePartialAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None):
        result = super().run_conversation(message, conversation_history=conversation_history, task_id=task_id)
        result["response_previewed"] = True
        return result


class VoicePartialMediaAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
            self.stream_delta_callback(", voice.")
        return {
            "final_response": "Hello, voice.",
            "messages": [
                {"role": "tool", "content": '{"audio": "MEDIA:/tmp/voice-reply.wav"}'},
            ],
            "api_calls": 1,
        }


class QueuedVoicePartialMediaAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1:
            if self.stream_delta_callback:
                self.stream_delta_callback("Hello")
                self.stream_delta_callback(", voice.")
            return {
                "final_response": "Hello, voice.",
                "messages": [
                    {"role": "tool", "content": '{"audio": "MEDIA:/tmp/voice-reply.wav"}'},
                ],
                "api_calls": 1,
            }
        return {
            "final_response": "follow-up done",
            "messages": [],
            "api_calls": 1,
        }


class VoicePartialHistoryAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
            self.stream_delta_callback(", voice.")
        return {
            "final_response": "Hello, voice.",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Hello, voice."},
            ],
            "api_calls": 1,
        }


class VoicePartialToolCallContentAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Tool-call final.")
        return {
            "final_response": "Tool-call final.",
            "messages": [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": "Tool-call final.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                },
            ],
            "api_calls": 1,
        }


class CapturingHistoryAgent:
    conversation_history = None

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).conversation_history = conversation_history
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class VoiceFullHistoryAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": "Hello, voice.",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Hello, voice."},
            ],
            "api_calls": 1,
        }


class VoiceToolPreambleAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Let me check that first.")
        return {
            "final_response": "The final answer is 42.",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Let me check that first."},
                {"role": "tool", "content": "42", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "The final answer is 42."},
            ],
            "api_calls": 2,
        }


class VoicePreambleHistoryAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Let me check that first.")
        return {
            "final_response": "The final answer is 42.",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Let me check that first."},
                {"role": "assistant", "content": "The final answer is 42."},
            ],
            "api_calls": 2,
        }


class VoiceToolBoundaryAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Let me check that first.")
            self.stream_delta_callback(None)
            self.stream_delta_callback("\n\nThe final answer is 42.")
        return {
            "final_response": "The final answer is 42.",
            "messages": [],
            "api_calls": 2,
        }


class RaisingVoicePartialAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
        raise RuntimeError("voice partial failure")


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
    user_id=None,
    adapter_cls=ProgressCaptureAdapter,
    history=None,
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
        user_id=user_id,
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
        history=[] if history is None else history,
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
async def test_run_agent_previewed_split_keeps_final_delivery_pending(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PreviewedSplitAfterCommentaryAgent,
        session_id="sess-split",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result["session_id"] == "sess-split-child"
    assert result.get("already_sent") is not True
    assert [call["content"] for call in adapter.sent] == ["I'll inspect the repo first."]


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
async def test_run_agent_voice_server_default_partial_stream_suppresses_final_reply(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialAgent,
        session_id="sess-voice-partials",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    assert result["final_response"] == "Hello, voice."
    assert result.get("already_sent") is True
    assert [call["content"] for call in adapter.sent] == []
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_text",
        "assistant_llm_end",
    ]
    assert [call.get("text") for call in adapter.voice_events if call["type"] == "assistant_llm_text"] == [
        "Hello",
        ", voice.",
    ]


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_tags_final_assistant_message(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialHistoryAgent,
        session_id="sess-voice-partials-history",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                            "partial_llm_streaming": True,
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    assert result["messages"][-1]["voice_turn_id"] == turn_id


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_tags_tool_call_content_for_reconciliation(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialToolCallContentAgent,
        session_id="sess-voice-partials-tool-call-content",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                            "partial_llm_streaming": True,
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    assert result["voice_turn_id"] == turn_id
    assert result["voice_turn_ids"] == [turn_id]
    assert result["messages"][-1]["voice_turn_id"] == turn_id


@pytest.mark.asyncio
async def test_run_agent_strips_voice_metadata_from_rich_replay_messages(monkeypatch, tmp_path):
    history = [
        {
            "role": "assistant",
            "content": "Tool-call final.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
            "timestamp": 123,
            "voice_turn_id": "voice-turn-1",
            "voice_interrupted": True,
            "voice_planned_content": "Tool-call final.",
            "voice_spoken_content": "Tool-call",
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
    ]

    await _run_with_agent(
        monkeypatch,
        tmp_path,
        CapturingHistoryAgent,
        session_id="sess-rich-replay-voice-metadata",
        history=history,
    )

    replay_msg = CapturingHistoryAgent.conversation_history[0]
    assert replay_msg["tool_calls"] == history[0]["tool_calls"]
    assert "timestamp" not in replay_msg
    assert "voice_turn_id" not in replay_msg
    assert "voice_interrupted" not in replay_msg
    assert "voice_planned_content" not in replay_msg
    assert "voice_spoken_content" not in replay_msg


@pytest.mark.asyncio
async def test_run_agent_voice_server_full_reply_tags_final_assistant_message(monkeypatch, tmp_path):
    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoiceFullHistoryAgent,
        session_id="sess-voice-full-history",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                            "partial_llm_streaming": False,
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    turn_id = result["messages"][-1]["voice_turn_id"]
    assert turn_id.startswith("voice_server-")
    assert result["voice_turn_id"] == turn_id


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_media_suffix_suppresses_replayed_text(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialMediaAgent,
        session_id="sess-voice-partials-media",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                            "partial_llm_streaming": True,
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    assert result["final_response"] == "Hello, voice.\nMEDIA:/tmp/voice-reply.wav"
    assert result.get("already_sent") is True
    assert [call["content"] for call in adapter.sent] == []
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_text",
        "assistant_llm_end",
    ]

@pytest.mark.asyncio
async def test_run_agent_voice_server_queued_partial_media_still_delivers_attachment(monkeypatch, tmp_path):
    QueuedVoicePartialMediaAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedVoicePartialMediaAgent,
        session_id="sess-voice-queued-partial-media",
        pending_text="queued follow-up",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {
                            "url": "ws://127.0.0.1:7860/events",
                            "room_id": "personal-room",
                            "partial_llm_streaming": True,
                        },
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    assert result["final_response"] == "follow-up done"
    assert "Hello, voice." not in [call["content"] for call in adapter.sent]
    assert any(call["content"].endswith("/tmp/voice-reply.wav") for call in adapter.sent), adapter.sent


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_preamble_keeps_final_reply(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoiceToolPreambleAgent,
        session_id="sess-voice-partial-preamble",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    assert result["final_response"] == "The final answer is 42."
    assert result.get("already_sent") is not True
    assert [call["content"] for call in adapter.sent] == []
    assert [call.get("text") for call in adapter.voice_events if call["type"] == "assistant_llm_text"] == [
        "Let me check that first.",
    ]


@pytest.mark.asyncio
async def test_run_agent_voice_server_preamble_history_uses_full_reply_turn_id(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePreambleHistoryAgent,
        session_id="sess-voice-preamble-history",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    preamble_turn_id = next(
        call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start"
    )
    final_turn_id = result["messages"][-1]["voice_turn_id"]
    assert final_turn_id == result["voice_turn_id"]
    assert final_turn_id != preamble_turn_id
    assert final_turn_id.startswith("voice_server-")
    assert result["voice_turn_ids"] == [preamble_turn_id, final_turn_id]
    assert result["messages"][1]["voice_turn_id"] == preamble_turn_id
    assert result.get("already_sent") is not True


@pytest.mark.asyncio
async def test_run_agent_voice_server_preamble_keeps_turn_id_when_final_segment_fails(monkeypatch, tmp_path):
    """A completed streamed preamble must keep its voice_turn_id even when
    the later segment's end_assistant_stream fails.

    Regression for run.py's `_apply_failed_voice_stream_result`: an earlier
    blanket pop of `voice_turn_id` / `voice_turn_ids` after scrub-of-stale
    wiped the completed preamble's id, so post-persistence spoken-turn
    drain would not run for it. The helper now does per-id scrubbing only;
    this end-to-end test asserts the actual run.py path keeps the preamble
    turn id in the returned `voice_turn_ids` list.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoiceToolBoundaryAgent,
        adapter_cls=VoiceStreamSecondEndFailureAdapter,
        session_id="sess-voice-preamble-survives-final-fail",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    starts = [call for call in adapter.voice_events if call["type"] == "assistant_llm_start"]
    assert len(starts) == 2, "expected one start per segment (preamble + final)"
    preamble_turn_id = starts[0]["turn_id"]
    final_turn_id = starts[1]["turn_id"]
    assert preamble_turn_id != final_turn_id

    surviving_ids = result.get("voice_turn_ids") or []
    assert preamble_turn_id in surviving_ids, (
        "completed preamble turn id was wiped after the final segment's "
        "end_assistant_stream failed: surviving voice_turn_ids = "
        f"{surviving_ids}"
    )


@pytest.mark.asyncio
async def test_run_agent_voice_server_tool_boundary_resets_partial_delivery_match(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoiceToolBoundaryAgent,
        session_id="sess-voice-partial-tool-boundary",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
    )

    assert result["final_response"] == "The final answer is 42."
    assert result.get("already_sent") is True
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_end",
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_end",
    ]
    assert [call.get("text") for call in adapter.voice_events if call["type"] == "assistant_llm_text"] == [
        "Let me check that first.",
        "The final answer is 42.",
    ]
    starts = [call for call in adapter.voice_events if call["type"] == "assistant_llm_start"]
    assert starts[0]["turn_id"] != starts[1]["turn_id"]
    assert result["voice_turn_id"] == starts[1]["turn_id"]
    assert result["voice_turn_ids"] == [starts[0]["turn_id"], starts[1]["turn_id"]]


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_text_failure_sends_full_fallback(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialAgent,
        session_id="sess-voice-partial-text-failure",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=VoiceStreamTextFailureAdapter,
    )

    stream_turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    assert result["final_response"] == "Hello, voice."
    assert result.get("already_sent") is not True
    assert result["voice_turn_id"] != stream_turn_id
    assert result.get("response_previewed") is False
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_abort",
    ]
    assert adapter.cleaned_voice_streams


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_failure_clears_previewed_for_prefix(monkeypatch, tmp_path):
    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePreviewedPartialFailureAgent,
        session_id="sess-voice-partial-previewed-failure",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=VoiceStreamTextFailureAdapter,
    )

    assert result["final_response"] == "Hello, voice."
    assert result.get("response_previewed") is False
    assert result.get("already_sent") is not True


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_end_failure_suppresses_replay(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialAgent,
        session_id="sess-voice-partial-end-failure",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=VoiceStreamEndFailureAdapter,
    )

    stream_turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    assert result["final_response"] == "Hello, voice."
    assert result.get("already_sent") is True
    assert result["voice_turn_id"] == stream_turn_id
    assert [call["content"] for call in adapter.sent] == []
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_text",
    ]


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_end_failure_keeps_stream_turn_id(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialHistoryAgent,
        session_id="sess-voice-partial-end-failure-history",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=VoiceStreamEndFailureAdapter,
    )

    stream_turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    final_turn_id = result["messages"][-1]["voice_turn_id"]
    assert final_turn_id == stream_turn_id
    assert result["voice_turn_id"] == final_turn_id


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_timeout_cancels_late_delta(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialAgent,
        session_id="sess-voice-partial-timeout",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=SlowVoiceStreamTextAdapter,
    )

    await asyncio.sleep(0.05)

    assert result["final_response"] == "Hello, voice."
    assert result.get("already_sent") is not True
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_abort",
    ]
    assert adapter.cleaned_voice_streams


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_finish_timeout_cleans_stream(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VoicePartialAgent,
        session_id="sess-voice-partial-finish-timeout",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": True},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        adapter_cls=SlowVoiceStreamEndAdapter,
    )

    await asyncio.sleep(0.05)

    stream_turn_id = next(call["turn_id"] for call in adapter.voice_events if call["type"] == "assistant_llm_start")
    assert result["final_response"] == "Hello, voice."
    assert result.get("already_sent") is not True
    assert result.get("response_previewed") is False
    assert result["voice_turn_id"] != stream_turn_id
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_text",
        "assistant_llm_abort",
    ]
    assert adapter.cleaned_voice_streams


@pytest.mark.asyncio
async def test_run_agent_voice_server_partial_stream_closes_when_agent_raises(monkeypatch, tmp_path):
    RecordingVoiceStreamAdapter.instances = []
    with pytest.raises(RuntimeError, match="voice partial failure"):
        await _run_with_agent(
            monkeypatch,
            tmp_path,
            RaisingVoicePartialAgent,
            session_id="sess-voice-partial-raises",
            config_data={
                "display": {"tool_progress": "off", "interim_assistant_messages": False},
                "gateway": {
                    "platforms": {
                        "voice_server": {
                            "enabled": True,
                            "extra": {"partial_llm_streaming": True},
                        }
                    }
                },
            },
            platform=Platform.VOICE_SERVER,
            chat_id="personal-room",
            chat_type="channel",
            thread_id=None,
            adapter_cls=RecordingVoiceStreamAdapter,
        )

    adapter = RecordingVoiceStreamAdapter.instances[-1]
    assert [call["type"] for call in adapter.voice_events] == [
        "assistant_llm_start",
        "assistant_llm_text",
        "assistant_llm_abort",
    ]
    assert adapter.cleaned_voice_streams


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
async def test_run_agent_voice_server_queued_full_reply_uses_persisted_turn_id(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-voice-queued-full",
        pending_text="queued follow-up",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "gateway": {
                "platforms": {
                    "voice_server": {
                        "enabled": True,
                        "extra": {"partial_llm_streaming": False},
                    }
                }
            },
        },
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        thread_id=None,
        user_id="caller",
    )

    first_response_send = next(call for call in adapter.sent if call["content"] == "final response 1")
    assert first_response_send["metadata"]["turn_id"].startswith("voice_server-")
    assert first_response_send["metadata"]["participant_id"] == "caller"
    assert first_response_send["metadata"]["turn_id"] in result["voice_turn_ids"]
    assert result["voice_turn_id"] in result["voice_turn_ids"]
    assert adapter.held_spoken_turns == [first_response_send["metadata"]["turn_id"]]
    assert result["final_response"] == "final response 2"


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


@pytest.mark.asyncio
async def test_run_agent_relays_thinking_when_tool_progress_off(monkeypatch, tmp_path):
    """_thinking scratch text relays as a bubble when thinking_progress is on,
    even with tool_progress off.

    Regression: agent.tool_progress_callback used to be gated on
    tool_progress_enabled alone, so enabling only thinking_progress left the
    callback None and _thinking never relayed — despite the progress queue
    being created for it (needs_progress_queue = tool OR thinking).
    """
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ThinkingAgent,
        session_id="sess-thinking-on",
        config_data={"display": {"thinking_progress": True, "tool_progress": "off"}},
    )

    assert result["final_response"] == "done"
    blob = "\n".join(
        [c["content"] for c in adapter.sent] + [c["content"] for c in adapter.edits]
    )
    assert "weighing the options here" in blob


@pytest.mark.asyncio
async def test_run_agent_suppresses_thinking_when_thinking_off(monkeypatch, tmp_path):
    """With thinking_progress off and tool_progress off, _thinking is suppressed
    (no callback wired → no relay)."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ThinkingAgent,
        session_id="sess-thinking-off",
        config_data={"display": {"thinking_progress": False, "tool_progress": "off"}},
    )

    assert result["final_response"] == "done"
    blob = "\n".join(
        [c["content"] for c in adapter.sent] + [c["content"] for c in adapter.edits]
    )
    assert "weighing the options here" not in blob
