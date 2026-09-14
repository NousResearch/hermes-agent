"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import json
import threading
import sys
import time
import types
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

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


class DiscordProgressCaptureAdapter(ProgressCaptureAdapter):
    """Capture sends while exercising Discord's real preview formatter."""

    def __init__(self):
        super().__init__(platform=Platform.DISCORD)

    def format_tool_preview(self, preview, **kwargs):
        from plugins.platforms.discord.adapter import DiscordAdapter

        return DiscordAdapter.format_tool_preview(self, preview, **kwargs)


class MediaCaptureProgressAdapter(ProgressCaptureAdapter):
    """Capture native image batches without contacting a platform API."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.image_batches = []

    async def send_multiple_images(
        self, chat_id, images, metadata=None, human_delay=0.0
    ) -> None:
        self.image_batches.append(
            {
                "chat_id": chat_id,
                "images": images,
                "metadata": metadata,
            }
        )


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


class RetryableFirstEditProgressCaptureAdapter(ProgressCaptureAdapter):
    """Fail one progress edit transiently, then accept later edits."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.edit_outcomes = []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        if not self.edit_outcomes:
            self.edit_outcomes.append(False)
            return SendResult(
                success=False,
                error="temporary network failure",
                retryable=True,
                error_kind="transient",
            )
        self.edit_outcomes.append(True)
        return SendResult(success=True, message_id=message_id)


class RetryableOverflowEditProgressAdapter(SmallLimitProgressAdapter):
    """Fail the first split edit transiently, then keep editing."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.retryable_edit_failures = 0

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if self.retryable_edit_failures == 0:
            self.retryable_edit_failures += 1
            self.edits.append(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                }
            )
            return SendResult(
                success=False,
                error="temporary network failure",
                retryable=True,
                error_kind="transient",
            )
        return await super().edit_message(chat_id, message_id, content)


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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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


class NativeTaskCardAdapter(ProgressCaptureAdapter):
    def __init__(self, platform=Platform.SLACK):
        super().__init__(platform=platform)
        self.native_updates = []
        self.native_stops = 0

    def native_task_cards_enabled(self):
        return True

    async def send_native_task_card_progress(
        self,
        chat_id,
        tasks,
        *,
        title,
        reply_to=None,
        metadata=None,
        fallback_text=None,
    ) -> SendResult:
        self.native_updates.append(
            {
                "chat_id": chat_id,
                "tasks": [dict(task) for task in tasks],
                "metadata": dict(metadata or {}),
                "fallback_text": fallback_text,
            }
        )
        return SendResult(success=True, message_id="native-stream-1")

    async def stop_native_task_card_progress(
        self, chat_id, *, reply_to=None, metadata=None
    ):
        self.native_stops += 1

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
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


class FailingNativeTaskCardAdapter(NativeTaskCardAdapter):
    async def send_native_task_card_progress(self, *args, **kwargs) -> SendResult:
        await super().send_native_task_card_progress(*args, **kwargs)
        return SendResult(success=False, error="native stream unavailable", retryable=True)


class DuplicateNativeToolsAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tool_start_callback = kwargs.get("tool_start_callback")
        self.tool_complete_callback = kwargs.get("tool_complete_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_start_callback("call-a", "web_search", {"query": "alpha"})
        time.sleep(0.15)
        self.tool_start_callback("call-b", "web_search", {"query": "beta"})
        time.sleep(0.15)
        # Complete the second same-name call first. Correlation by tool name
        # would incorrectly mark call-a as failed here.
        self.tool_complete_callback(
            "call-b", "web_search", {"query": "beta"}, '{"error": "boom"}'
        )
        time.sleep(0.15)
        self.tool_complete_callback(
            "call-a", "web_search", {"query": "alpha"}, '{"success": true}'
        )
        time.sleep(0.15)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class ThinkingAgent:
    """Agent that emits _thinking scratch text (no tool calls).

    Used to prove the progress callback relays _thinking bubbles when
    thinking_progress is enabled but tool_progress is off.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_progress_callback("tool.started", "terminal", self.LONG_CMD, {})
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class UrlPreviewAgent:
    URL = "https://hermes-agent.nousresearch.com/docs/gateway/discord/tool-progress"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_progress_callback(
            "tool.started",
            "web_extract",
            self.URL,
            {"urls": [self.URL]},
        )
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_progress_callback("tool.started", "terminal", "first command", {})
        time.sleep(0.45)
        self.tool_progress_callback("tool.started", "terminal", "second command", {})
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class RetryableEditProgressAgent:
    """Keep the turn alive long enough to retry the same progress bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "first command", {})
        time.sleep(0.5)
        callback("tool.started", "terminal", "second command", {})
        time.sleep(1.7)
        callback("tool.started", "terminal", "third command", {})
        time.sleep(0.5)
        callback("tool.started", "terminal", "fourth command", {})
        time.sleep(0.6)
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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
    expected_metadata = {
        "thread_id": "1234567890.000001",
        "message_id": "1234567890.000001",
    }
    assert adapter.sent[0]["metadata"] == expected_metadata
    assert all(call["metadata"] == expected_metadata for call in adapter.typing)


@pytest.mark.asyncio
async def test_progress_carries_anchor_for_relay_discord_auto_thread(monkeypatch, tmp_path):
    """Relay Discord channel-initiate: the thread doesn't exist at ingest, so
    the connector auto-threads on the reply anchor and stamps
    prospective_thread_id. The tool-progress / status bubbles must carry that
    anchor (reply_to + metadata.reply_to_message_id) so they route into the
    SAME auto-thread as the final reply — otherwise the search-status updates
    leak into the parent channel (staging repro 2026-08-02)."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    import yaml
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"platforms": {"discord": {"tool_progress": "all"}}}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.RELAY)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    # Channel-initiating message: no thread_id yet, but the connector stamped
    # the prospective thread id (== the triggering message id). Relay ingress
    # keeps the underlying platform (discord) on the source for display policy,
    # but delivery/progress route through the one live RelayAdapter.
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-parent",
        chat_type="group",
        thread_id=None,
        prospective_thread_id="msg-anchor-1",
        delivered_via_upstream_relay=True,
    )

    result = await runner._run_agent(
        message="find me a gift",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-relay-thread",
        session_key="agent:main:discord:thread:chan-parent:msg-anchor-1",
        event_message_id="msg-anchor-1",
    )

    assert result["final_response"] == "done"
    assert adapter.sent, "expected at least one progress send"
    # Every progress send must carry the anchor so the connector threads it.
    for call in adapter.sent:
        assert call["reply_to"] == "msg-anchor-1", call
        assert (call["metadata"] or {}).get("reply_to_message_id") == "msg-anchor-1", call
        # Discord lifecycle/status sends are marked non-conversational.
        assert (call["metadata"] or {}).get("non_conversational") is True, call


@pytest.mark.asyncio
async def test_progress_no_anchor_for_native_discord_thread_event(monkeypatch, tmp_path):
    """A message ARRIVING in an existing Discord thread (not the relay
    auto-thread lane) must NOT get the synthetic prospective anchor — it already
    routes by its real thread. Guards against over-broadening the relay fix."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    import yaml
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"platforms": {"discord": {"tool_progress": "all"}}}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.RELAY)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    # No prospective_thread_id (event is IN a real thread already).
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="real-thread-9",
        chat_type="thread",
        thread_id="real-thread-9",
        delivered_via_upstream_relay=True,
    )

    result = await runner._run_agent(
        message="continue",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-in-thread",
        session_key="agent:main:discord:thread:real-thread-9:real-thread-9",
        event_message_id="msg-2",
    )

    assert result["final_response"] == "done"
    # The relay-prospective synthetic anchor path must NOT engage; progress
    # routes by the real thread's own metadata, not a forced reply_to anchor.
    for call in adapter.sent:
        meta = call["metadata"] or {}
        # The real thread id drives routing; we did not inject the anchor
        # reply_to that the prospective lane uses.
        assert meta.get("thread_id") == "real-thread-9" or call["reply_to"] != "msg-2", call


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


def test_discord_truncated_tool_url_links_to_full_destination(monkeypatch, tmp_path):
    """The real gateway path must retain the URL beyond its visible cap."""
    import yaml

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = UrlPreviewAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"tool_preview_length": 0}}),
        encoding="utf-8",
    )

    adapter = DiscordProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    source = SessionSource(
        platform=Platform.DISCORD,
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
            session_id="sess-discord-url",
            session_key="agent:main:discord:dm:12345",
        )
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    visible = UrlPreviewAgent.URL[:37] + "..."
    label = visible.removeprefix("https://")
    assert f"[{label}](<{UrlPreviewAgent.URL}>)" in adapter.sent[0]["content"]


class CommentaryAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1 and self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        return {
            "final_response": f"final response {type(self).calls}",
            "messages": [],
            "api_calls": 1,
        }


class QueuedMediaAgent:
    """Return an explicit image attachment before a queued follow-up."""

    calls = 0
    media_path = None

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            final_response = f"first response\nMEDIA:{type(self).media_path}"
            if self.stream_delta_callback:
                self.stream_delta_callback("first response")
        else:
            final_response = "follow-up processed"
            if self.stream_delta_callback:
                self.stream_delta_callback(final_response)
        return {
            "final_response": final_response,
            "messages": [],
            "api_calls": 1,
        }


class QueuedSilenceAgent:
    """First turn is intentionally silent; queued follow-up still runs."""

    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls += 1
        return {
            "final_response": "NO_REPLY" if type(self).calls == 1 else "follow-up processed",
            "messages": [],
            "api_calls": 1,
        }


class QueuedFailedEmptyAgent:
    """First turn fails empty; its normalized error must send before follow-up."""

    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "final_response": "",
                "messages": [],
                "api_calls": 1,
                "failed": True,
                "error": "provider exploded",
            }
        return {
            "final_response": "follow-up processed",
            "messages": [],
            "api_calls": 1,
        }


class BackgroundReviewAgent:
    def __init__(self, **kwargs):
        self.background_review_callback = kwargs.get("background_review_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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
    user_id=None,
    scope_id=None,
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
        scope_id=scope_id,
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
async def test_slack_native_progress_correlates_concurrent_duplicate_tools_by_id(
    monkeypatch, tmp_path
):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-ids",
        config_data={
            "display": {"platforms": {"slack": {"tool_progress": "off"}}}
        },
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="thread-1",
        adapter_cls=NativeTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert adapter.native_updates
    second_completed = next(
        update
        for update in adapter.native_updates
        if {task["id"]: task["status"] for task in update["tasks"]}
        == {"call-a": "in_progress", "call-b": "error"}
    )
    assert second_completed["metadata"]["recipient_team_id"] == "T1"
    assert second_completed["metadata"]["recipient_user_id"] == "U1"
    assert adapter.native_updates[-1]["tasks"] == [
        {
            "id": "call-a",
            "title": "web_search - alpha",
            "status": "complete",
        },
        {
            "id": "call-b",
            "title": "web_search - beta",
            "status": "error",
        },
    ]
    assert adapter.sent == []
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_slack_native_failure_keeps_editing_one_live_text_fallback(
    monkeypatch, tmp_path
):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-fallback",
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="thread-1",
        adapter_cls=FailingNativeTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert len(adapter.native_updates) == 1
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"].endswith("web_search - alpha - running")
    assert len(adapter.edits) >= 2
    assert {edit["message_id"] for edit in adapter.edits} == {"progress-1"}
    assert adapter.edits[-1]["content"].endswith("web_search - beta - error")
    assert "web_search - alpha - complete" in adapter.edits[-1]["content"]
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_retryable_overflow_edit_keeps_editable_bubble_identity(monkeypatch, tmp_path):
    """A transient split edit must retain can_edit and the current message ID."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-retry-overflow-same-message",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            }
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=RetryableOverflowEditProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, RetryableOverflowEditProgressAdapter)
    assert adapter.retryable_edit_failures == 1
    assert len(adapter.sent) >= 2
    assert adapter.edits[0]["message_id"] == "progress-1"
    assert any(call["message_id"] == "progress-1" for call in adapter.edits[1:])
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []


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


class TransformedStreamAgent:
    """Streams a response, then signals the gateway that a plugin hook
    (``transform_llm_output``) modified the final text after streaming
    finished. ``run_conversation`` returns ``response_transformed=True``
    plus a ``final_response`` that diverges from what was streamed.
    """

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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
async def test_run_agent_queued_message_delivers_first_response_media(monkeypatch, tmp_path):
    """Queued follow-ups must preserve explicit attachments from the first turn."""
    media_path = tmp_path / "queued-first-response.png"
    media_path.write_bytes(b"not-a-real-png-but-a-real-file")
    QueuedMediaAgent.calls = 0
    QueuedMediaAgent.media_path = media_path

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedMediaAgent,
        session_id="sess-queued-media",
        pending_text="queued follow-up",
        platform=Platform.DISCORD,
        chat_id="discord-thread",
        chat_type="group",
        thread_id="discord-thread",
        adapter_cls=MediaCaptureProgressAdapter,
    )

    assert result["final_response"] == "follow-up processed"
    assert isinstance(adapter, MediaCaptureProgressAdapter)
    assert {
        "sent_texts": [call["content"] for call in adapter.sent],
        "image_batches": adapter.image_batches,
    } == {
        "sent_texts": ["first response"],
        "image_batches": [
            {
                "chat_id": "discord-thread",
                "images": [(media_path.as_uri(), "")],
                "metadata": {"thread_id": "discord-thread"},
            }
        ],
    }


@pytest.mark.asyncio
async def test_run_agent_queued_message_delivers_streamed_first_response_media(
    monkeypatch, tmp_path,
):
    """Streaming first-turn text must not suppress its explicit attachment."""
    media_path = tmp_path / "queued-streamed-first-response.png"
    media_path.write_bytes(b"not-a-real-png-but-a-real-file")
    QueuedMediaAgent.calls = 0
    QueuedMediaAgent.media_path = media_path

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedMediaAgent,
        session_id="sess-queued-streamed-media",
        pending_text="queued follow-up",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1},
        },
        platform=Platform.DISCORD,
        chat_id="discord-thread",
        chat_type="group",
        thread_id="discord-thread",
        adapter_cls=MediaCaptureProgressAdapter,
    )

    assert result["final_response"] == "follow-up processed"
    assert isinstance(adapter, MediaCaptureProgressAdapter)
    all_text = [call["content"] for call in adapter.sent + adapter.edits]
    assert all("MEDIA:" not in text for text in all_text)
    assert adapter.image_batches == [
        {
            "chat_id": "discord-thread",
            "images": [(media_path.as_uri(), "")],
            "metadata": {"thread_id": "discord-thread"},
        }
    ]


@pytest.mark.asyncio
async def test_run_agent_suppresses_silent_first_turn_and_processes_queued_followup(
    monkeypatch, tmp_path,
):
    """Regression: queued direct-send must not leak NO_REPLY to the channel."""
    QueuedSilenceAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedSilenceAgent,
        session_id="sess-queued-silence",
        pending_text="queued follow-up",
        platform=Platform.SLACK,
        chat_id="C123",
        thread_id="1712345678.000100",
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert QueuedSilenceAgent.calls == 2
    assert result["final_response"] == "follow-up processed"
    assert "NO_REPLY" not in sent_texts


@pytest.mark.asyncio
async def test_run_agent_sends_normalized_failure_before_queued_followup(
    monkeypatch, tmp_path,
):
    """Queued delivery uses finalized output, not the raw empty agent result."""
    QueuedFailedEmptyAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedFailedEmptyAgent,
        session_id="sess-queued-failed-empty",
        pending_text="queued follow-up",
        platform=Platform.SLACK,
        chat_id="C123",
        thread_id="1712345678.000100",
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert QueuedFailedEmptyAgent.calls == 2
    assert result["final_response"] == "follow-up processed"
    assert any("The request failed: provider exploded" in text for text in sent_texts)


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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
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


class TestSlackReplyInThreadProgressRouting:
    """#18859: reply_in_thread=false must stop progress from creating threads."""

    def test_slack_reply_in_thread_false_drops_synthetic_thread(self):
        from gateway.run import _resolve_progress_thread_id

        # source.thread_id == event ts is the adapter's synthetic
        # session-keying thread for top-level messages — not a real thread.
        assert _resolve_progress_thread_id(
            Platform.SLACK,
            source_thread_id="1700000000.000100",
            event_message_id="1700000000.000100",
            reply_in_thread=False,
        ) is None

    def test_buzz_uses_event_message_id_as_progress_thread(self):
        """Buzz has no native thread_id; progress must reply-to the trigger."""
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz",
            source_thread_id=None,
            event_message_id="evt-trigger-001",
            reply_in_thread=True,
        ) == "evt-trigger-001"


class SmallLimitCodeBlockProgressAdapter(SmallLimitProgressAdapter):
    """Tiny-limit adapter that also renders terminal commands as fences."""

    supports_code_blocks = True


class Utf16SmallLimitProgressAdapter(SmallLimitProgressAdapter):
    """Counts UTF-16 code units, as Telegram-compatible adapters do."""

    @property
    def message_len_fn(self):
        return lambda text: len(text.encode("utf-16-le")) // 2

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if self.message_len_fn(content) > self.MAX_MESSAGE_LENGTH:
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


class FailFirstEditProgressAdapter(SmallLimitProgressAdapter):
    """Permanently rejects the first accumulated progress edit."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.failed_edit_ids = []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if not self.failed_edit_ids:
            self.failed_edit_ids.append(message_id)
            self.edits.append(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                }
            )
            return SendResult(success=False, error="message cannot be edited")
        return await super().edit_message(chat_id, message_id, content)


class TrackingSmallLimitProgressAdapter(SmallLimitProgressAdapter):
    """Records IDs so a continuation chunk cannot be edited later."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.sent_message_ids = []
        self.edited_message_ids = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(
            chat_id, content, reply_to=reply_to, metadata=metadata
        )
        self.sent_message_ids.append(result.message_id)
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edited_message_ids.append(message_id)
        return await super().edit_message(chat_id, message_id, content)


class FullTerminalArgsAgent:
    """Emits every terminal option so full mode cannot collapse to command-only."""

    ARGS = {
        "command": "python -c \"print('full terminal payload')\"",
        "workdir": "/tmp/full-mode-project",
        "timeout": 321,
        "background": True,
        "pty": False,
        "notify_on_complete": True,
        "watch_patterns": ["ready", "finished"],
    }
    RESULT_MARKER = "FULL-MODE-RESULT-MUST-NOT-RENDER"
    THINKING_MARKER = "FULL-MODE-THINKING-MUST-NOT-RENDER"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", self.ARGS["command"], dict(self.ARGS))
        cb(
            "tool.completed",
            "terminal",
            None,
            None,
            result={"output": self.RESULT_MARKER},
        )
        cb("_thinking", self.THINKING_MARKER)
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullMediaArgsAgent:
    """Emits ordinary non-terminal media/path/question arguments."""

    ARGS = {
        "media": ["/tmp/frame-α.png", "/tmp/frame-β.jpg"],
        "path": "/tmp/screenshots",
        "question": "Compare every frame without dropping metadata",
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "vision_analyze", self.ARGS["question"], dict(self.ARGS)
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullSecretArgsAgent:
    """Emits a credential while global redaction is disabled by the test."""

    SECRET = "opaque-credential-604"
    ARGS = {
        "token": SECRET,
        "path": "/tmp/non-secret-path",
        "nested": {"question": "keep this structure"},
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "custom_upload", "upload", dict(self.ARGS)
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedSingleFullEntryAgent:
    """Emits one progress entry larger than the adapter's message limit."""

    PAYLOAD = "lossless-" + "0123456789abcdef" * 40 + "-end"
    ARGS = {"question": PAYLOAD}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "full_payload_tool", "oversized-preview", dict(self.ARGS)
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullStrictRedactionAgent:
    """Exercises full-mode's chat-strict structured redaction boundary."""

    QUERY_SECRET = "opaque-query-value-917263"
    QUERY_VISIBLE_VALUE = "ordinary-query-value-381"
    URL_USER = "opaque-url-user-624"
    URL_PASSWORD = "opaque-url-password-735"
    URL_FRAGMENT = "opaque-url-fragment-846"
    AUTHORIZATION = "Custom opaque-header-authorization-159"
    COOKIE = "session=opaque-cookie-value-260"
    QUOTED_SECRET = 'opaque-before-quote"opaque-after-quote-371'
    TOKEN_PREFIX_VALUE = "opaque-token-prefix-key-value-593"
    REDACTED_KEY_ONE = "https://example.test/key?q=opaque-one"
    REDACTED_KEY_TWO = "https://example.test/key?q=opaque-two"
    ARGS = {
        "request_url": (
            f"prefix https://{URL_USER}:{URL_PASSWORD}@api.example.test/v1/items"
            f"?access_token={QUERY_SECRET}&visible={QUERY_VISIBLE_VALUE}"
            f"#{URL_FRAGMENT} suffix"
        ),
        "headers": {
            "Authorization": AUTHORIZATION,
            "Cookie": COOKIE,
        },
        "secret_value": QUOTED_SECRET,
        "tokenPrefixData": TOKEN_PREFIX_VALUE,
        "safe": {"path": "/tmp/visible", "items": ["alpha", ("beta",)]},
        REDACTED_KEY_ONE: "first-collision-value",
        REDACTED_KEY_TWO: "second-collision-value",
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "strict_redaction_tool", "strict", self.ARGS
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullCliCredentialAgent:
    """Emits opaque CLI and structured credentials for the enqueue boundary."""

    SECRETS = {
        "curl_short": "opaque-curl-short-481",
        "curl_long": "opaque-curl-long-592",
        "password": "opaque-password-quote-'-603",
        "token": "opaque-cli-token-714",
        "bearer": "opaque-body-bearer-825",
        "jwt": "opaque-body-jwt-936",
        "env_password": "opaque env password phrase 147",
        "recovery_key": "opaque recovery key material 258",
    }
    RECOVERY_ENV_NAME = "MATRIX_RECOVERY_" + "KEY"
    COMMAND = (
        f"export MY_DATABASE_PASSWORD='{SECRETS['env_password']}' "
        f"export {RECOVERY_ENV_NAME}='{SECRETS['recovery_key']}' "
        f"curl -u alice:{SECRETS['curl_short']} "
        f"--user=alice:{SECRETS['curl_long']} "
        f"--password \"{SECRETS['password']}\" "
        f"--token='{SECRETS['token']}' https://example.test/health"
    )
    ARGS = {
        "command": COMMAND,
        "body": {"bearer": SECRETS["bearer"], "jwt": SECRETS["jwt"]},
        "safe": {"path": "/tmp/visible", "token_count": 4},
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "credential_cli", "strict", self.ARGS
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedVerboseTerminalAgent:
    """Emits a verbose fenced entry that exceeds a tiny adapter limit."""

    COMMAND = "printf 'verbose fence'\n" + "x" * 500

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "terminal", self.COMMAND, {"command": self.COMMAND}
        )
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class EditFailureThenOversizedFullAgent:
    """Disables editing before emitting an oversized marked full entry."""

    BIG_ARGS = {"question": "after-edit-failure-" + "abcdef0123456789" * 40}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "first_full_tool", "first", {"value": "first"})
        time.sleep(1.7)
        cb("tool.started", "second_full_tool", "second", {"value": "second"})
        time.sleep(0.4)
        cb("tool.started", "full_payload_tool", "big", self.BIG_ARGS)
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedThenSmallFullAgent:
    """Emits a continuation-split full entry followed by another tool."""

    BIG_ARGS = {"question": "split-state-" + "0123456789abcdef" * 40}
    SMALL_ARGS = {"question": "fresh bubble after split"}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "full_payload_tool", "big", self.BIG_ARGS)
        time.sleep(0.35)
        cb("tool.started", "post_split_tool", "small", self.SMALL_ARGS)
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class NativeCardsProgressAdapter(ProgressCaptureAdapter):
    def native_task_cards_enabled(self):
        return True

    async def send_native_task_card_progress(self, **kwargs):
        raise AssertionError("full arguments must not become compact native cards")


def _first_progress_json(adapter):
    """Return the JSON dict from a one-entry full-mode progress message."""
    assert adapter.sent, "expected a full-mode progress bubble"
    content = adapter.sent[0]["content"]
    _, json_text = content.split("\n", 1)
    return content, json.loads(json_text)


def _expected_full_message(tool_name, args):
    from agent.display import get_tool_emoji
    from gateway.run_turn_runner import _redact_full_progress_args

    safe_args = _redact_full_progress_args(args)
    return (
        f"{get_tool_emoji(tool_name, default='⚙️')} {tool_name}\n"
        f"{json.dumps(safe_args, ensure_ascii=False, default=str)}"
    )


def test_full_mode_structured_redactor_is_recursive_and_non_mutating():
    from gateway.run_turn_runner import _redact_full_progress_args

    original_tuple = ("visible", {"password": "opaque-password-482"})
    original_list = [original_tuple]
    original = {
        "safe": original_list,
        "apiKey": "opaque-api-key-604",
        "x-api-key": "opaque-x-api-key-609",
        "private_key": {"nested": "opaque-private-key-715"},
        "key_material": ["opaque-key-material-826"],
    }

    redacted = _redact_full_progress_args(original)

    assert redacted is not original
    assert redacted["safe"] is not original_list
    assert redacted["safe"][0] is not original_tuple
    assert isinstance(redacted["safe"][0], tuple)
    assert original["safe"] == [
        ("visible", {"password": "opaque-password-482"})
    ]
    assert redacted["safe"][0][1]["password"] == "***"
    assert redacted["apiKey"] == "***"
    assert redacted["x-api-key"] == "***"
    assert redacted["private_key"] == "***"
    assert redacted["key_material"] == "***"


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "前https://user:password@example.test/x?plain=opaque#fragment",
            "前https://***@example.test/x?plain=***#***",
        ),
        (
            "prefix（https://name:pass@example.test/path?q=hidden#anchor",
            "prefix（https://***@example.test/path?q=***#***",
        ),
    ],
)
def test_full_mode_redacts_urls_adjacent_to_cjk_and_punctuation(value, expected):
    from gateway.run_turn_runner import _redact_full_progress_string

    rendered = _redact_full_progress_string(value)

    assert rendered == expected
    secrets = (
        "user",
        "password",
        "opaque",
        "fragment",
        "name",
        "pass",
        "hidden",
        "anchor",
    )
    for secret in secrets:
        if secret in value:
            assert secret not in rendered


def test_full_mode_masks_lowercase_compound_secret_keys_without_renaming_keys():
    from gateway.run_turn_runner import _redact_full_progress_args

    original = {
        "dbpassword": "raw-db-password-183",
        "oauth_token": "raw-oauth-token-294",
        "authtoken": "raw-auth-token-305",
        "clientcredentialsblob": "raw-client-credentials-416",
        "safevalue": "keep-visible",
    }

    redacted = _redact_full_progress_args(original)

    assert set(redacted) == set(original)
    assert redacted["safevalue"] == "keep-visible"
    for key in original.keys() - {"safevalue"}:
        assert redacted[key] == "***"
        assert original[key] not in json.dumps(redacted)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "前custom+v1.2://url-user:url-pass@example.test/items"
            "?visible=opaque-visible&empty=#opaque-fragment 后",
            "前custom+v1.2://***@example.test/items"
            "?visible=***&empty=***#*** 后",
        ),
        (
            "git+ssh://git@example.test/repo?ref=opaque-ref&flag#opaque-anchor",
            "git+ssh://***@example.test/repo?ref=***&flag#***",
        ),
        (
            "s3://opaque-bucket-user@bucket.example.test/object?version=opaque-version",
            "s3://***@bucket.example.test/object?version=***",
        ),
        (
            "authorization://header-user:header-pass@example.test/session"
            "?scope=opaque-scope#opaque-fragment",
            "authorization://***@example.test/session?scope=***#***",
        ),
    ],
)
def test_full_mode_redacts_every_rfc_scheme_url(value, expected):
    from gateway.run_turn_runner import _redact_full_progress_string

    assert _redact_full_progress_string(value) == expected


def test_full_mode_masks_dsn_and_connection_url_compound_keys_fail_closed():
    from gateway.run_turn_runner import _redact_full_progress_args

    original = {
        "dsn": "opaque-dsn-100",
        "database_dsn": "opaque-database-dsn-200",
        "primaryDatabaseUrl": "opaque-primary-database-url-300",
        "readonly_connection_string": "opaque-readonly-connection-string-400",
        "db_uri": {"nested": "opaque-db-uri-500"},
        "monkey": "keep-monkey-visible",
        "monkey_business": "keep-business-visible",
        "database_name": "inventory",
        "connection_timeout": 30,
        "url": "https://example.test/health",
    }

    redacted = _redact_full_progress_args(original)

    for key in {
        "dsn",
        "database_dsn",
        "primaryDatabaseUrl",
        "readonly_connection_string",
        "db_uri",
    }:
        assert redacted[key] == "***"
        assert str(original[key]) not in json.dumps(redacted)
    assert redacted["monkey"] == "keep-monkey-visible"
    assert redacted["monkey_business"] == "keep-business-visible"
    assert redacted["database_name"] == "inventory"
    assert redacted["connection_timeout"] == 30
    assert redacted["url"] == "https://example.test/health"


def test_full_mode_masks_sensitive_string_and_list_headers():
    from gateway.run_turn_runner import _redact_full_progress_args

    header_values = [
        "Authorization: Custom opaque-authorization-510",
        "Proxy-Authorization: Basic opaque-proxy-authorization-620",
        "Cookie: session=opaque-cookie-730; theme=dark",
        "Set-Cookie: session=opaque-set-cookie-840; HttpOnly; Secure",
        "X-API-Key: opaque-x-api-key-950",
        "API-Key: opaque-api-key-061",
        "X-Auth-Token: opaque-auth-token-172",
        "CSRF-Token: opaque-csrf-token-283",
    ]
    original = {
        "headers": header_values,
        "header_block": (
            "Authorization: Bearer opaque-block-authorization-394\n"
            "Set-Cookie: session=opaque-block-cookie-405; SameSite=Lax"
        ),
        "safe_headers": [
            "Accept: application/json",
            "Content-Type: application/json",
            "X-Request-ID: public-trace-123",
        ],
    }

    redacted = _redact_full_progress_args(original)

    assert redacted["headers"] == [
        f"{header.partition(':')[0]}: ***" for header in header_values
    ]
    assert redacted["header_block"] == (
        "Authorization: ***\nSet-Cookie: ***"
    )
    assert redacted["safe_headers"] == original["safe_headers"]
    assert all(secret not in json.dumps(redacted) for secret in (
        "opaque-authorization-510",
        "opaque-proxy-authorization-620",
        "opaque-cookie-730",
        "opaque-set-cookie-840",
        "opaque-x-api-key-950",
        "opaque-api-key-061",
        "opaque-auth-token-172",
        "opaque-csrf-token-283",
        "opaque-block-authorization-394",
        "opaque-block-cookie-405",
    ))


def test_full_mode_masks_sensitive_headers_inside_curl_strings_losslessly():
    from gateway.run_turn_runner import _redact_full_progress_string

    command = (
        "curl -H 'Authorization: Bearer opaque-curl-auth-516' "
        "-H \"Proxy-Authorization: Basic opaque-curl-proxy-627\" "
        "-H 'Cookie: sid=opaque-curl-cookie-738; theme=dark' "
        "-H 'X-API-Key: opaque-curl-api-key-849' "
        "-H 'X-CSRF-Token: opaque-curl-csrf-950' "
        "-H 'Accept: application/json' https://example.test/health"
    )

    redacted = _redact_full_progress_string(command)

    assert redacted == (
        "curl -H 'Authorization: ***' "
        "-H \"Proxy-Authorization: ***\" "
        "-H 'Cookie: ***' "
        "-H 'X-API-Key: ***' "
        "-H 'X-CSRF-Token: ***' "
        "-H 'Accept: application/json' https://example.test/health"
    )


def test_full_mode_masks_cli_credentials_quote_safely():
    from gateway.run_turn_runner import _redact_full_progress_string

    command = (
        "curl -u alice:opaque-short-password-101 "
        "--user bob:opaque-long-password-202 "
        "--user=carol:opaque-equals-password-303 "
        "--password 'opaque spaced password 404' "
        "--password=opaque-equals-password-505 "
        '"--token" "opaque quoted token 606" '
        "'--api-key=opaque api key 707' "
        "--jwt opaque-jwt-value-808 --bearer=opaque-bearer-value-909 "
        "https://example.test/health"
    )

    redacted = _redact_full_progress_string(command)

    assert redacted == (
        "curl -u *** "
        "--user *** "
        "--user=*** "
        "--password '***' "
        "--password=*** "
        '"--token" "***" '
        "'--api-key=***' "
        "--jwt *** --bearer=*** "
        "https://example.test/health"
    )
    for secret in (
        "opaque-short-password-101",
        "opaque-long-password-202",
        "opaque-equals-password-303",
        "opaque spaced password 404",
        "opaque-equals-password-505",
        "opaque quoted token 606",
        "opaque api key 707",
        "opaque-jwt-value-808",
        "opaque-bearer-value-909",
    ):
        assert secret not in redacted


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "MY_DATABASE_PASSWORD='opaque database phrase 101' command",
            "MY_DATABASE_PASSWORD='***' command",
        ),
        (
            'export API_TOKEN="opaque exported token phrase 202"; command',
            'export API_TOKEN="***"; command',
        ),
        (
            "env CLIENT_SECRET='opaque client phrase 303' command",
            "env CLIENT_SECRET='***' command",
        ),
        (
            "'MY_DATABASE_PASSWORD=opaque whole word phrase 404' command",
            "'MY_DATABASE_PASSWORD=***' command",
        ),
        (
            "MY_DATABASE_PASSWORD=opaque-single-value command",
            "MY_DATABASE_PASSWORD=*** command",
        ),
        (
            "MY_DATABASE_LABEL='visible database label' command",
            "MY_DATABASE_LABEL='visible database label' command",
        ),
    ],
)
def test_full_mode_masks_quoted_secret_environment_assignments(command, expected):
    from gateway.run_turn_runner import _redact_full_progress_string

    assert _redact_full_progress_string(command) == expected


def test_full_mode_masks_key_suffix_environment_assignment():
    from gateway.run_turn_runner import _redact_full_progress_string

    env_name = "MATRIX_RECOVERY_" + "KEY"
    secret = "opaque recovery material 515"
    command = env_name + "='" + secret + "' command"
    expected = env_name + "='***' command"

    rendered = _redact_full_progress_string(command)

    assert rendered == expected
    assert secret not in rendered


def test_full_mode_keeps_benign_cli_flag_lookalikes():
    from gateway.run_turn_runner import _redact_full_progress_string

    command = (
        "curl --user alice --username alice:visible --user-agent alice:visible "
        "--password-policy strict --password-file ./fixture.txt "
        "--token-count 4 --tokenizer local --api-key-file ./public.json "
        "--jwt-decoder local --bearer-format compact; "
        "tool --password ; echo still-visible"
    )

    assert _redact_full_progress_string(command) == command


def test_full_mode_masks_canonical_body_secret_keys_plus_bearer():
    _SENSITIVE_BODY_KEYS = {'private_key', 'api_key', 'key', 'apikey', 'token', 'access_token', 'password', 'authorization', 'secret', 'client_secret', 'id_token', 'jwt', 'refresh_token', 'auth'}
    from gateway.run_turn_runner import _redact_full_progress_args

    original: dict[str, object] = {
        key: f"opaque-body-value-{index}"
        for index, key in enumerate(sorted(_SENSITIVE_BODY_KEYS))
    }
    original["bearer"] = "opaque-body-bearer-extra"
    original["safe_payload"] = {"path": "/tmp/visible", "count": 4}

    redacted = _redact_full_progress_args(original)

    for key in _SENSITIVE_BODY_KEYS | {"bearer"}:
        assert redacted[key] == "***"
        assert str(original[key]) not in json.dumps(redacted)
    assert redacted["safe_payload"] == original["safe_payload"]


def test_full_mode_masks_query_only_oauth_fragments():
    from gateway.run_turn_runner import _redact_full_progress_string

    value = (
        "exchange ?client_id=public-client&code=opaque-oauth-code-111"
        "&state=opaque-oauth-state-222#opaque-oauth-fragment then continue"
    )

    assert _redact_full_progress_string(value) == (
        "exchange ?client_id=***&code=***&state=***#*** then continue"
    )


def test_full_mode_masks_relative_and_network_url_credentials():
    from gateway.run_turn_runner import _redact_full_progress_string

    assert _redact_full_progress_string(
        "POST /callback?client_id=public-client&code=opaque-oauth-code"
        "&state=opaque-state#opaque-fragment"
    ) == "POST /callback?client_id=***&code=***&state=***#***"
    assert _redact_full_progress_string(
        "fetch //alice:opaque-password@example.test/path?token=opaque-token"
        "&visible=public-value"
    ) == "fetch //alice:***@example.test/path?token=***&visible=***"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fetch //alice:opaque@host/path", "fetch //alice:***@host/path"),
        ("redirect /callback#opaque", "redirect /callback#***"),
        ("open callback?code=opaque", "open callback?code=***"),
        ("open callback#opaque", "open callback#***"),
    ],
)
def test_full_mode_masks_queryless_and_bare_relative_url_credentials(
    value, expected
):
    from gateway.run_turn_runner import _redact_full_progress_string

    assert _redact_full_progress_string(value) == expected


def test_full_mode_keeps_noncredential_prose_and_invalid_query_syntax():
    from gateway.run_turn_runner import _redact_full_progress_args

    original = {
        "monkey": "The monkey checks which code path handles state transitions.",
        "question": "Ready? code = example and state = documented.",
        "header_docs": "Authorization guide: public documentation",
        "safe_headers": ["CookieJar: local.txt", "X-API-Version: 2026-07-16"],
        "invalid_scheme": "1custom://user@example.test/path?visible=value",
    }

    assert _redact_full_progress_args(original) == original


@pytest.mark.asyncio
async def test_full_mode_terminal_shows_all_args_without_code_block(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullTerminalArgsAgent,
        session_id="sess-full-terminal-all-fields",
        config_data={
            "display": {
                "tool_progress": "off",
                "tool_preview_length": 8,
                "platforms": {"telegram": {"tool_progress": "full"}},
            }
        },
        adapter_cls=CodeBlockProgressAdapter,
    )

    assert result["final_response"] == "done"
    content, rendered_args = _first_progress_json(adapter)
    assert rendered_args == FullTerminalArgsAgent.ARGS
    assert "```" not in content
    all_progress = "\n".join(
        call["content"] for call in adapter.sent + adapter.edits
    )
    assert FullTerminalArgsAgent.RESULT_MARKER not in all_progress
    assert FullTerminalArgsAgent.THINKING_MARKER not in all_progress


@pytest.mark.asyncio
async def test_full_mode_ignores_tool_preview_length(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullTerminalArgsAgent,
        session_id="sess-full-ignores-preview-length",
        config_data={
            "display": {"tool_progress": "full", "tool_preview_length": 1}
        },
        adapter_cls=CodeBlockProgressAdapter,
    )

    _, rendered_args = _first_progress_json(adapter)
    assert rendered_args["command"] == FullTerminalArgsAgent.ARGS["command"]
    assert rendered_args["watch_patterns"] == ["ready", "finished"]


@pytest.mark.asyncio
async def test_full_mode_nonterminal_preserves_media_path_and_question(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullMediaArgsAgent,
        session_id="sess-full-media-args",
        config_data={"display": {"tool_progress": "full", "tool_preview_length": 2}},
    )

    _, rendered_args = _first_progress_json(adapter)
    assert rendered_args == FullMediaArgsAgent.ARGS


@pytest.mark.asyncio
async def test_full_mode_force_redacts_secret_and_preserves_structure(monkeypatch, tmp_path):
    import agent.redact as redact

    # Full-mode chat is a hard safety boundary: force=True must win even when
    # the operator disabled ordinary global/log redaction.
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullSecretArgsAgent,
        session_id="sess-full-force-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    assert FullSecretArgsAgent.SECRET not in content
    assert rendered_args["token"] != FullSecretArgsAgent.SECRET
    assert rendered_args["path"] == FullSecretArgsAgent.ARGS["path"]
    assert rendered_args["nested"] == FullSecretArgsAgent.ARGS["nested"]
    assert set(rendered_args) == set(FullSecretArgsAgent.ARGS)


@pytest.mark.asyncio
async def test_full_mode_cli_and_body_credentials_are_redacted_before_enqueue(
    monkeypatch, tmp_path
):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullCliCredentialAgent,
        session_id="sess-full-cli-credential-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    assert all(
        secret not in content for secret in FullCliCredentialAgent.SECRETS.values()
    )
    assert rendered_args["command"] == (
        "export MY_DATABASE_PASSWORD='***' "
        "export MATRIX_RECOVERY_KEY='***' "
        "curl -u *** --user=*** --password \"***\" "
        "--token='***' https://example.test/health"
    )
    assert rendered_args["body"] == {"bearer": "***", "jwt": "***"}
    assert rendered_args["safe"]["path"] == "/tmp/visible"


@pytest.mark.asyncio
async def test_full_mode_strict_redaction_happens_before_json_serialization(
    monkeypatch, tmp_path
):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullStrictRedactionAgent,
        session_id="sess-full-strict-structured-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    raw_secrets = {
        FullStrictRedactionAgent.QUERY_SECRET,
        FullStrictRedactionAgent.QUERY_VISIBLE_VALUE,
        FullStrictRedactionAgent.URL_USER,
        FullStrictRedactionAgent.URL_PASSWORD,
        FullStrictRedactionAgent.URL_FRAGMENT,
        FullStrictRedactionAgent.AUTHORIZATION,
        FullStrictRedactionAgent.COOKIE,
        FullStrictRedactionAgent.QUOTED_SECRET,
        FullStrictRedactionAgent.TOKEN_PREFIX_VALUE,
        FullStrictRedactionAgent.REDACTED_KEY_ONE,
        FullStrictRedactionAgent.REDACTED_KEY_TWO,
        "opaque-after-quote-371",
    }
    assert all(secret not in content for secret in raw_secrets)
    assert rendered_args["headers"] == {
        "Authorization": "***",
        "Cookie": "***",
    }
    assert rendered_args["secret_value"] == "***"
    assert rendered_args["tokenPrefixData"] == "***"
    assert rendered_args["safe"] == {
        "path": "/tmp/visible",
        "items": ["alpha", ["beta"]],
    }
    assert rendered_args["request_url"] == (
        "prefix https://***@api.example.test/v1/items"
        "?access_token=***&visible=***#*** suffix"
    )
    assert "first-collision-value" in rendered_args.values()
    assert "second-collision-value" in rendered_args.values()
    assert len(rendered_args) == len(FullStrictRedactionAgent.ARGS)


@pytest.mark.asyncio
async def test_oversized_full_entry_splits_losslessly_in_separate_mode(
    monkeypatch, tmp_path
):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-overflow-separate",
        config_data={
            "display": {
                "tool_progress": "full",
                "tool_progress_grouping": "separate",
            }
        },
        adapter_cls=SmallLimitProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", OversizedSingleFullEntryAgent.ARGS
    )
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_oversized_full_entry_splits_after_edit_failure(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EditFailureThenOversizedFullAgent,
        session_id="sess-full-overflow-after-edit-failure",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=FailFirstEditProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", EditFailureThenOversizedFullAgent.BIG_ARGS
    )
    start = next(
        idx
        for idx, call in enumerate(adapter.sent)
        if call["content"].startswith("⚙️ full_payload_tool\n")
    )
    assert adapter.failed_edit_ids
    assert adapter.oversized_sends == []
    assert "".join(call["content"] for call in adapter.sent[start:]) == expected


@pytest.mark.asyncio
async def test_full_split_uses_utf16_adapter_length_losslessly(monkeypatch, tmp_path):
    original_args = OversizedSingleFullEntryAgent.ARGS
    unicode_args = {"question": "astral-" + "🧪" * 150 + "-done"}
    monkeypatch.setattr(OversizedSingleFullEntryAgent, "ARGS", unicode_args)
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-overflow-utf16",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=Utf16SmallLimitProgressAdapter,
    )

    expected = _expected_full_message("full_payload_tool", unicode_args)
    assert original_args != unicode_args
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert all(
        adapter.message_len_fn(call["content"]) <= adapter.MAX_MESSAGE_LENGTH
        for call in adapter.sent
    )
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_full_split_finalizes_chunks_before_later_progress(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedThenSmallFullAgent,
        session_id="sess-full-overflow-state-reset",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=TrackingSmallLimitProgressAdapter,
    )

    expected_big = _expected_full_message(
        "full_payload_tool", OversizedThenSmallFullAgent.BIG_ARGS
    )
    expected_small = _expected_full_message(
        "post_split_tool", OversizedThenSmallFullAgent.SMALL_ARGS
    )
    combined = "".join(call["content"] for call in adapter.sent)
    assert combined == expected_big + expected_small
    assert adapter.edited_message_ids == []


@pytest.mark.asyncio
async def test_exact_child_awaiter_propagates_direct_child_cancellation():
    from gateway.run_turn_runner import _await_exact_task_through_cancellation

    attempts = 0

    async def cancel_directly():
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    child = asyncio.create_task(cancel_directly())
    waiter = asyncio.create_task(_await_exact_task_through_cancellation(child))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1.0)

    assert attempts == 1
    assert child.cancelled() is True
    assert waiter.cancelled() is True


@pytest.mark.asyncio
async def test_oversized_verbose_fence_keeps_upstream_single_entry_behavior(
    monkeypatch, tmp_path
):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedVerboseTerminalAgent,
        session_id="sess-verbose-oversized-fence-unsplit",
        config_data={"display": {"tool_progress": "verbose"}},
        adapter_cls=SmallLimitCodeBlockProgressAdapter,
    )

    assert len(adapter.sent) == 1
    assert adapter.oversized_sends == [adapter.sent[0]["content"]]
    from agent.display import get_tool_emoji

    expected = (
        f"{get_tool_emoji('terminal', default='⚙️')} terminal\n```\n"
        f"{OversizedVerboseTerminalAgent.COMMAND}\n```"
    )
    assert adapter.sent[0]["content"] == expected
    assert all(call["content"] == expected for call in adapter.edits)


@pytest.mark.asyncio
async def test_oversized_single_full_entry_splits_losslessly(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-single-entry-overflow",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=SmallLimitProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", OversizedSingleFullEntryAgent.ARGS
    )
    assert result["final_response"] == "done"
    assert isinstance(adapter, SmallLimitProgressAdapter)
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_explicit_full_progress_survives_native_slack_task_card_opt_in(
    monkeypatch, tmp_path
):
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, FullTerminalArgsAgent,
        session_id="sess-full-native-slack-cards",
        config_data={"display": {"tool_progress": "full"}},
        platform=Platform.SLACK,
        adapter_cls=NativeCardsProgressAdapter,
    )
    assert result["final_response"] == "done"
    _, rendered = _first_progress_json(adapter)
    assert rendered == FullTerminalArgsAgent.ARGS


@pytest.mark.parametrize("mode", ["full", "all"])
def test_full_uses_lossless_queue_while_compact_keeps_native_stream_owner(mode):
    from queue import Queue
    from gateway.run_turn_runner import TurnRunner
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.turn_context import TurnContext

    consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    consumer._use_native_streaming = True
    consumer._queue = Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="native-chat"),
        _run_still_current=lambda: True,
        progress_mode=mode,
        tool_progress_enabled=True,
        progress_queue=Queue(),
        stream_consumer_holder=[consumer],
    )
    turn = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _s: None), ctx)
    turn.progress_callback("tool.started", "terminal", "pwd", {"command": "pwd"})
    if mode == "full":
        marker, text = ctx.progress_queue.get_nowait()
        assert marker == "__full__"
        assert json.loads(text.split("\n", 1)[1]) == {"command": "pwd"}
        assert consumer._queue.empty()
    else:
        assert ctx.progress_queue.empty()
        assert "pwd" in consumer._queue.get_nowait()[1]


class FullReceiptProgressAdapter(SmallLimitProgressAdapter):
    """Capture native last-ID receipts and exact operation order."""

    partial = False
    multipart = True
    block_at = None
    instance = None

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        type(self).instance = self
        self.operations = []
        self.receipts = []
        self.deleted = []
        self.cleanup_done = asyncio.Event()
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.first_sent = threading.Event()
        self.send_attempts = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_attempts.append(content)
        if self.block_at == "chunk" and len(self.send_attempts) == 2:
            self.blocked.set()
            await self.release.wait()
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        previous = (self._mint_id(),) if self.multipart else ()
        last = self._mint_id()
        receipt = SendResult(
            success=not self.partial, message_id=last,
            continuation_message_ids=previous,
            error="synthetic partial delivery" if self.partial else None,
            retryable=self.partial,
        )
        self.receipts.append(receipt)
        self.operations.append(("send", previous + (last,), content))
        self.first_sent.set()
        return receipt

    async def edit_message(self, chat_id, message_id, content):
        if self.block_at == "edit":
            self.blocked.set()
            await self.release.wait()
        self.operations.append(("edit", (message_id,), content))
        return await super().edit_message(chat_id, message_id, content)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)
        self.operations.append(("delete", (message_id,), None))
        expected_count = sum(len(r.continuation_message_ids) + 1 for r in self.receipts)
        if len(self.deleted) == expected_count:
            self.cleanup_done.set()
        return True

    async def confirm_final_delivery(self):
        """Exercise the native one-shot callback after simulated final delivery."""
        assert self.deleted == []
        assert len(self._post_delivery_callbacks) == 1
        session_key = next(iter(self._post_delivery_callbacks))
        callback = self.pop_post_delivery_callback(session_key)
        assert callable(callback)
        self.operations.append(("post_delivery", (), None))
        result = callback()
        if asyncio.iscoroutine(result):
            await result
        await asyncio.wait_for(self.cleanup_done.wait(), 3)
        assert self.pop_post_delivery_callback(session_key) is None


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.asyncio
async def test_full_cleanup_owns_continuations_and_partial_receipts_once(monkeypatch, tmp_path, partial):
    monkeypatch.setattr(FullReceiptProgressAdapter, "partial", partial)
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, OversizedSingleFullEntryAgent,
        session_id="full-receipts", adapter_cls=FullReceiptProgressAdapter,
        config_data={"display": {"tool_progress": "full", "cleanup_progress": True}},
    )
    assert result["final_response"] == "done"
    acknowledged = [ident for receipt in adapter.receipts
                    for ident in (*receipt.continuation_message_ids, receipt.message_id)]
    assert acknowledged
    assert len(set(acknowledged)) == len(acknowledged)
    assert isinstance(adapter, FullReceiptProgressAdapter)
    await adapter.confirm_final_delivery()
    assert adapter.deleted == acknowledged
    assert len(adapter.send_attempts) == len(adapter.receipts)
    assert adapter.edits == []
    expected = _expected_full_message("full_payload_tool", OversizedSingleFullEntryAgent.ARGS)
    if partial:
        # The first partial receipt owns visible IDs, but must not replay or continue
        # a failed logical entry whose delivered text range is unknown.
        assert len(adapter.send_attempts) == 1
        assert expected.startswith(adapter.send_attempts[0])
    else:
        assert "".join(adapter.send_attempts) == expected
    assert all(op[0] == "send" for op in adapter.operations[:len(adapter.receipts)])


@pytest.mark.parametrize("block_at", ["edit", "chunk"])
@pytest.mark.asyncio
async def test_full_split_survives_two_parent_cancellations_without_retry(monkeypatch, tmp_path, block_at):
    from gateway.run_turn_runner import TurnRunner

    release_agent = threading.Event()
    ready = asyncio.Event()
    captured = {}
    original_sender = TurnRunner.send_progress_messages

    async def capture_sender(self):
        captured["task"] = asyncio.current_task()
        ready.set()
        return await original_sender(self)

    monkeypatch.setattr(TurnRunner, "send_progress_messages", capture_sender)
    monkeypatch.setattr(FullReceiptProgressAdapter, "block_at", block_at)
    monkeypatch.setattr(FullReceiptProgressAdapter, "multipart", False)
    small = {"question": "pending-before-cancel"}
    big = {"question": "cancel-" + "0123456789" * 70}

    class Agent(FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None):
            self.tool_progress_callback("tool.started", "pending_tool", "pending", small)
            assert FullReceiptProgressAdapter.instance.first_sent.wait(3)
            self.tool_progress_callback("tool.started", "full_payload_tool", "big", big)
            assert release_agent.wait(8)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    run_task = asyncio.create_task(_run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="full-double-cancel",
        adapter_cls=FullReceiptProgressAdapter,
        config_data={"display": {"tool_progress": "full", "cleanup_progress": True}},
    ))
    try:
        await asyncio.wait_for(ready.wait(), 3)
        adapter = FullReceiptProgressAdapter.instance
        await asyncio.wait_for(adapter.blocked.wait(), 4)
        parent = captured["task"]
        assert parent.cancel()
        await asyncio.sleep(0)
        assert not parent.done()
        assert parent.cancel()
        await asyncio.sleep(0)
        assert not parent.done()
        adapter.release.set()
        await asyncio.wait_for(asyncio.shield(parent), 3)
    finally:
        if FullReceiptProgressAdapter.instance:
            FullReceiptProgressAdapter.instance.release.set()
        release_agent.set()
        adapter, result = await asyncio.wait_for(run_task, 4)
    assert result["final_response"] == "done"
    expected_small = _expected_full_message("pending_tool", small)
    expected_big = _expected_full_message("full_payload_tool", big)
    assert adapter.send_attempts[0] == expected_small
    assert "".join(adapter.send_attempts[1:]) == expected_big
    assert [op[0] for op in adapter.operations[:3]] == ["send", "edit", "send"]
    assert [edit["content"] for edit in adapter.edits] == [expected_small]
    ids = [r.message_id for r in adapter.receipts]
    assert isinstance(adapter, FullReceiptProgressAdapter)
    await adapter.confirm_final_delivery()
    assert adapter.deleted == ids
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_log_mode_real_turn_remains_chat_silent(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, FullTerminalArgsAgent, session_id="native-log-control",
        config_data={"display": {"tool_progress": "log"}},
    )
    assert result["final_response"] == "done"
    assert adapter.sent == [] and adapter.edits == []
    log_text = (tmp_path / "logs" / "tool_calls.log").read_text()
    assert "terminal:" in log_text
    assert FullTerminalArgsAgent.ARGS["command"] in log_text


def test_full_mode_nested_header_pairs_preserve_shape_and_inputs():
    from copy import deepcopy
    from gateway.run_turn_runner import _redact_full_progress_args

    original = {"headers": [["Authorization", "opaque-pair-71"],
                            ("X-API-Key", "opaque-pair-72"),
                            ["Accept", "application/json"]],
                "lines": ["Cookie: opaque-cookie-73", "Accept: text/plain"]}
    before = deepcopy(original)
    safe = _redact_full_progress_args(original)
    assert safe["headers"] == [["Authorization", "***"], ("X-API-Key", "***"),
                               ["Accept", "application/json"]]
    assert safe["lines"] == ["Cookie: ***", "Accept: text/plain"]
    assert original == before


@pytest.mark.parametrize("secret_kind", ["vendor", "vault"])
def test_full_mode_untrusted_mask_markers_cannot_bypass_enqueue_redaction(
    monkeypatch, secret_kind
):
    from copy import deepcopy
    from queue import Queue

    import agent.redact as redact
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    secret = "ghp_" + "a" * 36 if secret_kind == "vendor" else "isolated-vault-value-519"
    if secret_kind == "vault":
        redact.register_vault_redaction_value(secret)
    marked = "***" + secret
    scanned = redact.redact_sensitive_text(marked, force=True)
    assert secret not in scanned
    literal_marker = "__HERMES_FULL_MASKED_WORD_0__"
    args = {
        "question": marked,
        "command": (
            f"{literal_marker} export API_TOKEN='opaque phrase' "
            f"echo '{marked}' --label='visible ***'"
        ),
        "nearby": ["visible ***", {literal_marker: "🧪  keep\n\twhitespace"}],
    }
    original = deepcopy(args)
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="enqueue-test"),
        _run_still_current=lambda: True,
        progress_mode="full", tool_progress_enabled=True, progress_queue=Queue(),
    )
    turn = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _s: None), ctx)
    turn.progress_callback("tool.started", "full_payload_tool", "short", args)

    marker, content = ctx.progress_queue.get_nowait()
    assert marker == "__full__"
    assert args == original
    assert secret not in content
    decoded = json.loads(content.split("\n", 1)[1])
    assert decoded == {
        "question": scanned,
        "command": (
            f"{literal_marker} export API_TOKEN='***' "
            f"echo '{scanned}' --label='visible ***'"
        ),
        "nearby": original["nearby"],
    }
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("repeats", [1, 100], ids=["editable", "single-long-arg"])
async def test_full_mode_mattermost_http_preserves_json_fidelity(
    monkeypatch, tmp_path, repeats
):
    from copy import deepcopy
    from unittest.mock import AsyncMock, MagicMock

    from plugins.platforms.mattermost.adapter import MAX_POST_LENGTH, MattermostAdapter

    unit = '🧪  漢字\n\t![diagram](https://example.test/a.png)  '
    escaped = r'\![escaped](https://example.test/b.png) \\![double](https://example.test/c.png) \u0021[literal]'
    initial = {r'\![label](https://example.test/d.png)': [unit, escaped]}
    long_args = {"question": (unit + escaped + '  "quoted"  ') * repeats}
    originals = deepcopy([initial, long_args])
    requests = []
    posts = {}
    first_post = threading.Event()
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True, token="synthetic-test-token",
        extra={"url": "https://mm.example.test", "reply_mode": "thread"},
    ))
    adapter._resolve_root_id = AsyncMock(return_value="thread-root")
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter._session = MagicMock()

    def http_response(method, url, **kwargs):
        payload = deepcopy(kwargs["json"])
        requests.append((method, url, payload))
        if method == "POST":
            assert url.endswith("/api/v4/posts")
            message_id = f"post-{len(posts) + 1}"
        else:
            assert url.endswith("/patch")
            message_id = url.split("/")[-2]
            assert message_id in posts
        posts[message_id] = payload["message"]
        response = AsyncMock()
        response.status = 200
        response.json.return_value = {"id": message_id}
        response.__aenter__.return_value = response
        if method == "POST":
            first_post.set()
        return response

    adapter._session.post.side_effect = lambda url, **kw: http_response("POST", url, **kw)
    adapter._session.put.side_effect = lambda url, **kw: http_response("PUT", url, **kw)

    class Agent(FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            self.tool_progress_callback("tool.started", "initial_payload_tool", "short", initial)
            assert first_post.wait(2), "initial progress was not acknowledged"
            self.tool_progress_callback("tool.started", "full_payload_tool", "short", long_args)
            assert [initial, long_args] == originals
            return {"final_response": "done", "messages": [], "api_calls": 1}

    _, result = await _run_with_agent(
        monkeypatch, tmp_path, Agent, session_id=f"mm-json-{repeats}",
        platform=Platform.MATTERMOST, chat_id="channel-test", thread_id="thread-root",
        adapter_cls=lambda platform: adapter,
        config_data={"display": {"tool_progress": "full", "tool_preview_length": 1}},
    )

    assert result["final_response"] == "done"
    assert [initial, long_args] == originals
    assert {method for method, _, _ in requests} == {"POST", "PUT"}
    assert all(adapter.message_len_fn(body["message"]) <= MAX_POST_LENGTH
               for _, _, body in requests)
    assert all(body["root_id"] == "thread-root" for method, _, body in requests if method == "POST")
    if repeats > 1:
        assert adapter.message_len_fn(json.dumps(long_args, ensure_ascii=False)) > MAX_POST_LENGTH
        assert len(posts) > 2
    remaining = "".join(posts.values())
    for tool, expected in zip(("initial_payload_tool", "full_payload_tool"), originals):
        header, separator, body = remaining.partition("\n")
        assert separator and header.endswith(tool)
        decoded, end = json.JSONDecoder().raw_decode(body)
        assert decoded == expected
        remaining = body[end:].removeprefix("\n")
    assert remaining == ""
    # The adapter's ordinary image-Markdown policy remains native.
    assert adapter.format_message("![diagram](https://example.test/a.png)") == "https://example.test/a.png"
