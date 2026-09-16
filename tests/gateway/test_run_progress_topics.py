"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import logging
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote
from unittest.mock import MagicMock

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _isolate_goal_continuation_claims(tmp_path, monkeypatch):
    """Keep durable-claim regressions away from the live Hermes profile."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    token = hermes_constants.set_hermes_home_override(None)
    try:
        yield home
    finally:
        hermes_constants.reset_hermes_home_override(token)


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


def _resolved_file_images(images):
    return [
        (Path(unquote(url.removeprefix("file://"))).resolve(), caption)
        for url, caption in images
    ]


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
        # Production (agent/tool_executor.py) fires these through _safe_callback, which skips a
        # None callback; the card lane leaves them unset when cards are disabled for the turn.
        start = self.tool_start_callback or (lambda *a, **k: None)
        complete = self.tool_complete_callback or (lambda *a, **k: None)
        start("call-a", "web_search", {"query": "alpha"})
        time.sleep(0.15)
        start("call-b", "web_search", {"query": "beta"})
        time.sleep(0.15)
        # Complete the second same-name call first. Correlation by tool name
        # would incorrectly mark call-a as failed here.
        complete("call-b", "web_search", {"query": "beta"}, '{"error": "boom"}')
        time.sleep(0.15)
        complete("call-a", "web_search", {"query": "alpha"}, '{"success": true}')
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


class QueuedGoalDispatchAgent:
    """Record whether a queued synthetic goal turn reaches the agent."""

    calls = 0
    messages = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls += 1
        type(self).messages.append(message)
        return {
            "final_response": (
                "first response" if type(self).calls == 1 else "goal continuation processed"
            ),
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
    pending_event=None,
    overflow_events=(),
    before_run=None,
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
    if pending_event is not None:
        adapter._pending_messages[session_key] = pending_event
    elif pending_text is not None:
        adapter._pending_messages[session_key] = MessageEvent(
            text=pending_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id="queued-1",
        )
    if overflow_events:
        runner._session_state(session_key).conversation.queued_events.extend(
            overflow_events
        )
    if before_run is not None:
        before_run(runner, adapter, session_key, source)

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
    # No display config: Slack's tier default (tool_progress off) keeps the TEXT lane quiet while
    # the card lane stays on. An operator-written ``off`` is a different thing (see below).
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-ids",
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


class UnsupportedDestinationTaskCardAdapter(NativeTaskCardAdapter):
    """Relay connector shape for a flat Slack DM: cards need a thread anchor, so the
    connector rejects every card frame with a deterministic unsupported-destination error."""

    async def send_native_task_card_progress(self, *args, **kwargs) -> SendResult:
        await super().send_native_task_card_progress(*args, **kwargs)
        return SendResult(success=False, error="slack task_card requires a thread anchor")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "display_cfg",
    [
        {"platforms": {"slack": {"tool_progress": "off"}}},
        {"tool_progress": "off"},
        {"tool_progress_overrides": {"slack": "off"}},
    ],
    ids=["platform-override", "global", "legacy-overrides"],
)
async def test_slack_operator_tool_progress_off_disables_task_cards(monkeypatch, tmp_path, display_cfg):
    # Task cards ARE tool progress rendered natively. When the operator writes ``off`` (not the
    # tier default), neither the card lane nor its text fallback may publish anything.
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-operator-off",
        config_data={"display": display_cfg},
        platform=Platform.SLACK,
        chat_id="D1",
        chat_type="dm",
        thread_id=None,
        adapter_cls=NativeTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert adapter.native_updates == []
    assert adapter.native_stops == 0
    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "display_cfg",
    [
        {"platforms": {"slack": {"tool_progress": None}}},
        {"tool_progress": None},
        {"tool_progress_overrides": {"slack": None}},
        # null at the platform level over a global "all": the platform inherits "all", cards stay.
        {"tool_progress": "all", "platforms": {"slack": {"tool_progress": None}}},
    ],
    ids=["platform-null", "global-null", "legacy-null", "platform-null-over-global-all"],
)
@pytest.mark.parametrize("env_mode", [None, "all", "new"])
async def test_slack_null_tool_progress_is_inheritance_not_explicit_off(monkeypatch, tmp_path, display_cfg, env_mode):
    # A bare key with ``null`` inherits (the resolver skips None); it is not an operator saying "off".
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)
    if env_mode is not None:
        monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", env_mode)
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-null",
        config_data={"display": display_cfg},
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="thread-1",
        adapter_cls=NativeTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert adapter.native_updates
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_slack_operator_tool_progress_new_keeps_task_cards(monkeypatch, tmp_path):
    # An explicit non-off mode keeps the native card lane engaged (text lane stays swallowed by it).
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-operator-new",
        config_data={"display": {"platforms": {"slack": {"tool_progress": "new"}}}},
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="thread-1",
        adapter_cls=NativeTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert adapter.native_updates
    assert adapter.sent == []
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_slack_unsupported_card_destination_does_not_degrade_to_text_progress(
    monkeypatch, tmp_path
):
    # Flat Slack DM, no operator config (tier default: tool_progress off). The connector cannot
    # render a card without a thread anchor; that is a property of the destination, not a
    # transient native failure, so the lane must NOT fall back to text tool progress the operator
    # never asked for. Transient failures keep the fallback (see the test above).
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-unsupported-dest",
        platform=Platform.SLACK,
        chat_id="D1",
        chat_type="dm",
        thread_id=None,
        adapter_cls=UnsupportedDestinationTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert len(adapter.native_updates) == 1
    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_slack_explicit_all_in_unsupported_card_destination_falls_back_to_text(
    monkeypatch, tmp_path
):
    # Same flat DM, but the operator WROTE ``tool_progress: all``: they asked for text progress,
    # so an un-cardable chat carries it through the editable fallback instead of going silent.
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-native-unsupported-dest-all",
        config_data={"display": {"platforms": {"slack": {"tool_progress": "all"}}}},
        platform=Platform.SLACK,
        chat_id="D1",
        chat_type="dm",
        thread_id=None,
        adapter_cls=UnsupportedDestinationTaskCardAdapter,
        user_id="U1",
        scope_id="T1",
    )

    assert result["final_response"] == "done"
    assert len(adapter.native_updates) == 1
    assert len(adapter.sent) == 1
    assert adapter.edits
    assert "web_search" in adapter.edits[-1]["content"]


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
    assert [call["content"] for call in adapter.sent] == ["first response"]
    assert len(adapter.image_batches) == 1
    image_batch = adapter.image_batches[0]
    assert image_batch["chat_id"] == "discord-thread"
    assert image_batch["metadata"] == {"thread_id": "discord-thread"}
    assert _resolved_file_images(image_batch["images"]) == [
        (media_path.resolve(), "")
    ]


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
    assert len(adapter.image_batches) == 1
    image_batch = adapter.image_batches[0]
    assert image_batch["chat_id"] == "discord-thread"
    assert image_batch["metadata"] == {"thread_id": "discord-thread"}
    assert _resolved_file_images(image_batch["images"]) == [
        (media_path.resolve(), "")
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
    # Sanitized failure copy (raw "provider exploded" stays in the log), then the queued follow-up.
    assert any("couldn't finish this reply" in text and "/retry" in text for text in sent_texts)
    assert not any("provider exploded" in text for text in sent_texts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_notice"),
    [
        (
            "missing",
            "Goal status unavailable: session goal storage is unavailable",
        ),
        (
            "read",
            "Goal status unavailable: persisted goal read failed",
        ),
        (
            "invalid",
            "Goal status unavailable: persisted goal state is invalid",
        ),
    ],
)
async def test_run_agent_queued_goal_read_failure_reports_after_main_response(
    monkeypatch, tmp_path, caplog, failure, expected_notice
):
    """A failed queued-goal recheck must not look like verified inactivity."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    attempts = 0

    class FirstAttemptDB:
        def get_meta(self, _key):
            if failure == "read":
                raise OSError("sensitive queued-read details")
            return '{"goal": "", "status": "active"}'

    class ActiveGoalDB:
        def get_meta(self, _key):
            return active_raw

    def get_goal_db():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None if failure == "missing" else FirstAttemptDB()
        return ActiveGoalDB()

    monkeypatch.setattr(goals, "_get_session_db", get_goal_db)
    judge = MagicMock(side_effect=AssertionError("goal judge must not run"))
    enqueue = MagicMock(side_effect=AssertionError("goal enqueue must not run"))
    monkeypatch.setattr(goals, "judge_goal", judge)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run.GatewayRunner, "_enqueue_fifo", enqueue)

    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    pending = goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=pending,
        message_type=MessageType.TEXT,
        user_id="synthetic-user",
        user_name="synthetic-name",
        source=source,
        raw_message={"opaque": "identity"},
        message_id="goal-continuation-1",
        platform_update_id=41,
        reply_to_message_id="prior-message",
        reply_to_author_id="prior-author",
        channel_prompt="keep this channel prompt",
        internal=False,
        metadata={"identity": "keep-me"},
        allow_gateway_control=False,
        goal_continuation=True,
    )
    holder = {}

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(
            runner=runner,
            adapter=adapter,
            key=key,
            source=current_source,
        )

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        adapter, result = await _run_with_agent(
            monkeypatch,
            tmp_path,
            QueuedGoalDispatchAgent,
            session_id=f"queued-goal-{failure}",
            pending_event=continuation,
            before_run=before_run,
        )

    assert result["final_response"] == "goal continuation processed"
    assert QueuedGoalDispatchAgent.calls == 2
    assert QueuedGoalDispatchAgent.messages == ["hello", pending]
    judge.assert_not_called()
    enqueue.assert_not_called()
    assert [call["content"] for call in adapter.sent] == [
        "first response",
        expected_notice,
    ]
    assert expected_notice in caplog.text
    assert "sensitive queued-read details" not in caplog.text
    assert all("sensitive queued-read details" not in call["content"] for call in adapter.sent)
    assert adapter._pending_messages == {}
    assert holder["runner"]._queued_events.get(holder["key"], []) == []
    assert attempts == 2
    assert continuation.source is source
    assert continuation.raw_message == {"opaque": "identity"}
    assert continuation.message_id == "goal-continuation-1"
    assert continuation.platform_update_id == 41
    assert continuation.metadata == {"identity": "keep-me"}
    assert continuation.allow_gateway_control is False


@pytest.mark.asyncio
async def test_run_agent_goal_read_failure_restores_ahead_of_fifo_overflow(
    monkeypatch, tmp_path
):
    """The consumed continuation remains the exact FIFO head after failure."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class RecoveringGoalDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise OSError("private fifo read detail")
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: RecoveringGoalDB())
    judge = MagicMock(side_effect=AssertionError("goal judge must not run"))
    monkeypatch.setattr(goals, "judge_goal", judge)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        message_id="continuation-head",
        metadata={"identity": "continuation"},
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    followups = [
        MessageEvent(text="user follow-up 1", source=source, message_id="user-1"),
        MessageEvent(text="user follow-up 2", source=source, message_id="user-2"),
    ]
    holder = {}
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(runner=runner, key=key)

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="queued-goal-fifo-failure",
        pending_event=continuation,
        overflow_events=followups,
        before_run=before_run,
    )

    assert result["final_response"] == "goal continuation processed"
    assert reads == 2
    assert QueuedGoalDispatchAgent.messages == [
        "hello",
        continuation.text,
        followups[0].text,
        followups[1].text,
    ]
    judge.assert_not_called()
    assert adapter._pending_messages == {}
    assert holder["runner"]._queued_events.get(holder["key"], []) == []


@pytest.mark.asyncio
async def test_run_agent_goal_read_failure_preserves_concurrent_slot_arrival(
    monkeypatch, tmp_path
):
    """A slot arrival during the failed read is shifted behind the continuation."""
    from hermes_cli import goals

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        message_id="continuation-concurrent",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    concurrent = MessageEvent(
        text="arrived during read",
        source=source,
        message_id="concurrent-user",
        metadata={"identity": "concurrent"},
    )
    holder = {}

    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class ConcurrentReadFailureDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if reads == 1:
                holder["adapter"]._pending_messages[holder["key"]] = concurrent
                raise OSError("private concurrent read detail")
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: ConcurrentReadFailureDB())
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(runner=runner, adapter=adapter, key=key)

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="queued-goal-concurrent-failure",
        pending_event=continuation,
        before_run=before_run,
    )

    assert result["final_response"] == "goal continuation processed"
    assert reads == 2
    assert QueuedGoalDispatchAgent.messages == [
        "hello",
        continuation.text,
        concurrent.text,
    ]
    assert adapter._pending_messages == {}
    assert holder["runner"]._queued_events.get(holder["key"], []) == []


@pytest.mark.asyncio
async def test_run_agent_bounded_initial_retry_preserves_claimed_head(
    monkeypatch, tmp_path
):
    """The current owner retries once without executing a later wake first."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class RecoveringGoalDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise OSError("private transient read detail")
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: RecoveringGoalDB())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        message_id="continuation-retry",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    holder = {}
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(runner=runner, key=key)

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="queued-goal-retry-owner",
        pending_event=continuation,
        before_run=before_run,
    )

    assert result["final_response"] == "goal continuation processed"
    assert reads == 2
    assert QueuedGoalDispatchAgent.messages == ["hello", continuation.text]
    assert [call["content"] for call in adapter.sent] == [
        "first response",
        "Goal status unavailable: persisted goal read failed",
    ]
    assert adapter._pending_messages == {}
    assert holder["runner"]._queued_events.get(holder["key"], []) == []
    assert holder["key"] not in getattr(
        holder["runner"], "_goal_continuation_retries", {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted_state", ["absent", "paused"])
async def test_run_agent_queued_goal_verified_inactivity_remains_silent(
    monkeypatch, tmp_path, persisted_state
):
    """Authoritative absence or inactivity still discards a stale continuation."""
    from hermes_cli import goals

    raw = (
        None
        if persisted_state == "absent"
        else goals.GoalState(goal="finish the task", status="paused").to_json()
    )

    class InactiveGoalDB:
        def get_meta(self, _key):
            return raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: InactiveGoalDB())
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        message_id="stale-goal-continuation",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    followups = [
        MessageEvent(text="real user 1", source=source, message_id="real-1"),
        MessageEvent(text="real user 2", source=source, message_id="real-2"),
    ]
    holder = {}

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id=f"queued-goal-{persisted_state}",
        pending_event=continuation,
        overflow_events=followups,
        before_run=lambda runner, adapter, key, current_source: holder.update(
            runner=runner,
            key=key,
        ),
    )

    assert result["final_response"] == "first response"
    assert QueuedGoalDispatchAgent.calls == 1
    assert QueuedGoalDispatchAgent.messages == ["hello"]
    assert [call["content"] for call in adapter.sent] == ["first response"]
    assert adapter._pending_messages[holder["key"]] is followups[0]
    assert holder["runner"]._queued_events[holder["key"]] == [followups[1]]
    assert all(
        event is not continuation
        for event in [
            *adapter._pending_messages.values(),
            *holder["runner"]._queued_events[holder["key"]],
        ]
    )


@pytest.mark.asyncio
async def test_inactive_top_level_continuation_stages_real_successor():
    """Dropping a stale adapter-owned head cannot strand runner overflow."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
    from gateway.platforms.base import MessageEvent

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="stale goal"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    successor = MessageEvent(text="real successor", source=source)
    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    key = "inactive-top-level-key"
    adapter._pending_messages[key] = continuation
    runner._queued_events[key] = [continuation, successor]
    runner._goal_still_active_for_session = lambda _session_id: False

    admitted = await runner._wait_for_goal_continuation_admission(
        event=continuation,
        source=source,
        session_id="inactive-top-level",
        session_key=key,
        adapter=adapter,
    )

    assert admitted is False
    assert adapter._pending_messages[key] is successor
    assert runner._queued_events.get(key, []) == []


def test_inactive_continuation_removes_alias_behind_distinct_slot():
    """A stale same-object alias cannot survive behind an older real slot head."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
    from gateway.platforms.base import MessageEvent

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="stale goal"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    slot_head = MessageEvent(text="older slot head", source=source)
    successor = MessageEvent(text="real successor", source=source)
    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    key = "inactive-distinct-slot-key"
    adapter._pending_messages[key] = slot_head
    runner._queued_events[key] = [continuation, successor]

    staged = runner._stage_next_queued_event(
        key, adapter, exclude_event=continuation
    )

    assert staged is False
    assert adapter._pending_messages[key] is slot_head
    assert runner._queued_events[key] == [successor]


@pytest.mark.asyncio
async def test_goal_pause_clear_race_drops_claimed_synthetic_only(
    monkeypatch, tmp_path
):
    """Pause/clear wakes the owner and keeps every real FIFO successor."""
    from hermes_cli import goals

    reads = 0

    class ReadFailureDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            raise OSError("private pause race detail")

    monkeypatch.setattr(goals, "_get_session_db", lambda: ReadFailureDB())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        message_id="continuation-before-pause",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    followups = [
        MessageEvent(text="real user 1", source=source, message_id="real-1"),
        MessageEvent(text="real user 2", source=source, message_id="real-2"),
    ]
    holder = {}
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(runner=runner, adapter=adapter, key=key)

    run_task = asyncio.create_task(
        _run_with_agent(
            monkeypatch,
            tmp_path,
            QueuedGoalDispatchAgent,
            session_id="queued-goal-pause-race",
            pending_event=continuation,
            overflow_events=followups,
            before_run=before_run,
        )
    )
    for _ in range(200):
        retries = getattr(holder.get("runner"), "_goal_continuation_retries", {})
        if holder.get("key") in retries:
            break
        await asyncio.sleep(0.005)

    retry = holder["runner"]._goal_continuation_retries[holder["key"]]
    assert retry.event is continuation
    assert holder["adapter"]._pending_messages[holder["key"]] is followups[0]
    assert holder["runner"]._queued_events[holder["key"]] == [followups[1]]
    for _ in range(200):
        if reads >= 2:
            break
        await asyncio.sleep(0.005)
    assert reads == 2
    await asyncio.sleep(0.05)
    assert reads == 2

    removed = holder["runner"]._clear_goal_pending_continuations(
        holder["key"], holder["adapter"]
    )
    _adapter, result = await asyncio.wait_for(run_task, timeout=2)

    assert result["final_response"] == "first response"
    assert removed == 1
    assert holder["adapter"]._pending_messages[holder["key"]] is followups[0]
    assert holder["runner"]._queued_events[holder["key"]] == [followups[1]]
    assert holder["key"] not in holder["runner"]._goal_continuation_retries


@pytest.mark.asyncio
async def test_run_agent_queued_goal_active_state_runs_continuation(monkeypatch, tmp_path):
    """An authoritative active row still permits the queued continuation."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()

    class ActiveGoalDB:
        def get_meta(self, _key):
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: ActiveGoalDB())
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    pending = goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=pending,
        message_type=MessageType.TEXT,
        source=source,
        message_id="active-goal-continuation",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="queued-goal-active",
        pending_event=continuation,
    )

    assert result["final_response"] == "goal continuation processed"
    assert QueuedGoalDispatchAgent.calls == 2
    assert QueuedGoalDispatchAgent.messages == ["hello", pending]
    assert [call["content"] for call in adapter.sent] == ["first response"]


def test_goal_continuation_classifier_requires_typed_authorized_provenance():
    """User-controlled continuation text is not trusted as synthetic provenance."""
    from gateway.run import GatewayRunner
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    text = CONTINUATION_PROMPT_TEMPLATE.format(goal="user-controlled prefix")
    real_user = MessageEvent(text=text, source=source, internal=False)
    plugin_collision = MessageEvent(
        text=text,
        source=source,
        internal=True,
        allow_gateway_control=False,
    )
    synthetic = MessageEvent(
        text=text,
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    internal_marked = MessageEvent(
        text=text,
        source=source,
        internal=True,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    gate_failed = MessageEvent(
        text=(
            "[Continuing toward your standing goal — a quality gate failed]\n"
            "Goal: user-controlled prefix\n\nGate details"
        ),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    control_enabled = MessageEvent(
        text=text,
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=True,
    )
    malformed = MessageEvent(
        text="not a canonical goal continuation",
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )

    assert GatewayRunner._is_goal_continuation_event(text) is False
    assert GatewayRunner._is_goal_continuation_event(real_user) is False
    assert GatewayRunner._is_goal_continuation_event(plugin_collision) is False
    assert GatewayRunner._is_goal_continuation_event(internal_marked) is False
    assert GatewayRunner._is_goal_continuation_event(synthetic) is True
    assert GatewayRunner._is_goal_continuation_event(gate_failed) is True
    assert GatewayRunner._is_goal_continuation_event(control_enabled) is False
    assert GatewayRunner._is_goal_continuation_event(malformed) is False


@pytest.mark.asyncio
async def test_real_prefix_collision_pending_runs_as_user_input_when_goal_absent(
    monkeypatch, tmp_path
):
    """A genuine user prefix collision is never silently discarded as stale."""
    from hermes_cli import goals

    class EmptyGoalDB:
        def get_meta(self, _key):
            return None

    monkeypatch.setattr(goals, "_get_session_db", lambda: EmptyGoalDB())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    collision = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="typed by the user"),
        message_type=MessageType.TEXT,
        source=source,
        message_id="real-prefix-collision",
        internal=False,
    )
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="real-prefix-collision",
        pending_event=collision,
    )

    assert result["final_response"] == "goal continuation processed"
    assert QueuedGoalDispatchAgent.messages == ["hello", collision.text]


def test_pause_clear_removes_only_proven_synthetic_prefix_collisions():
    """Text-identical real user events survive slot and overflow cleanup."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    key = "agent:main:telegram:group:-1001:17585"
    text = CONTINUATION_PROMPT_TEMPLATE.format(goal="same bytes, different provenance")
    real_slot = MessageEvent(text=text, source=source, message_id="real-slot")
    synthetic = MessageEvent(
        text=text,
        source=source,
        message_id="synthetic",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    real_overflow = MessageEvent(text=text, source=source, message_id="real-overflow")
    adapter._pending_messages[key] = real_slot
    runner._session_state(key).conversation.queued_events[:] = [
        synthetic,
        real_overflow,
    ]

    removed = runner._clear_goal_pending_continuations(key, adapter)

    assert removed == 1
    assert adapter._pending_messages[key] is real_slot
    assert runner._queued_events[key] == [real_overflow]


def test_busy_arrival_appends_behind_restored_continuation_when_slot_is_empty():
    """An empty adapter slot cannot bypass an older runner-owned FIFO head."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    key = "agent:main:telegram:group:-1001:17585"
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    predecessor = MessageEvent(text="older user follow-up", source=source)
    arrival = MessageEvent(text="arrived during notice", source=source)
    runner._session_state(key).conversation.queued_events[:] = [
        continuation,
        predecessor,
    ]

    runner._queue_or_replace_pending_event(key, arrival)

    assert adapter._pending_messages == {}
    assert runner._queued_events[key] == [continuation, predecessor, arrival]


@pytest.mark.asyncio
async def test_retry_owned_media_arrival_never_merges_with_slot_head():
    """Claimed continuation FIFO disables media coalescing for later arrivals."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
    from gateway.platforms.base import MessageEvent, MessageType

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    slot_head = MessageEvent(
        text="older photo",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["older.jpg"],
    )
    arrival = MessageEvent(
        text="newer photo",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["newer.jpg"],
    )
    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    key = "retry-media-key"
    adapter._pending_messages[key] = slot_head
    retry = runner._claim_goal_continuation_retry(
        key, adapter, continuation, session_id="retry-media-session"
    )
    assert retry is not None
    runner._is_user_authorized = lambda _source: True
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "queue"

    handled = await runner._handle_active_session_busy_message(arrival, key)

    assert handled is True
    assert retry.wake.is_set()
    assert adapter._pending_messages[key] is slot_head
    assert slot_head.media_urls == ["older.jpg"]
    assert runner._queued_events[key] == [arrival]


@pytest.mark.asyncio
async def test_busy_queue_mode_keeps_existing_overflow_ahead_of_new_arrival():
    """An active turn never lets the adapter slot overtake an older queue tail."""
    from gateway.platforms.base import MessageEvent

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    runner._busy_text_mode = "queue"
    runner._busy_input_mode = "queue"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="backlog-user",
        chat_id="backlog-chat",
    )
    key = "busy-backlog-order"
    predecessor = MessageEvent(text="older predecessor", source=source)
    arrival = MessageEvent(text="new arrival", source=source)
    runner._queued_events[key] = [predecessor]

    handled = await runner._handle_active_session_busy_message(arrival, key)

    assert handled is True
    assert adapter._pending_messages == {}
    assert runner._queued_events[key] == [predecessor, arrival]


def test_queue_cap_preserves_restored_continuation_and_existing_fifo(caplog):
    """A full retry queue drops only the later arrival, never its proven head."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    runner._BUSY_QUEUE_MAX_PENDING = 2
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    key = "agent:main:telegram:dm:-1001"
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    predecessor = MessageEvent(text="older user follow-up", source=source)
    arrival = MessageEvent(text="arrived after the queue filled", source=source)
    runner._session_state(key).conversation.queued_events[:] = [
        continuation,
        predecessor,
    ]

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._queue_or_replace_pending_event(key, arrival)

    assert adapter._pending_messages == {}
    assert runner._queued_events[key] == [continuation, predecessor]
    assert "pending queue at cap (2)" in caplog.text


def test_restore_to_adapter_head_deduplicates_same_object_aliases():
    """Cancellation restoration publishes one head and keeps distinct successors."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    key = "agent:main:telegram:dm:-1001"
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    trailing = MessageEvent(text="trailing", source=source)
    adapter._pending_messages[key] = continuation
    runner._session_state(key).conversation.queued_events[:] = [
        continuation,
        trailing,
        continuation,
    ]

    runner._restore_dequeued_event_front(key, adapter, continuation)

    assert adapter._pending_messages[key] is continuation
    assert runner._queued_events[key] == [trailing]


def test_promote_with_unsupported_slot_shape_preserves_overflow_head():
    """Promotion cannot pop an overflow event it has nowhere to publish."""
    from gateway.platforms.base import MessageEvent

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    claimed = MessageEvent(text="claimed", source=source)
    head = MessageEvent(text="overflow head", source=source)
    tail = MessageEvent(text="overflow tail", source=source)
    adapter = SimpleNamespace(_pending_messages=None)
    runner = _make_runner(ProgressCaptureAdapter())
    key = "unsupported-promote-key"
    runner._queued_events[key] = [head, tail]

    result = runner._promote_queued_event(key, adapter, claimed)

    assert result is claimed
    assert runner._queued_events[key] == [head, tail]


def test_restore_without_adapter_slot_deduplicates_alias_without_dropping():
    """Unsupported adapter shapes keep one exact continuation at FIFO head."""
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    adapter = SimpleNamespace()
    runner = _make_runner(ProgressCaptureAdapter())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    key = "agent:main:telegram:dm:-1001"
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    trailing = MessageEvent(text="trailing", source=source)
    runner._session_state(key).conversation.queued_events[:] = [
        continuation,
        trailing,
        continuation,
    ]

    runner._restore_dequeued_event_front(key, adapter, continuation)

    assert runner._queued_events[key] == [continuation, trailing]
    assert runner._queued_events[key][0] is continuation


@pytest.mark.asyncio
async def test_actual_arrival_during_notice_preserves_total_fifo_and_wakes_retry(
    monkeypatch, tmp_path
):
    """A production busy-path arrival cannot overtake during awaited notice I/O."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class RecoveringGoalDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise OSError("private asynchronous read detail")
            return active_raw

    db = RecoveringGoalDB()
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = QueuedGoalDispatchAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "queue"
    runner._BUSY_QUEUE_MAX_PENDING = 4
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._goal_continuation_retry_initial_delay_seconds = 0.01
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
        user_id="authorized-user",
    )
    key = "agent:main:telegram:group:-1001:17585"
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        message_type=MessageType.TEXT,
        source=source,
        message_id="continuation-head",
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    older = [
        MessageEvent(text="older follow-up 1", source=source, message_id="older-1"),
        MessageEvent(text="older follow-up 2", source=source, message_id="older-2"),
    ]
    concurrent = MessageEvent(
        text="arrived during awaited notice",
        source=source,
        message_id="actual-concurrent",
    )
    rejected_at_cap = MessageEvent(
        text="arrival beyond queue cap",
        message_type=MessageType.TEXT,
        source=source,
        message_id="actual-cap-rejected",
    )
    outer = MessageEvent(text="hello", source=source, message_id="outer")
    adapter._pending_messages[key] = continuation
    runner._session_state(key).conversation.queued_events[:] = older
    notice_started = asyncio.Event()
    release_notice = asyncio.Event()

    async def blocked_notice(_source, _notice):
        notice_started.set()
        await release_notice.wait()

    runner._send_goal_status_notice = blocked_notice

    handled = []

    async def handler(event):
        handled.append(event)
        result = await runner._run_agent(
            message=event.text,
            context_prompt="",
            history=[],
            source=event.source,
            session_id="async-arrival-session",
            session_key=key,
        )
        return result["final_response"]

    adapter.set_message_handler(handler)
    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []

    await adapter.handle_message(outer)
    await asyncio.wait_for(notice_started.wait(), timeout=2)
    owner_task = adapter._session_tasks[key]
    await adapter.handle_message(concurrent)
    await adapter.handle_message(rejected_at_cap)
    retry_during_notice = getattr(runner, "_goal_continuation_retries", {}).get(key)
    slot_during_notice = adapter._pending_messages.get(key)
    overflow_during_notice = list(runner._queued_events.get(key, []))
    release_notice.set()

    for _ in range(400):
        if (
            len(QueuedGoalDispatchAgent.messages) == 5
            and key not in adapter._active_sessions
        ):
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()
    assert retry_during_notice is not None
    assert retry_during_notice.event is continuation
    assert slot_during_notice is older[0]
    assert len(overflow_during_notice) == 2
    assert overflow_during_notice[0] is older[1]
    assert overflow_during_notice[1] is concurrent
    assert handled[0] is outer
    assert handled[-1] is concurrent
    assert continuation not in handled
    assert owner_task.done()
    assert reads >= 2
    assert QueuedGoalDispatchAgent.messages == [
        "hello",
        continuation.text,
        older[0].text,
        older[1].text,
        concurrent.text,
    ]
    assert rejected_at_cap.text not in QueuedGoalDispatchAgent.messages
    assert adapter._pending_messages == {}
    assert runner._queued_events.get(key, []) == []


@pytest.mark.asyncio
async def test_notice_send_exception_keeps_claimed_event_under_same_owner(
    monkeypatch, tmp_path
):
    """Network notice failure cannot undo the current owner's bounded retry."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class RecoveringGoalDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise OSError("private read failure")
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: RecoveringGoalDB())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    holder = {}

    async def failing_notice(_source, _notice):
        raise OSError("notice transport failed")

    def before_run(runner, adapter, key, current_source):
        runner._send_goal_status_notice = failing_notice
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        holder.update(runner=runner, key=key)

    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedGoalDispatchAgent,
        session_id="notice-exception-session",
        pending_event=continuation,
        before_run=before_run,
    )

    assert result["final_response"] == "goal continuation processed"
    assert reads == 2
    assert QueuedGoalDispatchAgent.messages == ["hello", continuation.text]
    assert adapter._pending_messages == {}
    assert holder["runner"]._queued_events.get(holder["key"], []) == []
    assert holder["key"] not in getattr(
        holder["runner"], "_goal_continuation_retries", {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_before_cancel", [False, True])
async def test_notice_cancellation_restores_claimed_event_to_adapter_head(
    monkeypatch, tmp_path, drop_before_cancel
):
    """Cancellation leaves one exact adapter-owned continuation for finalization."""
    from hermes_cli import goals

    class ReadFailureDB:
        def get_meta(self, _key):
            raise OSError("private read failure")

    monkeypatch.setattr(goals, "_get_session_db", lambda: ReadFailureDB())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
    )
    holder = {}
    notice_started = asyncio.Event()
    block_notice = asyncio.Event()

    async def cancelled_notice(_source, _notice):
        notice_started.set()
        await block_notice.wait()

    def before_run(runner, adapter, key, current_source):
        runner._send_goal_status_notice = cancelled_notice
        holder.update(runner=runner, adapter=adapter, key=key)

    QueuedGoalDispatchAgent.calls = 0
    run_task = asyncio.create_task(
        _run_with_agent(
            monkeypatch,
            tmp_path,
            QueuedGoalDispatchAgent,
            session_id="notice-cancellation-session",
            pending_event=continuation,
            before_run=before_run,
        )
    )
    await asyncio.wait_for(notice_started.wait(), timeout=2)
    if drop_before_cancel:
        removed = holder["runner"]._clear_goal_pending_continuations(
            holder["key"], holder["adapter"]
        )
        assert removed == 1
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    if drop_before_cancel:
        assert holder["adapter"]._pending_messages == {}
    else:
        assert holder["adapter"]._pending_messages[holder["key"]] is continuation
    assert holder["runner"]._queued_events.get(holder["key"], []) == []
    assert holder["key"] not in getattr(
        holder["runner"], "_goal_continuation_retries", {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first_read", [False, True])
async def test_cancellation_after_goal_admission_restores_before_successors(
    monkeypatch, tmp_path, fail_first_read
):
    """Admission remains owned until recursive dispatch accepts the continuation."""
    from hermes_cli import goals
    from gateway.platforms.base import (
        MessageEvent,
        MessageType,
        Platform,
        SessionSource,
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="post-admission-user",
        chat_id="post-admission-chat",
    )
    continuation = MessageEvent(
        text=goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="finish the task"),
        message_type=MessageType.TEXT,
        source=source,
        internal=False,
        goal_continuation=True,
        allow_gateway_control=False,
        message_id="post-admission-continuation",
    )
    successor = MessageEvent(
        text="older queued successor",
        message_type=MessageType.TEXT,
        source=source,
        message_id="post-admission-successor",
    )
    active_raw = goals.GoalState(goal="finish the task").to_json()
    reads = 0

    class RecoveringGoalDB:
        def get_meta(self, _key):
            nonlocal reads
            reads += 1
            if fail_first_read and reads == 1:
                raise OSError("private read detail")
            return active_raw

    monkeypatch.setattr(goals, "_get_session_db", lambda: RecoveringGoalDB())
    prepare_started = asyncio.Event()
    holder = {}

    def before_run(runner, adapter, key, current_source):
        runner._goal_continuation_retry_initial_delay_seconds = 0.01
        original_prepare = runner._prepare_profile_scoped_inbound_message_text

        async def blocking_prepare(**kwargs):
            if kwargs.get("event") is continuation:
                prepare_started.set()
                await asyncio.Event().wait()
            return await original_prepare(**kwargs)

        runner._prepare_profile_scoped_inbound_message_text = blocking_prepare
        holder.update(runner=runner, adapter=adapter, key=key)

    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    run_task = asyncio.create_task(
        _run_with_agent(
            monkeypatch,
            tmp_path,
            QueuedGoalDispatchAgent,
            session_id="post-admission-cancellation",
            pending_event=continuation,
            overflow_events=[successor],
            before_run=before_run,
        )
    )
    await asyncio.wait_for(prepare_started.wait(), timeout=2)
    retry = holder["runner"]._goal_continuation_retries[holder["key"]]
    assert retry.event is continuation
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert holder["adapter"]._pending_messages[holder["key"]] is continuation
    overflow = holder["runner"]._queued_events[holder["key"]]
    assert overflow == [successor]
    assert overflow[0] is successor
    assert holder["key"] not in holder["runner"]._goal_continuation_retries


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
