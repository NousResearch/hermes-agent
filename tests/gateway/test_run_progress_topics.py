"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import queue
import sys
import threading
import time
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import gateway.platforms.base as base_platform
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


_single_progress_sent = threading.Event()
_overflow_initial_sent = threading.Event()
_overflow_send_failed = threading.Event()
_overflow_missing_id_seen = threading.Event()
_initial_send_failed = threading.Event()
_initial_no_id_delivered = threading.Event()
_retryable_overflow_initial_sent = threading.Event()
_retryable_overflow_edit_failed = threading.Event()
_cancelled_overflow_initial_sent = threading.Event()
_cancelled_overflow_send_started = threading.Event()
_interim_content_sent = threading.Event()
_retained_tail_flushed = threading.Event()
_grouped_initial_sent = threading.Event()
_grouped_continuation_edited = threading.Event()
_ordering_draft_sent = threading.Event()
_ordering_initial_progress_failed = threading.Event()
_ordering_progress_retry_started = threading.Event()
_ordering_progress_retry_release = threading.Event()
_ordering_final_sent = threading.Event()


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

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        if len(self.sent) == 1:
            _retryable_overflow_initial_sent.set()
        return result

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
            _retryable_overflow_edit_failed.set()
            return SendResult(
                success=False,
                error="temporary network failure",
                retryable=True,
                error_kind="transient",
            )
        return await super().edit_message(chat_id, message_id, content)


class AmbiguousOverflowEditProgressAdapter(SmallLimitProgressAdapter):
    """Keep every overflow edit ACK-ambiguous to test stable retries."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        if len(self.sent) == 1:
            _retryable_overflow_initial_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        _retryable_overflow_edit_failed.set()
        return SendResult(
            success=False,
            error="relay acknowledgement timed out",
            retryable=True,
            error_kind="transient",
            ambiguous=True,
        )


class TooLongOverflowEditProgressAdapter(SmallLimitProgressAdapter):
    """Reject the first progress edit with Slack's size-limit error."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.too_long_edit_failures = 0
        self.visible = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        if result.success and result.message_id:
            self.sent[-1]["message_id"] = result.message_id
            self.visible[result.message_id] = content
            if len(self.sent) == 1:
                _grouped_initial_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if self.too_long_edit_failures == 0:
            self.too_long_edit_failures += 1
            self.edits.append(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                }
            )
            return SendResult(success=False, error="Slack API error: msg_too_long")
        result = await super().edit_message(chat_id, message_id, content)
        if result.success:
            self.visible[message_id] = content
            if message_id != "progress-1":
                _grouped_continuation_edited.set()
        return result


class TooLongFinalEditProgressAdapter(SmallLimitProgressAdapter):
    """Synchronize one delivered line, then reject its identical final edit."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.too_long_edit_failures = 0

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        _single_progress_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.too_long_edit_failures += 1
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=False, error="Slack API error: msg_too_long")


class TypedTransientTooLongEditProgressAdapter(SmallLimitProgressAdapter):
    """Return a typed transient whose detail happens to say ``too long``."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        _single_progress_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(
            success=False,
            error="relay request took too long awaiting acknowledgement",
            retryable=True,
            error_kind="transient",
        )


class AmbiguousEditProgressAdapter(SmallLimitProgressAdapter):
    """Lose an edit ACK without classifying the result as retryable."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(chat_id, content, reply_to, metadata)
        _single_progress_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(
            success=False,
            error="relay outbound timed out",
            ambiguous=True,
        )


class RetryFreshSendAfterTooLongAdapter(SmallLimitProgressAdapter):
    """Fail the first continuation send after a final ``msg_too_long`` edit."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 2:
            return SendResult(
                success=False,
                error="temporary continuation send failure",
                retryable=True,
                error_kind="transient",
            )
        result = await super().send(chat_id, content, reply_to, metadata)
        if len(self.send_attempts) == 1:
            _single_progress_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=False, error="Slack API error: msg_too_long")


class RaisingFreshSendAfterTooLongAdapter(RetryFreshSendAfterTooLongAdapter):
    """Post the first continuation, then raise before its ACK can be decoded."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 2:
            self.sent.append(
                {
                    "chat_id": chat_id,
                    "content": content,
                    "reply_to": reply_to,
                    "metadata": metadata,
                }
            )
            raise RuntimeError("temporary continuation send exception")
        result = await SmallLimitProgressAdapter.send(
            self, chat_id, content, reply_to, metadata
        )
        if len(self.send_attempts) == 1:
            _single_progress_sent.set()
        return result


class AmbiguousFreshSendAfterTooLongAdapter(RetryFreshSendAfterTooLongAdapter):
    """Lose the ACK for the first non-idempotent continuation send."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 2:
            return SendResult(
                success=False,
                error="relay outbound timed out",
                ambiguous=True,
            )
        result = await SmallLimitProgressAdapter.send(
            self, chat_id, content, reply_to, metadata
        )
        if len(self.send_attempts) == 1:
            _single_progress_sent.set()
        return result


class FailedOverflowGroupAdapter(SmallLimitProgressAdapter):
    """Fail one overflow group and expose the successfully visible messages."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []
        self.visible = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if content == "content between progress groups":
            result = await SmallLimitProgressAdapter.send(
                self, chat_id, content, reply_to, metadata
            )
            if result.success:
                _interim_content_sent.set()
            return result
        if len(self.send_attempts) == 2:
            _overflow_send_failed.set()
            return SendResult(
                success=False,
                error="temporary overflow group failure",
                retryable=True,
                error_kind="transient",
            )
        result = await super().send(chat_id, content, reply_to, metadata)
        if result.success and result.message_id:
            self.visible[result.message_id] = content
            if _interim_content_sent.is_set():
                _retained_tail_flushed.set()
        if len(self.send_attempts) == 1:
            _overflow_initial_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        result = await super().edit_message(chat_id, message_id, content)
        if result.success:
            self.visible[message_id] = content
        return result


class MissingIdOverflowGroupAdapter(SmallLimitProgressAdapter):
    """Deliver one continuation successfully without an editable message ID."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []
        self.visible = {}
        self.visible_without_ids = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 2:
            self.visible_without_ids.append(content)
            _overflow_missing_id_seen.set()
            return SendResult(success=True, message_id=None)
        result = await super().send(chat_id, content, reply_to, metadata)
        if result.success and result.message_id:
            self.visible[result.message_id] = content
        if len(self.send_attempts) == 1:
            _overflow_initial_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        result = await super().edit_message(chat_id, message_id, content)
        if result.success:
            self.visible[message_id] = content
        return result


class CancelledOverflowGroupAdapter(SmallLimitProgressAdapter):
    """Pause an overflow split after one successful no-ID continuation."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []
        self.visible = {}
        self.visible_without_ids = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        attempt = len(self.send_attempts)
        if attempt == 2:
            self.visible_without_ids.append(content)
            return SendResult(success=True, message_id=None)
        if attempt == 3:
            # The relay frame reached the wire before cancellation interrupted
            # its acknowledgement wait.  Treat this as possibly visible.
            self.visible_without_ids.append(content)
            _cancelled_overflow_send_started.set()
            await asyncio.Event().wait()
            raise AssertionError("blocked continuation send should be cancelled")
        result = await super().send(chat_id, content, reply_to, metadata)
        if result.success and result.message_id:
            self.visible[result.message_id] = content
        if attempt == 1:
            _cancelled_overflow_initial_sent.set()
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        result = await super().edit_message(chat_id, message_id, content)
        if result.success:
            self.visible[message_id] = content
        return result


class WedgedCancellationDrainAdapter(SmallLimitProgressAdapter):
    """Hang until a bounded cleanup wait cancels the finalizer send."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 1:
            _initial_send_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        await asyncio.Event().wait()
        raise AssertionError("wedged finalizer send should be cancelled")


class OrderedFinalProgressAdapter(ProgressCaptureAdapter):
    """Capture whether cancellation progress drains before streamed final text."""

    def __init__(self, platform=Platform.SLACK):
        super().__init__(platform=platform)
        self.send_attempts = []
        self.visible_order = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if "ordering progress" in content and not _ordering_initial_progress_failed.is_set():
            _ordering_initial_progress_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        if "ordering progress" in content:
            _ordering_progress_retry_started.set()
            while not _ordering_progress_retry_release.is_set():
                await asyncio.sleep(0)
        if "ordering progress" in content:
            self.visible_order.append("progress")
        elif "draft" in content:
            self.visible_order.append("draft")
            _ordering_draft_sent.set()
        elif content == "done":
            self.visible_order.append("final")
            _ordering_final_sent.set()
        return await super().send(chat_id, content, reply_to, metadata)

    async def edit_message(self, chat_id, message_id, content, **kwargs) -> SendResult:
        result = await super().edit_message(chat_id, message_id, content)
        if result.success and content == "done":
            self.visible_order.append("final")
            _ordering_final_sent.set()
        return result


class RetryNoIdDrainAdapter(SmallLimitProgressAdapter):
    """Fail the initial send and first cancellation-drain retry."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) <= 2:
            if len(self.send_attempts) == 1:
                _initial_send_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        return await super().send(chat_id, content, reply_to, metadata)


class SuccessfulInitialNoIdAdapter(SmallLimitProgressAdapter):
    """Deliver the first progress bubble without returning an editable ID."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.send_attempts = []
        self.visible_without_ids = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.send_attempts.append(content)
        if len(self.send_attempts) == 1:
            self.visible_without_ids.append(content)
            _initial_no_id_delivered.set()
            return SendResult(success=True, message_id=None)
        return await super().send(chat_id, content, reply_to, metadata)


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


class GroupedProgressLinesAgent:
    """Emit short lines so several fit in each continuation bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "grouped-seed", {})
        assert _grouped_initial_sent.wait(timeout=2)
        for index in range(1, 9):
            callback("tool.started", "terminal", f"grouped-item-{index}", {})
        assert _grouped_continuation_edited.wait(timeout=4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 9,
        }


class SingleProgressLineAgent:
    """Leave one already-delivered line for the cancellation finalizer."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "only command", {})
        assert _single_progress_sent.wait(timeout=2)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class PendingProgressLineAgent:
    """Queue one unseen line immediately before cancellation finalization."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "first command", {})
        assert _single_progress_sent.wait(timeout=2)
        callback("tool.started", "terminal", "second command", {})
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class OrderedFinalProgressAgent:
    """Finish a streamed answer while one definite progress failure is pending."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        progress = self.tool_progress_callback
        assert progress is not None
        if self.stream_delta_callback:
            self.stream_delta_callback("draft")
            assert _ordering_draft_sent.wait(timeout=2)
        progress("tool.started", "terminal", "ordering progress", {})
        assert _ordering_initial_progress_failed.wait(timeout=2)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class RetryableOverflowLinesAgent:
    """Drive a transient overflow edit using acknowledgements, not sleeps."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "first-short", {})
        assert _retryable_overflow_initial_sent.wait(timeout=2)
        for index in range(1, 8):
            callback(
                "tool.started",
                "terminal",
                f"overflow-line-{index}-" + ("x" * 45),
                {},
            )
        assert _retryable_overflow_edit_failed.wait(timeout=4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class OverflowSendFailureAgent:
    """Force a grouped overflow send, then return after one group fails."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "base command", {})
        assert _overflow_initial_sent.wait(timeout=2)
        for index in range(1, 7):
            callback(
                "tool.started",
                "terminal",
                f"overflow-item-{index}-" + ("x" * 45),
                {},
            )
        assert _overflow_send_failed.wait(timeout=4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class OverflowSendFailureThenInterimAgent:
    """Land content after a grouped overflow leaves an undelivered tail."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        progress = self.tool_progress_callback
        interim = self.interim_assistant_callback
        assert progress is not None
        assert interim is not None
        progress("tool.started", "terminal", "reset-base", {})
        assert _overflow_initial_sent.wait(timeout=2)
        for index in range(1, 7):
            progress(
                "tool.started",
                "terminal",
                f"reset-item-{index}-" + ("x" * 45),
                {},
            )
        assert _overflow_send_failed.wait(timeout=4)
        interim("content between progress groups")
        assert _interim_content_sent.wait(timeout=2)
        assert _retained_tail_flushed.wait(timeout=2)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class OverflowMissingIdAgent:
    """Force grouped overflow and wait for a successful no-ID continuation."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "base command", {})
        assert _overflow_initial_sent.wait(timeout=2)
        for index in range(1, 7):
            callback(
                "tool.started",
                "terminal",
                f"no-id-item-{index}-" + ("x" * 45),
                {},
            )
        assert _overflow_missing_id_seen.wait(timeout=4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class CancelledOverflowGroupAgent:
    """Finish while a later continuation group is still being sent."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "cancel-base", {})
        assert _cancelled_overflow_initial_sent.wait(timeout=2)
        for index in range(1, 7):
            callback(
                "tool.started",
                "terminal",
                f"cancel-item-{index}-" + ("x" * 45),
                {},
            )
        assert _cancelled_overflow_send_started.wait(timeout=4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class FailedInitialProgressSendAgent:
    """Return once the initial progress send has failed without an ID."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "retry me", {})
        assert _initial_send_failed.wait(timeout=2)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class InitialNoIdThenProgressAgent:
    """Emit another line after a successful uneditable initial delivery."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "first no-id command", {})
        assert _initial_no_id_delivered.wait(timeout=2)
        callback("tool.started", "terminal", "second command", {})
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


@asynccontextmanager
async def _running_progress_worker(adapter, *, grouping="accumulate"):
    """Run the direct progress worker with the shared minimal test context."""
    gateway_run = importlib.import_module("gateway.run")
    progress_queue = queue.Queue()
    ctx = SimpleNamespace(
        progress_queue=progress_queue,
        _native_slack_task_cards=False,
        source=SimpleNamespace(chat_id="C123"),
        progress_grouping=grouping,
        _run_still_current=lambda: True,
        _progress_reply_to=None,
        _progress_metadata=None,
        _cleanup_progress=False,
        _cleanup_msg_ids=[],
        agent_holder=[],
        last_progress_msg=[None],
        repeat_count=[0],
    )
    runner = SimpleNamespace(_adapter_for_source=lambda source: adapter)
    task = asyncio.create_task(
        gateway_run.TurnRunner(runner, ctx).send_progress_messages()
    )
    try:
        yield progress_queue, task
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_oversized_single_progress_line_is_split_before_adapter_delivery():
    adapter = SmallLimitProgressAdapter(platform=Platform.SLACK)
    progress_line = "x" * (adapter.MAX_MESSAGE_LENGTH + 50)

    async with _running_progress_worker(adapter) as (progress_queue, _task):
        progress_queue.put(progress_line)
        for _ in range(100):
            if sum(len(call["content"]) for call in adapter.sent) >= len(progress_line):
                break
            await asyncio.sleep(0.01)

    sent_chunks = [call["content"] for call in adapter.sent]
    assert "".join(sent_chunks) == progress_line
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []


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
    _retryable_overflow_initial_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RetryableOverflowLinesAgent,
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
async def test_ambiguous_overflow_edit_retries_exact_payload(monkeypatch, tmp_path):
    """Later lines must not mutate an unresolved same-ID edit retry."""
    _retryable_overflow_initial_sent.clear()
    _retryable_overflow_edit_failed.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RetryableOverflowLinesAgent,
        session_id="sess-progress-ambiguous-overflow-stable-payload",
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
        adapter_cls=AmbiguousOverflowEditProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert len(adapter.edits) >= 2
    assert {edit["message_id"] for edit in adapter.edits} == {"progress-1"}
    assert {edit["content"] for edit in adapter.edits} == {
        adapter.edits[0]["content"]
    }


@pytest.mark.asyncio
async def test_too_long_overflow_edit_starts_fresh_editable_bubble(
    monkeypatch, tmp_path
):
    """Slack ``msg_too_long`` should roll over, not disable edits for the turn."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        GroupedProgressLinesAgent,
        session_id="sess-progress-too-long-overflow-rollover",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
                "tool_preview_length": 60,
            }
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=TooLongOverflowEditProgressAdapter,
    )
    assert isinstance(adapter, TooLongOverflowEditProgressAdapter)

    assert result["final_response"] == "done"
    assert adapter.too_long_edit_failures == 1
    assert adapter.edits[0]["message_id"] == "progress-1"
    continuation_ids = {call["message_id"] for call in adapter.sent[1:]}
    assert any(call["message_id"] in continuation_ids for call in adapter.edits[1:]), (
        "expected later progress to edit a fresh continuation bubble"
    )
    assert len(adapter.sent) < 9, "progress fell back to one fresh send per tool line"
    assert any("\n" in call["content"] for call in adapter.sent[1:])
    visible_text = "\n".join(adapter.visible.values())
    assert visible_text.count("grouped-seed") == 1
    for index in range(1, 9):
        assert visible_text.count(f"grouped-item-{index}") == 1
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []


@pytest.mark.asyncio
async def test_identical_final_too_long_edit_does_not_duplicate_delivered_line(
    monkeypatch, tmp_path
):
    """A no-op final edit rejection must freeze the bubble without resending it."""
    _single_progress_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        SingleProgressLineAgent,
        session_id="sess-progress-too-long-identical-final",
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
        adapter_cls=TooLongFinalEditProgressAdapter,
    )
    assert isinstance(adapter, TooLongFinalEditProgressAdapter)

    assert result["final_response"] == "done"
    assert adapter.too_long_edit_failures == 1
    sent_contents = [call["content"] for call in adapter.sent]
    assert len(sent_contents) == 1
    assert sent_contents[0].endswith("Running only command")


@pytest.mark.asyncio
async def test_typed_transient_edit_error_text_does_not_trigger_size_rollover(
    monkeypatch, tmp_path
):
    """A populated error_kind is authoritative over ambiguous human text."""
    _single_progress_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PendingProgressLineAgent,
        session_id="sess-progress-typed-transient-not-overflow",
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
        adapter_cls=TypedTransientTooLongEditProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, TypedTransientTooLongEditProgressAdapter)
    assert len(adapter.edits) == 2
    assert adapter.edits[0]["content"] == adapter.edits[1]["content"]
    sent_contents = [call["content"] for call in adapter.sent]
    assert len(sent_contents) == 2
    assert sent_contents[0].endswith("Running first command")
    assert sent_contents[1].endswith("Running second command"), (
        "only the unseen tail may be fresh-sent after bounded transient edit retries"
    )


@pytest.mark.asyncio
async def test_ambiguous_edit_does_not_fall_back_to_fresh_send(monkeypatch, tmp_path):
    """An ACK-lost edit may have landed and must never trigger a duplicate bubble."""
    _single_progress_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PendingProgressLineAgent,
        session_id="sess-progress-ambiguous-edit",
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
        adapter_cls=AmbiguousEditProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert len(adapter.edits) == 2
    assert adapter.edits[0]["content"] == adapter.edits[1]["content"]
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_content_reset_retries_ambiguous_edit_in_place(monkeypatch):
    """A reset settles an ACK-lost edit before opening a fresh bubble."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    first_edit_attempted = asyncio.Event()
    edit_retry_succeeded = asyncio.Event()
    third_sent = asyncio.Event()
    original_send = adapter.send

    async def tracking_send(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        elif content == "third":
            third_sent.set()
        return result

    async def ambiguous_then_successful_edit(chat_id, message_id, content):
        adapter.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        if len(adapter.edits) == 1:
            first_edit_attempted.set()
            return SendResult(
                success=False,
                error="relay acknowledgement timed out",
                retryable=True,
                error_kind="transient",
                ambiguous=True,
            )
        edit_retry_succeeded.set()
        return SendResult(success=True, message_id=message_id)

    adapter.send = tracking_send
    adapter.edit_message = ambiguous_then_successful_edit
    async with _running_progress_worker(adapter) as (progress_queue, _task):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put("second")
        await asyncio.wait_for(first_edit_attempted.wait(), timeout=2)
        progress_queue.put(("__reset__",))
        await asyncio.wait_for(edit_retry_succeeded.wait(), timeout=2)
        progress_queue.put("third")
        await asyncio.wait_for(third_sent.wait(), timeout=2)

    assert [call["content"] for call in adapter.sent] == ["first", "third"]
    assert [edit["content"] for edit in adapter.edits[:2]] == [
        "first\nsecond",
        "first\nsecond",
    ]


@pytest.mark.parametrize(
    ("adapter_cls", "attempt_count", "visible_count"),
    [
        (RetryFreshSendAfterTooLongAdapter, 3, 2),
        (RaisingFreshSendAfterTooLongAdapter, 2, 2),
        (AmbiguousFreshSendAfterTooLongAdapter, 2, 1),
    ],
)
@pytest.mark.asyncio
async def test_continuation_send_outcome_controls_bounded_retry(
    monkeypatch, tmp_path, adapter_cls, attempt_count, visible_count
):
    """Retry definite failures once, but never replay an ambiguous send."""
    _single_progress_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PendingProgressLineAgent,
        session_id=f"sess-progress-{adapter_cls.__name__}",
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
        adapter_cls=adapter_cls,
    )

    assert result["final_response"] == "done"
    send_attempts = getattr(adapter, "send_attempts")
    assert len(send_attempts) == attempt_count
    assert send_attempts[1].endswith("Running second command")
    if attempt_count == 3:
        assert send_attempts[1] == send_attempts[2]
    successful_contents = [call["content"] for call in adapter.sent]
    assert len(successful_contents) == visible_count
    assert successful_contents[0].endswith("Running first command")
    if visible_count == 2:
        assert successful_contents[1].endswith("Running second command")


@pytest.mark.parametrize("outcome", ["ambiguous", "cancelled", "raised"])
@pytest.mark.asyncio
async def test_unknown_initial_send_is_never_replayed(monkeypatch, outcome):
    """An attempted non-idempotent send is never replayed after an unknown outcome."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    send_attempts = []
    first_attempted = asyncio.Event()
    second_attempted = asyncio.Event()

    async def unknown_send(chat_id, content, reply_to=None, metadata=None):
        send_attempts.append(content)
        if len(send_attempts) == 1:
            first_attempted.set()
        else:
            second_attempted.set()
        if outcome == "cancelled":
            await asyncio.Event().wait()
        if outcome == "raised" and len(send_attempts) == 1:
            raise RuntimeError("response decode failed after send became visible")
        return SendResult(
            success=False,
            error="relay outbound timed out",
            ambiguous=True,
        )

    adapter.send = unknown_send
    async with _running_progress_worker(adapter) as (progress_queue, task):
        progress_queue.put("first")
        await asyncio.wait_for(first_attempted.wait(), timeout=2)
        if outcome == "raised":
            progress_queue.put("second")
            await asyncio.wait_for(second_attempted.wait(), timeout=2)
        else:
            await asyncio.sleep(0)
            task.cancel()

    assert send_attempts == (["first", "second"] if outcome == "raised" else ["first"])


@pytest.mark.asyncio
async def test_separate_progress_is_not_replayed_after_content_reset(monkeypatch):
    """A reset must not resend a separate progress line that is already visible."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    second_sent = asyncio.Event()
    original_send = adapter.send

    async def tracking_send(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        elif content == "second":
            second_sent.set()
        return result

    adapter.send = tracking_send
    async with _running_progress_worker(adapter, grouping="separate") as (
        progress_queue,
        _task,
    ):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put(("__reset__",))
        progress_queue.put("second")
        await asyncio.wait_for(second_sent.wait(), timeout=2)

    sent_contents = [call["content"] for call in adapter.sent]
    assert sent_contents.count("first") == 1


@pytest.mark.parametrize("edit_failure", ["not_found", "flood"])
@pytest.mark.parametrize("fallback_outcome", ["success", "ambiguous", "raised"])
@pytest.mark.asyncio
async def test_edit_fallback_is_not_replayed_after_content_reset(
    monkeypatch,
    fallback_outcome,
    edit_failure,
):
    """A visible or unknown fresh fallback must leave the replay ledger."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    fallback_sent = asyncio.Event()
    third_sent = asyncio.Event()
    fourth_sent = asyncio.Event()
    send_attempts = []
    original_send = adapter.send

    async def tracking_send(chat_id, content, reply_to=None, metadata=None):
        send_attempts.append(content)
        if content == "second" and fallback_outcome == "ambiguous":
            fallback_sent.set()
            return SendResult(
                success=False,
                error="relay acknowledgement timed out",
                error_kind="transient",
                retryable=True,
                ambiguous=True,
            )
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        elif content == "second":
            fallback_sent.set()
        elif "third" in content.splitlines():
            third_sent.set()
        elif "fourth" in content.splitlines():
            fourth_sent.set()
        if content == "second" and fallback_outcome == "raised":
            raise RuntimeError("response decode failed after fallback became visible")
        return result

    async def permanently_failing_edit(chat_id, message_id, content):
        adapter.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        if edit_failure == "flood":
            return SendResult(
                success=False,
                error="flood control: retry after 1",
                error_kind="rate_limit",
            )
        return SendResult(
            success=False,
            error="message not found",
            error_kind="not_found",
        )

    adapter.send = tracking_send
    adapter.edit_message = permanently_failing_edit
    async with _running_progress_worker(adapter) as (progress_queue, _task):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put("second")
        await asyncio.wait_for(fallback_sent.wait(), timeout=2)
        progress_queue.put("third")
        await asyncio.wait_for(third_sent.wait(), timeout=2)
        progress_queue.put(("__reset__",))
        progress_queue.put("fourth")
        await asyncio.wait_for(fourth_sent.wait(), timeout=2)

    visible_lines = "\n".join(send_attempts).splitlines()
    assert visible_lines.count("first") == 1
    assert visible_lines.count("second") == 1
    assert visible_lines.count("third") == 1


@pytest.mark.asyncio
async def test_failed_edit_fallback_blocks_later_progress_until_retried(monkeypatch):
    """A definite failed fallback remains ahead of later progress lines."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    fallback_failed = asyncio.Event()
    third_sent = asyncio.Event()
    send_attempts = []
    original_send = adapter.send

    async def fail_fallback_once(chat_id, content, reply_to=None, metadata=None):
        send_attempts.append(content)
        if content == "second" and send_attempts.count("second") == 1:
            fallback_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        elif content == "third":
            third_sent.set()
        return result

    async def permanently_failing_edit(chat_id, message_id, content):
        adapter.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        return SendResult(
            success=False,
            error="message not found",
            error_kind="not_found",
        )

    adapter.send = fail_fallback_once
    adapter.edit_message = permanently_failing_edit
    async with _running_progress_worker(adapter) as (progress_queue, _task):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put("second")
        await asyncio.wait_for(fallback_failed.wait(), timeout=2)
        progress_queue.put("third")
        await asyncio.wait_for(third_sent.wait(), timeout=2)

    assert send_attempts == ["first", "second", "second", "third"]


@pytest.mark.asyncio
async def test_failed_separate_progress_blocks_newer_line_until_retry(monkeypatch):
    """Separate-mode retries preserve FIFO order after a definite failure."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    older_failed = asyncio.Event()
    newer_sent = asyncio.Event()
    send_attempts = []

    async def fail_older_once(chat_id, content, reply_to=None, metadata=None):
        send_attempts.append(content)
        if len(send_attempts) == 1:
            older_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        if content == "newer":
            newer_sent.set()
        return SendResult(success=True, message_id=f"progress-{len(send_attempts)}")

    adapter.send = fail_older_once
    async with _running_progress_worker(adapter, grouping="separate") as (
        progress_queue,
        _task,
    ):
        progress_queue.put("older")
        await asyncio.wait_for(older_failed.wait(), timeout=2)
        progress_queue.put("newer")
        await asyncio.wait_for(newer_sent.wait(), timeout=2)

    assert send_attempts == ["older", "older", "newer"]


@pytest.mark.asyncio
async def test_failed_separate_progress_is_retried_during_cancellation_drain(
    monkeypatch,
):
    """A definite failed separate send remains pending for the bounded drain."""
    gateway_run = importlib.import_module("gateway.run")
    clock = [0.0]

    def monotonic():
        clock[0] += 2.0
        return clock[0]

    monkeypatch.setattr(gateway_run.time, "monotonic", monotonic)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    send_attempts = []
    first_failed = asyncio.Event()

    async def fail_once(chat_id, content, reply_to=None, metadata=None):
        send_attempts.append(content)
        if len(send_attempts) == 1:
            first_failed.set()
            return SendResult(
                success=False,
                error="temporary progress send failure",
                retryable=True,
                error_kind="transient",
            )
        return SendResult(success=True, message_id="progress-retry")

    adapter.send = fail_once
    async with _running_progress_worker(adapter, grouping="separate") as (
        progress_queue,
        task,
    ):
        progress_queue.put("first")
        await asyncio.wait_for(first_failed.wait(), timeout=2)
        await asyncio.sleep(0)
        task.cancel()

    assert send_attempts == ["first", "first"]


@pytest.mark.parametrize("reset_before_cancel", [False, True])
@pytest.mark.asyncio
async def test_cancellation_drain_fresh_sends_unseen_tail_after_definite_edit_failure(
    monkeypatch,
    reset_before_cancel,
):
    """A definite drain edit failure must not discard unseen progress."""
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 2.0)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    original_send = adapter.send

    async def tracking_send(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        return result

    async def missing_edit(chat_id, message_id, content):
        adapter.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        return SendResult(
            success=False,
            error="message not found",
            error_kind="not_found",
        )

    adapter.send = tracking_send
    adapter.edit_message = missing_edit
    async with _running_progress_worker(adapter) as (progress_queue, task):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put("second")
        if reset_before_cancel:
            progress_queue.put(("__reset__",))
        task.cancel()

    assert [call["content"] for call in adapter.sent] == ["first", "second"]
    assert adapter.edits[-1]["content"] == "first\nsecond"


@pytest.mark.parametrize(
    ("first_failure", "second_failure"),
    [
        ("ambiguous", None),
        ("transient", None),
        ("exception", None),
        ("ambiguous", "ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_cancellation_drain_retries_idempotent_edit_before_fallback(
    monkeypatch,
    first_failure,
    second_failure,
):
    """Drain retries edits in place and never fresh-sends unresolved ambiguity."""
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 2.0)
    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    first_sent = asyncio.Event()
    original_send = adapter.send

    async def tracking_send(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to, metadata)
        if content == "first":
            first_sent.set()
        return result

    failures = [first_failure, second_failure]

    async def retrying_edit(chat_id, message_id, content):
        adapter.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        failure = failures[len(adapter.edits) - 1]
        if failure == "ambiguous":
            return SendResult(
                success=False,
                error="relay acknowledgement timed out",
                error_kind="transient",
                retryable=True,
                ambiguous=True,
            )
        if failure == "transient":
            return SendResult(
                success=False,
                error="temporary network failure",
                error_kind="transient",
                retryable=True,
            )
        if failure == "exception":
            raise RuntimeError("edit acknowledgement unavailable")
        return SendResult(success=True, message_id=message_id)

    adapter.send = tracking_send
    adapter.edit_message = retrying_edit
    async with _running_progress_worker(adapter) as (progress_queue, task):
        progress_queue.put("first")
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        progress_queue.put("second")
        task.cancel()

    assert [call["content"] for call in adapter.sent] == ["first"]
    assert [edit["content"] for edit in adapter.edits] == [
        "first\nsecond",
        "first\nsecond",
    ]


@pytest.mark.asyncio
async def test_failed_overflow_group_is_retained_and_retried(monkeypatch, tmp_path):
    """A failed overflow group must stop the split and preserve the exact tail."""
    _overflow_initial_sent.clear()
    _overflow_send_failed.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OverflowSendFailureAgent,
        session_id="sess-progress-overflow-group-retry",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            },
            "tool_preview_length": 60,
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=FailedOverflowGroupAdapter,
    )
    assert isinstance(adapter, FailedOverflowGroupAdapter)

    assert result["final_response"] == "done"
    assert len(adapter.send_attempts) >= 3
    assert adapter.send_attempts[2].splitlines()[0] == adapter.send_attempts[1]
    visible_text = "\n".join(adapter.visible.values())
    ordered_tokens = ["base command", *(f"overflow-item-{index}-" for index in range(1, 7))]
    assert [visible_text.count(token) for token in ordered_tokens] == [1] * len(
        ordered_tokens
    )
    assert [visible_text.index(token) for token in ordered_tokens] == sorted(
        visible_text.index(token) for token in ordered_tokens
    )


@pytest.mark.asyncio
async def test_content_reset_flushes_retained_failed_overflow_tail(monkeypatch, tmp_path):
    """A content-bubble reset must not discard a failed continuation tail."""
    _overflow_initial_sent.clear()
    _overflow_send_failed.clear()
    _interim_content_sent.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OverflowSendFailureThenInterimAgent,
        session_id="sess-progress-reset-retained-tail",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": True,
            },
            "tool_preview_length": 60,
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=FailedOverflowGroupAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, FailedOverflowGroupAdapter)
    visible_text = "\n".join(adapter.visible.values())
    ordered_tokens = ["reset-base", *(f"reset-item-{index}-" for index in range(1, 7))]
    assert [visible_text.count(token) for token in ordered_tokens] == [1] * len(
        ordered_tokens
    )


@pytest.mark.asyncio
async def test_successful_overflow_group_without_id_is_not_retried(monkeypatch, tmp_path):
    """Success without an ID is visible but frozen, not pending for resubmission."""
    _overflow_initial_sent.clear()
    _overflow_missing_id_seen.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OverflowMissingIdAgent,
        session_id="sess-progress-overflow-missing-id",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            },
            "tool_preview_length": 60,
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=MissingIdOverflowGroupAdapter,
    )
    assert isinstance(adapter, MissingIdOverflowGroupAdapter)

    assert result["final_response"] == "done"
    no_id_content = adapter.visible_without_ids[0]
    assert adapter.send_attempts.count(no_id_content) == 1
    visible_text = "\n".join(
        [*adapter.visible.values(), *adapter.visible_without_ids]
    )
    assert visible_text.count("base command") == 1
    for index in range(1, 7):
        assert visible_text.count(f"no-id-item-{index}-") == 1


@pytest.mark.asyncio
async def test_cancelled_overflow_send_does_not_retry_attempted_group(
    monkeypatch, tmp_path
):
    """Cancellation mid-split retains only groups that were never attempted."""
    _cancelled_overflow_initial_sent.clear()
    _cancelled_overflow_send_started.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CancelledOverflowGroupAgent,
        session_id="sess-progress-cancelled-overflow-tail",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            },
            "tool_preview_length": 60,
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=CancelledOverflowGroupAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, CancelledOverflowGroupAdapter)
    visible_with_ids = list(adapter.visible.values())
    visible_text = "\n".join(
        [visible_with_ids[0], *adapter.visible_without_ids, *visible_with_ids[1:]]
    )
    ordered_tokens = ["cancel-base", *(f"cancel-item-{index}-" for index in range(1, 7))]
    assert [visible_text.count(token) for token in ordered_tokens] == [1] * len(
        ordered_tokens
    ), (adapter.send_attempts, adapter.visible, adapter.visible_without_ids)
    assert [visible_text.index(token) for token in ordered_tokens] == sorted(
        visible_text.index(token) for token in ordered_tokens
    )


@pytest.mark.asyncio
async def test_cancellation_progress_drain_finishes_before_streamed_final(
    monkeypatch,
    tmp_path,
):
    """Pending progress must not appear below the completed final response."""
    _ordering_draft_sent.clear()
    _ordering_initial_progress_failed.clear()
    _ordering_progress_retry_started.clear()
    _ordering_progress_retry_release.clear()
    _ordering_final_sent.clear()
    gateway_run = importlib.import_module("gateway.run")
    original_cleanup_wait = gateway_run.GatewayRunner._await_adapter_cleanup_with_timeout
    cleanup_saw_final = []

    async def tracking_cleanup_wait(self, task, timeout):
        cleanup_saw_final.append(_ordering_final_sent.is_set())
        return await original_cleanup_wait(self, task, timeout)

    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_await_adapter_cleanup_with_timeout",
        tracking_cleanup_wait,
    )
    run_task = asyncio.create_task(
        _run_with_agent(
            monkeypatch,
            tmp_path,
            OrderedFinalProgressAgent,
            session_id="sess-progress-before-streamed-final",
            config_data={
                "display": {
                    "tool_progress": "all",
                    "interim_assistant_messages": False,
                },
                "streaming": {
                    "enabled": True,
                    "edit_interval": 0.01,
                    "buffer_threshold": 1,
                },
            },
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="direct",
            thread_id="1700000000.000100",
            adapter_cls=OrderedFinalProgressAdapter,
        )
    )
    retry_started = await asyncio.to_thread(
        _ordering_progress_retry_started.wait,
        2,
    )
    assert retry_started
    try:
        final_overtook_progress = await asyncio.to_thread(
            _ordering_final_sent.wait,
            0.2,
        )
    finally:
        _ordering_progress_retry_release.set()
    adapter, result = await run_task

    assert result["final_response"] == "done"
    assert isinstance(adapter, OrderedFinalProgressAdapter)
    assert not final_overtook_progress
    assert cleanup_saw_final == [False]
    assert adapter.visible_order == ["draft", "progress", "final"]


@pytest.mark.asyncio
async def test_wedged_cancellation_finalizer_does_not_block_turn(monkeypatch, tmp_path):
    """A stuck adapter send during cancellation cleanup must be time-bounded."""
    _initial_send_failed.clear()
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_PROGRESS_FINALIZE_TIMEOUT", 0.05, raising=False)

    started = asyncio.get_running_loop().time()
    adapter, result = await asyncio.wait_for(
        _run_with_agent(
            monkeypatch,
            tmp_path,
            FailedInitialProgressSendAgent,
            session_id="sess-progress-wedged-finalizer",
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
            adapter_cls=WedgedCancellationDrainAdapter,
        ),
        timeout=1.0,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, WedgedCancellationDrainAdapter)
    assert len(adapter.send_attempts) == 2
    assert asyncio.get_running_loop().time() - started < 0.7


@pytest.mark.asyncio
async def test_pending_no_id_drain_gets_bounded_retry(monkeypatch, tmp_path):
    """Cancellation finalization retries retained no-ID progress once more."""
    _initial_send_failed.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FailedInitialProgressSendAgent,
        session_id="sess-progress-no-id-drain-retry",
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
        adapter_cls=RetryNoIdDrainAdapter,
    )
    assert isinstance(adapter, RetryNoIdDrainAdapter)

    assert result["final_response"] == "done"
    assert len(adapter.send_attempts) == 3
    assert adapter.send_attempts[0] == adapter.send_attempts[1]
    assert adapter.send_attempts[1] == adapter.send_attempts[2]
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"].endswith("Running retry me")


@pytest.mark.asyncio
async def test_successful_initial_send_without_id_is_not_replayed(monkeypatch, tmp_path):
    """A visible initial no-ID bubble is frozen before later progress arrives."""
    _initial_no_id_delivered.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        InitialNoIdThenProgressAgent,
        session_id="sess-progress-initial-missing-id",
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
        adapter_cls=SuccessfulInitialNoIdAdapter,
    )
    assert isinstance(adapter, SuccessfulInitialNoIdAdapter)

    assert result["final_response"] == "done"
    assert len(adapter.send_attempts) == 2
    assert adapter.send_attempts[0].endswith("Running first no-id command")
    assert "first no-id command" not in adapter.send_attempts[1]
    assert adapter.send_attempts[1].endswith("Running second command")


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
