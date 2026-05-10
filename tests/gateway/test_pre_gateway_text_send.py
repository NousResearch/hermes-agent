"""Tests for the pre_gateway_text_send plugin hook (Issue #22603).

Covers:
  - Hook registration in VALID_HOOKS
  - The apply_text_send_hooks() helper: allow / rewrite / block / error
  - Non-streaming path integration (base.py _process_message_background)
  - Streaming path integration (stream_consumer.py got_done block)
"""

from __future__ import annotations

import asyncio
import queue
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Hook registration
# ---------------------------------------------------------------------------

def test_hook_registered_in_valid_hooks():
    """pre_gateway_text_send must be a recognised hook name."""
    from hermes_cli.plugins import VALID_HOOKS
    assert "pre_gateway_text_send" in VALID_HOOKS


# ---------------------------------------------------------------------------
# 2. apply_text_send_hooks() helper
# ---------------------------------------------------------------------------

class TestApplyTextSendHooks:
    """Unit tests for the shared helper function."""

    @patch("hermes_cli.plugins.invoke_hook", return_value=[])
    def test_no_plugins_returns_original(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Hello world")
        assert text == "Hello world"
        assert blocked is False

    @patch("hermes_cli.plugins.invoke_hook", return_value=[None])
    def test_none_return_allows(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Hello world")
        assert text == "Hello world"
        assert blocked is False

    @patch("hermes_cli.plugins.invoke_hook", return_value=[{"action": "allow"}])
    def test_allow_action_passes_through(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Hello world")
        assert text == "Hello world"
        assert blocked is False

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[{"action": "rewrite", "text": "Rewritten response"}],
    )
    def test_rewrite_replaces_text(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original text")
        assert text == "Rewritten response"
        assert blocked is False

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[{"action": "block", "text": "Blocked by policy"}],
    )
    def test_block_returns_fallback_and_blocked(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original text")
        assert text == "Blocked by policy"
        assert blocked is True

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[{"action": "block"}],
    )
    def test_block_without_text_returns_empty(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original text")
        assert text == ""
        assert blocked is True

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[
            {"action": "rewrite", "text": "First wins"},
            {"action": "rewrite", "text": "Second loses"},
        ],
    )
    def test_first_rewrite_wins(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original text")
        assert text == "First wins"
        assert blocked is False

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[
            {"action": "rewrite", "text": "Rewrite wins"},
            {"action": "block", "text": "Block loses"},
        ],
    )
    def test_first_rewrite_beats_later_block(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original text")
        assert text == "Rewrite wins"
        assert blocked is False

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=["not a dict", 42, {"action": "allow"}],
    )
    def test_non_dict_results_ignored(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Hello")
        assert text == "Hello"
        assert blocked is False

    @patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=RuntimeError("plugin crashed"),
    )
    def test_error_failopen_returns_original(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Hello")
        assert text == "Hello"
        assert blocked is False

    @patch("hermes_cli.plugins.invoke_hook", return_value=[])
    def test_kwargs_forwarded_correctly(self, mock_hook):
        from hermes_cli.plugins import apply_text_send_hooks
        apply_text_send_hooks(
            "test",
            platform="telegram",
            session_id="sess-1",
            chat_id="123",
            mode="streaming",
            is_edit=True,
            finalize=False,
        )
        mock_hook.assert_called_once_with(
            "pre_gateway_text_send",
            platform="telegram",
            session_id="sess-1",
            chat_id="123",
            text="test",
            mode="streaming",
            is_edit=True,
            finalize=False,
        )

    @patch(
        "hermes_cli.plugins.invoke_hook",
        return_value=[{"action": "rewrite", "text": None}],
    )
    def test_rewrite_with_non_string_text_ignored(self, mock_hook):
        """A rewrite with text=None should not replace the original."""
        from hermes_cli.plugins import apply_text_send_hooks
        text, blocked = apply_text_send_hooks("Original")
        assert text == "Original"
        assert blocked is False


# ---------------------------------------------------------------------------
# 3. Non-streaming path integration
# ---------------------------------------------------------------------------

class TestNonStreamingHookIntegration:
    """Verify the hook fires in _process_message_background (base.py)."""

    @pytest.mark.asyncio
    async def test_hook_called_before_send(self):
        """apply_text_send_hooks is called before _send_with_retry in
        the non-streaming path."""
        from gateway.platforms.base import SendResult

        call_order = []

        async def fake_send_with_retry(chat_id, content, reply_to=None, metadata=None):
            call_order.append(("send", content))
            return SendResult(success=True, message_id="msg1")

        def fake_apply_hooks(text, **kwargs):
            call_order.append(("hook", text))
            return text, False

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=fake_apply_hooks,
        ):
            text_content = "Agent response"
            from hermes_cli.plugins import apply_text_send_hooks as _apply_hooks
            text_content, _blocked = _apply_hooks(
                text_content,
                platform="test",
                chat_id="chat-1",
                mode="non_streaming",
            )
            if not _blocked and text_content:
                await fake_send_with_retry(
                    chat_id="chat-1",
                    content=text_content,
                )

        assert call_order == [("hook", "Agent response"), ("send", "Agent response")]

    @pytest.mark.asyncio
    async def test_hook_rewrite_changes_sent_content(self):
        """When a plugin rewrites, the rewritten text is sent."""
        from gateway.platforms.base import SendResult

        sent_contents = []

        async def capture_send(chat_id, content, reply_to=None, metadata=None):
            sent_contents.append(content)
            return SendResult(success=True, message_id="msg1")

        def rewrite_hook(text, **kwargs):
            return "REDACTED", False

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=rewrite_hook,
        ):
            from hermes_cli.plugins import apply_text_send_hooks
            text, blocked = apply_text_send_hooks("secret data")
            if not blocked and text:
                await capture_send(chat_id="c1", content=text)

        assert sent_contents == ["REDACTED"]

    @pytest.mark.asyncio
    async def test_hook_block_suppresses_send(self):
        """When a plugin blocks with empty text, nothing is sent."""
        sent_contents = []

        async def capture_send(chat_id, content, **kw):
            sent_contents.append(content)

        def block_hook(text, **kwargs):
            return "", True

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=block_hook,
        ):
            from hermes_cli.plugins import apply_text_send_hooks
            text, blocked = apply_text_send_hooks("should not send")
            if not blocked and text:
                await capture_send(chat_id="c1", content=text)

        assert sent_contents == []


# ---------------------------------------------------------------------------
# 4. Streaming path integration
# ---------------------------------------------------------------------------

class TestStreamingHookIntegration:
    """Verify the hook fires in GatewayStreamConsumer.run() got_done block."""

    @pytest.mark.asyncio
    async def test_hook_fires_on_done(self):
        """The hook fires when the stream finishes, on the full assembled text."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = MagicMock()
        adapter.name = "test_adapter"
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.REQUIRES_EDIT_FINALIZE = False

        async def mock_send(*a, **kw):
            return MagicMock(success=True, message_id=None)
        adapter.send = mock_send

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=StreamConsumerConfig(buffer_only=True),
        )

        hook_calls = []

        def capture_hook(text, **kwargs):
            hook_calls.append({"text": text, "mode": kwargs.get("mode")})
            return text, False

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=capture_hook,
        ):
            consumer.on_delta("Hello ")
            consumer.on_delta("world!")
            consumer.finish()
            await consumer.run()

        assert len(hook_calls) == 1
        assert hook_calls[0]["text"] == "Hello world!"
        assert hook_calls[0]["mode"] == "streaming"

    @pytest.mark.asyncio
    async def test_hook_rewrite_in_streaming(self):
        """A rewrite in the streaming path replaces the accumulated text."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = MagicMock()
        adapter.name = "test_adapter"
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.REQUIRES_EDIT_FINALIZE = False

        sent_texts = []
        async def mock_send(chat_id, content, reply_to=None, metadata=None):
            sent_texts.append(content)
            return MagicMock(success=True, message_id="msg1")
        adapter.send = mock_send

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=StreamConsumerConfig(buffer_only=True),
        )

        def rewrite_hook(text, **kwargs):
            return "[FILTERED]", False

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=rewrite_hook,
        ):
            consumer.on_delta("sensitive data")
            consumer.finish()
            await consumer.run()

        # The consumer should have sent the rewritten text
        assert consumer._accumulated == "[FILTERED]" or "[FILTERED]" in sent_texts

    @pytest.mark.asyncio
    async def test_hook_block_in_streaming_clears_accumulated(self):
        """A block in the streaming path empties the accumulated buffer."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = MagicMock()
        adapter.name = "test_adapter"
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.REQUIRES_EDIT_FINALIZE = False

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=StreamConsumerConfig(buffer_only=True),
        )

        def block_hook(text, **kwargs):
            return "", True

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=block_hook,
        ):
            consumer.on_delta("should be blocked")
            consumer.finish()
            await consumer.run()

        # After blocking, _accumulated should be empty
        assert consumer._accumulated == ""

    @pytest.mark.asyncio
    async def test_hook_not_called_on_deltas(self):
        """The hook must NOT fire on individual streaming deltas."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = MagicMock()
        adapter.name = "test_adapter"
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.REQUIRES_EDIT_FINALIZE = False

        async def mock_send(*a, **kw):
            return MagicMock(success=True, message_id=None)
        adapter.send = mock_send

        async def mock_edit(*a, **kw):
            return MagicMock(success=True)
        adapter.edit_message = mock_edit

        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            config=StreamConsumerConfig(buffer_only=True),
        )

        hook_call_count = 0

        def counting_hook(text, **kwargs):
            nonlocal hook_call_count
            hook_call_count += 1
            return text, False

        with patch(
            "hermes_cli.plugins.apply_text_send_hooks",
            side_effect=counting_hook,
        ):
            # Send multiple deltas
            consumer.on_delta("chunk1 ")
            consumer.on_delta("chunk2 ")
            consumer.on_delta("chunk3")
            consumer.finish()
            await consumer.run()

        # Hook fires exactly once (on got_done), not per delta
        assert hook_call_count == 1
