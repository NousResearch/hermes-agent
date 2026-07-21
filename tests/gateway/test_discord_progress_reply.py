"""Tests for Discord progress reply_to routing.

When auto_thread is disabled (no source.thread_id), Discord progress
messages should reply to the user's original message via Discord's
message-reference mechanism, mirroring Slack's behaviour where tool
progress nests under the user's message and the final text response
goes inline in the channel.
"""
import pytest

from gateway.config import Platform
from gateway.run import (
    _resolve_progress_reply_to,
    _resolve_progress_thread_id,
)


class TestDiscordProgressReplyTo:
    """Discord progress reply_to resolution."""

    def test_discord_no_thread_uses_event_message_id(self):
        """With auto_thread off, progress should reply to the user's message."""
        assert _resolve_progress_reply_to(
            Platform.DISCORD,
            source_thread_id=None,
            event_message_id="123456789",
        ) == "123456789"

    def test_discord_with_thread_returns_none(self):
        """With auto_thread on, progress goes to the thread, not reply_to."""
        assert _resolve_progress_reply_to(
            Platform.DISCORD,
            source_thread_id="987654321",
            event_message_id="123456789",
        ) is None

    def test_discord_no_event_message_id_returns_none(self):
        assert _resolve_progress_reply_to(
            Platform.DISCORD,
            source_thread_id=None,
            event_message_id=None,
        ) is None

    def test_discord_progress_thread_id_still_none(self):
        """Discord should NOT use event_message_id as a thread_id.

        Discord threads are real channels, not reply chains like Slack.
        Progress routing uses reply_to, not thread_id.
        """
        assert _resolve_progress_thread_id(
            Platform.DISCORD,
            source_thread_id=None,
            event_message_id="123456789",
        ) is None

    def test_discord_thread_progress_thread_id_uses_source_thread(self):
        """When inside a thread, progress should target that thread."""
        assert _resolve_progress_thread_id(
            Platform.DISCORD,
            source_thread_id="thread_abc",
            event_message_id="msg_123",
        ) == "thread_abc"


class TestProgressReplyToParity:
    """Ensure existing platform behaviour is unchanged."""

    def test_feishu_threaded_uses_reply_to(self):
        assert _resolve_progress_reply_to(
            Platform.FEISHU,
            source_thread_id="thread_abc",
            event_message_id="msg_123",
        ) == "msg_123"

    def test_feishu_no_thread_returns_none(self):
        assert _resolve_progress_reply_to(
            Platform.FEISHU,
            source_thread_id=None,
            event_message_id="msg_123",
        ) is None

    def test_mattermost_threaded_uses_reply_to(self):
        assert _resolve_progress_reply_to(
            Platform.MATTERMOST,
            source_thread_id="thread_abc",
            event_message_id="msg_123",
        ) == "msg_123"

    def test_telegram_returns_none(self):
        assert _resolve_progress_reply_to(
            Platform.TELEGRAM,
            source_thread_id=None,
            event_message_id="12345",
        ) is None

    def test_slack_returns_none(self):
        """Slack uses thread_id for progress, not reply_to."""
        assert _resolve_progress_reply_to(
            Platform.SLACK,
            source_thread_id=None,
            event_message_id="1234567890.000001",
        ) is None
