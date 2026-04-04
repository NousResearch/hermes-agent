"""
Tests for Slack platform adapter.

Covers: app_mention handler, send_document, send_video,
        incoming document handling, message routing.

Note: slack-bolt may not be installed in the test environment.
We mock the slack modules at import time to avoid collection errors.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
)


# ---------------------------------------------------------------------------
# Mock the slack-bolt package if it's not installed
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

# Patch SLACK_AVAILABLE before importing the adapter
import gateway.platforms.slack as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    # Mock the Slack app client
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    # Capture events instead of processing them
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


# ---------------------------------------------------------------------------
# TestAppMentionHandler
# ---------------------------------------------------------------------------

class TestAppMentionHandler:
    """Verify that the app_mention event handler is registered."""

    def test_app_mention_registered_on_connect(self):
        """connect() should register both 'message' and 'app_mention' handlers."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        # Track which events get registered
        registered_events = []
        registered_commands = []

        mock_app = MagicMock()

        def mock_event(event_type):
            def decorator(fn):
                registered_events.append(event_type)
                return fn
            return decorator

        def mock_command(cmd):
            def decorator(fn):
                registered_commands.append(cmd)
                return fn
            return decorator

        mock_app.event = mock_event
        mock_app.command = mock_command
        mock_app.client = AsyncMock()
        mock_app.client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT",
            "user": "testbot",
        })

        # Mock AsyncWebClient so multi-workspace auth_test is awaitable
        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT",
            "user": "testbot",
            "team_id": "T_FAKE",
            "team": "FakeTeam",
        })

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("asyncio.create_task"):
            asyncio.run(adapter.connect())

        assert "message" in registered_events
        assert "app_mention" in registered_events
        assert "/hermes" in registered_commands


# ---------------------------------------------------------------------------
# TestSendDocument
# ---------------------------------------------------------------------------

class TestSendDocument:
    @pytest.mark.asyncio
    async def test_send_document_success(self, adapter, tmp_path):
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake content")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            caption="Here's the report",
        )

        assert result.success
        adapter._app.client.files_upload_v2.assert_called_once()
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["channel"] == "C123"
        assert call_kwargs["file"] == str(test_file)
        assert call_kwargs["filename"] == "report.pdf"
        assert call_kwargs["initial_comment"] == "Here's the report"

    @pytest.mark.asyncio
    async def test_send_document_custom_name(self, adapter, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_bytes(b"a,b,c\n1,2,3")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            file_name="quarterly-report.csv",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["filename"] == "quarterly-report.csv"

    @pytest.mark.asyncio
    async def test_send_document_missing_file(self, adapter):
        result = await adapter.send_document(
            chat_id="C123",
            file_path="/nonexistent/file.pdf",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_not_connected(self, adapter):
        adapter._app = None
        result = await adapter.send_document(
            chat_id="C123",
            file_path="/some/file.pdf",
        )

        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_document_api_error_falls_back(self, adapter, tmp_path):
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"content")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=RuntimeError("Slack API error")
        )

        # Should fall back to base class (text message)
        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
        )

        # Base class send() is also mocked, so check it was attempted
        adapter._app.client.chat_postMessage.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_document_with_thread(self, adapter, tmp_path):
        test_file = tmp_path / "notes.txt"
        test_file.write_bytes(b"some notes")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            reply_to="1234567890.123456",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["thread_ts"] == "1234567890.123456"


# ---------------------------------------------------------------------------
# TestSendVideo
# ---------------------------------------------------------------------------

class TestSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_success(self, adapter, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")

        adapter._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})

        result = await adapter.send_video(
            chat_id="C123",
            video_path=str(video),
            caption="Check this out",
        )

        assert result.success
        call_kwargs = adapter._app.client.files_upload_v2.call_args[1]
        assert call_kwargs["filename"] == "clip.mp4"
        assert call_kwargs["initial_comment"] == "Check this out"

    @pytest.mark.asyncio
    async def test_send_video_missing_file(self, adapter):
        result = await adapter.send_video(
            chat_id="C123",
            video_path="/nonexistent/video.mp4",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_not_connected(self, adapter):
        adapter._app = None
        result = await adapter.send_video(
            chat_id="C123",
            video_path="/some/video.mp4",
        )

        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_video_api_error_falls_back(self, adapter, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=RuntimeError("Slack API error")
        )

        # Should fall back to base class (text message)
        result = await adapter.send_video(
            chat_id="C123",
            video_path=str(video),
        )

        adapter._app.client.chat_postMessage.assert_called_once()


# ---------------------------------------------------------------------------
# TestIncomingDocumentHandling
# ---------------------------------------------------------------------------

class TestIncomingDocumentHandling:
    def _make_event(self, files=None, text="hello", channel_type="im"):
        """Build a mock Slack message event with file attachments."""
        return {
            "text": text,
            "user": "U_USER",
            "channel": "C123",
            "channel_type": channel_type,
            "ts": "1234567890.000001",
            "files": files or [],
        }

    @pytest.mark.asyncio
    async def test_pdf_document_cached(self, adapter):
        """A PDF attachment should be downloaded, cached, and set as DOCUMENT type."""
        pdf_bytes = b"%PDF-1.4 fake content"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = pdf_bytes
            event = self._make_event(files=[{
                "mimetype": "application/pdf",
                "name": "report.pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "size": len(pdf_bytes),
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.DOCUMENT
        assert len(msg_event.media_urls) == 1
        assert os.path.exists(msg_event.media_urls[0])
        assert msg_event.media_types == ["application/pdf"]

    @pytest.mark.asyncio
    async def test_txt_document_injects_content(self, adapter):
        """A .txt file under 100KB should have its content injected into event text."""
        content = b"Hello from a text file"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(
                text="summarize this",
                files=[{
                    "mimetype": "text/plain",
                    "name": "notes.txt",
                    "url_private_download": "https://files.slack.com/notes.txt",
                    "size": len(content),
                }],
            )
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert "Hello from a text file" in msg_event.text
        assert "[Content of notes.txt]" in msg_event.text
        assert "summarize this" in msg_event.text

    @pytest.mark.asyncio
    async def test_md_document_injects_content(self, adapter):
        """A .md file under 100KB should have its content injected."""
        content = b"# Title\nSome markdown content"

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(files=[{
                "mimetype": "text/markdown",
                "name": "readme.md",
                "url_private_download": "https://files.slack.com/readme.md",
                "size": len(content),
            }], text="")
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert "# Title" in msg_event.text

    @pytest.mark.asyncio
    async def test_large_txt_not_injected(self, adapter):
        """A .txt file over 100KB should be cached but NOT injected."""
        content = b"x" * (200 * 1024)

        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = content
            event = self._make_event(files=[{
                "mimetype": "text/plain",
                "name": "big.txt",
                "url_private_download": "https://files.slack.com/big.txt",
                "size": len(content),
            }], text="")
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 1
        assert "[Content of" not in (msg_event.text or "")

    @pytest.mark.asyncio
    async def test_zip_file_cached(self, adapter):
        """A .zip file should be cached as a supported document."""
        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.return_value = b"PK\x03\x04zip"
            event = self._make_event(files=[{
                "mimetype": "application/zip",
                "name": "archive.zip",
                "url_private_download": "https://files.slack.com/archive.zip",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.DOCUMENT
        assert len(msg_event.media_urls) == 1
        assert msg_event.media_types == ["application/zip"]

    @pytest.mark.asyncio
    async def test_oversized_document_skipped(self, adapter):
        """A document over 20MB should be skipped."""
        event = self._make_event(files=[{
            "mimetype": "application/pdf",
            "name": "huge.pdf",
            "url_private_download": "https://files.slack.com/huge.pdf",
            "size": 25 * 1024 * 1024,
        }])
        await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 0

    @pytest.mark.asyncio
    async def test_document_download_error_handled(self, adapter):
        """If document download fails, handler should not crash."""
        with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
            dl.side_effect = RuntimeError("download failed")
            event = self._make_event(files=[{
                "mimetype": "application/pdf",
                "name": "report.pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        # Handler should still be called (the exception is caught)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_still_handled(self, adapter):
        """Image attachments should still go through the image path, not document."""
        with patch.object(adapter, "_download_slack_file", new_callable=AsyncMock) as dl:
            dl.return_value = "/tmp/cached_image.jpg"
            event = self._make_event(files=[{
                "mimetype": "image/jpeg",
                "name": "photo.jpg",
                "url_private_download": "https://files.slack.com/photo.jpg",
                "size": 1024,
            }])
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.PHOTO


# ---------------------------------------------------------------------------
# TestMessageRouting
# ---------------------------------------------------------------------------

class TestMessageRouting:
    @pytest.mark.asyncio
    async def test_dm_processed_without_mention(self, adapter):
        """DM messages should be processed without requiring a bot mention."""
        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "D123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_channel_message_requires_mention(self, adapter):
        """Channel messages without a bot mention should be ignored."""
        event = {
            "text": "just talking",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_mention_strips_bot_id(self, adapter):
        """When mentioned in a channel, the bot mention should be stripped."""
        event = {
            "text": "<@U_BOT> what's the weather?",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.text == "what's the weather?"
        assert "<@U_BOT>" not in msg_event.text

    @pytest.mark.asyncio
    async def test_bot_messages_ignored(self, adapter):
        """Messages from bots should be ignored."""
        event = {
            "text": "bot response",
            "bot_id": "B_OTHER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_edits_ignored(self, adapter):
        """Message edits should be ignored."""
        event = {
            "text": "edited message",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
            "subtype": "message_changed",
        }
        await adapter._handle_slack_message(event)
        adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# TestSendTyping — assistant.threads.setStatus
# ---------------------------------------------------------------------------


class TestSendTyping:
    """Test typing indicator via assistant.threads.setStatus."""

    @pytest.mark.asyncio
    async def test_sets_status_in_thread(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"thread_id": "parent_ts"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_noop_without_thread(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123")
        adapter._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_scope_gracefully(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope")
        )
        # Should not raise
        await adapter.send_typing("C123", metadata={"thread_id": "ts1"})

    @pytest.mark.asyncio
    async def test_uses_thread_ts_fallback(self, adapter):
        adapter._app.client.assistant_threads_setStatus = AsyncMock()
        await adapter.send_typing("C123", metadata={"thread_ts": "fallback_ts"})
        adapter._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="fallback_ts",
            status="is thinking...",
        )


# ---------------------------------------------------------------------------
# TestFormatMessage — Markdown → mrkdwn conversion
# ---------------------------------------------------------------------------


class TestFormatMessage:
    """Test markdown to Slack mrkdwn conversion."""

    def test_bold_conversion(self, adapter):
        assert adapter.format_message("**hello**") == "*hello*"

    def test_italic_asterisk_conversion(self, adapter):
        assert adapter.format_message("*hello*") == "_hello_"

    def test_italic_underscore_preserved(self, adapter):
        assert adapter.format_message("_hello_") == "_hello_"

    def test_header_to_bold(self, adapter):
        assert adapter.format_message("## Section Title") == "*Section Title*"

    def test_header_with_bold_content(self, adapter):
        # **bold** inside a header should not double-wrap
        assert adapter.format_message("## **Title**") == "*Title*"

    def test_link_conversion(self, adapter):
        result = adapter.format_message("[click here](https://example.com)")
        assert result == "<https://example.com|click here>"

    def test_strikethrough(self, adapter):
        assert adapter.format_message("~~deleted~~") == "~deleted~"

    def test_code_block_preserved(self, adapter):
        code = "```python\nx = **not bold**\n```"
        assert adapter.format_message(code) == code

    def test_inline_code_preserved(self, adapter):
        text = "Use `**raw**` syntax"
        assert adapter.format_message(text) == "Use `**raw**` syntax"

    def test_mixed_content(self, adapter):
        text = "**Bold** and *italic* with `code`"
        result = adapter.format_message(text)
        assert "*Bold*" in result
        assert "_italic_" in result
        assert "`code`" in result

    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_passthrough(self, adapter):
        assert adapter.format_message(None) is None


# ---------------------------------------------------------------------------
# TestReactions
# ---------------------------------------------------------------------------


class TestReactions:
    """Test emoji reaction methods."""

    @pytest.mark.asyncio
    async def test_add_reaction_calls_api(self, adapter):
        adapter._app.client.reactions_add = AsyncMock()
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is True
        adapter._app.client.reactions_add.assert_called_once_with(
            channel="C123", timestamp="ts1", name="eyes"
        )

    @pytest.mark.asyncio
    async def test_add_reaction_handles_error(self, adapter):
        adapter._app.client.reactions_add = AsyncMock(side_effect=Exception("already_reacted"))
        result = await adapter._add_reaction("C123", "ts1", "eyes")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_reaction_calls_api(self, adapter):
        adapter._app.client.reactions_remove = AsyncMock()
        result = await adapter._remove_reaction("C123", "ts1", "eyes")
        assert result is True

    @pytest.mark.asyncio
    async def test_reactions_in_message_flow(self, adapter):
        """Reactions should be added on receipt and swapped on completion."""
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)

        # Should have added 👀, then removed 👀, then added ✅
        add_calls = adapter._app.client.reactions_add.call_args_list
        remove_calls = adapter._app.client.reactions_remove.call_args_list
        assert len(add_calls) == 2
        assert add_calls[0].kwargs["name"] == "eyes"
        assert add_calls[1].kwargs["name"] == "white_check_mark"
        assert len(remove_calls) == 1
        assert remove_calls[0].kwargs["name"] == "eyes"


# ---------------------------------------------------------------------------
# TestUserNameResolution
# ---------------------------------------------------------------------------


class TestUserNameResolution:
    """Test user identity resolution."""

    @pytest.mark.asyncio
    async def test_resolves_display_name(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler", "real_name": "Tyler B"}}
        })
        name = await adapter._resolve_user_name("U123")
        assert name == "Tyler"

    @pytest.mark.asyncio
    async def test_falls_back_to_real_name(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "", "real_name": "Tyler B"}}
        })
        name = await adapter._resolve_user_name("U123")
        assert name == "Tyler B"

    @pytest.mark.asyncio
    async def test_caches_result(self, adapter):
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })
        await adapter._resolve_user_name("U123")
        await adapter._resolve_user_name("U123")
        # Only one API call despite two lookups
        assert adapter._app.client.users_info.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_api_error(self, adapter):
        adapter._app.client.users_info = AsyncMock(side_effect=Exception("rate limited"))
        name = await adapter._resolve_user_name("U123")
        assert name == "U123"  # Falls back to user_id

    @pytest.mark.asyncio
    async def test_user_name_in_message_source(self, adapter):
        """Message source should include resolved user name."""
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Tyler"}}
        })
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()

        event = {
            "text": "hello",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "1234567890.000001",
        }
        await adapter._handle_slack_message(event)

        # Check the source in the MessageEvent passed to handle_message
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.source.user_name == "Tyler"


# ---------------------------------------------------------------------------
# TestSlashCommands — expanded command set
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command routing."""

    @pytest.mark.asyncio
    async def test_compact_maps_to_compress(self, adapter):
        command = {"text": "compact", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/compress"

    @pytest.mark.asyncio
    async def test_resume_command(self, adapter):
        command = {"text": "resume my session", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/resume my session"

    @pytest.mark.asyncio
    async def test_background_command(self, adapter):
        command = {"text": "background run tests", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/background run tests"

    @pytest.mark.asyncio
    async def test_usage_command(self, adapter):
        command = {"text": "usage", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/usage"

    @pytest.mark.asyncio
    async def test_reasoning_command(self, adapter):
        command = {"text": "reasoning", "user_id": "U1", "channel_id": "C1"}
        await adapter._handle_slash_command(command)
        msg = adapter.handle_message.call_args[0][0]
        assert msg.text == "/reasoning"


# ---------------------------------------------------------------------------
# TestMessageSplitting
# ---------------------------------------------------------------------------


class TestMessageSplitting:
    """Test that long messages are split before sending."""

    @pytest.mark.asyncio
    async def test_long_message_split_into_chunks(self, adapter):
        """Messages over MAX_MESSAGE_LENGTH should be split."""
        long_text = "x" * 45000  # Over Slack's 40k API limit
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", long_text)
        # Should have been called multiple times
        assert adapter._app.client.chat_postMessage.call_count >= 2

    @pytest.mark.asyncio
    async def test_short_message_single_send(self, adapter):
        """Short messages should be sent in one call."""
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hello world")
        assert adapter._app.client.chat_postMessage.call_count == 1


# ---------------------------------------------------------------------------
# TestReplyBroadcast
# ---------------------------------------------------------------------------


class TestReplyBroadcast:
    """Test reply_broadcast config option."""

    @pytest.mark.asyncio
    async def test_broadcast_disabled_by_default(self, adapter):
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hi", metadata={"thread_id": "parent_ts"})
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "reply_broadcast" not in kwargs

    @pytest.mark.asyncio
    async def test_broadcast_enabled_via_config(self, adapter):
        adapter.config.extra["reply_broadcast"] = True
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "ts1"}
        )
        await adapter.send("C123", "hi", metadata={"thread_id": "parent_ts"})
        kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert kwargs.get("reply_broadcast") is True


# ---------------------------------------------------------------------------
# TestFallbackPreservesThreadContext
# ---------------------------------------------------------------------------

class TestFallbackPreservesThreadContext:
    """Bug fix: file upload fallbacks lost thread context (metadata) when
    calling super() without metadata, causing replies to appear outside
    the thread."""

    @pytest.mark.asyncio
    async def test_send_image_file_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_123"}
        await adapter.send_image_file(
            chat_id="C123",
            image_path=str(test_file),
            caption="test image",
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_123"

    @pytest.mark.asyncio
    async def test_send_video_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "clip.mp4"
        test_file.write_bytes(b"\x00\x00\x00\x1c")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_456"}
        await adapter.send_video(
            chat_id="C123",
            video_path=str(test_file),
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_456"

    @pytest.mark.asyncio
    async def test_send_document_fallback_preserves_thread(self, adapter, tmp_path):
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        metadata = {"thread_id": "parent_ts_789"}
        await adapter.send_document(
            chat_id="C123",
            file_path=str(test_file),
            caption="report",
            metadata=metadata,
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs.get("thread_ts") == "parent_ts_789"

    @pytest.mark.asyncio
    async def test_send_image_file_fallback_includes_caption(self, adapter, tmp_path):
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0")

        adapter._app.client.files_upload_v2 = AsyncMock(
            side_effect=Exception("upload failed")
        )
        adapter._app.client.chat_postMessage = AsyncMock(
            return_value={"ts": "msg_ts"}
        )

        await adapter.send_image_file(
            chat_id="C123",
            image_path=str(test_file),
            caption="important screenshot",
        )

        call_kwargs = adapter._app.client.chat_postMessage.call_args.kwargs
        assert "important screenshot" in call_kwargs["text"]


# ---------------------------------------------------------------------------
# TestProgressMessageThread
# ---------------------------------------------------------------------------

class TestProgressMessageThread:
    """Verify that progress messages go to the correct thread.

    Issue #2954: For Slack DM top-level messages, source.thread_id is None
    but the final reply is threaded under the user's message via reply_to.
    Progress messages must use the same thread anchor (the original message's
    ts) so they appear in the thread instead of the DM root.
    """

    @pytest.mark.asyncio
    async def test_dm_toplevel_progress_uses_message_ts_as_thread(self, adapter):
        """Progress messages for a top-level DM should go into the reply thread."""
        # Simulate a top-level DM: no thread_ts in the event
        event = {
            "channel": "D_DM",
            "channel_type": "im",
            "user": "U_USER",
            "text": "Hello bot",
            "ts": "1234567890.000001",
            # No thread_ts — this is a top-level DM
        }

        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        # Patch _resolve_user_name to avoid async Slack API call
        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        msg_event = captured_events[0]
        source = msg_event.source

        # For a top-level DM: source.thread_id should remain None
        # (session keying must not be affected)
        assert source.thread_id is None, (
            "source.thread_id must stay None for top-level DMs "
            "so they share one continuous session"
        )

        # The message_id should be the event's ts — this is what the gateway
        # passes as event_message_id so progress messages can thread correctly
        assert msg_event.message_id == "1234567890.000001", (
            "message_id must equal the event ts so _run_agent can use it as "
            "the fallback thread anchor for progress messages"
        )

        # Verify that the Slack send() method correctly threads a message
        # when metadata contains thread_id equal to the original ts
        adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "reply_ts"})
        result = await adapter.send(
            chat_id="D_DM",
            content="⚙️ working...",
            metadata={"thread_id": msg_event.message_id},
        )
        assert result.success
        call_kwargs = adapter._app.client.chat_postMessage.call_args[1]
        assert call_kwargs.get("thread_ts") == "1234567890.000001", (
            "send() must pass thread_ts when metadata has thread_id, "
            "ensuring progress messages land in the thread"
        )

    @pytest.mark.asyncio
    async def test_channel_mention_progress_uses_thread_ts(self, adapter):
        """Progress messages for a channel @mention should go into the reply thread."""
        # Simulate an @mention in a channel: the event ts becomes the thread anchor
        event = {
            "channel": "C_CHAN",
            "channel_type": "channel",
            "user": "U_USER",
            "text": f"<@U_BOT> help me",
            "ts": "2000000000.000001",
            # No thread_ts — top-level channel message
        }

        captured_events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: captured_events.append(e))

        with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="testuser")):
            await adapter._handle_slack_message(event)

        assert len(captured_events) == 1
        msg_event = captured_events[0]
        source = msg_event.source

        # For channel @mention: thread_id should equal the event ts (fallback)
        assert source.thread_id == "2000000000.000001", (
            "source.thread_id must equal the event ts for channel messages "
            "so each @mention starts its own thread"
        )
        assert msg_event.message_id == "2000000000.000001"
