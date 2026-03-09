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

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
             patch("asyncio.create_task"):
            asyncio.get_event_loop().run_until_complete(adapter.connect())

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
    async def test_unsupported_file_type_skipped(self, adapter):
        """A .zip file should be silently skipped."""
        event = self._make_event(files=[{
            "mimetype": "application/zip",
            "name": "archive.zip",
            "url_private_download": "https://files.slack.com/archive.zip",
            "size": 1024,
        }])
        await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.TEXT
        assert len(msg_event.media_urls) == 0

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
