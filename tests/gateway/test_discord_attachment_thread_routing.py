"""Tests for Discord attachment thread_id routing.

Regression coverage for #12174: file/image/video/document sends should
honor metadata["thread_id"] the same way text send() does.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Discord mock setup
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    """Create a DiscordAdapter with a mocked client."""
    _ensure_discord_mock()
    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    a._client = MagicMock()
    a._client.user = SimpleNamespace(id=999)
    return a


# ---------------------------------------------------------------------------
# send_image_file / send_video / send_document metadata forwarding
# ---------------------------------------------------------------------------

class TestMediaMethodsMetadataForwarding:
    async def test_send_image_file_forwards_metadata(self, adapter):
        """send_image_file should forward metadata to _send_file_attachment."""
        adapter._send_file_attachment = AsyncMock()
        adapter._send_file_attachment.return_value = SimpleNamespace(
            success=True, message_id="123"
        )

        await adapter.send_image_file(
            chat_id="100",
            image_path="/tmp/img.png",
            caption="look",
            metadata={"thread_id": "200"},
        )

        adapter._send_file_attachment.assert_called_once_with(
            "100", "/tmp/img.png", "look", metadata={"thread_id": "200"}
        )

    async def test_send_video_forwards_metadata(self, adapter):
        """send_video should forward metadata to _send_file_attachment."""
        adapter._send_file_attachment = AsyncMock()
        adapter._send_file_attachment.return_value = SimpleNamespace(
            success=True, message_id="123"
        )

        await adapter.send_video(
            chat_id="100",
            video_path="/tmp/vid.mp4",
            caption="watch",
            metadata={"thread_id": "200"},
        )

        adapter._send_file_attachment.assert_called_once_with(
            "100", "/tmp/vid.mp4", "watch", metadata={"thread_id": "200"}
        )

    async def test_send_document_forwards_metadata(self, adapter):
        """send_document should forward metadata to _send_file_attachment."""
        adapter._send_file_attachment = AsyncMock()
        adapter._send_file_attachment.return_value = SimpleNamespace(
            success=True, message_id="123"
        )

        await adapter.send_document(
            chat_id="100",
            file_path="/tmp/doc.pdf",
            caption="read this",
            file_name="doc.pdf",
            metadata={"thread_id": "200"},
        )

        adapter._send_file_attachment.assert_called_once_with(
            "100", "/tmp/doc.pdf", "read this",
            file_name="doc.pdf", metadata={"thread_id": "200"},
        )

    async def test_send_image_file_no_metadata(self, adapter):
        """send_image_file without metadata should pass None."""
        adapter._send_file_attachment = AsyncMock()
        adapter._send_file_attachment.return_value = SimpleNamespace(
            success=True, message_id="123"
        )

        await adapter.send_image_file(
            chat_id="100",
            image_path="/tmp/img.png",
        )

        adapter._send_file_attachment.assert_called_once_with(
            "100", "/tmp/img.png", None, metadata=None
        )
