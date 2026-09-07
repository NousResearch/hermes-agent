import inspect
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from plugins.platforms.discord.adapter import DiscordAdapter


def test_discord_media_methods_accept_metadata_kwarg():
    for method_name in (
        "send_voice",
        "send_image_file",
        "send_image",
        "send_video",
        "send_document",
        "send_animation",
        "send_multiple_images",
        "_send_file_attachment",
    ):
        signature = inspect.signature(getattr(DiscordAdapter, method_name))
        assert "metadata" in signature.parameters, method_name


import asyncio
from gateway.config import PlatformConfig


def test_discord_media_resolves_thread_id_from_metadata(tmp_path):
    async def _test():
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

        adapter._client = MagicMock()
        mock_channel = MagicMock()
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_msg.attachments = [MagicMock()]
        mock_channel.send = AsyncMock(return_value=mock_msg)
        adapter._resolve_channel = AsyncMock(return_value=mock_channel)

        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"dummy")

        # When metadata carries thread_id, _resolve_channel must receive thread_id, not guild/chat_id
        res = await adapter.send_image_file(
            chat_id="11111111",
            image_path=str(test_file),
            metadata={"thread_id": "22222222"},
        )
        assert res.success is True
        adapter._resolve_channel.assert_called_with("22222222")

        # Document send
        adapter._resolve_channel.reset_mock()
        res_doc = await adapter.send_document(
            chat_id="11111111",
            file_path=str(test_file),
            metadata={"thread_id": "33333333"},
        )
        assert res_doc.success is True
        adapter._resolve_channel.assert_called_with("33333333")

    asyncio.run(_test())


