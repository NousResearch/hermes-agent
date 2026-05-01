"""
Tests for ``send_multiple_images`` native batching across platforms.

Covers:
    - Base default loop (per-image fallback for platforms without native batching)
    - Telegram: ``bot.send_media_group`` with chunking at 10
    - Discord: ``channel.send(files=[...])`` with chunking at 10
    - Slack: ``files_upload_v2(file_uploads=[...])`` with chunking at 10
    - Mattermost: single post with ``file_ids`` list (chunk at 5)
    - Email: single email with multiple MIME attachments

Signal's native implementation is covered by test_signal.py.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Base default loop
# ---------------------------------------------------------------------------


class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter that records per-image send calls."""

    name = "stub"

    def __init__(self):
        self.sent_images = []
        self.sent_animations = []
        self.sent_files = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, **kwargs):
        from gateway.platforms.base import SendResult
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}

    async def send_image(self, chat_id, image_url, caption=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_images.append((chat_id, image_url, caption))
        return SendResult(success=True, message_id=str(len(self.sent_images)))

    async def send_animation(self, chat_id, animation_url, caption=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_animations.append((chat_id, animation_url, caption))
        return SendResult(success=True, message_id=str(len(self.sent_animations)))

    async def send_image_file(self, chat_id, image_path, caption=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_files.append((chat_id, image_path, caption))
        return SendResult(success=True, message_id=str(len(self.sent_files)))


class TestBaseDefaultLoop:
    def test_loops_per_image_by_default(self):
        a = _StubAdapter()
        images = [
            ("https://x.com/a.png", "alt 1"),
            ("https://x.com/b.png", "alt 2"),
            ("file:///tmp/foo.png", "local"),
            ("https://x.com/c.gif", ""),
        ]
        _run(a.send_multiple_images("chat1", images))
        # 2 URL images + 1 animation + 1 local file
        assert len(a.sent_images) == 2
        assert len(a.sent_animations) == 1
        assert len(a.sent_files) == 1
        assert a.sent_files[0][1] == "/tmp/foo.png"

    def test_empty_batch_is_noop(self):
        a = _StubAdapter()
        _run(a.send_multiple_images("chat1", []))
        assert a.sent_images == []
        assert a.sent_animations == []
        assert a.sent_files == []


# ---------------------------------------------------------------------------
# Telegram mocks setup (shared with test_send_image_file pattern)
# ---------------------------------------------------------------------------


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


class TestTelegramMultiImage:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = TelegramAdapter(config)
        a._bot = MagicMock()
        a._bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1)])
        return a

    def test_single_batch_under_10_calls_send_media_group_once(self, adapter):
        """3 photos → one send_media_group call with 3 items."""
        import telegram
        images = [(f"https://x.com/{i}.png", f"alt{i}") for i in range(3)]
        # Make InputMediaPhoto a concrete class that records its args
        telegram.InputMediaPhoto = MagicMock(side_effect=lambda media, caption=None: {"media": media, "caption": caption})

        _run(adapter.send_multiple_images("12345", images))

        adapter._bot.send_media_group.assert_awaited_once()
        call_kwargs = adapter._bot.send_media_group.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert len(call_kwargs["media"]) == 3

    def test_batch_over_10_chunks(self, adapter):
        """15 photos → two send_media_group calls (10 + 5)."""
        import telegram
        images = [(f"https://x.com/{i}.png", "") for i in range(15)]
        telegram.InputMediaPhoto = MagicMock(side_effect=lambda media, caption=None: {"media": media})

        _run(adapter.send_multiple_images("12345", images))

        assert adapter._bot.send_media_group.await_count == 2
        sizes = [len(c.kwargs["media"]) for c in adapter._bot.send_media_group.await_args_list]
        assert sizes == [10, 5]

    def test_animations_routed_to_send_animation(self, adapter):
        """GIFs are peeled off and sent individually via send_animation."""
        import telegram
        telegram.InputMediaPhoto = MagicMock(side_effect=lambda media, caption=None: {"media": media})
        adapter.send_animation = AsyncMock()
        # 2 photos + 1 gif
        images = [
            ("https://x.com/a.png", ""),
            ("https://x.com/b.gif", ""),
            ("https://x.com/c.png", ""),
        ]
        _run(adapter.send_multiple_images("12345", images))

        adapter.send_animation.assert_awaited_once()
        assert adapter._bot.send_media_group.await_count == 1
        photos = adapter._bot.send_media_group.await_args.kwargs["media"]
        assert len(photos) == 2

    def test_fallback_to_per_image_on_send_media_group_failure(self, adapter):
        """If send_media_group raises, each photo falls back to send_image."""
        import telegram
        telegram.InputMediaPhoto = MagicMock(side_effect=lambda media, caption=None: {"media": media})
        adapter._bot.send_media_group = AsyncMock(side_effect=Exception("boom"))
        adapter.send_image = AsyncMock(return_value=MagicMock(success=True))
        adapter.send_animation = AsyncMock(return_value=MagicMock(success=True))
        adapter.send_image_file = AsyncMock(return_value=MagicMock(success=True))

        images = [(f"https://x.com/{i}.png", "") for i in range(3)]
        _run(adapter.send_multiple_images("12345", images))

        # Three per-image fallback calls
        assert adapter.send_image.await_count == 3

    def test_empty_noop(self, adapter):
        _run(adapter.send_multiple_images("12345", []))
        adapter._bot.send_media_group.assert_not_called()


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    for name in ("discord", "discord.ext", "discord.ext.commands"):
        sys.modules.setdefault(name, discord_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class TestDiscordMultiImage:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = DiscordAdapter(config)
        a._client = MagicMock()
        return a

    def test_single_batch_of_local_files_sends_once(self, adapter, tmp_path):
        """3 local images → one channel.send with files=[...] of length 3."""
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 20)
            paths.append(p)

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=MagicMock(id=1))
        adapter._client.get_channel = MagicMock(return_value=mock_channel)
        # Non-forum channel
        adapter._is_forum_parent = MagicMock(return_value=False)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("67890", images))

        mock_channel.send.assert_awaited_once()
        assert len(mock_channel.send.call_args.kwargs["files"]) == 3

    def test_batch_over_10_chunks_into_two_messages(self, adapter, tmp_path):
        """15 local images → two channel.send calls (10 + 5)."""
        paths = []
        for i in range(15):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 10)
            paths.append(p)

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=MagicMock(id=1))
        adapter._client.get_channel = MagicMock(return_value=mock_channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("67890", images))

        assert mock_channel.send.await_count == 2
        sizes = [len(c.kwargs["files"]) for c in mock_channel.send.await_args_list]
        assert sizes == [10, 5]

    def test_empty_noop(self, adapter):
        adapter._client = MagicMock()
        _run(adapter.send_multiple_images("67890", []))


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_mod = MagicMock()
    for name in (
        "slack_bolt", "slack_bolt.app", "slack_bolt.app.async_app",
        "slack_bolt.adapter", "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.async_handler",
        "slack_sdk", "slack_sdk.web", "slack_sdk.web.async_client",
        "slack_sdk.errors",
    ):
        sys.modules.setdefault(name, slack_mod)


_ensure_slack_mock()

from gateway.platforms.slack import SlackAdapter  # noqa: E402


class TestSlackMultiImage:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._resolve_thread_ts = MagicMock(return_value=None)
        a._record_uploaded_file_thread = MagicMock()
        client = MagicMock()
        client.files_upload_v2 = AsyncMock(return_value={"ok": True})
        a._get_client = MagicMock(return_value=client)
        return a

    def test_single_batch_of_local_files_sends_one_upload(self, adapter, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 20)
            paths.append(p)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("C12345", images))

        client = adapter._get_client("C12345")
        client.files_upload_v2.assert_awaited_once()
        kwargs = client.files_upload_v2.await_args.kwargs
        assert len(kwargs["file_uploads"]) == 3

    def test_batch_over_10_chunks(self, adapter, tmp_path):
        paths = []
        for i in range(12):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 5)
            paths.append(p)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("C12345", images))

        client = adapter._get_client("C12345")
        assert client.files_upload_v2.await_count == 2
        sizes = [len(c.kwargs["file_uploads"]) for c in client.files_upload_v2.await_args_list]
        assert sizes == [10, 2]

    def test_empty_noop(self, adapter):
        _run(adapter.send_multiple_images("C12345", []))
        client = adapter._get_client("C12345")
        client.files_upload_v2.assert_not_called()


# ---------------------------------------------------------------------------
# Mattermost
# ---------------------------------------------------------------------------


from gateway.platforms.mattermost import MattermostAdapter  # noqa: E402


class TestMattermostMultiImage:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake")
        # Minimal construction via object.__new__ to avoid full setup
        a = object.__new__(MattermostAdapter)
        a._base_url = "https://mm.example.com"
        a._token = "fake"
        a._session = MagicMock()
        a._reply_mode = "thread"
        a._api_post = AsyncMock(return_value={"id": "post123"})
        a._upload_file = AsyncMock(side_effect=lambda *args, **kwargs: f"fid_{a._upload_file.await_count}")
        return a

    def test_local_files_uploaded_and_single_post(self, adapter, tmp_path):
        """3 local images → 3 uploads + 1 post with 3 file_ids."""
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 20)
            paths.append(p)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("channel123", images))

        assert adapter._upload_file.await_count == 3
        adapter._api_post.assert_awaited_once()
        payload = adapter._api_post.await_args.args[1]
        assert payload["channel_id"] == "channel123"
        assert len(payload["file_ids"]) == 3

    def test_batch_over_5_chunks(self, adapter, tmp_path):
        """7 images → 2 posts (5 + 2)."""
        paths = []
        for i in range(7):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 10)
            paths.append(p)

        images = [(f"file://{p}", "") for p in paths]
        _run(adapter.send_multiple_images("channel123", images))

        assert adapter._api_post.await_count == 2
        sizes = [len(c.args[1]["file_ids"]) for c in adapter._api_post.await_args_list]
        assert sizes == [5, 2]

    def test_empty_noop(self, adapter):
        _run(adapter.send_multiple_images("channel123", []))
        adapter._api_post.assert_not_called()


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


from gateway.platforms.email import EmailAdapter  # noqa: E402


class TestEmailMultiImage:
    @pytest.fixture
    def adapter(self):
        a = object.__new__(EmailAdapter)
        a._address = "bot@example.com"
        a._password = "secret"
        a._smtp_host = "smtp.example.com"
        a._smtp_port = 587
        a._thread_context = {}
        return a

    def test_local_files_attached_in_single_email(self, adapter, tmp_path):
        """3 local images → one SMTP send with 3 attachments."""
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 20)
            paths.append(p)

        images = [(f"file://{p}", f"alt {i}") for i, p in enumerate(paths)]

        with patch.object(
            adapter, "_send_email_with_attachments", MagicMock(return_value="<msgid@x>")
        ) as mock_send:
            _run(adapter.send_multiple_images("user@example.com", images))

        mock_send.assert_called_once()
        to_addr, body, file_paths = mock_send.call_args.args
        assert to_addr == "user@example.com"
        assert len(file_paths) == 3
        assert "alt 0" in body

    def test_remote_urls_linked_in_body(self, adapter, tmp_path):
        """Remote URL images get their URL appended to the body, no attachment."""
        images = [
            ("https://x.com/a.png", "first"),
            ("https://x.com/b.png", "second"),
        ]
        with patch.object(
            adapter, "_send_email_with_attachments", MagicMock(return_value="<msgid@x>")
        ) as mock_send:
            _run(adapter.send_multiple_images("user@example.com", images))

        mock_send.assert_called_once()
        to_addr, body, file_paths = mock_send.call_args.args
        assert file_paths == []
        assert "https://x.com/a.png" in body
        assert "https://x.com/b.png" in body

    def test_empty_noop(self, adapter):
        with patch.object(
            adapter, "_send_email_with_attachments", MagicMock()
        ) as mock_send:
            _run(adapter.send_multiple_images("user@example.com", []))
        mock_send.assert_not_called()
