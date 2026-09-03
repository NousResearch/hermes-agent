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
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def _run(coro):
    return asyncio.run(coro)


async def _allow_safe_url(_url):
    return True


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

    async def connect(self, *, is_reconnect: bool = False):
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


from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class TestDiscordMultiImage:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = DiscordAdapter(config)
        a._client = MagicMock()
        return a


    def test_url_batch_follows_safe_redirect_location_header(self, adapter, monkeypatch):
        """Redirect handling preserves case-insensitive Location behavior."""
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        public_url = "https://cdn.example.test/image.png"
        redirected_url = "https://assets.example.test/image.png"
        requested_urls = []

        class RedirectResponse:
            status_code = 302
            headers = {"Location": redirected_url}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                if False:
                    yield b""
                raise AssertionError("redirect responses must not be read")

        class ImageResponse:
            status_code = 200
            headers = {"Content-Type": "image/png"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                yield b"\x89PNG\r\n\x1a\n"

        class FakeClient:
            def stream(self, method, url, **kwargs):
                assert method == "GET"
                assert kwargs.get("follow_redirects") is False
                requested_urls.append(url)
                if url == public_url:
                    return RedirectResponse()
                assert url == redirected_url
                return ImageResponse()

            async def aclose(self):
                return None

        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: FakeClient(),
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=MagicMock()),
        )

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=MagicMock(id=1))
        adapter._client.get_channel = MagicMock(return_value=mock_channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        _run(adapter.send_multiple_images("67890", [(public_url, "caption")]))

        assert requested_urls == [public_url, redirected_url]
        mock_channel.send.assert_awaited_once()

    def test_send_image_blocks_private_redirect_before_send(self, adapter, monkeypatch):
        send_image_globals = DiscordAdapter.send_image.__globals__

        public_url = "https://cdn.example.test/image.png"
        private_url = "http://169.254.169.254/latest/meta-data/"

        class FakeResponse:
            status_code = 302
            headers = {"Location": private_url}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                if False:
                    yield b""
                raise AssertionError("redirect responses must not be read")

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, **kwargs):
                assert method == "GET"
                assert kwargs.get("follow_redirects") is False
                return FakeResponse()

        async def allow_non_metadata_url(url):
            return not str(url).startswith("http://169.254.169.254")

        monkeypatch.setitem(
            send_image_globals, "async_is_safe_url", allow_non_metadata_url
        )
        monkeypatch.setitem(
            send_image_globals,
            "_create_discord_image_http_client",
            lambda _proxy: FakeClient(),
        )
        monkeypatch.setitem(
            send_image_globals,
            "discord",
            SimpleNamespace(File=MagicMock()),
        )
        adapter._is_forum_parent = MagicMock(return_value=False)
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=MagicMock(id=1))
        adapter._client.get_channel = MagicMock(return_value=mock_channel)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)
        adapter.send = AsyncMock()

        _run(adapter.send_image("67890", public_url, "caption"))

        mock_channel.send.assert_not_awaited()

    @pytest.mark.parametrize(
        ("body", "extension"),
        [
            (b"\x89PNG\r\n\x1a\npng-body", "png"),
            (b"\xff\xd8\xff\xe0jpeg-body", "jpg"),
            (b"GIF89agif-body", "gif"),
            (b"RIFF\x00\x00\x00\x00WEBPwebp-body", "webp"),
        ],
    )
    def test_url_batch_uses_image_magic_bytes_for_filename(
        self, adapter, monkeypatch, body, extension
    ):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        class ImageResponse:
            status_code = 200
            headers = {"Content-Type": "image/png"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                yield body

        class FakeClient:
            def stream(self, method, url, **kwargs):
                assert method == "GET"
                assert kwargs.get("follow_redirects") is False
                return ImageResponse()

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id="sent"))
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(adapter.send_multiple_images("67890", [("https://cdn.example.test/asset.png", "")]))

        assert file_cls.call_args.kwargs["filename"] == f"image_0.{extension}"
        channel.send.assert_awaited_once()

    def test_url_batch_does_not_upload_invalid_body_to_forum_starter(
        self, adapter, monkeypatch
    ):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        class ImageResponse:
            status_code = 200
            headers = {"Content-Type": "image/jpeg"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                yield b'{"error":"not an image"}'

        class FakeClient:
            def stream(self, method, url, **kwargs):
                return ImageResponse()

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=True)
        adapter._forum_post_file = AsyncMock()

        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(adapter.send_multiple_images("67890", [("https://cdn.example.test/asset.jpg", "")]))

        file_cls.assert_not_called()
        adapter._forum_post_file.assert_not_awaited()

    def test_url_batch_cumulative_budget_spans_discord_chunks(
        self, adapter, monkeypatch
    ):
        """The remote budget is shared by the 10-image chunks, not reset per chunk."""
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        body = b"GIF89a"
        urls = [f"https://cdn.example.test/{index}.gif" for index in range(11)]

        class ImageResponse:
            status_code = 200

            def __init__(self):
                self.headers = {"Content-Type": "image/gif"}
                self.consumed_chunks = 0
                self.close_called = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                self.consumed_chunks += 1
                yield body

            async def aclose(self):
                self.close_called += 1

        responses = {url: ImageResponse() for url in urls}

        class FakeClient:
            def __init__(self):
                self.requested_urls = []

            def stream(self, method, url, **kwargs):
                assert method == "GET"
                assert kwargs.get("follow_redirects") is False
                self.requested_urls.append(url)
                return responses[url]

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id="sent"))
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
            len(body) * 10,
        )
        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(adapter.send_multiple_images("67890", [(url, "") for url in urls]))

        assert client.requested_urls == urls
        assert channel.send.await_count == 1
        assert len(channel.send.await_args.kwargs["files"]) == 10
        assert responses[urls[-1]].consumed_chunks == 0

    @pytest.mark.parametrize(
        ("status", "body", "headers"),
        [
            (200, b"<html>", {"Content-Type": "image/png"}),
            (503, b"GIF89a", {"Content-Type": "image/gif"}),
        ],
    )
    def test_url_batch_budget_counts_rejected_remote_bodies(
        self, adapter, monkeypatch, status, body, headers
    ):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        later_body = b"GIF89a"
        first_url = "https://cdn.example.test/rejected"
        later_url = "https://cdn.example.test/later.gif"

        class ImageResponse:
            def __init__(self, response_status, response_body, response_headers):
                self.status_code = response_status
                self.body = response_body
                self.headers = response_headers
                self.consumed_chunks = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                self.consumed_chunks += 1
                yield self.body

            async def aclose(self):
                return None

        responses = {
            first_url: ImageResponse(status, body, headers),
            later_url: ImageResponse(200, later_body, {"Content-Type": "image/gif"}),
        }

        class FakeClient:
            def stream(self, method, url, **kwargs):
                return responses[url]

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id="sent"))
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
            10,
        )
        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(
            adapter.send_multiple_images(
                "67890",
                [(first_url, ""), (later_url, "")],
            )
        )

        assert responses[first_url].consumed_chunks == 1
        assert responses[later_url].consumed_chunks == 1
        file_cls.assert_not_called()
        channel.send.assert_not_awaited()

    def test_url_batch_overdeclared_length_does_not_burn_remaining_budget(
        self, adapter, monkeypatch
    ):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        body = b"GIF89a"
        first_url = "https://cdn.example.test/first.gif"
        overdeclared_url = "https://cdn.example.test/overdeclared.gif"
        later_url = "https://cdn.example.test/later.gif"

        class ImageResponse:
            status_code = 200

            def __init__(self, response_body, headers):
                self.body = response_body
                self.headers = headers
                self.consumed_chunks = 0
                self.close_called = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                self.consumed_chunks += 1
                yield self.body

            async def aclose(self):
                self.close_called += 1

        responses = {
            first_url: ImageResponse(body, {"Content-Length": str(len(body))}),
            overdeclared_url: ImageResponse(body, {"Content-Length": "11"}),
            later_url: ImageResponse(body, {"Content-Length": str(len(body))}),
        }

        class FakeClient:
            def stream(self, method, url, **kwargs):
                return responses[url]

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id="sent"))
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
            16,
        )
        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(
            adapter.send_multiple_images(
                "67890",
                [(first_url, ""), (overdeclared_url, ""), (later_url, "")],
            )
        )

        assert responses[first_url].consumed_chunks == 1
        assert responses[overdeclared_url].consumed_chunks == 0
        assert responses[overdeclared_url].close_called == 1
        assert responses[later_url].consumed_chunks == 1
        assert file_cls.call_count == 2
        channel.send.assert_awaited_once()
        assert len(channel.send.await_args.kwargs["files"]) == 2

    def test_url_batch_missing_and_underreported_lengths_cannot_bypass_budget(
        self, adapter, monkeypatch
    ):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        body = b"GIF89a"
        urls = [
            "https://cdn.example.test/missing-length.gif",
            "https://cdn.example.test/underreported-length.gif",
            "https://cdn.example.test/after-budget.gif",
        ]

        class ImageResponse:
            status_code = 200

            def __init__(self, headers):
                self.headers = headers
                self.consumed_chunks = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                self.consumed_chunks += 1
                yield body

            async def aclose(self):
                return None

        responses = {
            urls[0]: ImageResponse({}),
            urls[1]: ImageResponse({"Content-Length": "1"}),
            urls[2]: ImageResponse({"Content-Length": "1"}),
        }

        class FakeClient:
            def stream(self, method, url, **kwargs):
                return responses[url]

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=SimpleNamespace(id="sent"))
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)

        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
            10,
        )
        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(adapter.send_multiple_images("67890", [(url, "") for url in urls]))

        assert responses[urls[0]].consumed_chunks == 1
        assert responses[urls[1]].consumed_chunks == 1
        assert responses[urls[2]].consumed_chunks == 0
        assert file_cls.call_count == 1
        channel.send.assert_awaited_once()

    def test_url_batch_fallback_reuses_cumulative_budget(self, adapter, monkeypatch):
        """A failed native upload must not re-upload the URL as a second file."""
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        image_url = "https://cdn.example.test/fallback.png"
        body = b"\x89PNG\r\n\x1a\n"
        batch_budget = len(body) + len(body) // 2
        assert batch_budget - len(body) < len(body)

        class ImageResponse:
            status_code = 200
            headers = {}

            def __init__(self):
                self.body_read_calls = 0
                self.bytes_yielded = 0
                self.close_calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self, *, chunk_size=None):
                self.body_read_calls += 1
                self.bytes_yielded += len(body)
                yield body

            async def aclose(self):
                self.close_calls += 1

        class FakeClient:
            def __init__(self):
                self.requested_urls = []
                self.responses = []

            def stream(self, method, url, **kwargs):
                assert method == "GET"
                assert kwargs.get("follow_redirects") is False
                self.requested_urls.append(url)
                response = ImageResponse()
                self.responses.append(response)
                return response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aclose(self):
                return None

        client = FakeClient()
        file_cls = MagicMock()
        channel = MagicMock()
        channel.send = AsyncMock(
            side_effect=[
                RuntimeError("native batch upload failed"),
                SimpleNamespace(id="unexpected second upload"),
            ]
        )
        adapter._client.get_channel = MagicMock(return_value=channel)
        adapter._is_forum_parent = MagicMock(return_value=False)
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

        budget_cls = send_multiple_images_globals["_DiscordImageDownloadBudget"]
        budgets = []

        def make_budget(limit):
            budget = budget_cls(limit)
            budgets.append(budget)
            return budget

        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
            batch_budget,
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_DiscordImageDownloadBudget",
            make_budget,
        )
        monkeypatch.setitem(
            send_multiple_images_globals, "async_is_safe_url", _allow_safe_url
        )
        monkeypatch.setitem(
            send_multiple_images_globals,
            "_create_discord_image_http_client",
            lambda _proxy: client,
        )
        monkeypatch.setitem(
            sys.modules,
            "discord",
            SimpleNamespace(File=file_cls),
        )

        _run(adapter.send_multiple_images("67890", [(image_url, "")]))

        assert client.requested_urls == [image_url, image_url]
        assert [response.body_read_calls for response in client.responses] == [1, 1]
        assert [response.bytes_yielded for response in client.responses] == [
            len(body),
            len(body),
        ]
        assert file_cls.call_count == 1
        assert file_cls.call_args.kwargs["filename"] == "image_0.png"
        assert file_cls.call_args.args[0].read() == body
        assert client.responses[1].close_calls == 1
        assert len(budgets) == 1
        assert budgets[0].bytes_read == len(body) * 2

        channel.send.assert_awaited_once_with(
            content=None,
            files=[file_cls.return_value],
        )
        adapter.send.assert_awaited_once_with(
            chat_id="67890",
            content=image_url,
            reply_to=None,
            metadata=None,
        )

    def test_batch_remote_download_budget_is_100_mib(self):
        send_multiple_images_globals = DiscordAdapter.send_multiple_images.__globals__

        assert (
            send_multiple_images_globals["_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES"]
            == 100 * 1024 * 1024
        )


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

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


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


# ---------------------------------------------------------------------------
# Mattermost
# ---------------------------------------------------------------------------


from plugins.platforms.mattermost.adapter import MattermostAdapter  # noqa: E402


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


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


from plugins.platforms.email.adapter import EmailAdapter  # noqa: E402


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
