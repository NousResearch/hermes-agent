"""Tests for the WeChat gateway adapter, transport, and state helpers."""

import base64
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_source(*parts: str) -> str:
    return (REPO_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _media_item(wechat_mod, item_type: int, *, stt_text: str = "") -> dict:
    media = {"encrypt_query_param": "eqp", "aes_key": base64.b64encode(b"0123456789abcdef").decode()}
    if item_type == wechat_mod.WX_ITEM_IMAGE:
        return {"type": item_type, "image_item": {"media": media}}
    if item_type == wechat_mod.WX_ITEM_VIDEO:
        return {"type": item_type, "video_item": {"media": media}}
    if item_type == wechat_mod.WX_ITEM_FILE:
        return {"type": item_type, "file_item": {"media": media}}
    return {"type": item_type, "voice_item": {"media": media, "text": stt_text}}


class _ClosedTask:
    def __init__(self, coro):
        coro.close()

    def cancel(self):
        return None


class TestPlatformEnum:
    def test_wechat_in_platform_enum(self):
        assert Platform.WECHAT.value == "wechat"


class TestConfigLoading:
    def test_apply_env_overrides_registers_wechat(self, monkeypatch):
        monkeypatch.setenv("WECHAT_BOT_TOKEN", "token-123")
        monkeypatch.setenv("WECHAT_ACCOUNT_ID", "bot-account")
        monkeypatch.setenv("WECHAT_API_BASE_URL", "https://wx.example")
        monkeypatch.setenv("WECHAT_CDN_BASE_URL", "https://cdn.example")
        monkeypatch.setenv("WECHAT_ILINK_APP_ID", "my-bot")
        monkeypatch.setenv("WECHAT_ILINK_CLIENT_VERSION", "12345")
        monkeypatch.setenv("WECHAT_HOME_CHANNEL", "user-123")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.WECHAT in config.platforms
        platform_config = config.platforms[Platform.WECHAT]
        assert platform_config.enabled is True
        assert platform_config.token == "token-123"
        assert platform_config.extra["account_id"] == "bot-account"
        assert platform_config.extra["base_url"] == "https://wx.example"
        assert platform_config.extra["cdn_base_url"] == "https://cdn.example"
        assert platform_config.extra["ilink_app_id"] == "my-bot"
        assert platform_config.extra["ilink_client_version"] == "12345"
        assert platform_config.home_channel is not None
        assert platform_config.home_channel.chat_id == "user-123"


class TestWeChatAdapterInit:
    def test_reads_config_values(self):
        from gateway.platforms.wechat import WeChatAdapter

        config = PlatformConfig(
            enabled=True,
            token="cfg-token",
            extra={
                "account_id": "acct-1",
                "base_url": "https://wx.example",
                "cdn_base_url": "https://cdn.example",
                "ilink_app_id": "cfg-app",
                "ilink_client_version": "123",
            },
        )

        adapter = WeChatAdapter(config)

        assert adapter._transport._token == "cfg-token"
        assert adapter._account_id == "acct-1"
        assert adapter._transport._base_url == "https://wx.example"
        assert adapter._transport._cdn_base_url == "https://cdn.example"
        assert adapter._transport._ilink_app_id == "cfg-app"
        assert adapter._transport._ilink_client_version == 123

    def test_uses_default_base_urls_when_not_configured(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        monkeypatch.delenv("WECHAT_API_BASE_URL", raising=False)
        monkeypatch.delenv("WECHAT_CDN_BASE_URL", raising=False)

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="cfg-token"))

        assert adapter._transport._base_url == wechat.DEFAULT_BASE_URL
        assert adapter._transport._cdn_base_url == wechat.CDN_BASE_URL
        assert adapter._account_id == ""

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("WECHAT_BOT_TOKEN", "env-token")
        monkeypatch.setenv("WECHAT_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("WECHAT_API_BASE_URL", "https://env.example")
        monkeypatch.setenv("WECHAT_CDN_BASE_URL", "https://cdn.env.example")
        monkeypatch.setenv("WECHAT_ILINK_APP_ID", "env-app")
        monkeypatch.setenv("WECHAT_ILINK_CLIENT_VERSION", "777")

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True))

        assert adapter._transport._token == "env-token"
        assert adapter._account_id == "env-account"
        assert adapter._transport._base_url == "https://env.example"
        assert adapter._transport._cdn_base_url == "https://cdn.env.example"
        assert adapter._transport._ilink_app_id == "env-app"
        assert adapter._transport._ilink_client_version == 777


class TestExtractText:
    def test_extracts_plain_text(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_TEXT

        items = [{"type": WX_ITEM_TEXT, "text_item": {"text": "hello"}}]

        assert WeChatAdapter._extract_text(items) == "hello"

    def test_extracts_empty_items(self):
        from gateway.platforms.wechat import WeChatAdapter

        assert WeChatAdapter._extract_text([]) == ""

    def test_extracts_voice_stt(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_VOICE

        items = [{"type": WX_ITEM_VOICE, "voice_item": {"text": "voice transcript"}}]

        assert WeChatAdapter._extract_text(items) == "voice transcript"

    def test_extracts_ref_with_title(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "reply"},
                "ref_msg": {
                    "title": "Alice",
                    "message_item": {"type": WX_ITEM_TEXT, "text_item": {"text": "quoted message"}},
                },
            }
        ]

        assert WeChatAdapter._extract_text(items) == "[Quote: Alice | quoted message]\nreply"

    def test_extracts_ref_with_message_item_only(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "reply"},
                "ref_msg": {
                    "message_item": {"type": WX_ITEM_TEXT, "text_item": {"text": "quoted only"}},
                },
            }
        ]

        assert WeChatAdapter._extract_text(items) == "[Quote: quoted only]\nreply"

    def test_ref_msg_with_title_and_content(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "fresh text"},
                "ref_msg": {
                    "title": "Thread title",
                    "message_item": {"type": WX_ITEM_TEXT, "text_item": {"text": "quoted body"}},
                },
            }
        ]

        assert WeChatAdapter._extract_text(items) == "[Quote: Thread title | quoted body]\nfresh text"

    def test_ref_media_returns_text_only(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_IMAGE, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "reply only"},
                "ref_msg": {
                    "title": "Alice",
                    "message_item": {
                        "type": WX_ITEM_IMAGE,
                        "image_item": {"media": {"encrypt_query_param": "eqp"}},
                    },
                },
            }
        ]

        assert WeChatAdapter._extract_text(items) == "reply only"

    def test_ref_msg_media_returns_text_only(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_VIDEO, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "caption"},
                "ref_msg": {
                    "message_item": {
                        "type": WX_ITEM_VIDEO,
                        "video_item": {"media": {"encrypt_query_param": "eqp"}},
                    },
                },
            }
        ]

        assert WeChatAdapter._extract_text(items) == "caption"


class TestMediaSelection:
    def test_find_media_item_skips_voice_with_stt(self):
        import gateway.platforms.wechat as wechat

        item = {
            "type": wechat.WX_ITEM_VOICE,
            "voice_item": {
                "text": "already transcribed",
                "media": {"encrypt_query_param": "eqp"},
            },
        }

        assert wechat.WeChatAdapter._find_media_item([item]) is None

    @pytest.mark.parametrize(
        ("items_factory", "expected_type"),
        [
            (
                lambda m: [
                    _media_item(m, m.WX_ITEM_VOICE),
                    _media_item(m, m.WX_ITEM_FILE),
                    _media_item(m, m.WX_ITEM_IMAGE),
                    _media_item(m, m.WX_ITEM_VIDEO),
                ],
                "WX_ITEM_IMAGE",
            ),
            (
                lambda m: [
                    _media_item(m, m.WX_ITEM_VOICE),
                    _media_item(m, m.WX_ITEM_FILE),
                    _media_item(m, m.WX_ITEM_VIDEO),
                ],
                "WX_ITEM_VIDEO",
            ),
            (
                lambda m: [
                    _media_item(m, m.WX_ITEM_VOICE),
                    _media_item(m, m.WX_ITEM_FILE),
                ],
                "WX_ITEM_FILE",
            ),
            (
                lambda m: [
                    _media_item(m, m.WX_ITEM_VOICE),
                ],
                "WX_ITEM_VOICE",
            ),
        ],
    )
    def test_media_priority_is_image_then_video_then_file_then_voice(self, items_factory, expected_type):
        import gateway.platforms.wechat as wechat

        item = wechat.WeChatAdapter._find_media_item(items_factory(wechat))

        assert item is not None
        assert item["type"] == getattr(wechat, expected_type)

    def test_find_media_item_uses_quoted_media(self):
        import gateway.platforms.wechat as wechat

        items = [
            {
                "type": wechat.WX_ITEM_TEXT,
                "text_item": {"text": "reply"},
                "ref_msg": {
                    "message_item": _media_item(wechat, wechat.WX_ITEM_FILE),
                },
            }
        ]

        item = wechat.WeChatAdapter._find_media_item(items)

        assert item is not None
        assert item["type"] == wechat.WX_ITEM_FILE

    def test_find_media_item_returns_none_when_missing_encrypt_query_param(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_ITEM_IMAGE

        items = [{"type": WX_ITEM_IMAGE, "image_item": {"media": {}}}]

        assert WeChatAdapter._find_media_item(items) is None


class TestCryptoHelpers:
    def test_aes_ecb_encrypt_decrypt_round_trip(self):
        from gateway.platforms.wechat_transport import aes_ecb_decrypt, aes_ecb_encrypt

        key = b"0123456789abcdef"
        plaintext = b"hello wechat media payload"

        ciphertext = aes_ecb_encrypt(plaintext, key)

        assert ciphertext != plaintext
        assert aes_ecb_decrypt(ciphertext, key) == plaintext

    def test_parse_aes_key_accepts_raw_bytes_base64(self):
        from gateway.platforms.wechat_transport import parse_aes_key

        key = b"0123456789abcdef"
        encoded = base64.b64encode(key).decode()

        assert parse_aes_key(encoded) == key

    def test_parse_aes_key_accepts_hex_string_base64(self):
        from gateway.platforms.wechat_transport import parse_aes_key

        key = b"0123456789abcdef"
        encoded = base64.b64encode(key.hex().encode("ascii")).decode()

        assert parse_aes_key(encoded) == key


class TestMimeHelpers:
    @pytest.mark.parametrize(
        ("file_path", "expected"),
        [
            ("photo.jpg", "image/jpeg"),
            ("video.mp4", "video/mp4"),
            ("doc.pdf", "application/pdf"),
            ("voice.ogg", "audio/ogg"),
        ],
    )
    def test_mime_from_path(self, file_path, expected):
        from gateway.platforms.wechat_transport import mime_from_path

        assert mime_from_path(file_path) == expected


class TestMarkdownToPlain:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("**bold**", "bold"),
            ("*italic*", "italic"),
            ("__strong__", "strong"),
            ("_emphasis_", "emphasis"),
            ("~~strike~~", "strike"),
            ("`inline`", "inline"),
            ("# Heading", "Heading"),
            ("## Heading", "Heading"),
            ("### Heading", "Heading"),
            ("#### Heading", "Heading"),
            ("##### Heading", "Heading"),
            ("###### Heading", "Heading"),
            ("[link](https://example.com)", "link"),
            ("![img](https://example.com/x.png)", ""),
            ("| a | b |", "a  b"),
            ("| --- | --- |", ""),
            ("```python\nprint('x')\n```", "print('x')"),
            ("plain text", "plain text"),
            ("before **bold** after", "before bold after"),
            ("mix [site](https://example.com) and `code`", "mix site and code"),
        ],
    )
    def test_strips_supported_markdown(self, text, expected):
        from gateway.platforms.wechat import _markdown_to_plain

        assert _markdown_to_plain(text) == expected


class TestDeduplication:
    def test_same_key_is_rejected(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))

        assert adapter._is_duplicate("msg-1") is False
        assert adapter._is_duplicate("msg-1") is True

    def test_different_key_is_accepted(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))

        assert adapter._is_duplicate("msg-1") is False
        assert adapter._is_duplicate("msg-2") is False

    def test_window_expiry_prunes_old_entries(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(wechat.time, "time", lambda: now)
        adapter._seen_messages = {
            f"old-{i}": now - wechat.DEDUP_WINDOW_S - 10 for i in range(wechat.DEDUP_MAX_SIZE + 1)
        }

        assert adapter._is_duplicate("fresh") is False
        assert list(adapter._seen_messages) == ["fresh"]


class TestTypingTickets:
    @pytest.mark.asyncio
    async def test_fresh_ticket_not_refetched(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(wechat.time, "time", lambda: now)
        adapter._typing_tickets["user-1"] = ("cached-ticket", now)
        adapter._typing_ticket_ttl_s = 60
        adapter._transport.get_config = AsyncMock(return_value={"typing_ticket": "new-ticket"})

        await adapter._cache_typing_ticket("user-1", "ctx-123")

        adapter._transport.get_config.assert_not_called()
        assert adapter._typing_tickets["user-1"] == ("cached-ticket", now)

    @pytest.mark.asyncio
    async def test_expired_ticket_is_refetched(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(wechat.time, "time", lambda: now)
        adapter._typing_tickets["user-1"] = ("stale-ticket", now - 120)
        adapter._typing_ticket_ttl_s = 60
        adapter._transport.get_config = AsyncMock(return_value={"typing_ticket": "fresh-ticket"})

        await adapter._cache_typing_ticket("user-1", "ctx-123")

        adapter._transport.get_config.assert_awaited_once_with("user-1", "ctx-123")
        assert adapter._typing_tickets["user-1"] == ("fresh-ticket", now)


class TestSend:
    @pytest.mark.asyncio
    async def test_send_requires_context_token(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))

        result = await adapter.send("user-1", "hello")

        assert result.success is False
        assert "context_token" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_skips_empty_text_after_markdown_stripping(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._transport.send_message = AsyncMock(return_value={"ret": 0})

        result = await adapter.send("user-1", "![img](https://example.com/a.png)")

        assert result.success is True
        assert result.message_id == "skipped-empty"
        adapter._transport.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_posts_plain_text_when_context_token_present(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._transport.send_message = AsyncMock(return_value={"ret": 0})

        result = await adapter.send("user-1", "**Hello** [there](https://example.com)")

        assert result.success is True
        adapter._transport.send_message.assert_awaited_once()
        body = adapter._transport.send_message.await_args.args[0]
        assert body["msg"]["to_user_id"] == "user-1"
        assert body["msg"]["context_token"] == "ctx-123"
        assert body["msg"]["item_list"][0]["text_item"]["text"] == "Hello there"

    @pytest.mark.asyncio
    async def test_send_chunks_at_4096(self):
        from gateway.platforms.wechat import MAX_MESSAGE_LENGTH, WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._transport.send_message = AsyncMock(return_value={"ret": 0})

        result = await adapter.send("user-1", "x" * (MAX_MESSAGE_LENGTH + 200))

        assert result.success is True
        assert adapter._transport.send_message.await_count == 2
        for call in adapter._transport.send_message.await_args_list:
            chunk = call.args[0]["msg"]["item_list"][0]["text_item"]["text"]
            assert len(chunk) <= MAX_MESSAGE_LENGTH


class TestInboundFiltering:
    @pytest.mark.asyncio
    async def test_non_user_message_is_skipped(self):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter.handle_message = AsyncMock()

        await adapter._on_message(
            {
                "from_user_id": "user-1",
                "message_type": 2,
                "message_id": "m1",
                "seq": "1",
                "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
            }
        )

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_own_account_id_is_skipped(self):
        from gateway.platforms.wechat import WeChatAdapter, WX_MSG_TYPE_USER

        adapter = WeChatAdapter(
            PlatformConfig(enabled=True, token="token", extra={"account_id": "bot-account"})
        )
        adapter.handle_message = AsyncMock()

        await adapter._on_message(
            {
                "from_user_id": "bot-account",
                "message_type": WX_MSG_TYPE_USER,
                "message_id": "m1",
                "seq": "1",
                "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
            }
        )

        adapter.handle_message.assert_not_awaited()


class TestCdnUploadGuards:
    @pytest.mark.asyncio
    async def test_cdn_upload_rejects_files_over_limit(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_transport as wechat_transport

        transport = wechat_transport.WeChatTransport(token="token")
        transport._http = AsyncMock()

        file_path = tmp_path / "too-big.bin"
        file_path.write_bytes(b"x")
        real_stat = wechat_transport.Path.stat

        def fake_stat(path_obj):
            if path_obj == file_path:
                return SimpleNamespace(st_size=wechat_transport.MEDIA_MAX_BYTES + 1)
            return real_stat(path_obj)

        monkeypatch.setattr(wechat_transport.Path, "stat", fake_stat)

        with pytest.raises(ValueError, match="File too large"):
            await transport.cdn_upload(str(file_path), "user-1", wechat_transport.UPLOAD_MEDIA_FILE)


class TestOutboundAesKeyEncoding:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "kwargs", "type_key"),
        [
            ("send_image_file", {"image_path": "/tmp/demo.png"}, "image_item"),
            ("send_video", {"video_path": "/tmp/demo.mp4"}, "video_item"),
            ("send_document", {"file_path": "/tmp/demo.pdf"}, "file_item"),
            ("send_voice", {"audio_path": "/tmp/demo.wav"}, "file_item"),
        ],
    )
    async def test_outbound_aes_key_is_base64_of_hex_string(self, method_name, kwargs, type_key, monkeypatch):
        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._transport.cdn_upload = AsyncMock(
            return_value={
                "download_param": "enc-param",
                "aeskey": "683c5d59ca1efb1b93989745deae27c4",
                "ciphertext_size": 128,
                "plaintext_size": 100,
            }
        )
        adapter._transport.send_message = AsyncMock(return_value={"ret": 0})

        if method_name == "send_voice":
            monkeypatch.setattr("pathlib.Path.stat", lambda _path, **_kw: SimpleNamespace(st_size=10))

        result = await getattr(adapter, method_name)("user-1", **kwargs)

        assert result.success is True
        body = adapter._transport.send_message.await_args_list[-1].args[0]
        media = body["msg"]["item_list"][0][type_key]["media"]
        assert base64.b64decode(media["aes_key"]).decode("ascii") == "683c5d59ca1efb1b93989745deae27c4"


class TestILinkHeaders:
    def test_ilink_headers_present(self):
        from gateway.platforms.wechat_transport import WeChatTransport

        transport = WeChatTransport(token="token", ilink_app_id="bot", ilink_client_version=123)
        headers = transport._build_headers()

        assert "iLink-App-Id" in headers
        assert "iLink-App-ClientVersion" in headers

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("2.1.7", 0x00020107),
            ("0.2.0", 512),
        ],
    )
    def test_ilink_client_version_encoding(self, version, expected):
        from gateway.platforms.wechat_transport import _build_client_version

        assert _build_client_version(version) == expected

    def test_build_common_headers(self):
        from gateway.platforms.wechat_transport import WeChatTransport

        transport = WeChatTransport(token="token", ilink_app_id="my-app", ilink_client_version=131335)

        assert transport._build_common_headers() == {
            "iLink-App-Id": "my-app",
            "iLink-App-ClientVersion": "131335",
        }

    def test_build_channel_version(self):
        from gateway.platforms.wechat_transport import _build_channel_version

        assert _build_channel_version().startswith("hermes-wechat/")


class TestWeChatState:
    def test_context_token_load(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)
        state_dir = tmp_path / "wechat"
        state_dir.mkdir()
        (state_dir / "context_tokens.json").write_text('{"user-1":"ctx-123"}', encoding="utf-8")

        assert wechat_state.load_context_tokens() == {"user-1": "ctx-123"}

    def test_context_token_save_load_roundtrip(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)

        wechat_state.save_context_tokens({"user-1": "ctx-123", "user-2": "ctx-456"})

        assert wechat_state.load_context_tokens() == {"user-1": "ctx-123", "user-2": "ctx-456"}

    def test_context_token_clear(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)
        wechat_state.save_context_tokens({"user-1": "ctx-123"})

        wechat_state.clear_context_tokens()

        assert not (tmp_path / "wechat" / "context_tokens.json").exists()

    def test_context_token_load_empty(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)

        assert wechat_state.load_context_tokens() == {}

    def test_sync_buf_save_load_roundtrip(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)

        wechat_state.save_sync_buf("acct-1", "cursor-123")

        assert wechat_state.load_sync_buf("acct-1") == "cursor-123"

    def test_sync_buf_empty_account_is_blank(self, tmp_path, monkeypatch):
        import gateway.platforms.wechat_state as wechat_state

        monkeypatch.setattr(wechat_state, "get_hermes_home", lambda: tmp_path)

        assert wechat_state.load_sync_buf("") == ""


class TestWeChatConnectAndPolling:
    @pytest.mark.asyncio
    async def test_context_token_reload_at_connect(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._transport.open = AsyncMock()
        monkeypatch.setattr(wechat, "check_wechat_requirements", lambda: True)
        monkeypatch.setattr(wechat, "load_context_tokens", lambda: {"user-1": "ctx-123"})
        monkeypatch.setattr(wechat.asyncio, "create_task", lambda coro: _ClosedTask(coro))

        connected = await adapter.connect()

        assert connected is True
        assert adapter._context_tokens == {"user-1": "ctx-123"}
        adapter._transport.open.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_token_clear_on_session_expiry(self, monkeypatch):
        import gateway.platforms.wechat as wechat

        adapter = wechat.WeChatAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._running = True
        adapter._context_tokens = {"user-1": "ctx-123"}
        adapter._transport.get_updates = AsyncMock(return_value={"ret": 0, "errcode": -14})
        clear_mock = Mock()

        monkeypatch.setattr(wechat, "load_sync_buf", lambda _account_id: "")
        monkeypatch.setattr(wechat, "clear_context_tokens", clear_mock)
        monkeypatch.setattr(wechat.time, "time", lambda: 0.0)

        async def fake_sleep(_seconds):
            adapter._running = False

        monkeypatch.setattr(wechat.asyncio, "sleep", fake_sleep)

        await adapter._poll_loop()

        assert adapter._context_tokens == {}
        clear_mock.assert_called_once()


class TestTransportUploadBehavior:
    @pytest.mark.asyncio
    async def test_cdn_upload_full_url_priority(self, tmp_path):
        from gateway.platforms.wechat_transport import UPLOAD_MEDIA_IMAGE, WeChatTransport

        file_path = tmp_path / "demo.bin"
        file_path.write_bytes(b"hello world")

        post_mock = AsyncMock(return_value=SimpleNamespace(status_code=200, headers={"x-encrypted-param": "enc"}, text="OK"))
        transport = WeChatTransport(token="token")
        transport._http = SimpleNamespace(post=post_mock)
        transport.api_fetch = AsyncMock(
            return_value={
                "upload_full_url": "https://upload.example/full",
                "upload_param": "legacy-param",
            }
        )

        result = await transport.cdn_upload(str(file_path), "user-1", UPLOAD_MEDIA_IMAGE)

        assert result["download_param"] == "enc"
        assert post_mock.await_args.args[0] == "https://upload.example/full"

    @pytest.mark.asyncio
    async def test_cdn_retry_with_backoff(self, tmp_path, monkeypatch):
        from gateway.platforms.wechat_transport import UPLOAD_MEDIA_FILE, WeChatTransport

        file_path = tmp_path / "demo.bin"
        file_path.write_bytes(b"hello world")

        responses = [
            SimpleNamespace(status_code=500, headers={"x-error-message": "server error"}, text="bad"),
            SimpleNamespace(status_code=500, headers={"x-error-message": "server error"}, text="bad"),
            SimpleNamespace(status_code=200, headers={"x-encrypted-param": "enc"}, text="OK"),
        ]
        post_mock = AsyncMock(side_effect=responses)
        sleep_mock = AsyncMock()

        transport = WeChatTransport(token="token")
        transport._http = SimpleNamespace(post=post_mock)
        transport.api_fetch = AsyncMock(return_value={"upload_param": "param"})
        monkeypatch.setattr("gateway.platforms.wechat_transport.asyncio.sleep", sleep_mock)

        result = await transport.cdn_upload(str(file_path), "user-1", UPLOAD_MEDIA_FILE)

        assert result["download_param"] == "enc"
        assert post_mock.await_count == 3
        assert sleep_mock.await_count == 2


class TestSilkFallback:
    def test_silk_to_wav_fallback(self, monkeypatch):
        from gateway.platforms.wechat import _silk_to_wav

        def raise_missing(*args, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr("subprocess.run", raise_missing)

        assert _silk_to_wav(b"fake-silk") is None


class TestModuleImports:
    def test_transport_imports_clean(self):
        sys.modules.pop("gateway.platforms.wechat", None)
        sys.modules.pop("gateway.platforms.wechat_transport", None)

        mod = importlib.import_module("gateway.platforms.wechat_transport")

        assert mod.__name__ == "gateway.platforms.wechat_transport"
        assert "gateway.platforms.wechat" not in sys.modules

    def test_state_imports_clean(self):
        sys.modules.pop("gateway.platforms.wechat", None)
        sys.modules.pop("gateway.platforms.wechat_state", None)

        mod = importlib.import_module("gateway.platforms.wechat_state")

        assert mod.__name__ == "gateway.platforms.wechat_state"
        assert "gateway.platforms.wechat" not in sys.modules


class TestSourceRegistration:
    def test_platform_enum_registered(self):
        assert "WECHAT" in _read_source("gateway", "config.py")

    def test_toolset_registered(self):
        assert "hermes-wechat" in _read_source("toolsets.py")

    def test_gateway_run_adapter_factory(self):
        source = _read_source("gateway", "run.py")

        assert "Platform.WECHAT" in source
        assert "WeChatAdapter" in source

    def test_gateway_run_auth_maps(self):
        source = _read_source("gateway", "run.py")

        assert "WECHAT_ALLOWED_USERS" in source
        assert "WECHAT_ALLOW_ALL_USERS" in source
