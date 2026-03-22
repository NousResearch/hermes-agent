"""Tests for the WeChat (Weixin) platform adapter."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_source() -> str:
    return (REPO_ROOT / "gateway" / "run.py").read_text(encoding="utf-8")


def _send_message_source() -> str:
    return (REPO_ROOT / "tools" / "send_message_tool.py").read_text(encoding="utf-8")


def _media_item(weixin_mod, item_type: int, *, stt_text: str = "") -> dict:
    media = {"encrypt_query_param": "eqp"}
    if item_type == weixin_mod.WX_ITEM_IMAGE:
        return {"type": item_type, "image_item": {"media": media}}
    if item_type == weixin_mod.WX_ITEM_VIDEO:
        return {"type": item_type, "video_item": {"media": media}}
    if item_type == weixin_mod.WX_ITEM_FILE:
        return {"type": item_type, "file_item": {"media": media}}
    return {
        "type": item_type,
        "voice_item": {"media": media, "text": stt_text},
    }


class TestPlatformEnum:

    def test_weixin_in_platform_enum(self):
        assert Platform.WEIXIN.value == "weixin"


class TestWeixinRequirements:

    def test_returns_true_when_dependencies_available(self, monkeypatch):
        import gateway.platforms.weixin as weixin

        monkeypatch.setattr(weixin, "HTTPX_AVAILABLE", True)
        monkeypatch.setattr(weixin, "CRYPTO_AVAILABLE", True)

        assert weixin.check_weixin_requirements() is True


class TestConfigLoading:

    def test_apply_env_overrides_registers_weixin(self, monkeypatch):
        monkeypatch.setenv("WEIXIN_TOKEN", "token-123")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "bot-account")
        monkeypatch.setenv("WEIXIN_BASE_URL", "https://wx.example")
        monkeypatch.setenv("WEIXIN_CDN_BASE_URL", "https://cdn.example")
        monkeypatch.setenv("WEIXIN_HOME_CHANNEL", "user-123")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.WEIXIN in config.platforms
        platform_config = config.platforms[Platform.WEIXIN]
        assert platform_config.enabled is True
        assert platform_config.token == "token-123"
        assert platform_config.extra["account_id"] == "bot-account"
        assert platform_config.extra["base_url"] == "https://wx.example"
        assert platform_config.extra["cdn_base_url"] == "https://cdn.example"
        assert platform_config.home_channel is not None
        assert platform_config.home_channel.chat_id == "user-123"


class TestWeixinAdapterInit:

    def test_reads_config_values(self):
        from gateway.platforms.weixin import WeixinAdapter

        config = PlatformConfig(
            enabled=True,
            token="cfg-token",
            extra={
                "account_id": "acct-1",
                "base_url": "https://wx.example",
                "cdn_base_url": "https://cdn.example",
            },
        )

        adapter = WeixinAdapter(config)

        assert adapter._token == "cfg-token"
        assert adapter._account_id == "acct-1"
        assert adapter._base_url == "https://wx.example"
        assert adapter._cdn_base_url == "https://cdn.example"

    def test_uses_default_base_urls_when_not_configured(self, monkeypatch):
        import gateway.platforms.weixin as weixin

        monkeypatch.delenv("WEIXIN_BASE_URL", raising=False)
        monkeypatch.delenv("WEIXIN_CDN_BASE_URL", raising=False)

        adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="cfg-token"))

        assert adapter._base_url == weixin.DEFAULT_BASE_URL
        assert adapter._cdn_base_url == weixin.CDN_BASE_URL


class TestExtractText:

    def test_extracts_plain_text(self):
        from gateway.platforms.weixin import WeixinAdapter, WX_ITEM_TEXT

        items = [{"type": WX_ITEM_TEXT, "text_item": {"text": "hello"}}]

        assert WeixinAdapter._extract_text(items) == "hello"

    def test_extracts_quoted_text(self):
        from gateway.platforms.weixin import WeixinAdapter, WX_ITEM_TEXT

        items = [
            {
                "type": WX_ITEM_TEXT,
                "text_item": {"text": "reply"},
                "ref_msg": {
                    "title": "Alice",
                    "message_item": {
                        "type": WX_ITEM_TEXT,
                        "text_item": {"text": "quoted message"},
                    },
                },
            }
        ]

        assert WeixinAdapter._extract_text(items) == "[Quote: Alice | quoted message]\nreply"

    def test_extracts_voice_stt(self):
        from gateway.platforms.weixin import WeixinAdapter, WX_ITEM_VOICE

        items = [{"type": WX_ITEM_VOICE, "voice_item": {"text": "voice transcript"}}]

        assert WeixinAdapter._extract_text(items) == "voice transcript"


class TestMediaSelection:

    def test_find_media_item_skips_voice_with_stt(self):
        import gateway.platforms.weixin as weixin

        item = {
            "type": weixin.WX_ITEM_VOICE,
            "voice_item": {
                "text": "already transcribed",
                "media": {"encrypt_query_param": "eqp"},
            },
        }

        assert weixin.WeixinAdapter._find_media_item([item]) is None

    def test_media_priority_is_image_then_video_then_file_then_voice(self):
        import gateway.platforms.weixin as weixin

        items = [
            _media_item(weixin, weixin.WX_ITEM_VOICE),
            _media_item(weixin, weixin.WX_ITEM_FILE),
            _media_item(weixin, weixin.WX_ITEM_IMAGE),
            _media_item(weixin, weixin.WX_ITEM_VIDEO),
        ]
        assert weixin.WeixinAdapter._find_media_item(items)["type"] == weixin.WX_ITEM_IMAGE
        assert weixin.WeixinAdapter._find_media_item(items[:2] + items[3:])["type"] == weixin.WX_ITEM_VIDEO
        assert weixin.WeixinAdapter._find_media_item(items[:2])["type"] == weixin.WX_ITEM_FILE
        assert weixin.WeixinAdapter._find_media_item([items[0]])["type"] == weixin.WX_ITEM_VOICE


class TestCryptoHelpers:

    def test_aes_ecb_encrypt_decrypt_round_trip(self):
        from gateway.platforms.weixin import _aes_ecb_decrypt, _aes_ecb_encrypt

        key = b"0123456789abcdef"
        plaintext = b"hello weixin media payload"

        ciphertext = _aes_ecb_encrypt(plaintext, key)

        assert ciphertext != plaintext
        assert _aes_ecb_decrypt(ciphertext, key) == plaintext

    def test_parse_aes_key_accepts_raw_bytes_base64(self):
        import base64

        from gateway.platforms.weixin import _parse_aes_key

        key = b"0123456789abcdef"
        encoded = base64.b64encode(key).decode()

        assert _parse_aes_key(encoded) == key

    def test_parse_aes_key_accepts_hex_string_base64(self):
        import base64

        from gateway.platforms.weixin import _parse_aes_key

        key = b"0123456789abcdef"
        encoded = base64.b64encode(key.hex().encode("ascii")).decode()

        assert _parse_aes_key(encoded) == key


class TestMarkdownToPlain:

    def test_strips_supported_markdown(self):
        from gateway.platforms.weixin import _markdown_to_plain

        text = (
            "**bold** *italic* [link](https://example.com)\n"
            "```python\nprint('x')\n```\n"
            "| col1 | col2 |\n"
            "| ---- | ---- |\n"
            "| a | b |\n"
        )

        result = _markdown_to_plain(text)

        assert "bold" in result
        assert "italic" in result
        assert "link" in result
        assert "print('x')" in result
        assert "col1  col2" in result
        assert "a  b" in result
        assert "**" not in result
        assert "[link]" not in result


class TestDeduplication:

    def test_duplicate_detection(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token"))

        assert adapter._is_duplicate("msg-1") is False
        assert adapter._is_duplicate("msg-1") is True
        assert adapter._is_duplicate("msg-2") is False

    def test_dedup_window_prunes_old_entries(self, monkeypatch):
        import gateway.platforms.weixin as weixin

        adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(weixin.time, "time", lambda: now)
        adapter._seen_messages = {
            f"old-{i}": now - weixin.DEDUP_WINDOW_S - 10 for i in range(weixin.DEDUP_MAX_SIZE + 1)
        }

        assert adapter._is_duplicate("fresh") is False
        assert list(adapter._seen_messages) == ["fresh"]


class TestSend:

    @pytest.mark.asyncio
    async def test_send_fails_without_context_token(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token"))

        result = await adapter.send("user-1", "hello")

        assert result.success is False
        assert "context_token" in result.error

    @pytest.mark.asyncio
    async def test_send_posts_plain_text_when_context_token_present(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._api_fetch = AsyncMock(return_value={"ret": 0})

        result = await adapter.send("user-1", "**Hello** [there](https://example.com)")

        assert result.success is True
        adapter._api_fetch.assert_awaited_once()
        endpoint, body = adapter._api_fetch.await_args.args[:2]
        assert endpoint == "ilink/bot/sendmessage"
        assert body["msg"]["to_user_id"] == "user-1"
        assert body["msg"]["context_token"] == "ctx-123"
        assert body["msg"]["item_list"][0]["text_item"]["text"] == "Hello there"

    @pytest.mark.asyncio
    async def test_send_skips_empty_text_after_markdown_stripping(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._context_tokens["user-1"] = "ctx-123"
        adapter._api_fetch = AsyncMock(return_value={"ret": 0})

        result = await adapter.send("user-1", "![img](https://example.com/a.png)")

        assert result.success is True
        assert result.message_id == "skipped-empty"
        adapter._api_fetch.assert_not_called()

    def test_format_message_returns_plain_text(self):
        from gateway.platforms.weixin import WeixinAdapter

        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token"))

        assert adapter.format_message("**Hello** `world`") == "Hello world"


class TestStandaloneWeixinSendMessage:

    @pytest.mark.asyncio
    async def test_send_message_media_path_initializes_adapter_http(self, tmp_path, monkeypatch):
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_weixin

        tokens_dir = tmp_path / "weixin"
        tokens_dir.mkdir()
        (tokens_dir / "context_tokens.json").write_text('{"user-1":"ctx-123"}', encoding="utf-8")

        media_path = tmp_path / "image.png"
        media_path.write_bytes(b"fake-image")

        class FakeResponse:
            status_code = 200
            text = "OK"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

        class FakeAdapter:
            def __init__(self, config):
                self.config = config
                self._http = None
                self._context_tokens = {}

            async def send_image_file(self, chat_id, image_path, **kwargs):
                assert self._http is not None
                assert self._context_tokens["user-1"] == "ctx-123"
                assert chat_id == "user-1"
                assert image_path == str(media_path)
                return SimpleNamespace(success=True, error=None)

        monkeypatch.setattr("httpx.AsyncClient", FakeClient)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("gateway.platforms.weixin.WeixinAdapter", FakeAdapter)

        result = await _send_weixin(
            PlatformConfig(enabled=True, token="token"),
            "user-1",
            "",
            media_files=[(str(media_path), False)],
        )

        assert result["success"] is True
        assert "image.png" in result["message"]


class TestStandaloneRetHandling:

    @pytest.mark.asyncio
    async def test_send_weixin_text_fails_on_ret_nonzero(self, tmp_path, monkeypatch):
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_weixin

        tokens_dir = tmp_path / "weixin"
        tokens_dir.mkdir()
        (tokens_dir / "context_tokens.json").write_text('{"user-1":"ctx-123"}', encoding="utf-8")

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"ret": -2, "errmsg": "context_token expired"}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

        monkeypatch.setattr("httpx.AsyncClient", FakeClient)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

        result = await _send_weixin(
            PlatformConfig(enabled=True, token="token"),
            "user-1",
            "hello",
        )

        assert result.get("error")
        assert "ret=" in result["error"]


class TestOutboundAesKeyEncoding:

    def test_outbound_aes_key_is_base64_of_hex_string(self):
        import base64

        from gateway.platforms.weixin import WeixinAdapter

        # Simulate what _send_media_file does after _cdn_upload
        aes_key_hex = "683c5d59ca1efb1b93989745deae27c4"  # 32-char hex
        aes_key_b64 = base64.b64encode(aes_key_hex.encode()).decode()

        # SDK-style: base64 of hex string should be 44 chars
        assert len(aes_key_b64) == 44
        # Decoding should give back the hex string
        assert base64.b64decode(aes_key_b64).decode() == aes_key_hex

    def test_file_item_uses_ciphertext_size_for_len(self):
        """file_item.len should be ciphertext_size (padded), not plaintext_size."""
        source = (REPO_ROOT / "gateway" / "platforms" / "weixin.py").read_text(encoding="utf-8")
        assert '"len": str(uploaded["ciphertext_size"])' in source


class TestTypingTickets:

    @pytest.mark.asyncio
    async def test_fresh_typing_ticket_is_not_refetched(self, monkeypatch):
        import gateway.platforms.weixin as weixin

        adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(weixin.time, "time", lambda: now)
        adapter._typing_tickets["user-1"] = ("cached-ticket", now)
        adapter._typing_ticket_ttl_s = 60
        adapter._api_fetch = AsyncMock(return_value={"typing_ticket": "new-ticket"})

        await adapter._cache_typing_ticket("user-1", "ctx-123")

        adapter._api_fetch.assert_not_called()
        assert adapter._typing_tickets["user-1"] == ("cached-ticket", now)

    @pytest.mark.asyncio
    async def test_stale_typing_ticket_triggers_refresh(self, monkeypatch):
        import gateway.platforms.weixin as weixin

        adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        now = 1_000.0
        monkeypatch.setattr(weixin.time, "time", lambda: now)
        adapter._typing_tickets["user-1"] = ("stale-ticket", now - 120)
        adapter._typing_ticket_ttl_s = 60
        adapter._api_fetch = AsyncMock(return_value={"typing_ticket": "fresh-ticket"})

        await adapter._cache_typing_ticket("user-1", "ctx-123")

        adapter._api_fetch.assert_awaited_once()
        assert adapter._typing_tickets["user-1"] == ("fresh-ticket", now)


class TestInboundFiltering:

    @pytest.mark.asyncio
    async def test_self_message_is_skipped(self):
        from gateway.platforms.weixin import WeixinAdapter, WX_MSG_TYPE_USER

        adapter = WeixinAdapter(
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


class TestCdnUpload:

    @pytest.mark.asyncio
    async def test_cdn_upload_rejects_files_over_limit(self, tmp_path, monkeypatch):
        import gateway.platforms.weixin as weixin

        adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="token"))
        adapter._http = AsyncMock()

        file_path = tmp_path / "too-big.bin"
        file_path.write_bytes(b"x")
        real_stat = weixin.Path.stat

        def fake_stat(path_obj):
            if path_obj == file_path:
                return SimpleNamespace(st_size=weixin.MEDIA_MAX_BYTES + 1)
            return real_stat(path_obj)

        monkeypatch.setattr(weixin.Path, "stat", fake_stat)

        with pytest.raises(ValueError, match="File too large"):
            await adapter._cdn_upload(str(file_path), "user-1", weixin.UPLOAD_MEDIA_FILE)


class TestSourceIntegration:

    def test_authorization_maps_include_weixin(self):
        source = _run_source()

        assert 'Platform.WEIXIN: "WEIXIN_ALLOWED_USERS"' in source
        assert 'Platform.WEIXIN: "WEIXIN_ALLOW_ALL_USERS"' in source

    def test_toolset_maps_include_weixin_in_all_four_run_py_copies(self):
        source = _run_source()

        assert source.count('Platform.WEIXIN: "hermes-weixin"') == 2
        assert source.count('Platform.WEIXIN: "weixin"') == 2

    def test_send_message_tool_routes_weixin(self):
        source = _send_message_source()

        assert '"weixin": Platform.WEIXIN' in source
        assert "if platform == Platform.WEIXIN:" in source
        assert "async def _send_weixin" in source
        assert "media_files=" in source  # Weixin uses media_files pipeline
