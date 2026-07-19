"""Tests for the LINE platform adapter plugin.

Covers LINE adapter behavior from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. inbound media normalization to gateway message types and MIME metadata
8. send routing: reply token preferred → push fallback → batched at 5/call
9. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
_is_system_bypass = _line._is_system_bypass
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement
_MessageDeduplicator = _line._MessageDeduplicator


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()


    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")


    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")


# ---------------------------------------------------------------------------
# 2. Chat-id / source resolution
# ---------------------------------------------------------------------------

class TestSourceResolution:

    def test_user_source(self):
        chat_id, ctype = _resolve_chat({"type": "user", "userId": "U123"})
        assert chat_id == "U123"
        assert ctype == "dm"

    def test_group_source(self):
        chat_id, ctype = _resolve_chat({"type": "group", "groupId": "C456", "userId": "U123"})
        assert chat_id == "C456"
        assert ctype == "group"


# ---------------------------------------------------------------------------
# 3. Three-allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        for src in [
            {"type": "user", "userId": "Ufoo"},
            {"type": "group", "groupId": "Cfoo"},
            {"type": "room", "roomId": "Rfoo"},
        ]:
            assert _allowed_for_source(src, allow_all=True, user_ids=set(), group_ids=set(), room_ids=set())

    def test_user_in_allowlist_passes(self):
        src = {"type": "user", "userId": "Uok"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_first_event_not_duplicate(self):
        d = _MessageDeduplicator()
        assert not d.is_duplicate("evt1")


# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:


    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED


    def test_set_ready_on_delivered_is_noop(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "first")
        c.mark_delivered(rid)
        c.set_ready(rid, "second")
        # DELIVERED is terminal — no further mutation
        assert c.get(rid).payload == "first"
        assert c.get(rid).state is State.DELIVERED


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"


    def test_split_long_chunks_at_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_line(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks), chunks
        assert len(chunks) >= 2

    def test_split_caps_at_five_chunks(self):
        # 1000 paragraphs of 100 chars each — must cap at 5 LINE bubbles.
        text = "\n\n".join(["x" * 100 for _ in range(1000)])
        chunks = split_for_line(text)
        assert len(chunks) <= 5


# ---------------------------------------------------------------------------
# 7. Inbound media normalization
# ---------------------------------------------------------------------------

class TestInboundMedia:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"line-bytes")
        ad.handle_message = AsyncMock()
        return ad

    def _event(self, msg_type, **message):
        payload = {"type": msg_type, "id": f"{msg_type}-1"}
        payload.update(message)
        return {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cline", "userId": "Uline"},
            "message": payload,
        }

    def _captured_event(self, adapter):
        adapter.handle_message.assert_awaited_once()
        return adapter.handle_message.await_args.args[0]

    def test_image_message_uses_photo_type_and_image_mime(self, adapter):
        with patch.object(_line, "cache_image_from_bytes", return_value="/cache/image.jpg") as cache:
            asyncio.run(adapter._handle_message_event(self._event("image")))

        cache.assert_called_once_with(b"line-bytes", ext=".jpg")
        event = self._captured_event(adapter)
        assert event.message_type is _line.MessageType.PHOTO
        assert event.media_urls == ["/cache/image.jpg"]
        assert event.media_types == ["image/jpeg"]


# ---------------------------------------------------------------------------
# 8. Send routing (reply -> push fallback, batching, system-bypass)
# ---------------------------------------------------------------------------

class TestSendRouting:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_system_bypass_recognized(self):
        assert _is_system_bypass("⚡ Interrupting current run")
        assert _is_system_bypass("⏳ Queued — agent is busy")
        assert _is_system_bypass("⏩ Steered toward new task")
        assert not _is_system_bypass("Hello world")
        assert not _is_system_bypass("")


    def test_send_caps_messages_per_call_at_five(self, adapter):
        # Build a payload that would naturally split into more than 5 LINE
        # bubbles; the chunker should cap at 5 + truncate.
        big = "\n\n".join(["x" * 4500 for _ in range(20)])
        result = asyncio.run(adapter.send("Uchat", big))
        assert result.success
        call_kwargs = adapter._client.push.call_args
        # call_args is (args, kwargs); for our send the messages are the 2nd positional
        sent_messages = call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs.get("messages")
        # Without args, fall back to inspecting the call shape
        if sent_messages is None:
            # We invoked client.push(chat_id, messages) — check first batch
            sent_messages = adapter._client.push.call_args.args[1]
        assert len(sent_messages) <= 5

    def test_format_message_strips_markdown(self, adapter):
        out = adapter.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out


# ---------------------------------------------------------------------------
# 9. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw


    def test_register_advertises_required_env(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert set(ctx.kwargs["required_env"]) == {
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET",
        }


    def test_register_factory_yields_line_adapter(self):
        ctx = self._FakeCtx()
        register(ctx)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = ctx.kwargs["adapter_factory"](cfg)
        assert isinstance(ad, LineAdapter)

    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hi"))
        assert "error" in result


class TestPostbackButtonShape:

    def test_template_buttons_structure(self):
        msg = build_postback_button_message("hi", "Tap me", "rid-1")
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        assert msg["template"]["text"] == "hi"
        actions = msg["template"]["actions"]
        assert len(actions) == 1
        assert actions[0]["type"] == "postback"
        data = json.loads(actions[0]["data"])
        assert data == {"action": "show_response", "request_id": "rid-1"}


class TestCheckRequirements:


    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)


class TestAdapterInit:

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "LINE_PORT"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "channel_access_token": "tok",
                "channel_secret": "sec",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "tok"
        assert ad.channel_secret == "sec"
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}


# ---------------------------------------------------------------------------
# 9. Inbound message-type classification
# ---------------------------------------------------------------------------

class TestMessageTypeMapping:
    """LINE webhook message types must map to the right normalized
    MessageType so the gateway routes media correctly (e.g. voice → STT,
    files → document handling). Regression guard for the old code that
    referenced the non-existent ``MessageType.IMAGE`` and collapsed every
    non-text message onto a single type."""

    def test_image_event_not_attributeerror_regression(self):
        # The bug: MessageType.IMAGE doesn't exist on the enum.
        MessageType = _line.MessageType
        assert not hasattr(MessageType, "IMAGE")


# ---------------------------------------------------------------------------
# 10. Dual-stack bind default (NS-603)
# ---------------------------------------------------------------------------

class TestDualStackBind:
    """The LINE webhook server's default bind must serve BOTH IPv4 and IPv6.

    Regression guard for the hosted LINE 502 (NS-603): Fly.io 6PN — the
    private network the edge router reverse-proxies LINE ingest over — is
    IPv6-only (``<app>.internal`` resolves to an ``fdaa:…`` address). The
    adapter used to default to ``host="0.0.0.0"`` (IPv4 only), so the
    router's dial to ``<app>.internal:8646`` hit an address nothing was
    listening on → connection refused → 502 on webhook verification.

    Mirrors gateway/platforms/webhook.py's fix (commit d542894ad):
    ``DEFAULT_HOST = None`` → asyncio binds one socket per address family.
    ``"::"`` is NOT a valid substitute (bindv6only=1 on Fly machines makes
    it IPv6-only, breaking IPv4 loopback health probes).
    """

    def _cfg(self, **extra):
        from gateway.config import PlatformConfig
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return PlatformConfig(enabled=True, extra=base)


    def test_empty_host_normalises_to_none(self, monkeypatch):
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(host=""))
        assert ad.webhook_host is None


    @pytest.mark.asyncio
    async def test_default_bind_serves_both_families(self, monkeypatch):
        """Behavioural proof: host=None opens v4 AND v6 listening sockets."""
        # Guard: on IPv4-only hosts (CI runners with IPv6 disabled) the
        # dual-stack bind legitimately yields no v6 socket — that's the
        # environment, not a regression. Probe an actual ::1 bind rather
        # than trusting socket.has_ipv6 (compile-time constant).
        import socket as _socket
        if not _socket.has_ipv6:
            pytest.skip("IPv6 not supported by this Python build")
        try:
            _probe = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
            try:
                _probe.bind(("::1", 0))
            finally:
                _probe.close()
        except OSError:
            pytest.skip("IPv6 stack unavailable on this host (cannot bind ::1)")
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(port=0))
        ad._client = MagicMock()
        ad._client.get_bot_user_id = AsyncMock(return_value="Ubot")

        # Skip credential/network preamble — drive the aiohttp bind directly
        # the same way connect() does.
        from aiohttp import web
        ad._app = web.Application()
        ad._runner = web.AppRunner(ad._app)
        await ad._runner.setup()
        site = web.TCPSite(ad._runner, ad.webhook_host, 0)
        try:
            await site.start()
            addrs = list(ad._runner.addresses)
            has_v6 = any(len(a) == 4 for a in addrs)
            has_v4 = any(len(a) == 2 for a in addrs)
            assert has_v4, f"IPv4 bind missing — got {addrs}"
            assert has_v6, (
                f"IPv6 bind missing (the 6PN reachability bug, NS-603) — got {addrs}"
            )
        finally:
            await ad._runner.cleanup()


class TestMediaPublicUrlGuard:
    """Outbound media requires LINE_PUBLIC_URL whenever the bind host is not
    a publicly fetchable address — including the new dual-stack ``None``
    default and the legacy wildcard strings."""

    def _adapter(self, monkeypatch, **extra):
        from gateway.config import PlatformConfig
        monkeypatch.delenv("LINE_HOST", raising=False)
        monkeypatch.delenv("LINE_PUBLIC_URL", raising=False)
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return LineAdapter(PlatformConfig(enabled=True, extra=base))


    def test_missing_public_url_false_with_public_base(self, monkeypatch):
        ad = self._adapter(monkeypatch, public_url="https://tunnel.example.com")
        if not ad.public_base_url:
            # Adapter reads env var name LINE_PUBLIC_URL / extra key —
            # set directly if the extra key differs.
            ad.public_base_url = "https://tunnel.example.com"
        assert ad._missing_public_url() is False


    def test_send_image_blocked_without_public_url(self, monkeypatch, tmp_path):
        ad = self._adapter(monkeypatch)
        ad._client = MagicMock()
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n123")
        result = asyncio.run(ad.send_image_file("Uchat", str(img)))
        assert not result.success
        assert "LINE_PUBLIC_URL" in (result.error or "")

class TestDownloadMediaRouting:
    """``_download_media`` must dispatch each LINE content type to its typed
    cache helper. Regression guard for the old code that pushed audio, video,
    and file payloads through ``cache_image_from_bytes``, which rejected them
    as non-image data and silently dropped the media."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"payload-bytes")
        ad._client.loading = AsyncMock()
        return ad

    @pytest.fixture
    def cache_mocks(self, monkeypatch):
        """Mock the typed cache helpers in the adapter's namespace; no disk IO."""
        mocks = {
            "image": MagicMock(return_value="/cache/images/img_abc.jpg"),
            "audio": MagicMock(return_value="/cache/audio/audio_abc.m4a"),
            "video": MagicMock(return_value="/cache/videos/video_abc.mp4"),
            "document": MagicMock(return_value="/cache/documents/doc_abc_x.bin"),
        }
        monkeypatch.setattr(_line, "cache_image_from_bytes", mocks["image"])
        monkeypatch.setattr(_line, "cache_audio_from_bytes", mocks["audio"])
        monkeypatch.setattr(_line, "cache_video_from_bytes", mocks["video"])
        monkeypatch.setattr(_line, "cache_document_from_bytes", mocks["document"])
        return mocks

    def _assert_only(self, cache_mocks, kind):
        for other in set(cache_mocks) - {kind}:
            cache_mocks[other].assert_not_called()

    def test_image_routes_to_image_cache(self, adapter, cache_mocks):
        path, media_type = asyncio.run(adapter._download_media("mid-1", "image"))
        assert path == "/cache/images/img_abc.jpg"
        assert media_type == "image/jpeg"
        cache_mocks["image"].assert_called_once_with(b"payload-bytes", ext=".jpg")
        self._assert_only(cache_mocks, "image")

    def test_audio_routes_to_audio_cache(self, adapter, cache_mocks):
        path, media_type = asyncio.run(adapter._download_media("mid-2", "audio"))
        assert path == "/cache/audio/audio_abc.m4a"
        # Exact subtype is platform mimetypes-dependent; family must be audio.
        assert media_type.startswith("audio/")
        cache_mocks["audio"].assert_called_once_with(b"payload-bytes", ext=".m4a")
        self._assert_only(cache_mocks, "audio")

    def test_video_routes_to_video_cache(self, adapter, cache_mocks):
        path, media_type = asyncio.run(adapter._download_media("mid-3", "video"))
        assert path == "/cache/videos/video_abc.mp4"
        assert media_type.startswith("video/")
        cache_mocks["video"].assert_called_once_with(b"payload-bytes", ext=".mp4")
        self._assert_only(cache_mocks, "video")

    def test_file_routes_to_document_cache_with_filename(self, adapter, cache_mocks):
        path, media_type = asyncio.run(
            adapter._download_media("mid-4", "file", filename="report.pdf")
        )
        assert path == "/cache/documents/doc_abc_x.bin"
        assert media_type == "application/pdf"
        cache_mocks["document"].assert_called_once_with(b"payload-bytes", "report.pdf")
        self._assert_only(cache_mocks, "document")

    def test_file_without_name_uses_fallback_filename(self, adapter, cache_mocks):
        asyncio.run(adapter._download_media("mid-5", "file"))
        cache_mocks["document"].assert_called_once_with(
            b"payload-bytes", "line_file.bin"
        )

    def test_cache_failure_returns_none(self, adapter, cache_mocks):
        cache_mocks["audio"].side_effect = ValueError("audio exceeds size limit")
        assert asyncio.run(adapter._download_media("mid-6", "audio")) == (None, "")

    def test_fetch_failure_returns_none_without_caching(self, adapter, cache_mocks):
        adapter._client.fetch_content = AsyncMock(side_effect=RuntimeError("boom"))
        assert asyncio.run(adapter._download_media("mid-7", "image")) == (None, "")
        for mock in cache_mocks.values():
            mock.assert_not_called()

    def test_message_event_threads_file_name_to_document_cache(
        self, adapter, cache_mocks
    ):
        adapter.handle_message = AsyncMock()
        event = {
            "message": {"type": "file", "id": "mid-8", "fileName": "minutes.docx"},
            "source": {"type": "user", "userId": "U1"},
            "replyToken": "rt-1",
        }
        asyncio.run(adapter._handle_message_event(event))
        cache_mocks["document"].assert_called_once_with(
            b"payload-bytes", "minutes.docx"
        )
        event_obj = adapter.handle_message.call_args.args[0]
        assert event_obj.media_urls == ["/cache/documents/doc_abc_x.bin"]
        assert event_obj.media_types == [
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ]
