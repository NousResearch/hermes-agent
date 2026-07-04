"""Tests for the Meta platform plugins (Facebook Messenger + Instagram DM).

Both plugins share ``plugins/platforms/messenger/meta_common.py`` — one
Meta app, one webhook callback, one Graph API send pipeline. Coverage:

1. webhook signature verification (X-Hub-Signature-256) + fail-closed edges
2. ``object``-field routing (page → messenger, instagram → instagram)
3. echo / delivery / read receipt filtering
4. attachment classification (image / audio / document + ext normalization)
5. adapter env/config parsing and per-surface chunk limits
6. inbound ``_process_messaging`` → MessageEvent dispatch
7. outbound send chunking + payload shape, send_image, failure paths
8. webhook GET verification handshake + POST signature gate
9. register() metadata for BOTH plugins + shared-module identity
10. enablement gating (ENABLED flag + shared credentials) and env seeding
11. standalone (out-of-process cron) send
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    import aiohttp  # noqa: F401
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_messenger = load_plugin_adapter("messenger")
_instagram = load_plugin_adapter("instagram")

# The shared protocol module — imported via its canonical path so tests
# exercise the same module object both plugin adapters use.
from plugins.platforms.messenger import meta_common

MessengerAdapter = _messenger.MessengerAdapter
InstagramAdapter = _instagram.InstagramAdapter

CREDS = {
    "META_PAGE_ACCESS_TOKEN": "page-token",
    "META_APP_SECRET": "app-secret",
    "META_VERIFY_TOKEN": "verify-token",
}


def _set_creds(monkeypatch, **overrides):
    for key, value in {**CREDS, **overrides}.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def _clear_meta_env(monkeypatch):
    for key in (
        *CREDS,
        "MESSENGER_ENABLED",
        "INSTAGRAM_ENABLED",
        "META_WEBHOOK_HOST",
        "META_WEBHOOK_PORT",
        "META_WEBHOOK_PATH",
        "META_GRAPH_API_BASE",
        "MESSENGER_HOME_CHANNEL",
        "INSTAGRAM_HOME_CHANNEL",
    ):
        monkeypatch.delenv(key, raising=False)


def _sign(body: bytes, secret: str = "app-secret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _adapter(monkeypatch, cls=MessengerAdapter, **env):
    _clear_meta_env(monkeypatch)
    _set_creds(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return cls(PlatformConfig())


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def test_valid_signature_passes(self):
        body = b'{"object": "page"}'
        assert meta_common.verify_meta_signature(body, _sign(body), "app-secret")

    def test_tampered_body_rejected(self):
        body = b'{"object": "page"}'
        sig = _sign(body)
        assert not meta_common.verify_meta_signature(body + b" ", sig, "app-secret")

    def test_wrong_secret_rejected(self):
        body = b"x"
        assert not meta_common.verify_meta_signature(body, _sign(body), "other")

    def test_empty_signature_rejected(self):
        assert not meta_common.verify_meta_signature(b"x", "", "app-secret")

    def test_empty_secret_fails_closed(self):
        body = b"x"
        assert not meta_common.verify_meta_signature(body, _sign(body), "")


# ---------------------------------------------------------------------------
# 2. object-field routing
# ---------------------------------------------------------------------------

class TestObjectRouting:

    def test_page_routes_to_messenger(self):
        assert meta_common.platform_for_webhook_object("page") == "messenger"

    def test_instagram_routes_to_instagram(self):
        assert meta_common.platform_for_webhook_object("instagram") == "instagram"

    @pytest.mark.parametrize("obj", ["whatsapp_business_account", "user", "", "PAGE"])
    def test_unknown_objects_dropped(self, obj):
        assert meta_common.platform_for_webhook_object(obj) is None


# ---------------------------------------------------------------------------
# 3. receipt / echo filtering
# ---------------------------------------------------------------------------

class TestSkipReasons:

    def test_delivery_receipt_skipped(self):
        m = {"sender": {"id": "1"}, "delivery": {"mids": []}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) == "receipt"

    def test_read_receipt_skipped(self):
        m = {"sender": {"id": "1"}, "read": {"watermark": 1}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) == "receipt"

    def test_echo_skipped(self):
        m = {"sender": {"id": "1"}, "message": {"is_echo": True, "text": "hi"}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) == "echo"

    def test_non_message_event_skipped(self):
        m = {"sender": {"id": "1"}, "postback": {"payload": "x"}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) == "non-message"

    def test_missing_sender_skipped(self):
        m = {"message": {"text": "hi"}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) == "no-sender"

    def test_real_message_not_skipped(self):
        m = {"sender": {"id": "1"}, "message": {"mid": "m1", "text": "hi"}}
        assert meta_common.MetaBaseAdapter._skip_reason(m) is None


# ---------------------------------------------------------------------------
# 4. attachment classification
# ---------------------------------------------------------------------------

class TestAttachmentClassification:

    def test_image_known_ext(self):
        assert meta_common.classify_attachment(
            "image", "https://cdn/x.png?sig=1"
        ) == ("image", ".png")

    def test_image_unknown_ext_normalized(self):
        assert meta_common.classify_attachment(
            "image", "https://cdn/x.heic"
        ) == ("image", ".jpg")

    def test_audio(self):
        assert meta_common.classify_attachment(
            "audio", "https://cdn/voice.mp4"
        ) == ("audio", ".mp4")

    def test_video_is_document_kind(self):
        kind, ext = meta_common.classify_attachment("video", "https://cdn/v.mp4")
        assert kind == "document"

    def test_file_without_ext(self):
        assert meta_common.classify_attachment("file", "https://cdn/blob") == (
            "document", ".bin",
        )

    def test_redact_meta_id(self):
        assert meta_common.redact_meta_id("1234567890123") == "1234…23"
        assert meta_common.redact_meta_id("123") == "***"
        assert meta_common.redact_meta_id(None) == "***"


# ---------------------------------------------------------------------------
# 5. adapter construction / config parsing
# ---------------------------------------------------------------------------

class TestAdapterInit:

    def test_env_credentials_and_defaults(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        assert adapter.page_access_token == "page-token"
        assert adapter.app_secret == "app-secret"
        assert adapter.verify_token == "verify-token"
        assert adapter.webhook_host == meta_common.DEFAULT_WEBHOOK_HOST
        assert adapter.webhook_port == meta_common.DEFAULT_WEBHOOK_PORT
        assert adapter.webhook_path == meta_common.DEFAULT_WEBHOOK_PATH
        assert adapter.graph_api_base == meta_common.DEFAULT_GRAPH_API_BASE

    def test_extra_fallback_when_env_missing(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        cfg = PlatformConfig(
            extra={
                "page_access_token": "extra-token",
                "app_secret": "extra-secret",
                "verify_token": "extra-verify",
                "port": 9999,
            }
        )
        adapter = MessengerAdapter(cfg)
        assert adapter.page_access_token == "extra-token"
        assert adapter.webhook_port == 9999

    def test_bad_port_falls_back(self, monkeypatch):
        adapter = _adapter(monkeypatch, META_WEBHOOK_PORT="not-a-port")
        assert adapter.webhook_port == meta_common.DEFAULT_WEBHOOK_PORT

    def test_platform_identity_and_chunk_limits(self, monkeypatch):
        msgr = _adapter(monkeypatch)
        insta = _adapter(monkeypatch, cls=InstagramAdapter)
        assert msgr.platform.value == "messenger"
        assert insta.platform.value == "instagram"
        assert msgr.SAFE_CHUNK_CHARS == meta_common.MESSENGER_SAFE_CHARS
        assert insta.SAFE_CHUNK_CHARS == meta_common.INSTAGRAM_SAFE_CHARS
        assert msgr.MAX_MESSAGE_LENGTH == 2000
        assert insta.MAX_MESSAGE_LENGTH == 1000

    def test_both_plugins_share_one_common_module(self):
        # The instagram plugin must reuse the messenger plugin's shared
        # module (same object in sys.modules) so the webhook server state
        # is shared when both surfaces are enabled.
        assert _instagram.MetaBaseAdapter is meta_common.MetaBaseAdapter
        assert _messenger.MetaBaseAdapter is meta_common.MetaBaseAdapter
        assert issubclass(MessengerAdapter, meta_common.MetaBaseAdapter)
        assert issubclass(InstagramAdapter, meta_common.MetaBaseAdapter)


# ---------------------------------------------------------------------------
# 6. inbound message processing
# ---------------------------------------------------------------------------

class TestInbound:

    def _process(self, adapter, messaging):
        adapter.handle_message = AsyncMock()

        async def _run():
            await adapter._process_messaging(messaging)
            while adapter._background_tasks:
                await asyncio.gather(*list(adapter._background_tasks))

        asyncio.run(_run())
        return adapter.handle_message

    def test_text_message_dispatched(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        handle = self._process(
            adapter,
            {
                "sender": {"id": "111"},
                "recipient": {"id": "222"},
                "message": {"mid": "m-1", "text": "hello"},
            },
        )
        assert handle.await_count == 1
        event = handle.await_args.args[0]
        assert event.text == "hello"
        assert event.message_id == "m-1"
        assert event.source.platform.value == "messenger"
        assert event.source.chat_id == "111"
        assert event.source.chat_type == "dm"
        assert event.source.chat_id_alt == "222"

    def test_instagram_message_records_instagram_platform(self, monkeypatch):
        adapter = _adapter(monkeypatch, cls=InstagramAdapter)
        handle = self._process(
            adapter,
            {
                "sender": {"id": "ig-1"},
                "recipient": {"id": "ig-acct"},
                "message": {"mid": "m-2", "text": "hi"},
            },
        )
        event = handle.await_args.args[0]
        assert event.source.platform.value == "instagram"
        assert event.source.chat_name == "Instagram DM"

    def test_echo_not_dispatched(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        handle = self._process(
            adapter,
            {"sender": {"id": "111"}, "message": {"is_echo": True, "text": "x"}},
        )
        assert handle.await_count == 0

    def test_receipt_not_dispatched(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        handle = self._process(
            adapter, {"sender": {"id": "111"}, "delivery": {"mids": ["m"]}}
        )
        assert handle.await_count == 0

    def test_empty_message_not_dispatched(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        handle = self._process(
            adapter, {"sender": {"id": "111"}, "message": {"mid": "m-3"}}
        )
        assert handle.await_count == 0

    def test_image_attachment_cached(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch)
        adapter._download_attachment = AsyncMock(return_value=b"\x89PNG data")
        cached = tmp_path / "img.png"

        def fake_cache(data, ext=".jpg"):
            cached.write_bytes(data)
            return str(cached)

        monkeypatch.setattr(meta_common, "cache_image_from_bytes", fake_cache)
        handle = self._process(
            adapter,
            {
                "sender": {"id": "111"},
                "message": {
                    "mid": "m-4",
                    "attachments": [
                        {"type": "image", "payload": {"url": "https://cdn/x.png"}}
                    ],
                },
            },
        )
        event = handle.await_args.args[0]
        assert event.media_urls == [str(cached)]
        assert event.message_type.value == "photo"


# ---------------------------------------------------------------------------
# 7. outbound sends
# ---------------------------------------------------------------------------

class TestOutbound:

    def _patched(self, monkeypatch, cls=MessengerAdapter, ok=True):
        adapter = _adapter(monkeypatch, cls=cls)
        calls = []

        async def fake_graph_post(endpoint, payload):
            calls.append((endpoint, payload))
            if ok:
                return True, {"message_id": f"mid-{len(calls)}"}
            return False, {"error": {"message": "boom"}}

        adapter._graph_post = fake_graph_post
        return adapter, calls

    def test_send_payload_shape(self, monkeypatch):
        adapter, calls = self._patched(monkeypatch)
        result = asyncio.run(adapter.send("111", "hello"))
        assert result.success and result.message_id == "mid-1"
        endpoint, payload = calls[0]
        assert endpoint == "me/messages"
        assert payload["recipient"] == {"id": "111"}
        assert payload["messaging_type"] == "RESPONSE"
        assert payload["message"] == {"text": "hello"}

    def test_send_chunks_long_message(self, monkeypatch):
        adapter, calls = self._patched(monkeypatch)
        result = asyncio.run(adapter.send("111", "x" * 4000))
        assert result.success
        assert len(calls) > 1
        for _, payload in calls:
            assert len(payload["message"]["text"]) <= meta_common.MESSENGER_SAFE_CHARS

    def test_instagram_uses_tighter_chunks(self, monkeypatch):
        adapter, calls = self._patched(monkeypatch, cls=InstagramAdapter)
        asyncio.run(adapter.send("ig-1", "x" * 1500))
        assert len(calls) == 2
        for _, payload in calls:
            assert len(payload["message"]["text"]) <= meta_common.INSTAGRAM_SAFE_CHARS

    def test_send_empty_rejected(self, monkeypatch):
        adapter, calls = self._patched(monkeypatch)
        result = asyncio.run(adapter.send("111", "   "))
        assert not result.success
        assert calls == []

    def test_send_error_surfaced(self, monkeypatch):
        adapter, _ = self._patched(monkeypatch, ok=False)
        result = asyncio.run(adapter.send("111", "hello"))
        assert not result.success
        assert result.error == "boom"

    def test_send_image_attachment_then_caption(self, monkeypatch):
        adapter, calls = self._patched(monkeypatch)
        result = asyncio.run(
            adapter.send_image("111", "https://img/x.png", caption="look")
        )
        assert result.success
        _, first = calls[0]
        assert first["message"]["attachment"]["type"] == "image"
        assert first["message"]["attachment"]["payload"]["url"] == "https://img/x.png"
        _, second = calls[1]
        assert second["message"] == {"text": "look"}

    def test_get_chat_info(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        info = asyncio.run(adapter.get_chat_info("111"))
        assert info == {"name": "Messenger DM", "type": "dm", "chat_id": "111"}


# ---------------------------------------------------------------------------
# 8. webhook HTTP handlers
# ---------------------------------------------------------------------------

def _fake_request(body: bytes = b"", headers=None, query=None):
    async def _read():
        return body

    return SimpleNamespace(
        read=_read, headers=headers or {}, query=query or {},
    )


@pytest.mark.skipif(
    not _HAS_AIOHTTP,
    reason="aiohttp not installed (CI installs it via --extra all)",
)
class TestWebhookHandlers:
    """Handler tests build real aiohttp ``web.Response`` objects."""

    def _server_with(self, adapter):
        server = meta_common._SharedWebhookServer()
        server.adapters[adapter.PLATFORM_NAME] = adapter
        return server

    def test_verify_handshake_ok(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        server = self._server_with(adapter)
        request = _fake_request(query={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-token",
            "hub.challenge": "challenge-123",
        })
        response = asyncio.run(server._handle_verify(request))
        assert response.status == 200
        assert response.text == "challenge-123"

    def test_verify_handshake_bad_token(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        server = self._server_with(adapter)
        request = _fake_request(query={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "c",
        })
        response = asyncio.run(server._handle_verify(request))
        assert response.status == 403

    def test_event_bad_signature_rejected(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._process_messaging = AsyncMock()
        server = self._server_with(adapter)
        body = json.dumps({"object": "page", "entry": []}).encode()
        request = _fake_request(
            body=body, headers={"X-Hub-Signature-256": "sha256=deadbeef"}
        )
        response = asyncio.run(server._handle_event(request))
        assert response.status == 403
        assert adapter._process_messaging.await_count == 0

    def test_event_routed_to_matching_adapter(self, monkeypatch):
        msgr = _adapter(monkeypatch)
        insta = _adapter(monkeypatch, cls=InstagramAdapter)
        msgr._process_messaging = AsyncMock()
        insta._process_messaging = AsyncMock()
        server = self._server_with(msgr)
        server.adapters["instagram"] = insta

        messaging = {"sender": {"id": "ig"}, "message": {"text": "hi"}}
        body = json.dumps(
            {"object": "instagram", "entry": [{"messaging": [messaging]}]}
        ).encode()
        request = _fake_request(
            body=body, headers={"X-Hub-Signature-256": _sign(body)}
        )
        response = asyncio.run(server._handle_event(request))
        assert response.status == 200
        assert insta._process_messaging.await_count == 1
        assert msgr._process_messaging.await_count == 0

    def test_event_for_unconnected_surface_dropped(self, monkeypatch):
        msgr = _adapter(monkeypatch)
        msgr._process_messaging = AsyncMock()
        server = self._server_with(msgr)
        body = json.dumps(
            {"object": "instagram", "entry": [{"messaging": [{}]}]}
        ).encode()
        request = _fake_request(
            body=body, headers={"X-Hub-Signature-256": _sign(body)}
        )
        response = asyncio.run(server._handle_event(request))
        # Still 200 so Meta doesn't disable the webhook, but nothing dispatched.
        assert response.status == 200
        assert msgr._process_messaging.await_count == 0


class TestConnectGate:

    def test_connect_fails_without_credentials(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        adapter = MessengerAdapter(PlatformConfig())
        assert asyncio.run(adapter.connect()) is False


# ---------------------------------------------------------------------------
# 9-10. registration metadata + enablement gating
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self):
        self.platforms = {}

    def register_platform(self, **kwargs):
        self.platforms[kwargs["name"]] = kwargs


class TestRegistration:

    @pytest.mark.parametrize(
        "module,name,prefix,limit",
        [
            (_messenger, "messenger", "MESSENGER", 2000),
            (_instagram, "instagram", "INSTAGRAM", 1000),
        ],
    )
    def test_register_metadata(self, module, name, prefix, limit):
        ctx = _FakeCtx()
        module.register(ctx)
        entry = ctx.platforms[name]
        assert entry["allowed_users_env"] == f"{prefix}_ALLOWED_USERS"
        assert entry["allow_all_env"] == f"{prefix}_ALLOW_ALL_USERS"
        assert entry["cron_deliver_env_var"] == f"{prefix}_HOME_CHANNEL"
        assert entry["max_message_length"] == limit
        assert entry["platform_hint"]
        assert callable(entry["standalone_sender_fn"])
        assert callable(entry["env_enablement_fn"])
        assert f"{prefix}_ENABLED" in entry["required_env"]
        for var in CREDS:
            assert var in entry["required_env"]

    def test_not_connected_without_enable_flag(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        assert not _messenger.is_connected(PlatformConfig())
        assert not _messenger.check_requirements()

    def test_connected_with_enable_flag_and_creds(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        monkeypatch.setenv("MESSENGER_ENABLED", "true")
        # check_requirements needs an importable aiohttp; fake it so this
        # test also runs in minimal envs (CI installs the real one).
        monkeypatch.setitem(sys.modules, "aiohttp", types.ModuleType("aiohttp"))
        assert _messenger.is_connected(PlatformConfig())
        assert _messenger.check_requirements()
        # Instagram stays off — per-surface opt-in is independent.
        assert not _instagram.is_connected(PlatformConfig())

    def test_not_connected_without_credentials(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        monkeypatch.setenv("MESSENGER_ENABLED", "true")
        assert not _messenger.is_connected(PlatformConfig())
        assert not _messenger.check_requirements()

    def test_env_enablement_seeds_extras(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        monkeypatch.setenv("MESSENGER_ENABLED", "true")
        monkeypatch.setenv("META_WEBHOOK_PORT", "9001")
        monkeypatch.setenv("MESSENGER_HOME_CHANNEL", "psid-1")
        seeded = _messenger._env_enablement()
        assert seeded["port"] == 9001
        assert seeded["home_channel"] == {"chat_id": "psid-1", "name": "Home"}

    def test_env_enablement_none_when_disabled(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        assert _messenger._env_enablement() is None


# ---------------------------------------------------------------------------
# 11. standalone (cron) send
# ---------------------------------------------------------------------------

class TestStandaloneSend:

    def test_standalone_send_success(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        payloads = []

        async def fake_graph_post(self, endpoint, payload):
            payloads.append(payload)
            return True, {"message_id": "mid-9"}

        monkeypatch.setattr(
            meta_common.MetaBaseAdapter, "_graph_post", fake_graph_post
        )
        result = asyncio.run(
            _messenger._standalone_send(PlatformConfig(), "111", "cron says hi")
        )
        assert result == {"success": True, "message_id": "mid-9"}
        assert payloads[0]["message"] == {"text": "cron says hi"}

    def test_standalone_send_notes_undeliverable_media(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        _set_creds(monkeypatch)
        payloads = []

        async def fake_graph_post(self, endpoint, payload):
            payloads.append(payload)
            return True, {"message_id": "mid-10"}

        monkeypatch.setattr(
            meta_common.MetaBaseAdapter, "_graph_post", fake_graph_post
        )
        result = asyncio.run(
            _instagram._standalone_send(
                PlatformConfig(), "ig-1", "report", media_files=["/tmp/a.png"]
            )
        )
        assert result["success"]
        assert "1 attachment(s)" in payloads[0]["message"]["text"]

    def test_standalone_send_requires_token(self, monkeypatch):
        _clear_meta_env(monkeypatch)
        result = asyncio.run(
            _messenger._standalone_send(PlatformConfig(), "111", "hi")
        )
        assert "error" in result
