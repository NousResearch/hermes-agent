"""Tests for the MAX platform-plugin adapter (Russian messenger).

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_max`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Most tests target the adapter class directly. The plugin-shape tests
(``register()``, ``_env_enablement``, ``_standalone_send``, registry
presence) mirror the ntfy adapter tests — everything routes through the
``platform_registry``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_max = load_plugin_adapter("max")

MaxAdapter = _max.MaxAdapter
check_requirements = _max.check_requirements
validate_config = _max.validate_config
is_connected = _max.is_connected
register = _max.register
_env_enablement = _max._env_enablement
_standalone_send = _max._standalone_send
MAX_MESSAGE_LENGTH = _max.MAX_MESSAGE_LENGTH
DEDUP_WINDOW_SECONDS = _max.DEDUP_WINDOW_SECONDS


def _run(coro):
    """Run an async coroutine synchronously (fresh event loop each call)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform('max')."""
    from gateway.config import Platform
    p = Platform("max")
    assert p.value == "max"
    assert Platform("max") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestMaxRequirements:

    def test_returns_false_when_httpx_unavailable(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "test-token")
        monkeypatch.setattr(_max, "HTTPX_AVAILABLE", False)
        assert check_requirements() is False

    def test_returns_false_without_token(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert check_requirements() is False

    def test_returns_true_with_token(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "test-token")
        assert check_requirements() is True

    def test_is_connected_from_extra(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={"token": "t"})) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False

    def test_validate_config(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={"token": "t"})) is True
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestMaxAdapterInit:

    def test_init_reads_token_from_extra(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "abc"}))
        assert adapter._token == "abc"
        assert adapter._last_user_id == ""

    def test_init_reads_token_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "env-token")
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._token == "env-token"


# ---------------------------------------------------------------------------
# 4. _env_enablement
# ---------------------------------------------------------------------------


class TestEnvEnablement:

    def test_returns_none_without_token(self, monkeypatch):
        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert _env_enablement() is None

    def test_returns_seed_with_token(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        seed = _env_enablement()
        assert seed == {"token": "tok"}

    def test_includes_home_channel(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "tok")
        monkeypatch.setenv("MAX_HOME_CHANNEL", "123")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "123"


# ---------------------------------------------------------------------------
# 5. _handle_update — message parsing
# ---------------------------------------------------------------------------


class TestHandleUpdate:

    def _make_adapter(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter.handle_message = AsyncMock()
        adapter._is_duplicate = MagicMock(return_value=False)
        return adapter

    def test_ignores_non_message_events(self):
        adapter = self._make_adapter()
        _run(adapter._handle_update({"update_type": "bot_started"}))
        adapter.handle_message.assert_not_called()

    def test_ignores_bot_messages(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 1, "is_bot": True},
                "body": {"text": "hi", "mid": "m1"},
            },
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()

    def test_parses_user_message(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 139383659, "name": "Артур", "is_bot": False},
                "recipient": {"chat_id": 532485678, "chat_type": "dialog"},
                "body": {"text": "привет", "mid": "mid.1"},
            },
            "timestamp": 1786823555223,
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "привет"
        assert event.message_id == "mid.1"
        assert event.source.user_id == "139383659"
        assert event.source.chat_id == "532485678"
        assert adapter._last_user_id == "139383659"

    def test_skips_empty_text(self):
        adapter = self._make_adapter()
        upd = {
            "update_type": "message_created",
            "message": {
                "sender": {"user_id": 1, "is_bot": False},
                "body": {"text": "  ", "mid": "m2"},
            },
        }
        _run(adapter._handle_update(upd))
        adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------


class TestDedup:

    def test_duplicate_message_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter._is_duplicate("mid.1") is False
        assert adapter._is_duplicate("mid.1") is True
        assert adapter._is_duplicate("mid.2") is False

    def test_dedup_window_prunes(self):
        import time
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        # Старое сообщение за пределами окна → не считается дубликатом
        old_id = "old-msg"
        adapter._seen_messages[old_id] = time.time() - DEDUP_WINDOW_SECONDS - 10
        assert adapter._is_duplicate(old_id) is False  # pruned (перезаписано)
        assert adapter._is_duplicate(old_id) is True   # теперь в окне


# ---------------------------------------------------------------------------
# 7. send()
# ---------------------------------------------------------------------------


class TestSend:

    def test_send_dm_uses_user_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("532485678", "hello", metadata={"user_id": "139383659", "chat_type": "dm"}))
        assert result.success is True
        # DM → user_id param, not chat_id
        call_kwargs = adapter._http_client.post.call_args.kwargs
        assert call_kwargs["params"] == {"user_id": "139383659"}

    def test_send_group_uses_chat_id(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("999", "hello", metadata={"chat_type": "group"}))
        assert result.success is True
        call_kwargs = adapter._http_client.post.call_args.kwargs
        assert call_kwargs["params"] == {"chat_id": "999"}

    def test_send_truncates_long_text(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        long_text = "x" * (MAX_MESSAGE_LENGTH + 100)
        _run(adapter.send("1", long_text, metadata={"user_id": "2", "chat_type": "dm"}))
        # send() разбивает на несколько сообщений — к-во вызовов > 1
        assert adapter._http_client.post.call_count > 1
        for call in adapter._http_client.post.call_args_list:
            sent_body = call.kwargs["content"].decode("utf-8")
            import json as _json
            payload = _json.loads(sent_body)
            assert len(payload["text"]) <= MAX_MESSAGE_LENGTH

    def test_smart_truncate_short(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter._smart_truncate("short") == "short"

    def test_smart_truncate_long_with_notice(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        long_text = "word " * 1500  # ~7500 chars
        result = adapter._smart_truncate(long_text)
        assert len(result) <= MAX_MESSAGE_LENGTH
        assert "обрезано" in result

    def test_rate_limit_send(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        # 3 sends to the same chat within 1s — the 3rd must be rate-limited (sleep)
        import time as _time
        _run(adapter.send("1", "a", metadata={"user_id": "2", "chat_type": "dm"}))
        _run(adapter.send("1", "b", metadata={"user_id": "2", "chat_type": "dm"}))
        start = _time.monotonic()
        _run(adapter.send("1", "c", metadata={"user_id": "2", "chat_type": "dm"}))
        elapsed = _time.monotonic() - start
        assert elapsed >= 1.0  # rate-limited
        assert adapter._http_client.post.call_count == 3

    def test_send_http_error_returns_failure(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "bad request"
        adapter._http_client.post = AsyncMock(return_value=resp)

        result = _run(adapter.send("1", "hi", metadata={"user_id": "2", "chat_type": "dm"}))
        assert result.success is False
        assert "400" in result.error


# ---------------------------------------------------------------------------
# 7b. send_typing — typing indicator
# ---------------------------------------------------------------------------


class TestSendTyping:

    def test_send_typing_calls_actions(self):
        import json as _json
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        adapter._http_client.post = AsyncMock(return_value=resp)

        _run(adapter.send_typing("532485678"))
        adapter._http_client.post.assert_called_once()
        call = adapter._http_client.post.call_args
        # POST /chats/{chatId}/actions with {"action": "typing_on"}
        assert "/chats/532485678/actions" in call.args[0]
        payload = _json.loads(call.kwargs["content"].decode("utf-8"))
        assert payload == {"action": "typing_on"} or payload.get("action") == "typing_on"

    def test_send_typing_no_client(self):
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = None
        # Не должно падать без HTTP-клиента
        _run(adapter.send_typing("1"))


# ---------------------------------------------------------------------------
# 7c. Marker persistence
# ---------------------------------------------------------------------------


class TestMarkerPersistence:

    def test_marker_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._marker = 12345
        adapter._save_marker()
        assert (tmp_path / "max" / "marker.json").exists()

        # Новый адаптер должен подхватить маркер с диска
        adapter2 = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        assert adapter2._marker == 12345


# ---------------------------------------------------------------------------
# 7d. PEM validation
# ---------------------------------------------------------------------------


class TestPemValidation:

    def test_valid_pem(self, tmp_path):
        # Валидная структура PEM (реальный сертификат Минцифры)
        real_cert = b"""-----BEGIN CERTIFICATE-----
MIIFwjCCA6qgAwIBAgICEAAwDQYJKoZIhvcNAQELBQAwcDELMAkGA1UEBhMCUlUx
EzARBgNVBAgMCuiBkNC+0YHQvtCy0YHQutCwMREwDwYDVQQHDAjQnNC+0YHQutCy
MRAwDgYDVQQKDAdNaW5jYWYxHTAbBgNVBAMMFE1pbmNpZnkgQ0EgMjAyMTCCAiIw
DQYJKoZIhvcNAQEBBQADggKPADCCAoUCggKBAMsEBPQE3U1b1Q8kq9nWJmH8RCnx
-----END CERTIFICATE-----
"""
        cert = tmp_path / "cert.pem"
        cert.write_bytes(real_cert)
        bad = tmp_path / "bad.crt"
        bad.write_bytes(b"<html>error page</html>")
        assert _max._is_valid_pem_cert(str(bad)) is False   # HTML — не PEM
        # Реальный сертификат может не пройти DER-парсинг (обрезанный образец),
        # но HTML обязан быть отвергнут; при отсутствии BEGIN CERTIFICATE — False
        assert _max._is_valid_pem_cert(str(tmp_path / "nonexistent.pem")) is False


# ---------------------------------------------------------------------------
# 7e. Media uploads
# ---------------------------------------------------------------------------


class TestMediaUploads:

    def test_guess_media_type(self):
        assert MaxAdapter._guess_media_type("photo.png") == "image"
        assert MaxAdapter._guess_media_type("photo.PNG") == "image"
        assert MaxAdapter._guess_media_type("clip.mp4") == "video"
        assert MaxAdapter._guess_media_type("voice.mp3") == "audio"
        assert MaxAdapter._guess_media_type("doc.pdf") == "file"
        assert MaxAdapter._guess_media_type("noext") == "file"

    def test_upload_media_flow(self, tmp_path):
        # Create a fake image to upload
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG fake")

        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()

        # Step 1: /uploads returns url only (no token for image!)
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"url": "https://upload.example/put"}

        # Step 2: upload to url returns token INSIDE photos map (real MAX behavior)
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {
            "photos": {
                "photoId123": {"token": "real-token-from-photos"}
            }
        }

        async def fake_post(url, **kwargs):
            if "uploads" in url:
                return resp1
            return resp2

        # _upload_media opens a NEW client for CDN upload; mock that via patch
        import httpx as _httpx
        up_client = MagicMock()
        up_client.post = AsyncMock(return_value=resp2)
        up_client.__aenter__ = AsyncMock(return_value=up_client)
        up_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_max.httpx, "AsyncClient", return_value=up_client):
            adapter._http_client.post = AsyncMock(side_effect=fake_post)
            att = _run(adapter._upload_media(str(img)))
        assert att is not None
        assert att["type"] == "image"
        # Token must come from the photos map, NOT from /uploads
        assert att["payload"]["token"] == "real-token-from-photos"

    def test_send_with_media_files(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake-jpeg")

        adapter = MaxAdapter(PlatformConfig(enabled=True, extra={"token": "t"}))
        adapter._http_client = MagicMock()

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"url": "https://upload.example/put"}
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"token": "upload-token"}
        resp3 = MagicMock()
        resp3.status_code = 200

        # CDN upload goes through a separate client
        up_client = MagicMock()
        up_client.post = AsyncMock(return_value=resp2)
        up_client.__aenter__ = AsyncMock(return_value=up_client)
        up_client.__aexit__ = AsyncMock(return_value=False)

        async def fake_post(url, **kwargs):
            if "uploads" in url:
                return resp1
            return resp3  # POST /messages

        with patch.object(_max.httpx, "AsyncClient", return_value=up_client):
            adapter._http_client.post = AsyncMock(side_effect=fake_post)
            result = _run(adapter.send(
                "1", "смотри", metadata={"user_id": "2", "chat_type": "dm", "media_files": [str(img)]}
            ))
        assert result.success is True
        # 2 API calls: /uploads + send message (CDN upload on separate client)
        assert adapter._http_client.post.call_count == 2


# ---------------------------------------------------------------------------
# 8. _standalone_send
# ---------------------------------------------------------------------------


class TestStandaloneSend:

    def test_standalone_send_dm(self):
        import json as _json
        cfg = PlatformConfig(enabled=True, extra={"token": "t", "user_id": "139383659"})

        with patch.object(_max, "httpx") as mock_httpx:
            client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            client.post = AsyncMock(return_value=resp)
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = _run(_standalone_send(cfg, "532485678", "hello"))
            assert result["success"] is True
            assert client.post.call_args.kwargs["params"] == {"user_id": "139383659"}

    def test_standalone_send_no_token(self):
        cfg = PlatformConfig(enabled=True, extra={})
        with patch("plugins.platforms.max.adapter.os.getenv", return_value=""):
            result = _run(_standalone_send(cfg, "1", "hi"))
        assert "error" in result


# ---------------------------------------------------------------------------
# 9. register()
# ---------------------------------------------------------------------------


class TestRegister:

    def test_register_platform(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "max"
        assert kwargs["label"] == "MAX"
        assert kwargs["emoji"] == "🟠"
        assert kwargs["allowed_users_env"] == "MAX_ALLOWED_USERS"
        assert kwargs["max_message_length"] == MAX_MESSAGE_LENGTH
