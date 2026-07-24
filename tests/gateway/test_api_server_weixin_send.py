"""
Tests for POST /api/weixin/send input validation and audit logging.

Covers:
- chat_id validation (empty, wrong format)
- message validation (empty, too long)
- Content-Type enforcement
- Structured audit log output
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create a minimal aiohttp app with just the weixin send endpoint."""
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/weixin/send", adapter._handle_weixin_send)
    return app


def _fake_weixin_adapter(*, success: bool = True, message_id: str = "mid-123",
                          error: str = "") -> MagicMock:
    """Create a mock WeixinAdapter."""
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(
        success=success, message_id=message_id, error=error,
    ))
    return adapter


VALID_CHAT_ID = "user123@im.wechat"
VALID_GROUP_ID = "group456@chatroom"
VALID_MESSAGE = "hello"


# ---------------------------------------------------------------------------
# Tests: Content-Type
# ---------------------------------------------------------------------------


class TestContentType:
    @pytest.mark.asyncio
    async def test_rejects_non_json_content_type(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                data="not json",
                headers={"Content-Type": "text/plain"},
            )
            body = await resp.json()

        assert resp.status == 415
        assert "application/json" in body["error"]

    @pytest.mark.asyncio
    async def test_accepts_json_content_type(self):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter()
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID, "message": VALID_MESSAGE},
            )
            body = await resp.json()

        assert resp.status == 200
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Tests: chat_id validation
# ---------------------------------------------------------------------------


class TestChatIdValidation:
    @pytest.mark.asyncio
    async def test_missing_chat_id(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"message": "hello"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_chat_id(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": "", "message": "hello"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_chat_id_format_no_at_suffix(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": "user123", "message": "hello"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "@im.wechat" in body["error"] or "@chatroom" in body["error"]

    @pytest.mark.asyncio
    async def test_invalid_chat_id_wrong_suffix(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": "user@slack.com", "message": "hello"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "@im.wechat" in body["error"] or "@chatroom" in body["error"]

    @pytest.mark.asyncio
    async def test_valid_im_wechat_id(self):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter()
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID, "message": VALID_MESSAGE},
            )
            body = await resp.json()

        assert resp.status == 200
        wx.send.assert_awaited_once_with(chat_id=VALID_CHAT_ID, content=VALID_MESSAGE)

    @pytest.mark.asyncio
    async def test_valid_chatroom_id(self):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter()
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_GROUP_ID, "message": VALID_MESSAGE},
            )
            body = await resp.json()

        assert resp.status == 200
        wx.send.assert_awaited_once_with(chat_id=VALID_GROUP_ID, content=VALID_MESSAGE)


# ---------------------------------------------------------------------------
# Tests: message validation
# ---------------------------------------------------------------------------


class TestMessageValidation:
    @pytest.mark.asyncio
    async def test_missing_message(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_message(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID, "message": ""},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_message_too_long(self):
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID, "message": "x" * 4097},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "4096" in body["error"]

    @pytest.mark.asyncio
    async def test_message_at_max_length_accepted(self):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter()
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/weixin/send",
                json={"chat_id": VALID_CHAT_ID, "message": "x" * 4096},
            )
            body = await resp.json()

        assert resp.status == 200
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Tests: audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    @pytest.mark.asyncio
    async def test_audit_log_on_success(self, caplog):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter(success=True, message_id="mid-99")
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/weixin/send",
                    json={"chat_id": VALID_CHAT_ID, "message": "test msg"},
                )

        assert resp.status == 200

        # Find the audit log line
        audit_lines = [r.message for r in caplog.records
                       if "[weixin-send-audit]" in r.message]
        assert len(audit_lines) == 1
        line = audit_lines[0]
        assert "caller=" in line
        assert f"chat={VALID_CHAT_ID}" in line
        assert "len=8" in line
        assert "result=ok" in line

    @pytest.mark.asyncio
    async def test_audit_log_on_failure(self, caplog):
        adapter = _make_adapter()
        wx = _fake_weixin_adapter(success=False, error="timeout")
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/weixin/send",
                    json={"chat_id": VALID_CHAT_ID, "message": "fail msg"},
                )

        assert resp.status == 502

        audit_lines = [r.message for r in caplog.records
                       if "[weixin-send-audit]" in r.message]
        assert len(audit_lines) == 1
        line = audit_lines[0]
        assert "result=send_failed" in line
        assert "error=timeout" in line

    @pytest.mark.asyncio
    async def test_audit_log_on_adapter_unavailable(self, caplog):
        adapter = _make_adapter()
        # No weixin adapter registered
        app = _create_app(adapter)

        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/weixin/send",
                    json={"chat_id": VALID_CHAT_ID, "message": "test"},
                )

        assert resp.status == 503

        audit_lines = [r.message for r in caplog.records
                       if "[weixin-send-audit]" in r.message]
        assert len(audit_lines) == 1
        assert "result=adapter_unavailable" in audit_lines[0]

    @pytest.mark.asyncio
    async def test_audit_log_on_exception(self, caplog):
        adapter = _make_adapter()
        wx = MagicMock()
        wx.send = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)

        with caplog.at_level(logging.INFO, logger="gateway.platforms.api_server"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/weixin/send",
                    json={"chat_id": VALID_CHAT_ID, "message": "boom"},
                )

        assert resp.status == 500

        audit_lines = [r.message for r in caplog.records
                       if "[weixin-send-audit]" in r.message]
        assert len(audit_lines) == 1
        assert "result=exception" in audit_lines[0]
        assert "connection lost" in audit_lines[0]

    @pytest.mark.asyncio
    async def test_audit_log_never_contains_message_content(self, caplog):
        """Privacy: the message body must not appear in any log line."""
        adapter = _make_adapter()
        wx = _fake_weixin_adapter()
        adapter._peer_adapters[Platform.WEIXIN] = wx
        app = _create_app(adapter)
        secret = "super-secret-password-12345"

        with caplog.at_level(logging.DEBUG, logger="gateway.platforms.api_server"):
            async with TestClient(TestServer(app)) as cli:
                await cli.post(
                    "/api/weixin/send",
                    json={"chat_id": VALID_CHAT_ID, "message": secret},
                )

        for record in caplog.records:
            assert secret not in record.message, (
                f"Message content leaked in log: {record.message}"
            )
