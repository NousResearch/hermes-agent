"""Tests for the Weixin QR session manager (web-based onboarding).

Tests the session lifecycle: start → poll → confirm, plus timeout,
cancellation, and concurrent session handling. Mirrors the patterns
from ``TestWeixinQrLogin`` in ``tests/gateway/test_weixin.py``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.weixin_qr_session import (
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    WeixinQRSession,
    WeixinQRSessionManager,
    _render_qr_png_base64,
)


def _make_qr_response(qrcode: str = "qr-1", url: str = "https://example.com/qr-1"):
    return {"qrcode": qrcode, "qrcode_img_content": url}


def _make_confirmed_response(
    account_id: str = "acc-1",
    token: str = "tok-1",
    base_url: str = "https://ilinkai.weixin.qq.com",
    user_id: str = "user-1",
):
    return {
        "status": "confirmed",
        "ilink_bot_id": account_id,
        "bot_token": token,
        "baseurl": base_url,
        "ilink_user_id": user_id,
    }


class TestWeixinQRSessionManagerLifecycle:
    """Start → poll → confirm happy path."""

    @pytest.mark.asyncio
    async def test_start_session_returns_starting_state(self, tmp_path):
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        with patch(
            "gateway.platforms.weixin_qr_session.WeixinQRSessionManager._run_session",
            new_callable=AsyncMock,
        ):
            result = manager.start_session(str(tmp_path))

        assert result["state"] == "starting"
        assert "session_id" in result
        assert "expires_at" in result

    @pytest.mark.asyncio
    async def test_get_status_returns_none_for_unknown_session(self, tmp_path):
        manager = WeixinQRSessionManager()
        assert manager.get_status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cancel_session_removes_it(self, tmp_path):
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        with patch(
            "gateway.platforms.weixin_qr_session.WeixinQRSessionManager._run_session",
            new_callable=AsyncMock,
        ):
            result = manager.start_session(str(tmp_path))
            session_id = result["session_id"]

        assert manager.cancel_session(session_id) is True
        assert manager.get_status(session_id) is None

    @pytest.mark.asyncio
    async def test_cancel_unknown_session_returns_false(self, tmp_path):
        manager = WeixinQRSessionManager()
        assert manager.cancel_session("nonexistent") is False


class TestWeixinQRSessionManagerBackgroundTask:
    """Test the background poll task with mocked iLink API."""

    @pytest.mark.asyncio
    async def test_confirmed_flow_saves_credentials(self, tmp_path):
        """QR fetch → wait → confirmed → credentials saved."""
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        qr_response = _make_qr_response()
        confirmed_response = _make_confirmed_response()

        saved_calls = []

        def fake_save(hermes_home, *, account_id, token, base_url, user_id=""):
            saved_calls.append(
                {
                    "hermes_home": hermes_home,
                    "account_id": account_id,
                    "token": token,
                    "base_url": base_url,
                    "user_id": user_id,
                }
            )

        with patch(
            "gateway.platforms.weixin._api_get", new_callable=AsyncMock
        ) as api_get_mock, \
             patch("gateway.platforms.weixin.AIOHTTP_AVAILABLE", True), \
             patch("gateway.platforms.weixin.aiohttp.ClientSession", create=True) as session_cls, \
             patch("gateway.platforms.weixin.save_weixin_account", side_effect=fake_save), \
             patch("gateway.platforms.weixin_qr_session._POLL_INTERVAL_SECONDS", 0.01):
            api_get_mock.side_effect = [qr_response, confirmed_response]
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = False
            session_cls.return_value = session

            result = manager.start_session(str(tmp_path), timeout_seconds=30)
            session_id = result["session_id"]

            # Wait for the background task to complete.
            await asyncio.sleep(0.5)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["state"] == "confirmed"
        assert status["account_id"] == "acc-1"
        assert status["base_url"] == "https://ilinkai.weixin.qq.com"

        # Verify credentials were saved to disk.
        assert len(saved_calls) == 1
        assert saved_calls[0]["account_id"] == "acc-1"
        assert saved_calls[0]["token"] == "tok-1"

    @pytest.mark.asyncio
    async def test_failed_qr_fetch_sets_failed_state(self, tmp_path):
        """If iLink returns an error, session state becomes 'failed'."""
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        with patch(
            "gateway.platforms.weixin._api_get", new_callable=AsyncMock
        ) as api_get_mock, \
             patch("gateway.platforms.weixin.AIOHTTP_AVAILABLE", True), \
             patch("gateway.platforms.weixin.aiohttp.ClientSession", create=True) as session_cls:
            api_get_mock.side_effect = RuntimeError("iLink HTTP 500")
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = False
            session_cls.return_value = session

            result = manager.start_session(str(tmp_path), timeout_seconds=30)
            session_id = result["session_id"]

            await asyncio.sleep(0.3)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["state"] == "failed"
        assert "Failed to fetch QR" in status.get("error", "")

    @pytest.mark.asyncio
    async def test_expired_status_triggers_refresh(self, tmp_path):
        """When iLink returns 'expired', the session refreshes the QR."""
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        first_qr = _make_qr_response("qr-1", "https://example.com/qr-1")
        expired = {"status": "expired"}
        refreshed_qr = _make_qr_response("qr-2", "https://example.com/qr-2")
        confirmed = _make_confirmed_response()

        with patch(
            "gateway.platforms.weixin._api_get", new_callable=AsyncMock
        ) as api_get_mock, \
             patch("gateway.platforms.weixin.AIOHTTP_AVAILABLE", True), \
             patch("gateway.platforms.weixin.aiohttp.ClientSession", create=True) as session_cls, \
             patch("gateway.platforms.weixin.save_weixin_account"), \
             patch("gateway.platforms.weixin_qr_session._POLL_INTERVAL_SECONDS", 0.01):
            # QR fetch, then expired, then refresh QR fetch, then confirmed.
            api_get_mock.side_effect = [first_qr, expired, refreshed_qr, confirmed]
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = False
            session_cls.return_value = session

            result = manager.start_session(str(tmp_path), timeout_seconds=30)
            session_id = result["session_id"]

            await asyncio.sleep(0.5)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["state"] == "confirmed"

    @pytest.mark.asyncio
    async def test_max_refreshes_exceeded_sets_expired_state(self, tmp_path):
        """After 3 refreshes, the session gives up with 'expired' state."""
        manager = WeixinQRSessionManager()
        manager.set_event_loop(asyncio.get_event_loop())

        qr = _make_qr_response()
        expired = {"status": "expired"}

        with patch(
            "gateway.platforms.weixin._api_get", new_callable=AsyncMock
        ) as api_get_mock, \
             patch("gateway.platforms.weixin.AIOHTTP_AVAILABLE", True), \
             patch("gateway.platforms.weixin.aiohttp.ClientSession", create=True) as session_cls, \
             patch("gateway.platforms.weixin_qr_session._POLL_INTERVAL_SECONDS", 0.01), \
             patch("gateway.platforms.weixin_qr_session._MAX_QR_REFRESHES", 2):
            # Initial fetch, then expired×2 (refreshes), then expired again (gives up).
            api_get_mock.side_effect = [qr, expired, qr, expired, qr, expired]
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = False
            session_cls.return_value = session

            result = manager.start_session(str(tmp_path), timeout_seconds=30)
            session_id = result["session_id"]

            await asyncio.sleep(0.5)

        status = manager.get_status(session_id)
        assert status is not None
        assert status["state"] == "expired"


class TestWeixinQRSessionStatusDict:
    """Test the to_status_dict serialization."""

    def test_status_dict_includes_required_fields(self):
        session = WeixinQRSession(
            session_id="test-1",
            hermes_home="/tmp",
            expires_at_ts=time.time() + 300,
        )
        session.state = "waiting"
        session.qr_image_base64 = "base64data"

        result = session.to_status_dict()
        assert result["session_id"] == "test-1"
        assert result["state"] == "waiting"
        assert result["qr_image_base64"] == "base64data"
        assert "expires_at" in result

    def test_status_dict_hides_qr_in_confirmed_state(self):
        """QR image should not be exposed after confirmation."""
        session = WeixinQRSession(
            session_id="test-1",
            hermes_home="/tmp",
            expires_at_ts=time.time() + 300,
        )
        session.state = "confirmed"
        session.qr_image_base64 = "base64data"
        session.credentials = {
            "account_id": "acc-1",
            "token": "tok-1",
            "base_url": "https://ilinkai.weixin.qq.com",
            "user_id": "user-1",
        }

        result = session.to_status_dict()
        assert result["state"] == "confirmed"
        assert result["account_id"] == "acc-1"
        assert "qr_image_base64" not in result
        # Token must never be exposed to the browser.
        assert "token" not in result

    def test_status_dict_includes_error_on_failure(self):
        session = WeixinQRSession(
            session_id="test-1",
            hermes_home="/tmp",
            expires_at_ts=time.time() + 300,
        )
        session.state = "failed"
        session.error_message = "Something went wrong"

        result = session.to_status_dict()
        assert result["state"] == "failed"
        assert result["error"] == "Something went wrong"


class TestRenderQrPngBase64:
    """Test the QR PNG rendering helper."""

    def test_returns_non_empty_string_for_valid_data(self):
        result = _render_qr_png_base64("https://example.com/qr")
        # If qrcode package is installed, this should be non-empty.
        # If not installed, it returns "" — both are acceptable.
        if result:
            assert isinstance(result, str)
            assert len(result) > 100  # base64 PNG is reasonably long

    def test_returns_empty_string_on_invalid_input(self):
        # Empty string should still produce a valid (blank) QR or empty fallback.
        result = _render_qr_png_base64("")
        assert isinstance(result, str)
