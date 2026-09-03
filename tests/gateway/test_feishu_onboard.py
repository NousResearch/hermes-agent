"""Tests for plugins.platforms.feishu.adapter — Feishu scan-to-create registration."""

import json
from unittest.mock import patch, MagicMock
import pytest


def _mock_urlopen(response_data, status=200):
    """Create a mock for urllib.request.urlopen that returns JSON response_data."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_response.status = status
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


class TestPostRegistration:
    """Tests for the low-level HTTP helper."""


    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_post_registration_sends_form_encoded_body(self, mock_urlopen_fn):
        from plugins.platforms.feishu.adapter import _post_registration

        mock_urlopen_fn.return_value = _mock_urlopen({})
        _post_registration("https://accounts.feishu.cn", {"action": "init", "key": "val"})
        call_args = mock_urlopen_fn.call_args
        request = call_args[0][0]
        body = request.data.decode("utf-8")
        assert "action=init" in body
        assert "key=val" in body
        assert request.get_header("Content-type") == "application/x-www-form-urlencoded"


class TestInitRegistration:
    """Tests for the init step."""

    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_init_succeeds_when_client_secret_supported(self, mock_urlopen_fn):
        from plugins.platforms.feishu.adapter import _init_registration

        mock_urlopen_fn.return_value = _mock_urlopen({
            "nonce": "abc",
            "supported_auth_methods": ["client_secret"],
        })
        _init_registration("feishu")

    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_init_raises_when_client_secret_not_supported(self, mock_urlopen_fn):
        from plugins.platforms.feishu.adapter import _init_registration

        mock_urlopen_fn.return_value = _mock_urlopen({
            "nonce": "abc",
            "supported_auth_methods": ["other_method"],
        })
        with pytest.raises(RuntimeError, match="client_secret"):
            _init_registration("feishu")


class TestBeginRegistration:
    """Tests for the begin step."""

    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_begin_returns_device_code_and_qr_url(self, mock_urlopen_fn):
        from plugins.platforms.feishu.adapter import _begin_registration

        mock_urlopen_fn.return_value = _mock_urlopen({
            "device_code": "dc_123",
            "verification_uri_complete": "https://accounts.feishu.cn/qr/abc",
            "user_code": "ABCD-1234",
            "interval": 5,
            "expire_in": 600,
        })
        result = _begin_registration("feishu")
        assert result["device_code"] == "dc_123"
        assert "qr_url" in result
        assert "accounts.feishu.cn" in result["qr_url"]
        assert result["user_code"] == "ABCD-1234"
        assert result["interval"] == 5
        assert result["expire_in"] == 600


class TestPollRegistration:
    """Tests for the poll step."""

    @patch("plugins.platforms.feishu.adapter.time")
    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_poll_returns_credentials_on_success(self, mock_urlopen_fn, mock_time):
        from plugins.platforms.feishu.adapter import _poll_registration

        mock_time.monotonic.side_effect = [0, 1]
        mock_time.sleep = MagicMock()

        mock_urlopen_fn.return_value = _mock_urlopen({
            "client_id": "cli_app123",
            "client_secret": "secret456",
            "user_info": {"open_id": "ou_owner", "tenant_brand": "feishu"},
        })
        result = _poll_registration(
            device_code="dc_123", interval=1, expire_in=60, domain="feishu"
        )
        assert result is not None
        assert result["app_id"] == "cli_app123"
        assert result["app_secret"] == "secret456"
        assert result["domain"] == "feishu"
        assert result["open_id"] == "ou_owner"

    @patch("plugins.platforms.feishu.adapter.time")
    @patch("plugins.platforms.feishu.adapter.urlopen")
    def test_poll_switches_domain_on_lark_tenant_brand(self, mock_urlopen_fn, mock_time):
        from plugins.platforms.feishu.adapter import _poll_registration

        mock_time.monotonic.side_effect = [0, 1, 2]
        mock_time.sleep = MagicMock()

        pending_resp = _mock_urlopen({
            "error": "authorization_pending",
            "user_info": {"tenant_brand": "lark"},
        })
        success_resp = _mock_urlopen({
            "client_id": "cli_lark",
            "client_secret": "secret_lark",
            "user_info": {"open_id": "ou_lark", "tenant_brand": "lark"},
        })
        mock_urlopen_fn.side_effect = [pending_resp, success_resp]

        result = _poll_registration(
            device_code="dc_123", interval=0, expire_in=60, domain="feishu"
        )
        assert result is not None
        assert result["domain"] == "lark"


class TestRenderQr:
    """Tests for QR code terminal rendering."""

    @patch("plugins.platforms.feishu.adapter._qrcode_mod", create=True)
    def test_render_qr_returns_true_on_success(self, mock_qrcode_mod):
        from plugins.platforms.feishu.adapter import _render_qr

        mock_qr = MagicMock()
        mock_qrcode_mod.QRCode.return_value = mock_qr
        assert _render_qr("https://example.com/qr") is True
        mock_qr.add_data.assert_called_once_with("https://example.com/qr")
        mock_qr.make.assert_called_once_with(fit=True)
        mock_qr.print_ascii.assert_called_once()


class TestProbeBot:
    """Tests for bot connectivity verification."""

    @patch("plugins.platforms.feishu.adapter.FEISHU_AVAILABLE", True)
    def test_probe_returns_bot_info_on_success(self):
        from plugins.platforms.feishu.adapter import probe_bot

        with patch("plugins.platforms.feishu.adapter._probe_bot_sdk") as mock_sdk:
            mock_sdk.return_value = {"bot_name": "TestBot", "bot_open_id": "ou_bot123"}
            result = probe_bot("cli_app", "secret", "feishu")

        assert result is not None
        assert result["bot_name"] == "TestBot"
        assert result["bot_open_id"] == "ou_bot123"


class TestQrRegister:
    """Tests for the public qr_register entry point."""

    @patch("plugins.platforms.feishu.adapter.probe_bot")
    @patch("plugins.platforms.feishu.adapter._render_qr")
    @patch("plugins.platforms.feishu.adapter._poll_registration")
    @patch("plugins.platforms.feishu.adapter._begin_registration")
    @patch("plugins.platforms.feishu.adapter._init_registration")
    def test_qr_register_success_flow(
        self, mock_init, mock_begin, mock_poll, mock_render, mock_probe
    ):
        from plugins.platforms.feishu.adapter import qr_register

        mock_begin.return_value = {
            "device_code": "dc_123",
            "qr_url": "https://example.com/qr",
            "user_code": "ABCD",
            "interval": 1,
            "expire_in": 60,
        }
        mock_poll.return_value = {
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
            "open_id": "ou_owner",
        }
        mock_probe.return_value = {"bot_name": "MyBot", "bot_open_id": "ou_bot"}

        result = qr_register()
        assert result is not None
        assert result["app_id"] == "cli_app"
        assert result["app_secret"] == "secret"
        assert result["bot_name"] == "MyBot"
        mock_init.assert_called_once()
        mock_render.assert_called_once()


    @patch("plugins.platforms.feishu.adapter._render_qr")
    @patch("plugins.platforms.feishu.adapter._poll_registration")
    @patch("plugins.platforms.feishu.adapter._begin_registration")
    @patch("plugins.platforms.feishu.adapter._init_registration")
    def test_qr_register_returns_none_on_poll_failure(
        self, mock_init, mock_begin, mock_poll, mock_render
    ):
        from plugins.platforms.feishu.adapter import qr_register

        mock_begin.return_value = {
            "device_code": "dc_123",
            "qr_url": "https://example.com/qr",
            "user_code": "ABCD",
            "interval": 1,
            "expire_in": 60,
        }
        mock_poll.return_value = None

        result = qr_register()
        assert result is None

    # -- Contract: expected errors → None, unexpected errors → propagate --


    # -- Negative paths: partial/malformed server responses --


    @patch("plugins.platforms.feishu.adapter.probe_bot")
    @patch("plugins.platforms.feishu.adapter._render_qr")
    @patch("plugins.platforms.feishu.adapter._poll_registration")
    @patch("plugins.platforms.feishu.adapter._begin_registration")
    @patch("plugins.platforms.feishu.adapter._init_registration")
    def test_qr_register_succeeds_even_when_probe_fails(
        self, mock_init, mock_begin, mock_poll, mock_render, mock_probe
    ):
        """Registration succeeds but probe fails → result with bot_name=None."""
        from plugins.platforms.feishu.adapter import qr_register

        mock_begin.return_value = {
            "device_code": "dc_123",
            "qr_url": "https://example.com/qr",
            "user_code": "ABCD",
            "interval": 1,
            "expire_in": 60,
        }
        mock_poll.return_value = {
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
            "open_id": "ou_owner",
        }
        mock_probe.return_value = None  # probe failed

        result = qr_register()
        assert result is not None
        assert result["app_id"] == "cli_app"
        assert result["bot_name"] is None
        assert result["bot_open_id"] is None
