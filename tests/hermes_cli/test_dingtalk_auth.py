"""Unit tests for hermes_cli/dingtalk_auth.py (QR device-flow registration)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# API layer — _api_post + error mapping
# ---------------------------------------------------------------------------


class TestApiPost:

    def test_raises_on_network_error(self):
        import requests
        from hermes_cli.dingtalk_auth import _api_post, RegistrationError

        with patch("hermes_cli.dingtalk_auth.requests.post",
                   side_effect=requests.ConnectionError("nope")):
            with pytest.raises(RegistrationError, match="Network error"):
                _api_post("/app/registration/init", {"source": "hermes"})

    def test_raises_on_nonzero_errcode(self):
        from hermes_cli.dingtalk_auth import _api_post, RegistrationError

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"errcode": 42, "errmsg": "boom"}

        with patch("hermes_cli.dingtalk_auth.requests.post", return_value=mock_resp):
            with pytest.raises(RegistrationError, match=r"boom \(errcode=42\)"):
                _api_post("/app/registration/init", {"source": "hermes"})

    def test_returns_data_on_success(self):
        from hermes_cli.dingtalk_auth import _api_post

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"errcode": 0, "nonce": "abc"}

        with patch("hermes_cli.dingtalk_auth.requests.post", return_value=mock_resp):
            result = _api_post("/app/registration/init", {"source": "hermes"})
            assert result["nonce"] == "abc"


# ---------------------------------------------------------------------------
# begin_registration — 2-step nonce → device_code chain
# ---------------------------------------------------------------------------


class TestBeginRegistration:

    def test_chains_init_then_begin(self):
        from hermes_cli.dingtalk_auth import begin_registration

        responses = [
            {"errcode": 0, "nonce": "nonce123"},
            {
                "errcode": 0,
                "device_code": "dev-xyz",
                "verification_uri_complete": "https://open-dev.dingtalk.com/openapp/registration/openClaw?user_code=ABCD",
                "expires_in": 7200,
                "interval": 2,
            },
        ]
        with patch("hermes_cli.dingtalk_auth._api_post", side_effect=responses):
            result = begin_registration()

        assert result["device_code"] == "dev-xyz"
        assert "verification_uri_complete" in result
        assert result["interval"] == 2
        assert result["expires_in"] == 7200

    def test_missing_nonce_raises(self):
        from hermes_cli.dingtalk_auth import begin_registration, RegistrationError

        with patch("hermes_cli.dingtalk_auth._api_post",
                   return_value={"errcode": 0, "nonce": ""}):
            with pytest.raises(RegistrationError, match="missing nonce"):
                begin_registration()

    def test_missing_device_code_raises(self):
        from hermes_cli.dingtalk_auth import begin_registration, RegistrationError

        responses = [
            {"errcode": 0, "nonce": "n1"},
            {"errcode": 0, "verification_uri_complete": "http://x"},  # no device_code
        ]
        with patch("hermes_cli.dingtalk_auth._api_post", side_effect=responses):
            with pytest.raises(RegistrationError, match="missing device_code"):
                begin_registration()

    def test_missing_verification_uri_raises(self):
        from hermes_cli.dingtalk_auth import begin_registration, RegistrationError

        responses = [
            {"errcode": 0, "nonce": "n1"},
            {"errcode": 0, "device_code": "dev"},  # no verification_uri_complete
        ]
        with patch("hermes_cli.dingtalk_auth._api_post", side_effect=responses):
            with pytest.raises(RegistrationError,
                               match="missing verification_uri_complete"):
                begin_registration()


# ---------------------------------------------------------------------------
# wait_for_registration_success — polling loop
# ---------------------------------------------------------------------------


class TestWaitForSuccess:

    def test_returns_credentials_on_success(self):
        from hermes_cli.dingtalk_auth import wait_for_registration_success

        responses = [
            {"status": "WAITING"},
            {"status": "WAITING"},
            {"status": "SUCCESS", "client_id": "cid-1", "client_secret": "sec-1"},
        ]
        with patch("hermes_cli.dingtalk_auth.poll_registration", side_effect=responses), \
             patch("hermes_cli.dingtalk_auth.time.sleep"):
            cid, secret = wait_for_registration_success(
                device_code="dev", interval=0, expires_in=60
            )
            assert cid == "cid-1"
            assert secret == "sec-1"

    def test_success_without_credentials_raises(self):
        from hermes_cli.dingtalk_auth import wait_for_registration_success, RegistrationError

        with patch("hermes_cli.dingtalk_auth.poll_registration",
                   return_value={"status": "SUCCESS", "client_id": "", "client_secret": ""}), \
             patch("hermes_cli.dingtalk_auth.time.sleep"):
            with pytest.raises(RegistrationError, match="credentials are missing"):
                wait_for_registration_success(
                    device_code="dev", interval=0, expires_in=60
                )

    def test_invokes_waiting_callback(self):
        from hermes_cli.dingtalk_auth import wait_for_registration_success

        callback = MagicMock()
        responses = [
            {"status": "WAITING"},
            {"status": "WAITING"},
            {"status": "SUCCESS", "client_id": "cid", "client_secret": "sec"},
        ]
        with patch("hermes_cli.dingtalk_auth.poll_registration", side_effect=responses), \
             patch("hermes_cli.dingtalk_auth.time.sleep"):
            wait_for_registration_success(
                device_code="dev", interval=0, expires_in=60, on_waiting=callback
            )
        assert callback.call_count == 2


# ---------------------------------------------------------------------------
# QR rendering — terminal output
# ---------------------------------------------------------------------------


class TestRenderQR:

    def test_returns_false_when_qrcode_missing(self, monkeypatch):
        from hermes_cli import dingtalk_auth

        # Simulate qrcode import failure
        monkeypatch.setitem(sys.modules, "qrcode", None)
        assert dingtalk_auth.render_qr_to_terminal("https://example.com") is False

    def test_prints_when_qrcode_available(self, capsys):
        """End-to-end: render a real QR and verify SOMETHING got printed."""
        try:
            import qrcode  # noqa: F401
        except ImportError:
            pytest.skip("qrcode library not available")

        from hermes_cli.dingtalk_auth import render_qr_to_terminal
        result = render_qr_to_terminal("https://example.com/test")
        captured = capsys.readouterr()
        assert result is True
        assert len(captured.out) > 100  # rendered matrix is non-trivial


# ---------------------------------------------------------------------------
# Configuration — env var overrides
# ---------------------------------------------------------------------------


class TestConfigOverrides:

    def test_base_url_default(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_REGISTRATION_BASE_URL", raising=False)
        # Force module reload to pick up current env
        import importlib
        import hermes_cli.dingtalk_auth as mod
        importlib.reload(mod)
        assert mod.REGISTRATION_BASE_URL == "https://oapi.dingtalk.com"

    def test_base_url_override_via_env(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_REGISTRATION_BASE_URL",
                           "https://test.example.com/")
        import importlib
        import hermes_cli.dingtalk_auth as mod
        importlib.reload(mod)
        # Trailing slash stripped
        assert mod.REGISTRATION_BASE_URL == "https://test.example.com"

    def test_source_default(self, monkeypatch):
        monkeypatch.delenv("DINGTALK_REGISTRATION_SOURCE", raising=False)
        import importlib
        import hermes_cli.dingtalk_auth as mod
        importlib.reload(mod)
        assert mod.REGISTRATION_SOURCE == "openClaw"
