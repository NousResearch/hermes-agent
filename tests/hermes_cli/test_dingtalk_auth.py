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


# ---------------------------------------------------------------------------
# poll_registration — single-shot status check
# ---------------------------------------------------------------------------


class TestPollRegistration:

    def test_returns_normalized_dict_for_waiting(self):
        from hermes_cli.dingtalk_auth import poll_registration

        with patch("hermes_cli.dingtalk_auth._api_post",
                   return_value={"errcode": 0, "status": "waiting"}):
            result = poll_registration("dev")
        assert result["status"] == "WAITING"  # upper-cased
        assert result["client_id"] is None
        assert result["client_secret"] is None
        assert result["fail_reason"] is None

    def test_returns_credentials_for_success(self):
        from hermes_cli.dingtalk_auth import poll_registration

        with patch("hermes_cli.dingtalk_auth._api_post",
                   return_value={"errcode": 0, "status": "SUCCESS",
                                 "client_id": "cid", "client_secret": "sec"}):
            result = poll_registration("dev")
        assert result["status"] == "SUCCESS"
        assert result["client_id"] == "cid"
        assert result["client_secret"] == "sec"

    def test_unknown_status_normalized_to_unknown(self):
        from hermes_cli.dingtalk_auth import poll_registration

        with patch("hermes_cli.dingtalk_auth._api_post",
                   return_value={"errcode": 0, "status": "weird-future-state"}):
            result = poll_registration("dev")
        assert result["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# wait_for_registration_success — transient-error and terminal-failure paths
# ---------------------------------------------------------------------------


class TestWaitForSuccessFailurePaths:

    def test_transient_poll_error_within_retry_window_continues(self):
        """A short burst of RegistrationError should be retried, then succeed."""
        from hermes_cli.dingtalk_auth import wait_for_registration_success, RegistrationError

        side_effects = [
            RegistrationError("transient 1"),
            RegistrationError("transient 2"),
            {"status": "SUCCESS", "client_id": "cid", "client_secret": "sec"},
        ]
        with patch("hermes_cli.dingtalk_auth.poll_registration", side_effect=side_effects), \
             patch("hermes_cli.dingtalk_auth.time.sleep"):
            cid, secret = wait_for_registration_success(
                device_code="dev", interval=0, expires_in=60
            )
        assert cid == "cid"
        assert secret == "sec"

    def test_persistent_poll_error_eventually_raises(self):
        """After the retry window expires, the error should propagate."""
        from hermes_cli.dingtalk_auth import wait_for_registration_success, RegistrationError

        with patch("hermes_cli.dingtalk_auth.poll_registration",
                   side_effect=RegistrationError("permanent")), \
             patch("hermes_cli.dingtalk_auth.time.sleep"), \
             patch("hermes_cli.dingtalk_auth.time.monotonic",
                   side_effect=[0, 1, 2, 200, 400]):
            with pytest.raises(RegistrationError, match="permanent"):
                wait_for_registration_success(
                    device_code="dev", interval=0, expires_in=600
                )

    def test_fail_status_raises_with_reason(self):
        """A terminal FAIL status (after retry window) propagates the reason."""
        from hermes_cli.dingtalk_auth import wait_for_registration_success, RegistrationError

        with patch("hermes_cli.dingtalk_auth.poll_registration",
                   return_value={"status": "FAIL", "fail_reason": "user denied",
                                 "client_id": None, "client_secret": None}), \
             patch("hermes_cli.dingtalk_auth.time.sleep"), \
             patch("hermes_cli.dingtalk_auth.time.monotonic",
                   side_effect=[0, 1, 2, 200, 400]):
            with pytest.raises(RegistrationError, match="user denied"):
                wait_for_registration_success(
                    device_code="dev", interval=0, expires_in=600
                )

    def test_overall_timeout_raises(self):
        """If the deadline is reached without SUCCESS, raise a timeout error."""
        from hermes_cli.dingtalk_auth import wait_for_registration_success, RegistrationError

        with patch("hermes_cli.dingtalk_auth.poll_registration",
                   return_value={"status": "WAITING", "client_id": None,
                                 "client_secret": None, "fail_reason": None}), \
             patch("hermes_cli.dingtalk_auth.time.sleep"), \
             patch("hermes_cli.dingtalk_auth.time.monotonic",
                   side_effect=[0, 100, 200]):
            with pytest.raises(RegistrationError, match="timed out"):
                wait_for_registration_success(
                    device_code="dev", interval=0, expires_in=50
                )


# ---------------------------------------------------------------------------
# _ensure_qrcode_installed — already-installed / install-success / install-fail
# ---------------------------------------------------------------------------


class TestEnsureQrcodeInstalled:

    def test_returns_true_when_already_installed(self):
        """A real qrcode (or any object) in sys.modules short-circuits install."""
        from hermes_cli import dingtalk_auth

        # Inject a fake qrcode module — import succeeds, no install attempted.
        fake_qrcode = MagicMock()
        with patch.dict(sys.modules, {"qrcode": fake_qrcode}), \
             patch("hermes_cli.tools_config._pip_install") as mock_install:
            assert dingtalk_auth._ensure_qrcode_installed() is True
        mock_install.assert_not_called()

    def test_falls_through_to_install_when_missing(self):
        """When import fails, the function installs via _pip_install and
        re-imports. `_pip_install` (not subprocess.check_call — the real
        function delegates to it via hermes_cli.tools_config, which shells
        out with subprocess.run) is the actual seam to mock."""
        from hermes_cli import dingtalk_auth

        fake_qrcode = MagicMock()
        install_calls = []

        def fake_pip_install(args, **kwargs):
            install_calls.append(args)
            # After the install "succeeds", inject the module so the
            # function's re-import finds it.
            sys.modules["qrcode"] = fake_qrcode
            return MagicMock(returncode=0)

        # None in sys.modules is CPython's documented sentinel for "this
        # import is known to fail" — it forces ImportError deterministically
        # regardless of whether qrcode happens to be really installed in
        # the venv running this test.
        with patch.dict(sys.modules, {"qrcode": None}, clear=False), \
             patch("hermes_cli.tools_config._pip_install", side_effect=fake_pip_install):
            assert dingtalk_auth._ensure_qrcode_installed() is True
        assert any("qrcode" in args for args in install_calls)

    def test_returns_false_when_both_installers_fail(self):
        """If _pip_install exhausts uv/pip and reports failure, return False
        without raising — and without a real install attempt reaching the
        network (this is what the un-awaited subprocess.check_call mock
        used to miss: the real code path shells out via _pip_install)."""
        from hermes_cli import dingtalk_auth

        # None in sys.modules forces the initial (and, since _pip_install
        # reports failure below, only) import attempt to raise ImportError.
        with patch.dict(sys.modules, {"qrcode": None}, clear=False), \
             patch("hermes_cli.tools_config._pip_install",
                   return_value=MagicMock(returncode=1)):
            assert dingtalk_auth._ensure_qrcode_installed() is False


# ---------------------------------------------------------------------------
# render_qr_to_terminal — true rendering path with a synthetic qrcode module
# ---------------------------------------------------------------------------


def _make_fake_qrcode_module():
    """Build a minimal stand-in for the ``qrcode`` library.

    The real qrcode wheel may not be installed in this venv.  The renderer
    only uses ``qrcode.QRCode(...)``, ``qrcode.constants.ERROR_CORRECT_L``,
    and the ``add_data``/``make``/``get_matrix`` methods, so we stub exactly
    those.
    """
    fake = MagicMock()
    fake.constants.ERROR_CORRECT_L = "L"

    class _FakeQR:
        def __init__(self, *_, **__):
            self._data: str = ""
            # 4x4 matrix that exercises every branch of the half-block encoder
            # (top-only, bottom-only, both, neither).
            self._matrix = [
                [True, False, True, False],
                [False, True, True, False],
                [True, True, False, False],
                [False, False, False, True],
            ]

        def add_data(self, data: str) -> None:
            self._data = data

        def make(self, fit: bool = True) -> None:  # noqa: ARG002
            pass

        def get_matrix(self):
            return self._matrix

    fake.QRCode = _FakeQR
    return fake


class TestRenderQRTrueRendering:

    def test_renders_matrix_when_qrcode_present(self, capsys):
        """Inject a synthetic qrcode module and verify a matrix is printed."""
        from hermes_cli import dingtalk_auth

        with patch.dict(sys.modules, {"qrcode": _make_fake_qrcode_module()}):
            ok = dingtalk_auth.render_qr_to_terminal("https://example.com")
        assert ok is True
        captured = capsys.readouterr()
        # The half-block encoder uses U+2580/U+2584/U+2588; at least one must
        # appear in the output for a non-trivial matrix.
        assert any(ch in captured.out for ch in "▀▄█")


# ---------------------------------------------------------------------------
# dingtalk_qr_auth — orchestration: begin → render → wait → return creds
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_setup_printers(monkeypatch):
    """Provide a stand-in ``hermes_cli.setup`` so dingtalk_qr_auth can import
    its print_* helpers without pulling in the real wizard."""
    fake_setup = MagicMock()
    fake_setup.print_info = MagicMock()
    fake_setup.print_success = MagicMock()
    fake_setup.print_warning = MagicMock()
    fake_setup.print_error = MagicMock()
    monkeypatch.setitem(sys.modules, "hermes_cli.setup", fake_setup)
    return fake_setup


class TestDingtalkQrAuth:

    def test_returns_credentials_on_happy_path(self, _stub_setup_printers):
        from hermes_cli import dingtalk_auth

        begin_payload = {
            "device_code": "dev-abc",
            "verification_uri_complete": "https://example/auth",
            "expires_in": 7200,
            "interval": 2,
        }
        with patch.object(dingtalk_auth, "begin_registration", return_value=begin_payload), \
             patch.object(dingtalk_auth, "_ensure_qrcode_installed", return_value=True), \
             patch.object(dingtalk_auth, "render_qr_to_terminal", return_value=True), \
             patch.object(dingtalk_auth, "wait_for_registration_success",
                          return_value=("cid-final", "secret-final-extra")):
            result = dingtalk_auth.dingtalk_qr_auth()
        assert result == ("cid-final", "secret-final-extra")

    def test_returns_none_when_begin_fails(self, _stub_setup_printers):
        from hermes_cli import dingtalk_auth
        from hermes_cli.dingtalk_auth import RegistrationError

        with patch.object(dingtalk_auth, "begin_registration",
                          side_effect=RegistrationError("init failed")):
            assert dingtalk_auth.dingtalk_qr_auth() is None
        _stub_setup_printers.print_error.assert_called()

    def test_returns_none_when_wait_fails(self, _stub_setup_printers):
        from hermes_cli import dingtalk_auth
        from hermes_cli.dingtalk_auth import RegistrationError

        begin_payload = {
            "device_code": "dev",
            "verification_uri_complete": "https://example/auth",
            "expires_in": 60,
            "interval": 2,
        }
        with patch.object(dingtalk_auth, "begin_registration", return_value=begin_payload), \
             patch.object(dingtalk_auth, "_ensure_qrcode_installed", return_value=True), \
             patch.object(dingtalk_auth, "render_qr_to_terminal", return_value=True), \
             patch.object(dingtalk_auth, "wait_for_registration_success",
                          side_effect=RegistrationError("user cancelled")):
            assert dingtalk_auth.dingtalk_qr_auth() is None
        _stub_setup_printers.print_error.assert_called()

    def test_warns_when_qrcode_install_unavailable(self, _stub_setup_printers):
        """Even if qrcode can't be installed, the flow proceeds and warns."""
        from hermes_cli import dingtalk_auth

        begin_payload = {
            "device_code": "dev",
            "verification_uri_complete": "https://example/auth",
            "expires_in": 60,
            "interval": 2,
        }
        with patch.object(dingtalk_auth, "begin_registration", return_value=begin_payload), \
             patch.object(dingtalk_auth, "_ensure_qrcode_installed", return_value=False), \
             patch.object(dingtalk_auth, "render_qr_to_terminal", return_value=False), \
             patch.object(dingtalk_auth, "wait_for_registration_success",
                          return_value=("cid", "secretsecret")):
            result = dingtalk_auth.dingtalk_qr_auth()
        assert result == ("cid", "secretsecret")
        _stub_setup_printers.print_warning.assert_called()
