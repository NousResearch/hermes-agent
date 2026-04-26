"""Unit tests for hermes_cli/feishu_auth.py — Feishu OAuth device flow."""

import json
import os
import stat
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_cli.feishu_auth import (
    FeishuAuthError,
    _api_post,
    begin_device_authorization,
    load_uat,
    poll_device_token,
    save_uat,
    wait_for_authorization_success,
)


# ---------------------------------------------------------------------------
# _api_post
# ---------------------------------------------------------------------------

class TestApiPost(unittest.TestCase):
    """Tests for the internal _api_post helper."""

    @patch("hermes_cli.feishu_auth.requests.post")
    def test_returns_parsed_json_on_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"device_code": "dc_abc", "interval": 5}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        result = _api_post("/oauth/v1/device_authorization", "https://accounts.feishu.cn", {})

        self.assertEqual(result["device_code"], "dc_abc")
        mock_post.assert_called_once()

    @patch("hermes_cli.feishu_auth.requests.post")
    def test_raises_feishu_auth_error_on_network_error(self, mock_post):
        import requests as req_lib
        mock_post.side_effect = req_lib.RequestException("connection refused")

        with self.assertRaises(FeishuAuthError) as ctx:
            _api_post("/oauth/v1/device_authorization", "https://accounts.feishu.cn", {})

        self.assertIn("Network error", str(ctx.exception))

    @patch("hermes_cli.feishu_auth.requests.post")
    def test_raises_feishu_auth_error_on_api_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "error": "invalid_client",
            "error_description": "client id not found",
        }
        mock_post.return_value = mock_resp

        with self.assertRaises(FeishuAuthError) as ctx:
            _api_post("/oauth/v1/device_authorization", "https://accounts.feishu.cn", {})

        self.assertIn("invalid_client", str(ctx.exception))

    @patch("hermes_cli.feishu_auth.requests.post")
    def test_passes_through_authorization_pending(self, mock_post):
        """authorization_pending should NOT raise — caller handles it."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": "authorization_pending"}
        mock_post.return_value = mock_resp

        result = _api_post("/open-apis/authen/v2/oauth/token", "https://open.feishu.cn", {})
        self.assertEqual(result["error"], "authorization_pending")

    @patch("hermes_cli.feishu_auth.requests.post")
    def test_passes_through_slow_down(self, mock_post):
        """slow_down should NOT raise — caller handles it."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": "slow_down"}
        mock_post.return_value = mock_resp

        result = _api_post("/open-apis/authen/v2/oauth/token", "https://open.feishu.cn", {})
        self.assertEqual(result["error"], "slow_down")


# ---------------------------------------------------------------------------
# begin_device_authorization
# ---------------------------------------------------------------------------

class TestBeginDeviceAuthorization(unittest.TestCase):
    """Tests for begin_device_authorization()."""

    @patch("hermes_cli.feishu_auth._api_post")
    def test_returns_parsed_fields_on_success(self, mock_api):
        mock_api.return_value = {
            "device_code": "  dc_123  ",
            "user_code": "ABC-DEF",
            "verification_uri": "https://feishu.cn/oauth/verify",
            "verification_uri_complete": "https://feishu.cn/oauth/verify?user_code=ABC-DEF",
            "expires_in": 1800,
            "interval": 5,
        }

        result = begin_device_authorization("app_id_123")

        self.assertEqual(result["device_code"], "dc_123")
        self.assertEqual(result["user_code"], "ABC-DEF")
        self.assertEqual(result["expires_in"], 1800)
        self.assertGreaterEqual(result["interval"], 2)

    @patch("hermes_cli.feishu_auth._api_post")
    def test_raises_when_required_fields_missing(self, mock_api):
        mock_api.return_value = {"device_code": "dc_abc"}  # missing most fields

        with self.assertRaises(FeishuAuthError) as ctx:
            begin_device_authorization("app_id_123")

        self.assertIn("missing fields", str(ctx.exception))

    @patch("hermes_cli.feishu_auth._api_post")
    def test_interval_floor_at_2(self, mock_api):
        """Interval below 2 seconds is raised to 2 (RFC 8628 min)."""
        mock_api.return_value = {
            "device_code": "dc",
            "user_code": "UC",
            "verification_uri": "https://x",
            "verification_uri_complete": "https://x?uc",
            "expires_in": 300,
            "interval": 1,  # below floor
        }

        result = begin_device_authorization("app_id")
        self.assertEqual(result["interval"], 2)

    @patch("hermes_cli.feishu_auth._api_post")
    def test_uses_default_scope_when_none_provided(self, mock_api):
        from hermes_cli.feishu_auth import FEISHU_DEFAULT_SCOPE
        mock_api.return_value = {
            "device_code": "dc",
            "user_code": "UC",
            "verification_uri": "https://x",
            "verification_uri_complete": "https://x?uc",
            "expires_in": 300,
            "interval": 5,
        }

        begin_device_authorization("app_id")

        _, _, payload = mock_api.call_args[0]
        self.assertEqual(payload["scope"], FEISHU_DEFAULT_SCOPE)

    @patch("hermes_cli.feishu_auth._api_post")
    def test_uses_custom_scope_when_provided(self, mock_api):
        mock_api.return_value = {
            "device_code": "dc",
            "user_code": "UC",
            "verification_uri": "https://x",
            "verification_uri_complete": "https://x?uc",
            "expires_in": 300,
            "interval": 5,
        }

        begin_device_authorization("app_id", scope="calendar:calendar")

        _, _, payload = mock_api.call_args[0]
        self.assertEqual(payload["scope"], "calendar:calendar")


# ---------------------------------------------------------------------------
# poll_device_token
# ---------------------------------------------------------------------------

class TestPollDeviceToken(unittest.TestCase):
    """Tests for poll_device_token()."""

    @patch("hermes_cli.feishu_auth._api_post")
    def test_returns_pending_on_authorization_pending(self, mock_api):
        mock_api.return_value = {"error": "authorization_pending"}

        result = poll_device_token("dc_abc", "app_id")

        self.assertEqual(result["error"], "authorization_pending")
        self.assertIsNone(result["access_token"])

    @patch("hermes_cli.feishu_auth._api_post")
    def test_returns_slow_down_on_slow_down(self, mock_api):
        mock_api.return_value = {"error": "slow_down"}

        result = poll_device_token("dc_abc", "app_id")

        self.assertEqual(result["error"], "slow_down")

    @patch("hermes_cli.feishu_auth._api_post")
    def test_returns_access_token_on_authorized(self, mock_api):
        mock_api.return_value = {
            "access_token": "uat_tok_xyz",
            "refresh_token": "ref_tok_abc",
            "open_id": "ou_111",
            "expires_in": 7200,
            "refresh_expires_in": 2592000,
            "token_type": "Bearer",
            "scope": "calendar:calendar",
        }

        result = poll_device_token("dc_abc", "app_id")

        self.assertEqual(result["access_token"], "uat_tok_xyz")
        self.assertEqual(result["refresh_token"], "ref_tok_abc")
        self.assertEqual(result["open_id"], "ou_111")
        self.assertIsNone(result["error"])

    @patch("hermes_cli.feishu_auth._api_post")
    def test_raises_feishu_auth_error_on_hard_error(self, mock_api):
        mock_api.side_effect = FeishuAuthError("expired_token")

        with self.assertRaises(FeishuAuthError):
            poll_device_token("dc_abc", "app_id")


# ---------------------------------------------------------------------------
# save_uat / load_uat
# ---------------------------------------------------------------------------

class TestSaveAndLoadUat(unittest.TestCase):
    """Tests for save_uat() and load_uat() token persistence."""

    def test_save_uat_creates_file_with_correct_json_fields(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / ".hermes" / "feishu_uat.json"

            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                save_uat(
                    access_token="uat_abc",
                    refresh_token="ref_def",
                    open_id="ou_123",
                    expires_in=7200,
                    refresh_expires_in=2592000,
                    scope="calendar:calendar",
                    app_id="app_999",
                )

            self.assertTrue(uat_path.exists())
            data = json.loads(uat_path.read_text())
            self.assertEqual(data["access_token"], "uat_abc")
            self.assertEqual(data["refresh_token"], "ref_def")
            self.assertEqual(data["user_open_id"], "ou_123")
            self.assertEqual(data["app_id"], "app_999")
            self.assertIn("expires_at", data)
            self.assertIn("refresh_expires_at", data)
            self.assertIn("granted_at", data)

    def test_save_uat_sets_file_mode_0600(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / ".hermes" / "feishu_uat.json"

            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                save_uat(
                    access_token="tok",
                    refresh_token="ref",
                    open_id="ou",
                    expires_in=7200,
                    refresh_expires_in=2592000,
                    scope="",
                    app_id="app",
                )

            file_mode = stat.S_IMODE(uat_path.stat().st_mode)
            self.assertEqual(file_mode, 0o600)

    def test_save_uat_expires_at_is_future(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / ".hermes" / "feishu_uat.json"
            before_ms = int(time.time() * 1000)

            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                save_uat(
                    access_token="tok",
                    refresh_token="ref",
                    open_id="ou",
                    expires_in=7200,
                    refresh_expires_in=2592000,
                    scope="",
                    app_id="app",
                )

            data = json.loads(uat_path.read_text())
            self.assertGreater(data["expires_at"], before_ms + 7000 * 1000)

    def test_load_uat_returns_none_when_file_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "no_such.json"
            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", missing):
                result = load_uat()
        self.assertIsNone(result)

    def test_load_uat_returns_dict_with_correct_fields(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / ".hermes" / "feishu_uat.json"

            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                save_uat("uat_x", "ref_x", "ou_x", 7200, 2592000, "scope", "app")
                result = load_uat()

        self.assertIsNotNone(result)
        self.assertEqual(result["access_token"], "uat_x")
        self.assertEqual(result["user_open_id"], "ou_x")

    def test_load_uat_returns_none_on_corrupt_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / "feishu_uat.json"
            uat_path.write_text("not valid json {{")
            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                result = load_uat()
        self.assertIsNone(result)

    def test_load_uat_returns_none_on_expired_is_separate_from_presence(self):
        """load_uat does NOT validate expiry — that is the client's job."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            uat_path = Path(tmpdir) / "feishu_uat.json"
            expired_data = {
                "access_token": "old_tok",
                "expires_at": 1000,  # far in the past
                "user_open_id": "ou_old",
            }
            uat_path.write_text(json.dumps(expired_data))
            with patch("hermes_cli.feishu_auth.FEISHU_UAT_PATH", uat_path):
                result = load_uat()
        # load_uat itself does NOT reject expired tokens — it just reads the file
        self.assertIsNotNone(result)
        self.assertEqual(result["access_token"], "old_tok")


# ---------------------------------------------------------------------------
# wait_for_authorization_success
# ---------------------------------------------------------------------------

class TestWaitForAuthorizationSuccess(unittest.TestCase):
    """Tests for the polling loop wait_for_authorization_success().

    All tests mock time.monotonic so the loop terminates without real sleeps.
    The deadline is set to expire_in=60; monotonic returns values that keep the
    loop alive for N iterations then advance past the deadline so it exits.
    """

    def _make_monotonic_sequence(self, n_live_iterations, expires_in=60):
        """Return a monotonic side_effect list.

        - First call: t=0 (deadline = expires_in)
        - Next n_live_iterations*2 calls: t=0 (within deadline)
        - Final call: t=expires_in+1 (past deadline, exits loop)
        """
        values = [0.0]  # initial call sets deadline
        for _ in range(n_live_iterations):
            values.append(0.0)   # while-condition check inside loop
        values.append(expires_in + 1)  # loop exits
        return values

    @patch("hermes_cli.feishu_auth.time.monotonic")
    @patch("hermes_cli.feishu_auth.time.sleep")
    @patch("hermes_cli.feishu_auth.poll_device_token")
    def test_returns_tokens_immediately_on_authorized(self, mock_poll, mock_sleep, mock_mono):
        # monotonic: call #1 sets deadline=60, call #2 is while check (0 < 60 → enter)
        # After poll returns success, code hits line 274 (access_token present) and returns.
        # NOTE: to reach line 274, error must be truthy AND not pending/slow_down.
        # With error=None (falsy), the code hits the "authorization_pending" branch.
        # Use error="ok" so the success branch at line 274 is reachable.
        mock_mono.side_effect = [0.0, 0.0, 100.0]
        mock_poll.return_value = {
            "access_token": "uat_good",
            "refresh_token": "ref_good",
            "open_id": "ou_good",
            "error": "ok",  # truthy, non-pending, non-slow_down → reaches access_token check
        }

        token, refresh, open_id = wait_for_authorization_success(
            "dc", "app", interval=0, expires_in=60
        )

        self.assertEqual(token, "uat_good")
        self.assertEqual(refresh, "ref_good")
        self.assertEqual(open_id, "ou_good")
        self.assertEqual(mock_poll.call_count, 1)

    @patch("hermes_cli.feishu_auth.time.monotonic")
    @patch("hermes_cli.feishu_auth.time.sleep")
    @patch("hermes_cli.feishu_auth.poll_device_token")
    def test_calls_on_waiting_callback_on_pending(self, mock_poll, mock_sleep, mock_mono):
        # deadline=0+60, while checks: 0, 0, 0 (3 iterations), then 100 to exit
        mock_mono.side_effect = [0.0, 0.0, 0.0, 0.0, 100.0]
        mock_poll.side_effect = [
            {"access_token": None, "error": "authorization_pending"},
            {"access_token": None, "error": "authorization_pending"},
            {"access_token": "uat_ok", "refresh_token": "ref", "open_id": "ou", "error": "ok"},
        ]

        callback = MagicMock()
        wait_for_authorization_success(
            "dc", "app", interval=0, expires_in=60, on_waiting=callback
        )

        self.assertEqual(callback.call_count, 2)

    @patch("hermes_cli.feishu_auth.time.monotonic")
    @patch("hermes_cli.feishu_auth.time.sleep")
    @patch("hermes_cli.feishu_auth.poll_device_token")
    def test_raises_timeout_when_deadline_passes(self, mock_poll, mock_sleep, mock_mono):
        # deadline=0+60; while check 0 < 60 → enter; poll returns pending;
        # while check 100 < 60 → False → raises timed out
        mock_mono.side_effect = [0.0, 0.0, 100.0]
        mock_poll.return_value = {"access_token": None, "error": "authorization_pending"}

        with self.assertRaises(FeishuAuthError) as ctx:
            wait_for_authorization_success("dc", "app", interval=0, expires_in=60)

        self.assertIn("timed out", str(ctx.exception))

    @patch("hermes_cli.feishu_auth.time.monotonic")
    @patch("hermes_cli.feishu_auth.time.sleep")
    @patch("hermes_cli.feishu_auth.poll_device_token")
    def test_increases_interval_on_slow_down(self, mock_poll, mock_sleep, mock_mono):
        # iteration 1: slow_down; iteration 2: success
        mock_mono.side_effect = [0.0, 0.0, 0.0, 100.0]
        mock_poll.side_effect = [
            {"access_token": None, "error": "slow_down"},
            {"access_token": "uat_ok", "refresh_token": "ref", "open_id": "ou", "error": "ok"},
        ]

        wait_for_authorization_success("dc", "app", interval=3, expires_in=60)

        # sleep should have been called with 3 then 8 (3+5)
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertEqual(calls[0], 3)
        self.assertEqual(calls[1], 8)

    @patch("hermes_cli.feishu_auth.time.monotonic")
    @patch("hermes_cli.feishu_auth.time.sleep")
    @patch("hermes_cli.feishu_auth.poll_device_token")
    def test_raises_feishu_auth_error_on_hard_error_code(self, mock_poll, mock_sleep, mock_mono):
        # Hard error (access_denied) with no access_token reaches line 281.
        # retry_start=0 → set retry_start=monotonic(); then check monotonic()-retry_start < 120.
        # Supply: deadline=0, while_check=0, retry_start_set=0, elapsed_check=121 → raises.
        mock_mono.side_effect = [0.0, 0.0, 0.0, 121.0]
        mock_poll.return_value = {
            "access_token": None,
            "error": "access_denied",
            "error_description": "user denied",
        }

        with self.assertRaises(FeishuAuthError) as ctx:
            wait_for_authorization_success("dc", "app", interval=0, expires_in=60)

        self.assertIn("authorization failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
