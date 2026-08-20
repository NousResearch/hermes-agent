"""Tests for DEAD_SESSION_RET handling, _fire_alert, and SessionExpiredError."""
import subprocess
from unittest.mock import patch
from gateway.platforms import weixin


class TestDeadSessionRet:
    """Test DEAD_SESSION_RET constant and _is_stale_session_ret boundary."""

    def test_dead_session_ret_value(self):
        assert weixin.DEAD_SESSION_RET == -3

    def test_stale_ret_excludes_minus_3(self):
        """_is_stale_session_ret only handles ret=-2, not ret=-3."""
        assert weixin._is_stale_session_ret(-3, None, None) is False
        assert weixin._is_stale_session_ret(-3, None, "unknown error") is False

    def test_stale_ret_minus_2_unknown_error(self):
        assert weixin._is_stale_session_ret(-2, None, "unknown error") is True

    def test_stale_ret_minus_2_errcode_zero(self):
        """ret=-2 with errcode=0 -- errcode should not interfere."""
        assert weixin._is_stale_session_ret(-2, 0, "unknown error") is True

    def test_stale_ret_minus_2_empty_errmsg(self):
        assert weixin._is_stale_session_ret(-2, None, "") is False

    def test_stale_ret_minus_2_none_errmsg(self):
        assert weixin._is_stale_session_ret(-2, None, None) is False

    def test_stale_ret_minus_2_freq_limit(self):
        assert weixin._is_stale_session_ret(-2, None, "freq limit") is False

    def test_stale_ret_minus_2_rate_limited(self):
        assert weixin._is_stale_session_ret(-2, None, "rate limited") is False

    def test_stale_ret_minus_2_access_token_expired(self):
        """access_token expired is NOT 'unknown error' -- not stale by this helper."""
        assert weixin._is_stale_session_ret(-2, None, "access_token expired") is False

    def test_stale_ret_minus_2_invalid_credential(self):
        """invalid credential is NOT 'unknown error' -- not stale by this helper."""
        assert weixin._is_stale_session_ret(-2, None, "invalid credential") is False

    def test_stale_ret_minus_2_unknown_error_case_insensitive(self):
        """'Unknown Error' (mixed case) should match."""
        assert weixin._is_stale_session_ret(-2, None, "Unknown Error") is True

    def test_stale_ret_minus_2_unknown_error_whitespace(self):
        """Leading/trailing whitespace should NOT match."""
        assert weixin._is_stale_session_ret(-2, None, " unknown error ") is False

    def test_stale_ret_minus_2_errcode_also_minus_2(self):
        """Both ret and errcode are -2."""
        assert weixin._is_stale_session_ret(-2, -2, "unknown error") is True

    def test_stale_ret_none_errcode_minus_2_unknown(self):
        """errcode=-2 alone (ret=None) with 'unknown error'."""
        assert weixin._is_stale_session_ret(None, -2, "unknown error") is True

    def test_stale_ret_none_errcode_minus_2_empty_errmsg(self):
        """errcode=-2 with empty errmsg -- should return False."""
        assert weixin._is_stale_session_ret(None, -2, "") is False

    def test_stale_ret_zero_not_stale(self):
        assert weixin._is_stale_session_ret(0, None, None) is False

    def test_stale_errcode_minus_14_not_stale(self):
        """errcode=-14 is handled separately, not by _is_stale_session_ret."""
        assert weixin._is_stale_session_ret(None, -14, "session expired") is False

    def test_stale_ret_minus_1_not_stale(self):
        """ret=-1 is a generic error, not stale."""
        assert weixin._is_stale_session_ret(-1, None, "unknown error") is False


class TestFireAlert:
    """Test _fire_alert function with mocked subprocess."""

    def test_no_script_noop(self):
        """Empty script_path should be a no-op, no subprocess call."""
        with patch("subprocess.run") as mock_run:
            weixin._fire_alert("test", "detail", "")
            mock_run.assert_not_called()

    def test_no_script_default(self):
        """Default script_path should be a no-op."""
        with patch("subprocess.run") as mock_run:
            weixin._fire_alert("test", "detail")
            mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_calls_script(self, mock_run):
        """Script is called with correct arguments."""
        weixin._fire_alert("stale_session", "ret=-3", "/usr/bin/alert.sh")
        mock_run.assert_called_once_with(
            ["/usr/bin/alert.sh", "stale_session", "ret=-3"],
            timeout=30,
            capture_output=True,
        )

    @patch("subprocess.run", side_effect=OSError("not found"))
    def test_bad_script_no_raise(self, mock_run):
        """Script failure should not raise -- fire_alert is fire-and-forget."""
        weixin._fire_alert("test", "detail", "/nonexistent/script")

    @patch("subprocess.run")
    def test_timeout_no_raise(self, mock_run):
        """Script timeout should not raise."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="alert", timeout=30)
        weixin._fire_alert("test", "detail", "/usr/bin/slow-script")

    @patch("subprocess.run", side_effect=PermissionError("denied"))
    def test_permission_error_no_raise(self, mock_run):
        """Permission error should not raise."""
        weixin._fire_alert("test", "detail", "/usr/bin/alert.sh")


class TestSessionExpiredError:
    """Test SessionExpiredError exception."""

    def test_is_exception(self):
        assert issubclass(weixin.SessionExpiredError, Exception)

    def test_can_raise_and_catch(self):
        try:
            raise weixin.SessionExpiredError("session dead")
        except weixin.SessionExpiredError as e:
            assert "session dead" in str(e)
