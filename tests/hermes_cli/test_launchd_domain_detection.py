"""Tests for launchd domain detection in gateway.py.

Tests the fix for #40831: macOS launchd domain detection for both Aqua and
background sessions. The fix addresses both #23387 (background sessions failing)
and #40831 (Aqua sessions failing).
"""
import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from hermes_cli.gateway import _launchd_domain


class TestLaunchdDomainDetection:
    """Test launchd domain detection for different session types."""

    def test_detects_gui_domain_for_aqua_session(self):
        """When service is loaded in gui/<uid>, should return gui/<uid>."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    # First call (gui/<uid> probe) succeeds
                    mock_run.side_effect = [
                        MagicMock(returncode=0),  # launchctl print gui/501/ai.hermes.gateway
                    ]
                    
                    result = _launchd_domain()
                    assert result == "gui/501"
                    mock_run.assert_called_once()

    def test_falls_back_to_user_domain_when_gui_unavailable(self):
        """When gui/<uid> is unavailable, should fall back to user/<uid>."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    # First call (gui/<uid>) fails with CalledProcessError
                    # Second call (user/<uid>) succeeds
                    def side_effect(*args, **kwargs):
                        call_count = len(mock_run.call_args_list)
                        if call_count == 1:  # First call (gui)
                            raise subprocess.CalledProcessError(113, "launchctl")
                        # Second call (user) succeeds
                        return MagicMock(returncode=0)
                    
                    mock_run.side_effect = side_effect
                    
                    result = _launchd_domain()
                    assert result == "user/501"
                    assert mock_run.call_count == 2

    def test_detects_aqua_session_when_service_not_loaded(self):
        """When service not loaded but managername is Aqua, should return gui/<uid>."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    def side_effect(cmd, *args, **kwargs):
                        # Both launchctl print calls fail (service not loaded yet)
                        if "print" in cmd:
                            raise subprocess.CalledProcessError(113, "launchctl")
                        # managername returns Aqua
                        if cmd[0] == "launchctl" and "managername" in cmd:
                            result = MagicMock()
                            result.stdout = "Aqua\n"
                            result.returncode = 0
                            return result
                        raise subprocess.CalledProcessError(1, "launchctl")
                    
                    mock_run.side_effect = side_effect
                    
                    result = _launchd_domain()
                    assert result == "gui/501"

    def test_defaults_to_user_domain_for_background_session(self):
        """When managername is Background, should return user/<uid>."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    def side_effect(cmd, *args, **kwargs):
                        # Both launchctl print calls fail (service not loaded yet)
                        if "print" in cmd:
                            raise subprocess.CalledProcessError(113, "launchctl")
                        # managername returns something other than Aqua
                        if cmd[0] == "launchctl" and "managername" in cmd:
                            result = MagicMock()
                            result.stdout = "Background\n"
                            result.returncode = 0
                            return result
                        raise subprocess.CalledProcessError(1, "launchctl")
                    
                    mock_run.side_effect = side_effect
                    
                    result = _launchd_domain()
                    assert result == "user/501"

    def test_defaults_to_user_domain_when_managername_fails(self):
        """When managername probe fails, should default to user/<uid>."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    # All calls fail
                    mock_run.side_effect = subprocess.CalledProcessError(1, "launchctl")
                    
                    result = _launchd_domain()
                    assert result == "user/501"

    def test_handles_timeout_exceptions(self):
        """Should handle timeout exceptions gracefully."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=501):
                with patch("subprocess.run") as mock_run:
                    def side_effect(cmd, *args, **kwargs):
                        # First call times out, rest fail
                        if "print" in cmd:
                            raise subprocess.TimeoutExpired("launchctl", 5)
                        raise subprocess.CalledProcessError(1, "launchctl")
                    
                    mock_run.side_effect = side_effect
                    
                    result = _launchd_domain()
                    assert result == "user/501"

    def test_respects_different_uids(self):
        """Should respect different user IDs."""
        with patch("hermes_cli.gateway.get_launchd_label", return_value="ai.hermes.gateway"):
            with patch("os.getuid", return_value=1000):  # Different UID
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.CalledProcessError(113, "launchctl")
                    
                    result = _launchd_domain()
                    assert result == "user/1000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
