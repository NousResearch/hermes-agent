import argparse
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from hermes_cli.main import (
    cmd_dashboard,
    _dashboard_auto_install_allowed,
)


def _ns(**kw):
    """Build an argparse.Namespace with dashboard defaults plus overrides."""
    defaults = dict(
        port=9119, host="127.0.0.1", no_open=False, insecure=False,
        stop=False, status=False, skip_build=False, isolated=False,
        open_profile=""
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestDashboardAutoInstallAllowed:

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure env var is clean before each test."""
        old_val = os.environ.get("HERMES_DASHBOARD_AUTO_INSTALL")
        if "HERMES_DASHBOARD_AUTO_INSTALL" in os.environ:
            del os.environ["HERMES_DASHBOARD_AUTO_INSTALL"]
        yield
        if old_val is not None:
            os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = old_val
        elif "HERMES_DASHBOARD_AUTO_INSTALL" in os.environ:
            del os.environ["HERMES_DASHBOARD_AUTO_INSTALL"]

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", " 0 ", "FALSE"])
    def test_explicit_opt_out_env(self, val):
        os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = val
        # Should be False regardless of TTY status
        with patch("sys.stdin.isatty", return_value=True), \
             patch("sys.stdout.isatty", return_value=True):
            assert _dashboard_auto_install_allowed() is False

        with patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdout.isatty", return_value=False):
            assert _dashboard_auto_install_allowed() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", " 1 ", "TRUE"])
    def test_explicit_opt_in_env(self, val):
        os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = val
        # Should be True regardless of TTY status
        with patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdout.isatty", return_value=False):
            assert _dashboard_auto_install_allowed() is True

        with patch("sys.stdin.isatty", return_value=True), \
             patch("sys.stdout.isatty", return_value=True):
            assert _dashboard_auto_install_allowed() is True

    def test_default_tty_allowed(self):
        # Unset env, both stdin and stdout are TTY -> Allowed
        with patch("sys.stdin.isatty", return_value=True), \
             patch("sys.stdout.isatty", return_value=True):
            assert _dashboard_auto_install_allowed() is True

    @pytest.mark.parametrize("stdin_tty, stdout_tty", [
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_default_non_tty_blocked(self, stdin_tty, stdout_tty):
        # Unset env, either stdin or stdout is non-TTY -> Blocked
        with patch("sys.stdin.isatty", return_value=stdin_tty), \
             patch("sys.stdout.isatty", return_value=stdout_tty):
            assert _dashboard_auto_install_allowed() is False


class TestCmdDashboardAutoInstallGuard:

    @pytest.fixture(autouse=True)
    def mock_dashboard_deps(self):
        """Mock out downstream steps in cmd_dashboard to prevent server start."""
        with patch("hermes_cli.main._sync_bundled_skills_quietly"), \
             patch("hermes_cli.plugins.discover_plugins"), \
             patch("hermes_cli.mcp_startup.start_background_mcp_discovery"), \
             patch("hermes_cli.main._maybe_setup_dashboard_auth_interactively"), \
             patch("hermes_cli.web_server.start_server"), \
             patch.dict(sys.modules, {"fastapi": MagicMock(), "uvicorn": MagicMock()}):
            yield

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure env var is clean before each test."""
        old_val = os.environ.get("HERMES_DASHBOARD_AUTO_INSTALL")
        if "HERMES_DASHBOARD_AUTO_INSTALL" in os.environ:
            del os.environ["HERMES_DASHBOARD_AUTO_INSTALL"]
        yield
        if old_val is not None:
            os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = old_val
        elif "HERMES_DASHBOARD_AUTO_INSTALL" in os.environ:
            del os.environ["HERMES_DASHBOARD_AUTO_INSTALL"]

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui")
    @patch("hermes_cli.main._dashboard_auto_install_allowed", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)  # Mock package.json check
    def test_non_tty_build_needed_no_override_blocked(
        self, mock_exists, mock_allowed, mock_build, mock_build_needed, capsys
    ):
        with pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns())

        assert exc.value.code == 1
        mock_build.assert_not_called()
        out = capsys.readouterr().out
        assert "Web UI TUI dependencies not installed" in out
        assert "started in a non-interactive context and cannot run `npm install`" in out

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui", return_value=True)
    @patch("hermes_cli.main._dashboard_auto_install_allowed", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_tty_build_needed_allowed(
        self, mock_exists, mock_allowed, mock_build, mock_build_needed
    ):
        # Should not raise SystemExit(1)
        cmd_dashboard(_ns())
        mock_build.assert_called_once()

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui")
    @patch("pathlib.Path.exists", return_value=True)
    def test_env_opt_out_blocked(self, mock_exists, mock_build, mock_build_needed):
        os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = "0"
        with pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns())

        assert exc.value.code == 1
        mock_build.assert_not_called()

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_env_opt_in_allowed(self, mock_exists, mock_build, mock_build_needed):
        os.environ["HERMES_DASHBOARD_AUTO_INSTALL"] = "1"
        cmd_dashboard(_ns())
        mock_build.assert_called_once()

    @patch("hermes_cli.main._web_ui_build_needed", return_value=False)
    @patch("hermes_cli.main._build_web_ui", return_value=True)
    @patch("hermes_cli.main._dashboard_auto_install_allowed", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    def test_build_not_needed_unaffected(
        self, mock_exists, mock_allowed, mock_build, mock_build_needed
    ):
        # If dependencies are fresh (build not needed), dashboard startup is
        # permitted and does not fail closed, regardless of TTY or env flags.
        cmd_dashboard(_ns())
        mock_build.assert_called_once()

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui")
    @patch("hermes_cli.main._dashboard_auto_install_allowed", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    @patch("hermes_cli.main.Path.exists", return_value=True) # Mock index.html existence for skip-build check
    def test_skip_build_unaffected(
        self, mock_exists_path, mock_exists, mock_allowed, mock_build, mock_build_needed
    ):
        # With skip_build=True, the entire build step is bypassed.
        cmd_dashboard(_ns(skip_build=True))
        mock_build.assert_not_called()
        mock_build_needed.assert_not_called()

    @patch("hermes_cli.main._web_ui_build_needed", return_value=True)
    @patch("hermes_cli.main._build_web_ui")
    @patch("hermes_cli.main._dashboard_auto_install_allowed", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    def test_hermes_web_dist_env_var_set_unaffected(
        self, mock_exists, mock_allowed, mock_build, mock_build_needed
    ):
        # When HERMES_WEB_DIST env var is set, dashboard startup is permitted
        # and does not fail closed, and _build_web_ui is not called.
        os.environ["HERMES_WEB_DIST"] = "/some/prebuilt/dist"
        try:
            cmd_dashboard(_ns())
        finally:
            del os.environ["HERMES_WEB_DIST"]
        mock_build.assert_not_called()
        mock_build_needed.assert_not_called()
