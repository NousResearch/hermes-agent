"""Tests for the built-in RTK rewrite plugin."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import plugins.rtk_rewrite as rtk_rewrite


@pytest.fixture(autouse=True)
def _reset_cache():
    rtk_rewrite._rtk_available_cache.clear()
    yield
    rtk_rewrite._rtk_available_cache.clear()


class TestLoadRtkSettings:
    def test_defaults_when_config_missing(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            settings = rtk_rewrite._load_rtk_settings()
        assert settings == {
            "enabled": "auto",
            "binary": "rtk",
            "rewrite_timeout_seconds": 2.0,
            "log_rewrites": False,
        }

    def test_reads_terminal_rtk_config(self):
        config = {
            "terminal": {
                "rtk": {
                    "enabled": True,
                    "binary": "/opt/rtk/bin/rtk",
                    "rewrite_timeout_seconds": 5,
                    "log_rewrites": True,
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            settings = rtk_rewrite._load_rtk_settings()
        assert settings == {
            "enabled": "true",
            "binary": "/opt/rtk/bin/rtk",
            "rewrite_timeout_seconds": 5.0,
            "log_rewrites": True,
        }


class TestCheckRtk:
    def test_uses_shutil_for_path_lookup(self):
        with patch("plugins.rtk_rewrite.shutil.which", return_value="/usr/local/bin/rtk") as mock_which:
            assert rtk_rewrite._check_rtk("rtk") is True
            assert rtk_rewrite._check_rtk("rtk") is True
        mock_which.assert_called_once_with("rtk")

    def test_supports_explicit_binary_path(self, tmp_path):
        binary = tmp_path / "rtk"
        binary.write_text("#!/bin/sh\n")
        assert rtk_rewrite._check_rtk(str(binary)) is True


class TestTryRewrite:
    @staticmethod
    def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")

    def test_rewrites_when_rtk_returns_new_command(self):
        with patch("plugins.rtk_rewrite.subprocess.run", return_value=self._completed("rtk git status\n")):
            rewritten = rtk_rewrite._try_rewrite("git status", binary="rtk", timeout_seconds=2)
        assert rewritten == "rtk git status"

    def test_same_command_returns_none(self):
        with patch("plugins.rtk_rewrite.subprocess.run", return_value=self._completed("echo hi\n")):
            rewritten = rtk_rewrite._try_rewrite("echo hi", binary="rtk", timeout_seconds=2)
        assert rewritten is None

    def test_nonzero_exit_with_rewrite_still_returns_rewrite(self):
        with patch("plugins.rtk_rewrite.subprocess.run", return_value=self._completed("rtk git status\n", returncode=3)):
            rewritten = rtk_rewrite._try_rewrite("git status", binary="rtk", timeout_seconds=2)
        assert rewritten == "rtk git status"

    def test_nonzero_exit_returns_none(self):
        with patch("plugins.rtk_rewrite.subprocess.run", return_value=self._completed(returncode=1)):
            rewritten = rtk_rewrite._try_rewrite("git status", binary="rtk", timeout_seconds=2)
        assert rewritten is None

    def test_timeout_returns_none(self):
        with patch(
            "plugins.rtk_rewrite.subprocess.run",
            side_effect=subprocess.TimeoutExpired("rtk", 2),
        ):
            rewritten = rtk_rewrite._try_rewrite("git status", binary="rtk", timeout_seconds=2)
        assert rewritten is None


class TestPreToolCall:
    def test_rewrites_terminal_commands(self):
        args = {"command": "git status", "timeout": 30}
        with (
            patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": True}}}),
            patch.object(rtk_rewrite, "_check_rtk", return_value=True),
            patch.object(rtk_rewrite, "_try_rewrite", return_value="rtk git status"),
        ):
            rtk_rewrite._pre_tool_call(tool_name="terminal", args=args)
        assert args == {"command": "rtk git status", "timeout": 30}

    def test_skips_when_disabled(self):
        args = {"command": "git status"}
        with (
            patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": False}}}),
            patch.object(rtk_rewrite, "_check_rtk") as mock_check,
        ):
            rtk_rewrite._pre_tool_call(tool_name="terminal", args=args)
        mock_check.assert_not_called()
        assert args["command"] == "git status"

    def test_skips_non_terminal_tools(self):
        args = {"command": "git status"}
        with patch.object(rtk_rewrite, "_try_rewrite") as mock_rewrite:
            rtk_rewrite._pre_tool_call(tool_name="web_search", args=args)
        mock_rewrite.assert_not_called()

    def test_skips_missing_binary(self):
        args = {"command": "git status"}
        with (
            patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": True}}}),
            patch.object(rtk_rewrite, "_check_rtk", return_value=False),
            patch.object(rtk_rewrite, "_try_rewrite") as mock_rewrite,
        ):
            rtk_rewrite._pre_tool_call(tool_name="terminal", args=args)
        mock_rewrite.assert_not_called()
        assert args["command"] == "git status"


class TestRegister:
    def test_registers_hook_when_available(self):
        ctx = MagicMock()
        with (
            patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": "auto"}}}),
            patch.object(rtk_rewrite, "_check_rtk", return_value=True),
        ):
            rtk_rewrite.register(ctx)
        ctx.register_hook.assert_called_once_with("pre_tool_call", rtk_rewrite._pre_tool_call)

    def test_skips_registration_when_disabled(self):
        ctx = MagicMock()
        with patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": False}}}):
            rtk_rewrite.register(ctx)
        ctx.register_hook.assert_not_called()

    def test_skips_registration_when_binary_missing_in_auto_mode(self):
        ctx = MagicMock()
        with (
            patch("hermes_cli.config.load_config", return_value={"terminal": {"rtk": {"enabled": "auto"}}}),
            patch.object(rtk_rewrite, "_check_rtk", return_value=False),
        ):
            rtk_rewrite.register(ctx)
        ctx.register_hook.assert_not_called()
