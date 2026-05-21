"""Tests for uv-tool install detection in the update path (issue #29700).

``uv tool install hermes-agent`` lives outside any venv, so the previous
``uv pip install --upgrade`` update path failed with ``No virtual
environment found``. ``is_uv_tool_install`` should detect this layout and
both the user-facing recommended command and the actual
``_cmd_update_pip`` subprocess invocation should switch to
``uv tool upgrade hermes-agent``.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# is_uv_tool_install
# ---------------------------------------------------------------------------


class TestIsUvToolInstall:
    def test_returns_true_when_sys_prefix_matches_uv_tool_layout(self):
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/home/user/.local/share/uv/tools/hermes-agent"):
            assert config.is_uv_tool_install("uv") is True

    def test_returns_true_when_uv_tool_list_includes_hermes_agent(self):
        from hermes_cli import config

        completed = subprocess.CompletedProcess(
            ["uv", "tool", "list"],
            0,
            stdout="hermes-agent v0.14.0\n- hermes\n- hermes-bot\nblack v23.0.0\n- black\n",
            stderr="",
        )
        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("subprocess.run", return_value=completed) as mock_run:
            assert config.is_uv_tool_install("/usr/local/bin/uv") is True
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["/usr/local/bin/uv", "tool", "list"]

    def test_returns_false_when_uv_tool_list_lacks_hermes_agent(self):
        from hermes_cli import config

        completed = subprocess.CompletedProcess(
            ["uv", "tool", "list"], 0, stdout="black v23.0.0\n- black\nruff v0.5.0\n- ruff\n", stderr=""
        )
        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("subprocess.run", return_value=completed):
            assert config.is_uv_tool_install("uv") is False

    def test_returns_false_when_uv_tool_list_fails(self):
        from hermes_cli import config

        completed = subprocess.CompletedProcess(["uv", "tool", "list"], 2, stdout="", stderr="oops")
        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("subprocess.run", return_value=completed):
            assert config.is_uv_tool_install("uv") is False

    def test_returns_false_when_subprocess_raises(self):
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["uv"], 15)):
            assert config.is_uv_tool_install("uv") is False

    def test_returns_false_when_no_uv_available(self):
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("shutil.which", return_value=None):
            assert config.is_uv_tool_install() is False

    def test_indented_alias_line_does_not_false_positive(self):
        """A tool whose alias line is ``- hermes-agent`` shouldn't match."""
        from hermes_cli import config

        completed = subprocess.CompletedProcess(
            ["uv", "tool", "list"],
            0,
            stdout="some-other-tool v1.0.0\n- hermes-agent\n",
            stderr="",
        )
        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch("subprocess.run", return_value=completed):
            assert config.is_uv_tool_install("uv") is False


# ---------------------------------------------------------------------------
# recommended_update_command_for_method
# ---------------------------------------------------------------------------


class TestRecommendedUpdateCommandForUvTool:
    def test_uv_tool_install_recommends_uv_tool_upgrade(self):
        from hermes_cli import config

        with patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch.object(config, "is_uv_tool_install", return_value=True):
            cmd = config.recommended_update_command_for_method("pip")
            assert cmd == "uv tool upgrade hermes-agent"

    def test_uv_pip_install_keeps_legacy_recommendation(self):
        """Existing behavior: uv is on PATH but Hermes is a regular pip install."""
        from hermes_cli import config

        with patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch.object(config, "is_uv_tool_install", return_value=False):
            cmd = config.recommended_update_command_for_method("pip")
            assert cmd == "uv pip install --upgrade hermes-agent"

    def test_no_uv_falls_back_to_plain_pip(self):
        from hermes_cli.config import recommended_update_command_for_method

        with patch("shutil.which", return_value=None):
            cmd = recommended_update_command_for_method("pip")
            assert cmd == "pip install --upgrade hermes-agent"


# ---------------------------------------------------------------------------
# _cmd_update_pip subprocess command
# ---------------------------------------------------------------------------


class TestCmdUpdatePipUsesUvTool:
    @patch("subprocess.run")
    def test_runs_uv_tool_upgrade_when_uv_tool_install(self, mock_run):
        """The actual subprocess invocation must switch to ``uv tool upgrade``."""
        from hermes_cli.main import _cmd_update_pip

        mock_run.return_value = subprocess.CompletedProcess(["uv"], 0, stdout="", stderr="")
        with patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch("hermes_cli.config.is_uv_tool_install", return_value=True):
            _cmd_update_pip(SimpleNamespace())

        assert mock_run.call_args[0][0] == ["/usr/local/bin/uv", "tool", "upgrade", "hermes-agent"]

    @patch("subprocess.run")
    def test_runs_uv_pip_install_when_not_uv_tool(self, mock_run):
        """Existing behavior preserved when uv is present but Hermes isn't a tool install."""
        from hermes_cli.main import _cmd_update_pip

        mock_run.return_value = subprocess.CompletedProcess(["uv"], 0, stdout="", stderr="")
        with patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch("hermes_cli.config.is_uv_tool_install", return_value=False):
            _cmd_update_pip(SimpleNamespace())

        assert mock_run.call_args[0][0] == [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--upgrade",
            "hermes-agent",
        ]

    @patch("subprocess.run")
    def test_falls_back_to_pip_when_no_uv(self, mock_run):
        from hermes_cli.main import _cmd_update_pip

        mock_run.return_value = subprocess.CompletedProcess(["pip"], 0, stdout="", stderr="")
        with patch("shutil.which", return_value=None):
            _cmd_update_pip(SimpleNamespace())

        cmd = mock_run.call_args[0][0]
        assert cmd[1:] == ["-m", "pip", "install", "--upgrade", "hermes-agent"]

    @patch("subprocess.run")
    def test_exits_nonzero_on_subprocess_failure(self, mock_run):
        from hermes_cli.main import _cmd_update_pip

        mock_run.return_value = subprocess.CompletedProcess(["uv"], 1, stdout="", stderr="")
        with patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch("hermes_cli.config.is_uv_tool_install", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_update_pip(SimpleNamespace())
        assert exc_info.value.code == 1
