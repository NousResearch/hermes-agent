"""Tests for uv-tool install detection in the update path (issue #29700).

``uv tool install hermes-agent`` lives outside any venv, so the previous
``uv pip install --upgrade`` update path failed with ``No virtual
environment found``. ``is_uv_tool_install`` should detect this layout and
both the user-facing recommended command and the actual
``_cmd_update_pip`` subprocess invocation should switch to
``uv tool upgrade hermes-agent``.

Detection is restricted to properties of the running interpreter
(``sys.prefix`` / ``sys.executable``) so a pip/venv install on a machine
that also has ``uv tool install hermes-agent`` does not get misclassified.
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
            assert config.is_uv_tool_install() is True

    def test_returns_true_when_sys_executable_matches_uv_tool_layout(self):
        """Some uv-tool layouts surface the marker on ``sys.executable`` (bin/python)."""
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch.object(
                 config.sys,
                 "executable",
                 "/home/user/.local/share/uv/tools/hermes-agent/bin/python",
             ):
            assert config.is_uv_tool_install() is True

    def test_returns_false_when_neither_prefix_nor_executable_matches(self):
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch.object(config.sys, "executable", "/usr/bin/python3"):
            assert config.is_uv_tool_install() is False

    def test_does_not_consult_uv_tool_list(self):
        """Detection must NOT shell out: ``uv tool list`` would false-positive
        when the active install is pip/venv but the machine also has
        ``uv tool install hermes-agent`` somewhere on disk. Copilot review on
        PR #29703 flagged this; the fix is to never call ``uv tool list``
        from the detection path."""
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch.object(config.sys, "executable", "/usr/bin/python3"), \
             patch("subprocess.run") as mock_run:
            assert config.is_uv_tool_install() is False
            mock_run.assert_not_called()

    def test_case_insensitive_match(self):
        """Match must be case-insensitive — Windows paths preserve case
        (e.g. ``...AppData\\Local\\UV\\Tools\\hermes-agent``) and a case-sensitive
        check would miss them. We exercise the lower-cased compare path here
        without monkey-patching ``os.sep``, which would break the whole suite."""
        from hermes_cli import config

        with patch.object(
            config.sys, "prefix", "/HOME/USER/.local/share/UV/Tools/hermes-agent"
        ):
            assert config.is_uv_tool_install() is True

    def test_handles_empty_executable(self):
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch.object(config.sys, "executable", ""):
            assert config.is_uv_tool_install() is False


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

    def test_uv_tool_install_recommends_uv_tool_upgrade_even_without_uv_on_path(self):
        """Recommendation reflects the *install method*, not whether ``uv`` is
        currently on PATH — the user needs to know the right command to run."""
        from hermes_cli import config

        with patch("shutil.which", return_value=None), \
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
        from hermes_cli import config

        with patch("shutil.which", return_value=None), \
             patch.object(config, "is_uv_tool_install", return_value=False):
            cmd = config.recommended_update_command_for_method("pip")
            assert cmd == "pip install --upgrade hermes-agent"

    def test_recommendation_does_not_spawn_subprocess(self):
        """Computing the recommendation string must be cheap — no ``uv tool list``
        spawn. Copilot review on PR #29703 flagged the prior subprocess hop
        as adding overhead and a multi-second timeout window for what is
        purely a display string."""
        from hermes_cli import config

        with patch.object(config.sys, "prefix", "/some/unrelated/venv"), \
             patch.object(config.sys, "executable", "/usr/bin/python3"), \
             patch("shutil.which", return_value="/usr/local/bin/uv"), \
             patch("subprocess.run") as mock_run:
            cmd = config.recommended_update_command_for_method("pip")
            mock_run.assert_not_called()
            assert cmd == "uv pip install --upgrade hermes-agent"


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
        with patch("shutil.which", return_value=None), \
             patch("hermes_cli.config.is_uv_tool_install", return_value=False):
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

    @patch("subprocess.run")
    def test_uv_tool_install_without_uv_on_path_exits_with_hint(self, mock_run):
        """If the running interpreter looks like a uv-tool install but ``uv`` is
        somehow missing from PATH, surface a clear hint instead of silently
        falling back to ``python -m pip``, which would either fail (no venv)
        or upgrade the wrong copy."""
        from hermes_cli.main import _cmd_update_pip

        with patch("shutil.which", return_value=None), \
             patch("hermes_cli.config.is_uv_tool_install", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_update_pip(SimpleNamespace())
        assert exc_info.value.code == 1
        mock_run.assert_not_called()
