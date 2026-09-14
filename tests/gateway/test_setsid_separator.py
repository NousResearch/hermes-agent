"""Verify setsid receives ``--`` before bash so Termux doesn't swallow flags.

On Android/Termux, util-linux ``setsid`` parses ``-c``/``-lc`` as its own
options instead of forwarding them to ``bash``.  POSIX mandates ``--`` to
separate a command's own options from the child command.  These tests ensure
both the detached restart and the detached update spawn include the separator.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX setsid paths only")


class TestRestartSetsidSeparator:
    """``GatewayShutdownMixin._launch_detached_restart_command`` must pass
    ``"--"`` between the setsid binary and bash."""

    @pytest.mark.asyncio
    async def test_restart_command_includes_setsid_separator(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._detached_restart_helper_started = False
        fake_setsid = "/usr/bin/setsid"

        with (
            patch("gateway.run._resolve_hermes_bin", return_value=["hermes"]),
            patch("shutil.which", return_value=fake_setsid),
            patch("subprocess.Popen") as mock_popen,
            patch("os.getpid", return_value=12345),
        ):
            await runner._launch_detached_restart_command()

        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]

        assert argv[0] == fake_setsid, "first arg must be the setsid binary"
        assert argv[1] == "--", "setsid must be followed by '--' separator"
        assert argv[2] == "bash", "bash must follow the separator"

    @pytest.mark.asyncio
    async def test_restart_command_without_setsid_no_separator(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._detached_restart_helper_started = False

        with (
            patch("gateway.run._resolve_hermes_bin", return_value=["hermes"]),
            patch("shutil.which", return_value=None),
            patch("subprocess.Popen") as mock_popen,
            patch("os.getpid", return_value=12345),
        ):
            await runner._launch_detached_restart_command()

        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]

        assert argv[0] == "bash", "first arg must be bash when setsid is absent"
        assert "--" not in argv, "no '--' separator when setsid is not used"


class TestUpdateSetsidSeparator:
    """``_spawn_detached_update`` must pass ``"--"`` between the setsid
    binary and bash."""

    @pytest.mark.asyncio
    async def test_update_spawn_includes_setsid_separator(self, tmp_path):
        from gateway.slash_commands import _spawn_detached_update

        fake_setsid = "/usr/bin/setsid"
        output_path = tmp_path / ".update_output.txt"
        exit_code_path = tmp_path / ".update_exit_code"

        with (
            patch("shutil.which", return_value=fake_setsid),
            patch("subprocess.Popen") as mock_popen,
        ):
            _spawn_detached_update(["hermes"], output_path, exit_code_path)

        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]

        assert argv[0] == fake_setsid, "first arg must be the setsid binary"
        assert argv[1] == "--", "setsid must be followed by '--' separator"
        assert argv[2] == "bash", "bash must follow the separator"
        assert argv[3] == "-c", "update command runs via bash -c"

    @pytest.mark.asyncio
    async def test_update_spawn_without_setsid_no_separator(self, tmp_path):
        from gateway.slash_commands import _spawn_detached_update

        output_path = tmp_path / ".update_output.txt"
        exit_code_path = tmp_path / ".update_exit_code"

        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.Popen") as mock_popen,
        ):
            _spawn_detached_update(["hermes"], output_path, exit_code_path)

        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]

        assert argv[0] == "bash", "first arg must be bash when setsid is absent"
        assert "--" not in argv, "no '--' separator when setsid is not used"
        assert str(exit_code_path) in argv[-1], "exit code path must reach the shell command"
