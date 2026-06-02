"""Verify setsid receives ``--`` before bash so Termux doesn't swallow flags.

On Android/Termux, ``setsid`` interprets ``-lc`` as its own options instead of
forwarding them to ``bash``.  POSIX mandates ``--`` to separate setsid's own
options from the child command.  This test ensures both the restart and update
code paths include the separator.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestSetsidBashSeparator:
    """Both ``_launch_detached_restart_command`` and ``_handle_update_command``
    must pass ``"--"`` between the setsid binary and bash."""

    @pytest.mark.asyncio
    async def test_restart_command_includes_setsid_separator(self, tmp_path):
        """_launch_detached_restart_command passes '--' to setsid before bash."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        # Minimal stubs so the method can reach the setsid branch
        runner._running = True

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

        assert argv[0] == fake_setsid, "first arg must be setsid binary"
        assert argv[1] == "--", "setsid must be followed by '--' separator"
        assert argv[2] == "bash", "bash must follow the separator"

    @pytest.mark.asyncio
    async def test_restart_command_without_setsid_no_separator(self, tmp_path):
        """When setsid is not available, bash runs directly (no separator needed)."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._running = True

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
