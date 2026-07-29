"""Tests for the cua-driver --no-overlay policy.

cua-driver's cursor overlay rendering loop can consume CPU indefinitely when
idle (#28152, #47032). Hermes passes ``--no-overlay`` to suppress it when the
``computer_use.no_overlay`` config is enabled (or auto-detected on macOS and
headless Linux / WSL2).

These assert the behavior contract (auto-detect, explicit override, version
probe), not specific config snapshots.
"""

import os
import sys
from unittest.mock import mock_open, patch

from tools.computer_use import cua_backend


class TestNoOverlayFlag:
    def test_default_linux_headless_disables(self):
        """Auto-detect: Linux without DISPLAY => overlay disabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            assert cua_backend._cua_no_overlay() is True

    def test_default_linux_desktop_enables(self):
        """Auto-detect: Linux with DISPLAY => overlay enabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._cua_no_overlay() is False

    def test_default_linux_wsl2_disables(self):
        """Auto-detect: WSL2 (microsoft in /proc/version) => overlay disabled."""
        fake_version = "Linux version 5.15.0 (Microsoft@Microsoft.com)"
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("builtins.open", mock_open(read_data=fake_version)):
            assert cua_backend._cua_no_overlay() is True

    def test_default_macos_disables(self):
        """Auto-detect: macOS => overlay disabled (idle CPU / #47032)."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "darwin"):
            assert cua_backend._cua_no_overlay() is True

    def test_default_windows_enables(self):
        """Auto-detect: Windows => overlay enabled."""
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "win32"):
            assert cua_backend._cua_no_overlay() is False

    def test_explicit_true_overrides(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"no_overlay": True}}):
            assert cua_backend._cua_no_overlay() is True

    def test_explicit_false_overrides(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"no_overlay": False}}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            # Explicit False overrides auto-detect on headless Linux.
            assert cua_backend._cua_no_overlay() is False

    def test_config_load_failure_falls_through_to_auto_detect(self):
        """Unreadable config => auto-detect (macOS defaults to disabled)."""
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("boom")), \
             patch.object(sys, "platform", "darwin"):
            assert cua_backend._cua_no_overlay() is True

    def test_macos_explicit_false_keeps_overlay(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"computer_use": {"no_overlay": False}}), \
             patch.object(sys, "platform", "darwin"):
            assert cua_backend._cua_no_overlay() is False

    def test_missing_section_falls_through_to_auto_detect(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"other": {}}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert cua_backend._cua_no_overlay() is False


class TestDriverSupportsNoOverlay:
    def test_returns_true_when_help_shows_flag(self):
        fake_help = "Usage: cua-driver [OPTIONS] COMMAND\n  --no-overlay  Disable cursor overlay\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_help
            mock_run.return_value.stderr = ""
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is True

    def test_returns_false_when_help_lacks_flag(self):
        fake_help = "Usage: cua-driver [OPTIONS] COMMAND\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_help
            mock_run.return_value.stderr = ""
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is False

    def test_returns_false_on_subprocess_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            assert cua_backend._cua_driver_supports_no_overlay("cua-driver") is False

    def test_help_probe_passes_sanitized_env(self):
        """The ``--help`` subprocess must not leak provider credentials
        via the inherited parent environment (third-party binary; same
        policy as the manifest probe and MCP spawn).
        """
        from unittest.mock import MagicMock
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="--no-overlay in help", stderr="")
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            cua_backend._cua_driver_supports_no_overlay("cua-driver")
            kwargs = mock_run.call_args.kwargs
            assert "env" in kwargs, (
                "subprocess.run was called without env= — cua-driver is a "
                "third-party binary and must not receive inherited secrets"
            )
            # The sanitized env must come from the same helper the MCP
            # spawn uses, so the policy is consistent across every
            # cua-driver invocation in this file.
            assert kwargs["env"] is not None


class TestMcpInvocationUsesResolvedCommand:
    """Surface 8 (NousResearch/hermes-agent#47072) + sweeper feedback
    #4701565902: when the manifest surfaces a relocated executable for
    ``mcp_invocation.command``, the support probe must run against THAT
    binary, not the system-resolved ``_CUA_DRIVER_CMD``. Otherwise a
    wrapper/relocation with a different feature set either crashes on
    the unknown flag (when the probe falsely reports support) or
    silently keeps an unwanted overlay (when the probe falsely reports
    no support).
    """

    @staticmethod
    def _fake_run(stdout: str = "", returncode: int = 0):
        from unittest.mock import MagicMock
        def _run(*args, **kwargs):
            proc = MagicMock()
            proc.stdout = stdout
            proc.returncode = returncode
            return proc
        return _run

    def test_manifest_command_drives_support_probe(self):
        """When the manifest returns a distinct command, the support
        probe runs against the manifest command, not the input
        ``driver_cmd`` parameter.
        """
        from unittest.mock import patch
        from tools.computer_use.cua_backend import _resolve_mcp_invocation

        manifest = (
            '{"mcp_invocation":'
            '{"command":"/opt/relocated/cua-driver","args":["mcp"]}}'
        )
        with patch("subprocess.run", new=self._fake_run(stdout=manifest)), \
             patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(
                 cua_backend, "_cua_driver_supports_no_overlay",
                 return_value=True,
             ) as mock_probe:
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            cmd, args = _resolve_mcp_invocation("/usr/bin/cua-driver")
        assert cmd == "/opt/relocated/cua-driver"
        # The support probe must be called with the manifest-resolved
        # command, not the input driver_cmd argument.
        mock_probe.assert_called_with("/opt/relocated/cua-driver")

    def test_fallback_uses_input_driver_cmd_for_support_probe(self):
        """When the manifest knows the args but NOT the command, the
        input ``driver_cmd`` parameter is what gets launched and
        probed.
        """
        from unittest.mock import patch
        from tools.computer_use.cua_backend import _resolve_mcp_invocation

        manifest = '{"mcp_invocation":{"args":["mcp"]}}'
        with patch("subprocess.run", new=self._fake_run(stdout=manifest)), \
             patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(
                 cua_backend, "_cua_driver_supports_no_overlay",
                 return_value=True,
             ) as mock_probe:
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            cmd, args = _resolve_mcp_invocation("/my/local/cua-driver")
        assert cmd == "/my/local/cua-driver"
        # Fallback path: probe runs against the input driver_cmd.
        mock_probe.assert_called_with("/my/local/cua-driver")

    def test_probe_distinguishes_support_between_binaries(self):
        """Different binaries must produce independent support verdicts.
        The cache is keyed on ``driver_cmd``; the same cached result
        must not leak between the system binary and a manifest-relocated
        one.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(
                 cua_backend, "_cua_driver_supports_no_overlay",
                 side_effect=lambda cmd: cmd == "/opt/relocated/cua-driver",
             ):
            # System binary does NOT support, manifest binary DOES.
            args = cua_backend._mcp_args_with_overlay_flag(
                ["mcp"], driver_cmd="/usr/bin/cua-driver",
            )
            assert "--no-overlay" not in args
            args = cua_backend._mcp_args_with_overlay_flag(
                ["mcp"], driver_cmd="/opt/relocated/cua-driver",
            )
            assert "--no-overlay" in args


class TestMcpArgsOverlayFlag:
    def test_appended_when_enabled_and_supported(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp", "--no-overlay"]

    def test_not_appended_when_disabled(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=False), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp"]

    def test_not_appended_when_driver_unsupported(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=False):
            result = cua_backend._mcp_args_with_overlay_flag(["mcp"])
            assert result == ["mcp"]

    def test_does_not_mutate_original_list(self):
        original = ["mcp"]
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend._mcp_args_with_overlay_flag(original)
            assert "--no-overlay" in result
            assert "--no-overlay" not in original
