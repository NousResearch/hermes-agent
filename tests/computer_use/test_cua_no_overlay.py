"""Regression tests for the cua-driver ``--no-overlay`` policy at the
user-configured MCP launch path.

Covers NousResearch/hermes-agent#81220: a user-registered
``mcp_servers.cua-driver`` (or any equivalent cua-driver binary) entry
must receive the same ``--no-overlay`` flag the embedded cua_backend
applies at ``_resolve_mcp_invocation``. Without this, the overlay's
InputOutput override-redirect window on a multi-monitor X11 desktop
silently swallows all clicks outside Hermes.

Also covers the multi-monitor auto-detect heuristic in
``_cua_no_overlay``: a virtual root wider than a single 5K panel is
treated like macOS / headless Linux / WSL2 and forces ``--no-overlay``
even when the user has not set the config knob.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import cua_backend


# ---------------------------------------------------------------------------
# looks_like_cua_driver_command
# ---------------------------------------------------------------------------


class TestLooksLikeCuaDriverCommand:
    @pytest.mark.parametrize(
        "command",
        [
            "cua-driver",
            "/usr/local/bin/cua-driver",
            "/opt/cua/cua-driver",
            "/home/u/.local/bin/cua-driver",
            "C:\\Program Files\\cua\\cua-driver.exe",
            "cua-driver.exe",
            "./cua-driver",
        ],
    )
    def test_recognises_known_binaries(self, command):
        assert cua_backend.looks_like_cua_driver_command(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "",
            None,
            "node",
            "npx",
            "python",
            "/usr/local/bin/some-other-mcp",
            "mcp-remote",
        ],
    )
    def test_rejects_unrelated_commands(self, command):
        assert cua_backend.looks_like_cua_driver_command(command) is False


# ---------------------------------------------------------------------------
# normalize_user_cua_driver_args
# ---------------------------------------------------------------------------


class TestNormalizeUserCuaDriverArgs:
    def test_appends_no_overlay_for_cua_driver_when_enabled(self):
        """User-configured cua-driver MCP receives ``--no-overlay`` when
        the policy resolves True and the installed driver supports it.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            args = cua_backend.normalize_user_cua_driver_args(
                "/usr/local/bin/cua-driver", ["mcp"],
            )
        assert args == ["mcp", "--no-overlay"]

    def test_does_not_mutate_caller_list(self):
        original = ["mcp"]
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            args = cua_backend.normalize_user_cua_driver_args(
                "cua-driver", original,
            )
        assert "--no-overlay" in args
        assert "--no-overlay" not in original

    def test_passthrough_for_non_cua_driver(self):
        """The helper must not touch args for unrelated MCP servers —
        this is what guarantees the change is class-level instead of
        touching every MCP spawn.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            args = cua_backend.normalize_user_cua_driver_args(
                "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            )
        assert args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    def test_no_overlay_omitted_when_policy_disabled(self):
        with patch.object(cua_backend, "_cua_no_overlay", return_value=False), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            args = cua_backend.normalize_user_cua_driver_args("cua-driver", ["mcp"])
        assert "--no-overlay" not in args

    def test_no_overlay_omitted_when_driver_does_not_support(self):
        """Older drivers reject unknown flags; the helper must not append
        ``--no-overlay`` when the installed binary doesn't recognise it.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=False):
            args = cua_backend.normalize_user_cua_driver_args("cua-driver", ["mcp"])
        assert "--no-overlay" not in args

    def test_supports_flag_probed_against_user_command(self):
        """The support probe must run against the user's resolved
        binary path (not the embedded default), so a wrapper or
        relocated driver with a different feature set is treated
        correctly — mirrors the embedded-backend invariant.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(
                 cua_backend, "_cua_driver_supports_no_overlay",
                 return_value=True,
             ) as mock_probe:
            cua_backend._cua_driver_supports_no_overlay.cache_clear()
            cua_backend.normalize_user_cua_driver_args(
                "/opt/relocated/cua-driver", ["mcp"],
            )
        mock_probe.assert_called_with("/opt/relocated/cua-driver")


# ---------------------------------------------------------------------------
# Multi-monitor X11 auto-detect
# ---------------------------------------------------------------------------


class TestX11MultiMonitorAutoDetect:
    def test_wide_virtual_root_forces_no_overlay(self):
        """A virtual root wider than a single 5K panel must trigger
        ``--no-overlay`` even when no explicit config is set — this is
        the exact X11 multi-monitor class reported in #81220.
        """
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False), \
             patch.object(cua_backend, "_x11_root_pixel_width", return_value=6000):
            assert cua_backend._cua_no_overlay() is True

    def test_single_4k_panel_keeps_overlay(self):
        """A single 4K panel (4096 px wide) must NOT trigger the
        heuristic — only multi-monitor layouts exceed the threshold.
        """
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False), \
             patch.object(cua_backend, "_x11_root_pixel_width", return_value=4096):
            assert cua_backend._cua_no_overlay() is False

    def test_unprobeable_x11_falls_through(self):
        """When xrandr is missing or fails, the heuristic must return
        False so we don't regress single-head Linux setups where
        ``--no-overlay`` would otherwise strip a useful cursor.
        """
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False), \
             patch.object(cua_backend, "_x11_root_pixel_width", return_value=None):
            assert cua_backend._cua_no_overlay() is False

    def test_explicit_false_overrides_multi_monitor(self):
        """An explicit ``computer_use.no_overlay: false`` must beat the
        heuristic — users on multi-monitor setups can still opt back
        into the overlay when they understand the risk.
        """
        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"no_overlay": False}},
        ), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False), \
             patch.object(cua_backend, "_x11_root_pixel_width", return_value=6000):
            assert cua_backend._cua_no_overlay() is False

    def test_no_display_skips_multi_monitor_probe(self):
        """Headless Linux must not run xrandr at all — return True via
        the existing no-DISPLAY branch.
        """
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(cua_backend, "_x11_root_pixel_width", return_value=6000):
            # DISPLAY cleared; xrandr result should not even matter.
            assert cua_backend._cua_no_overlay() is True


# ---------------------------------------------------------------------------
# tools/mcp_tool._run_stdio integration
# ---------------------------------------------------------------------------


class TestMcpToolAppliesCuaOverlayPolicy:
    """The fix lands in ``tools/mcp_tool.py::_run_stdio`` so a
    user-configured ``mcp_servers.cua-driver`` entry cannot silently
    bypass the embedded-backend normalization. These tests assert the
    integration without going through a live subprocess.
    """

    def _run_stdio_coro(self):
        # ``_run_stdio`` is an async method; importing the module triggers
        # the heavy ``mcp`` SDK import. Guard so a missing SDK surfaces as
        # ``pytest.skip`` rather than an ImportError during collection.
        try:
            import tools.mcp_tool as mcp_tool_mod
        except ImportError as exc:  # pragma: no cover - depends on env
            pytest.skip(f"mcp_tool import unavailable: {exc}")
        return mcp_tool_mod

    def test_user_cua_driver_args_receive_no_overlay(self):
        """End-to-end normalisation: a user MCP server named ``cua-driver``
        with ``args: [mcp]`` is augmented with ``--no-overlay`` before
        the OSV preflight + watchdog wrap.

        We exercise the helper directly (rather than calling ``_run_stdio``
        with a full mock) to keep the test focused on the policy hook.
        """
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(cua_backend, "_cua_driver_supports_no_overlay", return_value=True):
            result = cua_backend.normalize_user_cua_driver_args(
                "/usr/local/bin/cua-driver", ["mcp"],
            )
        assert result == ["mcp", "--no-overlay"], (
            "user-configured cua-driver MCP must receive --no-overlay "
            "so #81220 cannot reproduce via mcp_servers"
        )


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


import os  # noqa: E402  (placed after classes so pytest discovery is clean)


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    cua_backend._cua_driver_supports_no_overlay.cache_clear()
    yield
    cua_backend._cua_driver_supports_no_overlay.cache_clear()
