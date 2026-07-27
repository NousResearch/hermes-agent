"""Tests for `hermes memory status` CLI command.

Covers:
- Status output shows config-aware indicators instead of hardcoded 'always active'
- memory_enabled, user_profile_enabled, and memory tool are each reflected
- Memory tool resolution uses the canonical _get_platform_tools resolver
- Original issue: 'Built-in: always active' was misleading when features were disabled
"""

import pytest
from unittest.mock import patch


def _run_cmd_status(capfd, mem_config=None, memory_tools=None):
    """Run cmd_status with a mocked config and return captured stdout.

    Args:
        mem_config: The "memory" section of config.
        memory_tools: Set of tool names returned by _get_platform_tools.
                      Defaults to {"memory"} (tool enabled).
    """
    from hermes_cli.memory_setup import cmd_status

    config = {"memory": mem_config or {}}
    if memory_tools is None:
        memory_tools = {"memory"}

    with patch("hermes_cli.config.load_config", return_value=config):
        with patch("hermes_cli.memory_setup._get_available_providers", return_value=[]):
            with patch(
                "hermes_cli.tools_config._get_platform_tools",
                return_value=memory_tools,
            ):
                cmd_status(args=None)

    captured = capfd.readouterr()
    return captured.out


class TestMemoryStatusLabels:
    """Status output should reflect actual config, not a hardcoded string."""

    def test_no_hardcoded_always_active(self, capfd):
        """The old 'always active' label must not appear."""
        out = _run_cmd_status(capfd)
        assert "always active" not in out

    def test_shows_memory_injection_enabled_by_default(self, capfd):
        """Memory injection defaults to enabled."""
        out = _run_cmd_status(capfd)
        assert "Memory injection:" in out
        assert "enabled ✓" in out

    def test_shows_memory_injection_disabled(self, capfd):
        """When memory_enabled is false, status reflects it."""
        out = _run_cmd_status(capfd, mem_config={"memory_enabled": False})
        assert "Memory injection:" in out
        assert "disabled ✗" in out

    def test_shows_user_profile_disabled(self, capfd):
        """When user_profile_enabled is false, status reflects it."""
        out = _run_cmd_status(
            capfd, mem_config={"user_profile_enabled": False}
        )
        assert "User profile:" in out
        assert "disabled ✗" in out

    def test_shows_user_profile_enabled(self, capfd):
        """When user_profile_enabled is true, status reflects it."""
        out = _run_cmd_status(
            capfd, mem_config={"user_profile_enabled": True}
        )
        assert "User profile:" in out
        assert "enabled ✓" in out

    def test_memory_tool_enabled_by_default(self, capfd):
        """Memory tool is enabled by default."""
        out = _run_cmd_status(capfd)
        assert "Memory tool:" in out
        assert "enabled ✓" in out

    def test_memory_tool_disabled_via_toolset(self, capfd):
        """When CLI toolset excludes 'memory', the tool shows disabled."""
        out = _run_cmd_status(
            capfd,
            memory_tools={"terminal", "file"},
        )
        assert "Memory tool:" in out
        assert "disabled ✗" in out

    def test_memory_tool_enabled_via_toolset(self, capfd):
        """When CLI toolset includes 'memory', the tool shows enabled."""
        out = _run_cmd_status(
            capfd,
            memory_tools={"terminal", "file", "memory"},
        )
        assert "Memory tool:" in out
        assert "enabled ✓" in out

    def test_provider_still_shown(self, capfd):
        """Provider line still appears alongside the config indicators."""
        out = _run_cmd_status(
            capfd, mem_config={"provider": "honcho", "memory_enabled": True}
        )
        assert "honcho" in out
        assert "Memory injection:" in out

    def test_all_disabled(self, capfd):
        """All three indicators show disabled when everything is off."""
        out = _run_cmd_status(
            capfd,
            mem_config={"memory_enabled": False, "user_profile_enabled": False},
            memory_tools=set(),
        )
        assert out.count("disabled ✗") == 3
        assert "always active" not in out
