"""Tests for `hermes memory status` CLI command.

Covers:
- Status output clarity when no provider is configured (built-in only)
- Status output clarity when an external provider is configured
- The status message should NOT confuse "built-in memory subsystem"
  with the "built-in storage backend" (issue #18404)
"""

import pytest
from unittest.mock import patch


def _run_cmd_status(capfd, mem_config=None):
    """Run cmd_status with a mocked config and return captured stdout."""
    from hermes_cli.memory_setup import cmd_status

    config = {"memory": mem_config or {}}

    with patch("hermes_cli.config.load_config", return_value=config):
        with patch("hermes_cli.memory_setup._get_available_providers", return_value=[]):
            cmd_status(args=None)

    captured = capfd.readouterr()
    return captured.out


class TestMemoryStatusClarity:
    """Status output should not mislead users about built-in store state."""

    def test_builtin_store_disabled_by_default(self, capfd):
        """When no memory flags are set, built-in store should be disabled."""
        out = _run_cmd_status(capfd)
        assert "Built-in store: disabled" in out
        assert "Built-in:  always active" not in out

    def test_builtin_store_active_when_memory_enabled(self, capfd):
        """When memory.memory_enabled is true, built-in store should be active."""
        out = _run_cmd_status(capfd, mem_config={"memory_enabled": True})
        assert "Built-in store: active" in out
        assert "Built-in:  always active" not in out

    def test_builtin_store_active_when_profile_enabled(self, capfd):
        """When memory.user_profile_enabled is true, built-in store should be active."""
        out = _run_cmd_status(capfd, mem_config={"user_profile_enabled": True})
        assert "Built-in store: active" in out
        assert "Built-in:  always active" not in out

    def test_provider_with_builtin_store_active(self, capfd):
        """When external provider AND built-in flags are set, both should show."""
        out = _run_cmd_status(
            capfd,
            mem_config={"provider": "mnemosyne", "memory_enabled": True}
        )
        assert "Built-in store: active" in out
        assert "mnemosyne" in out
        assert "Built-in:  always active" not in out

    def test_provider_with_builtin_store_disabled(self, capfd):
        """When external provider is set but built-in flags are disabled,
        built-in store should show as disabled."""
        out = _run_cmd_status(
            capfd,
            mem_config={"provider": "mnemosyne", "memory_enabled": False}
        )
        assert "Built-in store: disabled" in out
        assert "mnemosyne" in out
        assert "Built-in:  always active" not in out
