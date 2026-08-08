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


class TestMemoryStatusLazyPredicate:
    """Status must agree with the runtime retain path's lazy-deps
    predicate, not just provider.is_available() — e.g. hindsight reports
    "available" from config alone, but retain gates on
    ensure("memory.hindsight")."""

    def test_status_not_available_when_lazy_deps_unsatisfied(self, capfd, monkeypatch):
        """provider.is_available() True but lazy-deps unsatisfied
        must still print "not available" (mirrors the runtime predicate)."""
        from hermes_cli.memory_setup import cmd_status

        class ProviderAvailableNoDeps:
            def is_available(self):
                return True

            def get_config_schema(self):
                return []

        provider = ProviderAvailableNoDeps()

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"provider": "hindsight"}},
        )
        monkeypatch.setattr(
            "hermes_cli.memory_setup._get_available_providers",
            lambda: [("hindsight", "Hindsight memory", provider)],
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda *args, **kwargs: {"memory"},
        )
        monkeypatch.setattr("tools.lazy_deps.is_available", lambda feature: False)

        cmd_status(args=None)
        out = capfd.readouterr().out

        assert "Status:    available ✓" not in out
        assert "Status:    not available ✗" in out




