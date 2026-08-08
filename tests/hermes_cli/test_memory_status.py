"""Tests for `hermes memory status` CLI command.

Covers:
- Status output shows config-aware indicators instead of hardcoded 'always active'
- memory_enabled, user_profile_enabled, and memory tool are each reflected
- Memory tool resolution uses the canonical _get_platform_tools resolver
- Original issue: 'Built-in: always active' was misleading when features were disabled
"""

import pytest
from unittest.mock import patch


def _run_cmd_status(capfd, mem_config=None, memory_tools=None, providers=None):
    """Run cmd_status with a mocked config and return captured stdout.

    Args:
        mem_config: The "memory" section of config.
        memory_tools: Set of tool names returned by _get_platform_tools.
                      Defaults to {"memory"} (tool enabled).
        providers: Optional list of (name, kind, provider) tuples returned by
                   _get_available_providers. Defaults to [] (no plugins).
    """
    from hermes_cli.memory_setup import cmd_status

    config = {"memory": mem_config or {}}
    if memory_tools is None:
        memory_tools = {"memory"}

    with patch("hermes_cli.config.load_config", return_value=config):
        with patch("hermes_cli.memory_setup._get_available_providers", return_value=providers or []):
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


class _StubProvider:
    """Minimal memory provider with a controllable static availability check."""

    def __init__(self, available=True, schema=None):
        self._available = available
        self._schema = schema or []

    def is_available(self):
        return self._available

    def get_config_schema(self):
        return self._schema


class TestStatusUsesRuntimePredicate:
    """`hermes memory status` must share the runtime retain-path predicate
    (issue #80388): a provider whose plugin check passes but whose lazy deps
    are missing shows "not available" with the reason — status can no longer
    be green while every retain fails. Healthy output is unchanged."""

    def _run(self, capfd, provider_name, provider, versions=None):
        import importlib.metadata as _md
        from importlib.metadata import PackageNotFoundError

        providers = [(provider_name, "stub", provider)]
        base = dict(mem_config={"provider": provider_name}, providers=providers)
        if versions is None:
            return _run_cmd_status(capfd, **base)

        def _version(pkg):
            if pkg in versions:
                return versions[pkg]
            raise PackageNotFoundError(pkg)

        with patch.object(_md, "version", _version):
            return _run_cmd_status(capfd, **base)

    def test_provider_with_missing_lazy_deps_is_not_available(self, capfd):
        # Plugin check passes (is_available True) but the runtime retain path
        # would fail — hindsight-client isn't installed.
        out = self._run(capfd, "hindsight", _StubProvider(available=True), versions={})
        assert "available ✓" not in out
        assert "not available ✗" in out
        assert "hindsight-client>=0.6.1" in out  # reason surfaced

    def test_provider_with_off_pin_install_still_available(self, capfd):
        # Newer-than-pin installed: runtime is satisfied (never-downgrade),
        # and the status probe uses the same predicate as ensure().
        out = self._run(
            capfd, "hindsight", _StubProvider(available=True),
            versions={"hindsight-client": "0.8.6"},
        )
        assert "available ✓" in out
        assert "not available ✗" not in out

    def test_healthy_provider_output_unchanged(self, capfd):
        out = self._run(
            capfd, "hindsight", _StubProvider(available=True),
            versions={"hindsight-client": "0.6.1"},
        )
        assert "Plugin:    installed ✓" in out
        assert "Status:    available ✓" in out
        assert "not available" not in out

    def test_static_failure_still_reports_env_missing(self, capfd):
        # is_available() False keeps today's behavior: not available + the
        # schema's env-var "Missing:" block.
        schema = [{"env_var": "HINDSIGHT_API_KEY", "url": "https://example.com"}]
        out = self._run(
            capfd, "hindsight", _StubProvider(available=False, schema=schema),
            versions={"hindsight-client": "0.6.1"},
        )
        assert "Status:    not available ✗" in out
        assert "Missing:" in out
        assert "HINDSIGHT_API_KEY" in out

    def test_provider_without_lazy_feature_falls_back_to_plugin_check(self, capfd):
        # A provider with no LAZY_DEPS entry (e.g. a pure-plugin backend) is
        # judged by its own is_available() — no change in behavior.
        out = self._run(capfd, "holographic", _StubProvider(available=True))
        assert "Status:    available ✓" in out

    def test_no_provider_configured_still_works(self, capfd):
        out = self._run(capfd, "", _StubProvider())
        assert "Provider:  (none — built-in only)" in out



