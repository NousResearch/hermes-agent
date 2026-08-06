"""Behavior contract for the Claude-subscription release gate.

The gate is imported by the provider catalog and the dashboard web server, so
the invariants that matter are: it is off unless someone explicitly turned it
on, it never raises on a malformed config, and the SDK probe answers even when
the optional `claude-code` extra is not installed.
"""

from hermes_cli import claude_subscription
from hermes_cli.claude_subscription import (
    CLAUDE_AGENT_SDK_MIN_VERSION,
    CLAUDE_CLI_MIN_VERSION,
    claude_agent_sdk_available,
    claude_subscription_enabled,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_shipped_default_is_off():
    """The runtime ships default-off pending Anthropic policy clearance."""
    assert claude_subscription_enabled(DEFAULT_CONFIG) is False


def test_off_for_absent_empty_and_none_config():
    assert claude_subscription_enabled(None) is False
    assert claude_subscription_enabled({}) is False
    assert claude_subscription_enabled({"model": "", "agent": {}}) is False
    assert claude_subscription_enabled({"claude_subscription": {}}) is False


def test_on_only_when_explicitly_enabled():
    assert claude_subscription_enabled({"claude_subscription": {"enabled": True}}) is True
    assert claude_subscription_enabled({"claude_subscription": {"enabled": False}}) is False


def test_malformed_config_reads_as_off_without_raising():
    """A hand-edited config.yaml can put anything under the key; the gate must
    fail closed rather than blow up a startup path."""
    for section in ("yes", 1, [], ["enabled"], None):
        assert claude_subscription_enabled({"claude_subscription": section}) is False
    for config in ("not-a-dict", [], 0):
        assert claude_subscription_enabled(config) is False


def test_availability_probe_returns_bool_and_never_raises():
    assert isinstance(claude_agent_sdk_available(), bool)


def test_availability_probe_is_false_without_the_optional_extra(monkeypatch):
    claude_agent_sdk_available.cache_clear()
    monkeypatch.setattr(claude_subscription.importlib.util, "find_spec", lambda name: None)
    try:
        assert claude_agent_sdk_available() is False
    finally:
        claude_agent_sdk_available.cache_clear()


def test_availability_probe_swallows_a_broken_import_system(monkeypatch):
    """A shadowed/half-installed `claude_agent_sdk` makes find_spec raise. The
    probe runs on startup paths, so it must answer False, not propagate."""
    def _boom(name):
        raise ImportError(name)

    claude_agent_sdk_available.cache_clear()
    monkeypatch.setattr(claude_subscription.importlib.util, "find_spec", _boom)
    try:
        assert claude_agent_sdk_available() is False
    finally:
        claude_agent_sdk_available.cache_clear()


def test_pinned_versions_are_orderable_version_strings():
    """Downstream PRs compare an installed version against these floors, so
    both constants must parse as dotted numeric versions."""
    for pin in (CLAUDE_AGENT_SDK_MIN_VERSION, CLAUDE_CLI_MIN_VERSION):
        parts = pin.split(".")
        assert len(parts) >= 3
        assert all(part.isdigit() for part in parts)
