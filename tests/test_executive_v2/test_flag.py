"""Tests for Executive v2 flag resolution: default-off, opt-in."""

from __future__ import annotations

from agent.executive.flag import CONFIG_UNSET, resolve_v2_enabled
from tests.test_executive_v2.conftest import agent_stub  # noqa: F401


def test_resolve_v2_enabled_default_false(clean_env_executive, agent_stub):
    """Default-off: no env var, no attr flag -> False."""
    assert resolve_v2_enabled(agent=None) is False
    assert resolve_v2_enabled(agent=agent_stub) is False


def test_resolve_v2_enabled_via_legacy_env_var(clean_env_executive, monkeypatch):
    """Legacy env bridge enables only when config is unset."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    assert resolve_v2_enabled(agent=None) is True
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "true")
    assert resolve_v2_enabled(agent=None) is True
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "yes")
    assert resolve_v2_enabled(agent=None) is True


def test_resolve_v2_enabled_via_agent_attr_true(clean_env_executive):
    """Per-instance attr True overrides config."""
    class A:
        _executive_v2_enabled = True

    assert resolve_v2_enabled(agent=A(), config_value=False) is True


def test_resolve_v2_enabled_via_agent_attr_false(clean_env_executive):
    """Per-instance attr False is an explicit tri-state override."""
    class A:
        _executive_v2_enabled = False

    assert resolve_v2_enabled(agent=A(), config_value=True) is False


def test_resolve_v2_enabled_config_true(clean_env_executive):
    """Explicit config.yaml true enables Executive v2."""
    assert resolve_v2_enabled(config_value=True) is True


def test_resolve_v2_enabled_config_false_beats_env_true(clean_env_executive, monkeypatch):
    """Explicit config.yaml false has precedence over legacy env true."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    assert resolve_v2_enabled(config_value=False) is False


def test_resolve_v2_enabled_config_mapping_presence(clean_env_executive, monkeypatch):
    """Raw config presence distinguishes explicit false from default absence."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    assert resolve_v2_enabled(config={}) is True
    assert resolve_v2_enabled(config={"agent": {}}) is True
    assert resolve_v2_enabled(config={"agent": {"executive_v2_enabled": False}}) is False
    assert resolve_v2_enabled(config={"agent": {"executive_v2_enabled": True}}) is True


def test_resolve_v2_enabled_config_unset_sentinel_falls_through_to_env(
    clean_env_executive, monkeypatch
):
    """CONFIG_UNSET means absent user config, not explicit false."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    assert resolve_v2_enabled(config_value=CONFIG_UNSET) is True


def test_resolve_v2_enabled_env_var_falsy_values(clean_env_executive, monkeypatch):
    """Falsy env var values don't enable."""
    for v in ("0", "false", "no", "off", "", "definitely"):
        monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", v)
        assert resolve_v2_enabled(agent=None) is False, f"v={v!r} should be False"
