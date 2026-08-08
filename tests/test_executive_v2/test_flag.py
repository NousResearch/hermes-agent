"""Tests for Executive v2 flag resolution: default-off, opt-in."""

from __future__ import annotations

import pytest

from agent.executive.flag import resolve_v2_enabled
from tests.test_executive_v2.conftest import agent_stub  # noqa: F401


def test_resolve_v2_enabled_default_false(clean_env_executive, agent_stub):
    """Default-off: no env var, no attr flag -> False."""
    assert resolve_v2_enabled(agent=None) is False
    assert resolve_v2_enabled(agent=agent_stub) is False


def test_resolve_v2_enabled_via_env_var(clean_env_executive, monkeypatch):
    """Opt-in via env var."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    assert resolve_v2_enabled(agent=None) is True
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "true")
    assert resolve_v2_enabled(agent=None) is True
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "yes")
    assert resolve_v2_enabled(agent=None) is True


def test_resolve_v2_enabled_via_agent_attr(clean_env_executive):
    """Opt-in via per-instance attr."""
    class A:
        _executive_v2_enabled = True
    assert resolve_v2_enabled(agent=A()) is True


def test_resolve_v2_enabled_env_var_falsy_values(clean_env_executive, monkeypatch):
    """Falsy env var values don't enable."""
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", v)
        assert resolve_v2_enabled(agent=None) is False, f"v={v!r} should be False"
