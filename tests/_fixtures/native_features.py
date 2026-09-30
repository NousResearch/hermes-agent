"""Explicit opt-ins for tests of retained upstream implementations.

Employee surface tests must not use these: they exercise the real fixed policy.
"""

import pytest


@pytest.fixture
def native_skills(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.SKILLS_ENABLED", True)


@pytest.fixture
def manual_approvals(monkeypatch):
    from hermes_cli.config import DEFAULT_CONFIG

    monkeypatch.setitem(DEFAULT_CONFIG["approvals"], "mode", "manual")


@pytest.fixture
def native_tool_surface(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.select_tools", set)


@pytest.fixture
def native_browser_tools(monkeypatch):
    monkeypatch.setattr("tools.browser_use_cli.is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr("tools.browser_tool._is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
