"""Explicit opt-ins for tests of retained upstream implementations.

Employee surface tests must not use these: they exercise the real fixed policy.
"""

import pytest


@pytest.fixture
def native_skills(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.SKILLS_ENABLED", True)
    from hermes_cli import commands
    monkeypatch.setattr(commands, "EMPLOYEE_EXCLUDED_COMMAND_NAMES",
                        commands.EMPLOYEE_EXCLUDED_COMMAND_NAMES - {"skills", "reload-skills", "learn", "bundles", "curator"})


@pytest.fixture
def native_kanban(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.KANBAN_ENABLED", True)


@pytest.fixture
def native_cron_authoring(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.NATIVE_CRON_AUTHORING_ENABLED", True)


@pytest.fixture
def native_personality(monkeypatch):
    monkeypatch.setattr("agent.employee_policy.LEGACY_PERSONALITY_ENABLED", True)


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


@pytest.fixture
def native_skills_dashboard(native_skills, monkeypatch):
    """Mount retained upstream handlers only for their library contract tests."""
    from hermes_cli import web_server
    from hermes_cli.web_routers import skills

    monkeypatch.setattr(web_server.app.router, "routes", list(web_server.app.router.routes))
    if not any(getattr(route, "path", "") == "/api/skills/content" for route in web_server.app.routes):
        web_server.app.include_router(skills.hub_router)
        web_server.app.include_router(skills.router)


@pytest.fixture
def native_cron_dashboard(native_cron_authoring, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.employee_surface import responsibility_authoring_only

    monkeypatch.setitem(web_server.app.dependency_overrides, responsibility_authoring_only, lambda: None)


@pytest.fixture
def native_personality_dashboard(native_personality, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.employee_surface import employee_instructions_only

    monkeypatch.setitem(web_server.app.dependency_overrides, employee_instructions_only, lambda: None)
