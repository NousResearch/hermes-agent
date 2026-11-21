"""Per-turn browser cleanup preserves only locally launched headed browsers.

The idle reaper owns a headed local browser the user may be watching; configured CDP and cloud
backends still clean up per turn. The retired ``AGENT_BROWSER_HEADED`` env var does not override config.
"""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import cleanup_task_resources
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def run(tmp_path, monkeypatch):
    import run_agent

    closed = []
    monkeypatch.setattr(run_agent, "cleanup_browser", closed.append)
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda task_id: None)

    def go(config_text: str) -> list:
        (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
        token = set_hermes_home_override(str(tmp_path))
        try:
            cleanup_task_resources(SimpleNamespace(verbose_logging=False), "task-1")
        finally:
            reset_hermes_home_override(token)
        return closed

    return go


def test_headed_config_keeps_the_browser_for_the_reaper(run):
    assert run("browser:\n  headed: true\n") == []


def test_headless_session_closes_even_with_the_retired_env_var(run, monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_HEADED", "1")
    assert run("browser:\n  headed: false\n") == ["task-1"]


def test_headed_config_with_cdp_override_still_cleans_browser(run):
    """Headed is local-only: a configured external CDP supervisor must not outlive the turn."""
    assert run("browser:\n  headed: true\n  cdp_url: http://127.0.0.1:9333\n") == ["task-1"]


def test_headed_config_with_cloud_provider_still_cleans_browser(run, monkeypatch):
    """Cloud sessions do not become persistent just because local headed mode is enabled."""
    monkeypatch.setattr("tools.browser_session_cloud.get_cloud_provider", lambda: object())
    assert run("browser:\n  headed: true\n") == ["task-1"]
