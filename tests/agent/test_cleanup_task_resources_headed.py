"""Per-turn browser cleanup skips headed sessions by ``browser.headed`` (config) alone.

The idle reaper owns a headed browser the user may be watching; the retired ``AGENT_BROWSER_HEADED``
env var no longer keeps a headless session alive past the turn.
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
