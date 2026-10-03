"""On-agent-close callback registry: run-once pre-cleanup, late-register, error swallow."""

import pytest

from agent import agent_close_hooks as hooks
from agent.client_lifecycle import ClientLifecycleMixin


@pytest.fixture(autouse=True)
def _clean_hooks():
    hooks.reset_for_tests()
    yield
    hooks.reset_for_tests()


def _bare_mixin():
    mixin = ClientLifecycleMixin()
    mixin._process_owner_task_ids = ()
    return mixin


def test_close_hook_fires_once_pre_cleanup(monkeypatch):
    import run_agent

    events: list = []
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda task_id: events.append("cleanup_vm"))
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda task_id: events.append("cleanup_browser"))
    hooks.register_on_agent_close(lambda task_id: events.append("hook"))
    mixin = _bare_mixin()
    mixin._close_task_resources("task-1")
    mixin._close_task_resources("task-1")
    assert events.count("hook") == 1
    assert events.index("hook") < events.index("cleanup_vm")


def test_late_register_runs_immediately():
    _bare_mixin()._close_task_resources("task-9")
    seen: list = []
    hooks.register_on_agent_close(lambda task_id: seen.append(task_id))
    assert seen == ["task-9"]


def test_hook_errors_do_not_break_close(monkeypatch):
    import run_agent

    events: list = []
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda task_id: events.append("cleanup_vm"))

    def _bad(task_id):
        raise RuntimeError("boom")

    hooks.register_on_agent_close(_bad)
    hooks.register_on_agent_close(lambda task_id: events.append("good"))
    _bare_mixin()._close_task_resources("task-2")
    assert "good" in events and "cleanup_vm" in events
