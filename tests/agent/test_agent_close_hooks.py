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


def test_second_agent_generation_fires_on_own_close():
    events_a: list = []
    events_b: list = []
    hooks.register_on_agent_close(lambda task_id: events_a.append(task_id))
    _bare_mixin()._close_task_resources("task-A")
    hooks.register_on_agent_close(lambda task_id: events_b.append(task_id), task_id="task-B")
    assert events_b == []
    _bare_mixin()._close_task_resources("task-B")
    assert events_a == ["task-A"]
    assert events_b == ["task-B"]


def test_stale_close_does_not_steal_new_generation():
    events_b: list = []
    _bare_mixin()._close_task_resources("task-A")
    hooks.register_on_agent_close(lambda task_id: events_b.append(task_id), task_id="task-B")
    _bare_mixin()._close_task_resources("task-A")
    assert events_b == []
    _bare_mixin()._close_task_resources("task-B")
    assert events_b == ["task-B"]
