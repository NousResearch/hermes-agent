from types import SimpleNamespace

import pytest

from hermes_cli import kanban_continue as cont


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def task():
    return SimpleNamespace(
        id="t_123", status="ready", current_run_id=None, claim_lock=None,
    )


def _wire(monkeypatch, task, delegate):
    monkeypatch.setattr(cont, "continue_command_enabled", lambda: True)
    monkeypatch.setattr(
        cont, "resolve_status_reference",
        lambda _ref, board=None: SimpleNamespace(ok=True, scope="task", task_id=task.id, board=board or "default", error=None),
    )
    monkeypatch.setattr(cont.kbc, "connect", lambda board: _Conn())
    monkeypatch.setattr(cont.kb, "get_task", lambda _conn, _task_id: task)
    monkeypatch.setattr(cont.kb, "latest_run", lambda _conn, _task_id: None)
    monkeypatch.setattr(cont.fix_review, "_latest_changes_run", lambda _conn, _task_id: None)
    monkeypatch.setattr(cont, "_active_result", lambda *_args: None)
    monkeypatch.setattr(cont, "_recovery_result", lambda *_args: None)
    monkeypatch.setattr(cont.implement, "run_implement_slash", delegate)


def test_normal_ready_delegates_to_implement(monkeypatch, task):
    seen = []
    _wire(monkeypatch, task, lambda text: seen.append(text) or {"command": "implement", "dispatch_status": "started", "task_id": task.id})

    result = cont.run_continue_slash("t_123")

    assert result["selected_action"] == "implement"
    assert result["delegated_command"] == "implement"
    assert seen == ["t_123 --board default"]


def test_changes_requested_delegates_to_fix_review(monkeypatch, task):
    seen = []
    _wire(monkeypatch, task, lambda _text: {"command": "implement", "dispatch_status": "started", "task_id": task.id})
    monkeypatch.setattr(cont.fix_review, "_latest_changes_run", lambda *_args: 8)
    monkeypatch.setattr(cont.fix_review, "_implementation_run_after", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cont.fix_review, "run_fix_review_slash", lambda text: seen.append(text) or {"dispatch_status": "started", "task_id": task.id})

    result = cont.run_continue_slash("t_123")

    assert result["selected_action"] == "fix-review"
    assert result["delegated_command"] == "fix-review"
    assert seen == ["t_123 --board default"]


def test_done_is_terminal_without_delegation(monkeypatch, task):
    task.status = "done"
    _wire(monkeypatch, task, lambda _text: pytest.fail("delegation must not occur"))

    result = cont.run_continue_slash("t_123")

    assert result["terminal"] is True
    assert result["selected_action"] == "terminal"
    assert result["dispatch_status"] == "not_eligible"


def test_active_implementation_is_reported_without_delegation(monkeypatch, task):
    task.status = "running"
    task.current_run_id = 17
    _wire(monkeypatch, task, lambda _text: pytest.fail("delegation must not occur"))
    monkeypatch.setattr(cont, "_active_result", lambda *_args: cont._result(
        task_id=task.id, task_status="running", continuation_state="active-implementation",
        selected_action="already-active", dispatch_status="already_active", run_id=17,
    ))

    result = cont.run_continue_slash("t_123")

    assert result["dispatch_status"] == "already_active"
    assert result["run_id"] == 17


def test_disabled_gate_does_not_resolve_or_mutate(monkeypatch):
    monkeypatch.setattr(cont, "continue_command_enabled", lambda: False)
    monkeypatch.setattr(cont, "resolve_status_reference", lambda *_args, **_kwargs: pytest.fail("must not resolve"))

    result = cont.run_continue_slash("t_123")

    assert result["dispatch_status"] == "disabled"
    assert result["command"] == "continue"
