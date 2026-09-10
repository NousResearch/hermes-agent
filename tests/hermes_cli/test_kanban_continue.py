from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_continue as cont
from hermes_cli import kanban_db_dispatch as kbd


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
        lambda _ref, board=None, **_kwargs: SimpleNamespace(ok=True, scope="task", task_id=task.id, board=board or "default", error=None),
    )
    monkeypatch.setattr(cont.kbc, "connect", lambda board: _Conn())
    monkeypatch.setattr(cont.kb, "get_task", lambda _conn, _task_id: task)
    monkeypatch.setattr(cont.kb, "latest_run", lambda _conn, _task_id: None)
    monkeypatch.setattr(cont.fix_review, "_latest_changes_run", lambda _conn, _task_id: None)
    monkeypatch.setattr(cont, "_active_result", lambda *_args: None)
    monkeypatch.setattr(cont, "_recovery_result", lambda *_args: None)
    monkeypatch.setattr(cont.implement, "run_implement_slash", delegate)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_normal_ready_delegates_to_implement(monkeypatch, task):
    seen = []
    _wire(monkeypatch, task, lambda text: seen.append(text) or {"command": "implement", "dispatch_status": "started", "task_id": task.id})

    result = cont.run_continue_slash("t_123")

    assert result["selected_action"] == "implement"
    assert result["delegated_command"] == "implement"
    assert seen == ["t_123 --board default"]


def test_rate_limited_active_cooldown_stops_before_implementation(monkeypatch, task):
    seen = []
    recovery_result = cont._recovery_result
    _wire(monkeypatch, task, lambda text: seen.append(text) or pytest.fail("implementation must not occur"))
    monkeypatch.setattr(cont, "_recovery_result", recovery_result)
    monkeypatch.setattr(
        cont.kb,
        "latest_run",
        lambda _conn, _task_id: SimpleNamespace(outcome="rate_limited"),
    )
    monkeypatch.setattr(
        cont.kbd,
        "recovery_requirement_for_task",
        lambda _conn, _task_id: "rate_limit_cooldown",
    )

    result = cont.run_continue_slash("t_123")

    assert result["recovery_required"] is True
    assert result["selected_action"] == "recover-required"
    assert result["delegated_command"] is None
    assert result["dispatch_status"] == "recovery_required"
    assert seen == []


def test_rate_limited_real_sqlite_reproduction_is_read_only(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setattr(cont, "continue_command_enabled", lambda: True)
    delegated = []

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="rate limited", assignee="rozmilo-codex")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        assert kb.claim_task(conn, task_id, claimer="test:rate-limit") is not None
        run_id = kb.get_task(conn, task_id).current_run_id
        now = int(__import__("time").time())
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='rate_limited', outcome='rate_limited', "
                "ended_at=? WHERE id=?",
                (now, run_id),
            )
            conn.execute(
                "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (task_id,),
            )
            kb._append_event(conn, task_id, "rate_limited", {"retry_status": "ready"}, run_id=run_id)
        before = {
            "task": tuple(conn.execute(
                "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid "
                "FROM tasks WHERE id=?", (task_id,),
            ).fetchone()),
            "runs": conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0],
        }

    monkeypatch.setattr(
        cont.implement, "run_implement_slash",
        lambda text: delegated.append(text) or pytest.fail("implementation must not occur"),
    )
    result = cont.run_continue_slash(task_id)

    assert result["recovery_required"] is True
    assert result["selected_action"] == "recover-required"
    assert result["delegated_command"] is None
    assert result["dispatch_status"] == "recovery_required"
    assert delegated == []
    with kbc.connect() as conn:
        after = {
            "task": tuple(conn.execute(
                "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid "
                "FROM tasks WHERE id=?", (task_id,),
            ).fetchone()),
            "runs": conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0],
        }
    assert after == before


def test_rate_limit_expiry_matches_dispatcher_eligibility(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="expired rate limit", assignee="a")
        conn.execute("UPDATE tasks SET status='ready', last_failure_error=? WHERE id=?", ("rate-limited", task_id))
        kb.claim_task(conn, task_id, claimer="test:expired")
        run_id = kb.get_task(conn, task_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET status='rate_limited', outcome='rate_limited', ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?", (task_id,))
        conn.commit()
        monkeypatch.setattr(kbd.time, "time", lambda: now + 301)
        assert kbd.check_respawn_guard(conn, task_id) is None
        assert kbd.recovery_requirement_for_task(conn, task_id) is None


@pytest.mark.parametrize("outcome", [
    "crashed", "timed_out", "spawn_failed", "reclaimed", "stale", "gave_up",
])
def test_current_failure_outcomes_require_recovery(kanban_home, outcome):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title=outcome, assignee="a")
        kb.claim_task(conn, task_id, claimer=f"test:{outcome}")
        run_id = kb.get_task(conn, task_id).current_run_id
        now = int(__import__("time").time())
        conn.execute(
            "UPDATE task_runs SET status=?, outcome=?, ended_at=? WHERE id=?",
            (outcome, outcome, now, run_id),
        )
        conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?", (task_id,))
        conn.commit()
        assert kbd.recovery_requirement_for_task(conn, task_id) == outcome


def test_requeued_failure_is_not_permanently_recovery_required(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="requeued", assignee="a")
        kb.claim_task(conn, task_id, claimer="test:history")
        run_id = kb.get_task(conn, task_id).current_run_id
        now = int(__import__("time").time())
        conn.execute(
            "UPDATE task_runs SET status='crashed', outcome='crashed', ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, last_failure_error=? WHERE id=?",
            ("quota/auth failure from old attempt", task_id),
        )
        conn.commit()
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "status", {"status": "ready"})
        assert kbd.recovery_requirement_for_task(conn, task_id) is None


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


def test_archived_is_terminal_without_delegation_or_fabricated_dispatch(monkeypatch, task):
    task.status = "archived"
    seen = {}
    _wire(monkeypatch, task, lambda _text: pytest.fail("delegation must not occur"))
    monkeypatch.setattr(
        cont,
        "resolve_status_reference",
        lambda _ref, board=None, **kwargs: seen.update(kwargs) or SimpleNamespace(
            ok=True, scope="task", task_id=task.id, board=board or "default", error=None,
        ),
    )

    result = cont.run_continue_slash("t_123")

    assert seen == {"include_archived": True}
    assert result == {
        **cont._result(
            task_id=task.id,
            board="default",
            task_status="archived",
            continuation_state="terminal",
            selected_action="terminal",
            dispatch_status="not_eligible",
            terminal=True,
            message="task is archived; it will not be restarted",
        ),
    }


def test_archived_continue_is_a_read_only_terminal_noop(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Archived continue", assignee="rozmilo-codex")
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task_id,))
        conn.commit()
        before_runs = len(kb.list_runs(conn, task_id))
        before_events = [event.kind for event in kb.list_events(conn, task_id)]

    monkeypatch.setattr(cont, "continue_command_enabled", lambda: True)
    monkeypatch.setattr(cont.implement, "run_implement_slash", lambda *_args: pytest.fail("implemented"))
    monkeypatch.setattr(cont.review, "run_review_slash", lambda *_args: pytest.fail("reviewed"))
    monkeypatch.setattr(cont.fix_review, "run_fix_review_slash", lambda *_args: pytest.fail("fixed"))

    result = cont.run_continue_slash(task_id)

    assert result["command"] == "continue"
    assert result["task_id"] == task_id
    assert result["board"] == "default"
    assert result["task_status"] == "archived"
    assert result["terminal"] is True
    assert result["recovery_required"] is False
    assert result["selected_action"] == "terminal"
    assert result["delegated_command"] is None
    assert result["run_id"] is None
    assert result["decision_id"] is None
    assert result["implementation_provider"] is None
    assert result["implementation_model"] is None
    assert result["reviewer_provider"] is None
    assert result["reviewer_model"] is None

    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "archived"
        assert len(kb.list_runs(conn, task_id)) == before_runs == 0
        assert [event.kind for event in kb.list_events(conn, task_id)] == before_events == ["created"]


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
