from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from agent.routing_decision import build_routing_decision
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_routing as kbr


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(title="Fix review", assignee="rozmilo-codex"):
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, assignee=assignee)


def _identity(profile):
    return ("openai-codex", "codex-model") if profile == "rozmilo-codex" else ("anthropic", "claude-model")


def _enable(monkeypatch):
    from hermes_cli import kanban_fix_review as fix
    monkeypatch.setattr(fix, "fix_review_command_enabled", lambda: True)
    monkeypatch.setattr(fix, "_profile_runtime_identity", _identity)
    monkeypatch.setattr("hermes_cli.kanban_implement._profile_runtime_identity", _identity)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _profile: True)


def _seed_completed_implementation(task_id: str, profile="rozmilo-codex"):
    with kbc.connect() as conn:
        claimed = kb.claim_task(conn, task_id, claimer="implementation-a")
        decision = build_routing_decision(
            task_id=task_id, board="default", task_type="implementation", capability="implement",
            risk="normal", code_change=True, independent_review=True,
            preferred_profile=profile, reviewer_profile="rozmilo-claude" if profile == "rozmilo-codex" else "rozmilo-codex",
            selected_profile=profile, selected_provider=_identity(profile)[0], selected_model="impl-a",
            human_gate_required=False, independence_valid=True, policy_digest=None,
            selected_by="test", run_id=claimed.current_run_id, session_id="impl-a",
        )
        assert kbr.persist_run_routing_decision(conn, task_id=task_id, run_id=claimed.current_run_id,
                                                decision=decision, event_kind="routing_selected")
        assert kb.complete_task(conn, task_id, result="implemented", expected_run_id=claimed.current_run_id)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()


def _request_changes(task_id: str):
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.current_run_id is not None
        assert kb.request_changes(conn, task_id, reason="fix it", expected_run_id=task.current_run_id) == (True, "rozmilo-codex")


def _seed_changes_requested(task_id: str, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    _seed_completed_implementation(task_id)
    monkeypatch.setattr("hermes_cli.kanban_review.review_command_enabled", lambda: True)
    monkeypatch.setattr("hermes_cli.kanban_review._profile_runtime_identity", _identity)
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: 101)
    assert run_review_slash(task_id)["dispatch_status"] == "started"
    _request_changes(task_id)


def test_fix_review_rejects_without_changes_and_disabled_is_noop(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    task_id = _task()
    assert run_fix_review_slash(task_id)["dispatch_status"] == "disabled"
    _enable(monkeypatch)
    result = run_fix_review_slash(task_id)
    assert result["dispatch_status"] == "not_eligible"
    assert result["correction_required"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
        assert len(kb.list_runs(conn, task_id)) == 0


def test_fix_review_starts_fresh_codex_route_and_review_is_blocked_then_eligible(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    _enable(monkeypatch)
    _seed_changes_requested(task_id, monkeypatch)
    assert run_review_slash(task_id)["dispatch_status"] == "not_eligible"
    observed = {}

    def spawn(task, workspace, **_kwargs):
        with kbc.connect() as conn:
            observed["route"] = kb.get_run(conn, task.current_run_id).metadata["routing_decision"]
        return 202

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    result = run_fix_review_slash(task_id)
    assert result["dispatch_status"] == "started"
    assert result["implementation_profile"] == "rozmilo-codex"
    assert result["reviewer_profile"] == "rozmilo-claude"
    assert observed["route"]["selected_by"] == "fix-review"
    assert observed["route"]["task_type"] == "implementation"
    with kbc.connect() as conn:
        correction = kb.get_task(conn, task_id).current_run_id
        assert correction > result["changes_run_id"]
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 3
        assert kb.complete_task(conn, task_id, result="fixed", expected_run_id=correction)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()
    assert run_review_slash(task_id)["dispatch_status"] == "started"


def test_fix_review_is_idempotent_and_preserves_old_route(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    task_id = _task()
    _enable(monkeypatch)
    _seed_changes_requested(task_id, monkeypatch)
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: 303)
    first = run_fix_review_slash(task_id)
    second = run_fix_review_slash(task_id)
    assert second["dispatch_status"] == "already_active"
    assert first["run_id"] == second["run_id"]
    with kbc.connect() as conn:
        runs = kb.list_runs(conn, task_id)
        assert len(runs) == 4
        assert runs[0].metadata["routing_decision"]["selected_by"] == "test"
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 3


def test_fix_review_concurrent_calls_start_one_worker(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    task_id = _task()
    _enable(monkeypatch)
    _seed_changes_requested(task_id, monkeypatch)
    count, lock = 0, Lock()

    def spawn(*_args, **_kwargs):
        nonlocal count
        with lock:
            count += 1
        return 404

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    barrier = Barrier(4)

    def invoke(_):
        barrier.wait()
        return run_fix_review_slash(task_id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert sum(r["dispatch_status"] == "started" for r in results) == 1
    assert count == 1
    with kbc.connect() as conn:
        assert len(kb.list_runs(conn, task_id)) == 4
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 3


def test_fix_review_persistence_failure_does_not_spawn(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    task_id = _task()
    _enable(monkeypatch)
    _seed_changes_requested(task_id, monkeypatch)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: spawned.append(True))
    monkeypatch.setattr(kbr, "persist_run_routing_decision", lambda *_args, **_kwargs: False)
    result = run_fix_review_slash(task_id)
    assert result["dispatch_status"] == "routing_failed"
    assert spawned == []
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.current_run_id is None


def test_fix_review_supports_authorized_claude_and_rejects_blocked(kanban_home, monkeypatch):
    from hermes_cli.kanban_fix_review import run_fix_review_slash
    task_id = _task()
    _enable(monkeypatch)
    _seed_changes_requested(task_id, monkeypatch)
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: 505)
    result = run_fix_review_slash(f"{task_id} --profile rozmilo-claude")
    assert result["dispatch_status"] == "started"
    assert result["implementation_profile"] == "rozmilo-claude"
    assert result["reviewer_profile"] == "rozmilo-codex"

    with kbc.connect() as conn:
        correction = kb.get_task(conn, task_id).current_run_id
        assert kb.complete_task(conn, task_id, result="fixed", expected_run_id=correction)
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        conn.commit()
    assert run_fix_review_slash(task_id)["dispatch_status"] == "not_eligible"
