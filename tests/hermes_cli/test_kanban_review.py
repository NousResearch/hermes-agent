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
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(title="Review this", assignee="rozmilo-codex"):
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, assignee=assignee)


def _seed_implementation_route(task_id: str, profile="rozmilo-codex", provider="openai-codex"):
    with kbc.connect() as conn:
        claimed = kb.claim_task(conn, task_id, claimer="implementation-test")
        assert claimed is not None
        decision = build_routing_decision(
            task_id=task_id, board="default", task_type="implementation", capability="implement",
            risk="normal", code_change=True, independent_review=True,
            preferred_profile=profile, reviewer_profile="rozmilo-claude" if profile == "rozmilo-codex" else "rozmilo-codex",
            selected_profile=profile, selected_provider=provider, selected_model="impl-model",
            human_gate_required=False, independence_valid=True, policy_digest=None,
            selected_by="test", run_id=claimed.current_run_id, session_id="implementation-session",
        )
        assert kbr.persist_run_routing_decision(
            conn, task_id=task_id, run_id=claimed.current_run_id,
            decision=decision, event_kind="routing_selected",
        )
        assert kb.complete_task(conn, task_id, result="implemented", expected_run_id=claimed.current_run_id)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()


def _enable(monkeypatch):
    monkeypatch.setattr("hermes_cli.kanban_review.review_command_enabled", lambda: True)
    monkeypatch.setattr("hermes_cli.kanban_review._profile_runtime_identity", lambda profile: (
        ("anthropic", "review-model") if profile == "rozmilo-claude" else ("openai-codex", "impl-model")
    ))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _profile: True)


def test_review_gate_disabled_is_no_mutation(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    result = run_review_slash(task_id)
    assert result["dispatch_status"] == "disabled"
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.current_run_id is None


def test_exact_id_dispatches_and_persists_route_before_spawn(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    _seed_implementation_route(task_id)
    _enable(monkeypatch)
    observed = {}

    def spawn(task, workspace, **_kwargs):
        with kbc.connect() as conn:
            observed["decision"] = kb.get_run(conn, task.current_run_id).metadata["routing_decision"]
        return 123

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    result = run_review_slash(task_id)
    assert result["dispatch_status"] == "started"
    assert result["reviewer_profile"] == "rozmilo-claude"
    assert result["independence_valid"] is True
    assert observed["decision"]["selected_profile"] == "rozmilo-claude"
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 2


def test_ambiguous_or_missing_reference_does_not_mutate(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    _enable(monkeypatch)
    first, second = _task("same"), _task("same")
    ambiguous = run_review_slash("same")
    missing = run_review_slash("does-not-exist")
    assert ambiguous["dispatch_status"] == "not_eligible"
    assert missing["dispatch_status"] == "not_eligible"
    with kbc.connect() as conn:
        assert kb.get_task(conn, first).status == "ready"
        assert kb.get_task(conn, second).status == "ready"
        assert [e.kind for e in kb.list_events(conn, first)] == ["created"]


def test_terminal_and_blocked_tasks_are_not_restarted(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    _enable(monkeypatch)
    for status in ("done", "archived", "blocked"):
        task_id = _task(status)
        with kbc.connect() as conn:
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()
        result = run_review_slash(task_id)
        assert result["dispatch_status"] == "not_eligible"
        with kbc.connect() as conn:
            assert kb.get_task(conn, task_id).current_run_id is None


def test_missing_provenance_fails_closed(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    _enable(monkeypatch)
    task_id = _task()
    result = run_review_slash(task_id)
    assert result["dispatch_status"] == "routing_failed"
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_repeated_review_is_idempotent(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    _seed_implementation_route(task_id)
    _enable(monkeypatch)
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: 123)
    first, second = run_review_slash(task_id), run_review_slash(task_id)
    assert first["run_id"] == second["run_id"]
    assert second["dispatch_status"] == "already_active"
    with kbc.connect() as conn:
        assert len(kb.list_runs(conn, task_id)) == 3  # implementation, handoff, reviewer
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == "routing_selected"]) == 2


def test_concurrent_review_dispatches_once(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    _seed_implementation_route(task_id)
    _enable(monkeypatch)
    count, lock = 0, Lock()

    def spawn(*_args, **_kwargs):
        nonlocal count
        with lock:
            count += 1
        return 123

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    barrier = Barrier(4)

    def invoke(_):
        barrier.wait()
        return run_review_slash(task_id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert sum(r["dispatch_status"] == "started" for r in results) == 1
    assert count == 1


def test_routing_failure_prevents_spawn_and_rolls_back_review_claim(kanban_home, monkeypatch):
    from hermes_cli.kanban_review import run_review_slash
    task_id = _task()
    _seed_implementation_route(task_id)
    _enable(monkeypatch)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda *_args, **_kwargs: spawned.append(True))
    monkeypatch.setattr(kbr, "persist_run_routing_decision", lambda *_args, **_kwargs: False)
    result = run_review_slash(task_id)
    assert result["dispatch_status"] == "routing_failed"
    assert spawned == []
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.claim_lock is None


def test_structured_result_has_required_fields(kanban_home):
    from hermes_cli.kanban_review import run_review_slash
    result = run_review_slash("unknown")
    assert {"command", "task_id", "board", "task_status", "review_status", "run_id", "decision_id",
            "implementation_profile", "implementation_provider", "implementation_model", "reviewer_profile",
            "reviewer_provider", "reviewer_model", "independence_valid", "dispatch_status",
            "human_gate_required", "message"} <= result.keys()
