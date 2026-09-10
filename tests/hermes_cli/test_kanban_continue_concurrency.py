from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.routing_decision import build_routing_decision
from hermes_cli import kanban_continue as cont
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_routing as kbr
from hermes_cli import kanban_fix_review as fix_review
from hermes_cli import kanban_implement as implement
from hermes_cli import kanban_review as review
from hermes_cli.kanban_implement import ImplementationRoute


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def canonical_dispatch(monkeypatch: pytest.MonkeyPatch):
    route = ImplementationRoute(
        selected_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        implementation_provider="openai-codex",
        reviewer_provider="anthropic",
        selected_model="gpt-5.6-sol",
        independence_valid=True,
    )
    monkeypatch.setattr(cont, "continue_command_enabled", lambda: True)
    monkeypatch.setattr(implement, "implement_command_enabled", lambda: True)
    monkeypatch.setattr(fix_review, "fix_review_command_enabled", lambda: True)
    monkeypatch.setattr(review, "review_command_enabled", lambda: True)
    monkeypatch.setattr(implement.kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(implement, "select_implementation_route", lambda _profile: route)
    monkeypatch.setattr(fix_review, "select_implementation_route", lambda _profile: route)
    monkeypatch.setattr(review, "_profile_runtime_identity", lambda _profile: ("anthropic", "claude-sonnet"))

    workers: list[subprocess.Popen] = []

    def spawn(_task, _workspace, **_kwargs):
        worker = subprocess.Popen(["sleep", "30"])
        workers.append(worker)
        return worker.pid

    monkeypatch.setattr(implement.kbd, "_default_spawn", spawn)
    yield route
    for worker in workers:
        worker.terminate()
        worker.wait(timeout=5)


def _parallel_continue(task_id: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda _index: cont.run_continue_slash(task_id), range(4)))


def _assert_single_owner(results: list[dict], action: str) -> None:
    assert sum(result["dispatch_status"] == "started" for result in results) == 1
    assert all(result["selected_action"] == action for result in results)
    assert all(result["dispatch_status"] in {"started", "already_active", "already_running", "not_dispatched"} for result in results)
    started = next(result for result in results if result["dispatch_status"] == "started")
    assert started["run_id"] is not None


def _new_ready_task() -> str:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="concurrent continue", assignee="rozmilo-codex")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()
        return task_id


def _seed_review_task(*, changes_requested: bool = False) -> str:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="review continue", assignee="rozmilo-codex")
        claimed = kb.claim_task(conn, task_id, claimer="implementation")
        assert claimed is not None
        decision = build_routing_decision(
            task_id=task_id,
            board="default",
            task_type="implementation",
            capability="implement",
            risk="normal",
            code_change=True,
            independent_review=True,
            preferred_profile="rozmilo-codex",
            reviewer_profile="rozmilo-claude",
            selected_profile="rozmilo-codex",
            selected_provider="openai-codex",
            selected_model="gpt-5.6-sol",
            human_gate_required=False,
            independence_valid=True,
            policy_digest=None,
            selected_at="2026-09-10T00:00:00Z",
            selected_by="test",
            run_id=claimed.current_run_id,
            session_id=claimed.session_id,
        )
        assert kbr.persist_run_routing_decision(
            conn, task_id=task_id, run_id=claimed.current_run_id,
            decision=decision, event_kind="routing_selected",
        )
        assert kb.request_review(conn, task_id, expected_run_id=claimed.current_run_id)
        if changes_requested:
            review_claim = kb.claim_review_task(conn, task_id)
            assert review_claim is not None
            assert kb.request_changes(conn, task_id, reason="needs correction", expected_run_id=review_claim.current_run_id)
        return task_id


def test_ready_continue_uses_sqlite_cas_for_one_implementation(kanban_home, canonical_dispatch):
    results = _parallel_continue(_new_ready_task())

    _assert_single_owner(results, "implement")
    with kbc.connect() as conn:
        task = kb.get_task(conn, results[0]["task_id"])
        assert task.status == "running"
        assert len(kb.list_runs(conn, task.id)) == 1
        assert len([event for event in kb.list_events(conn, task.id) if event.kind == "routing_selected"]) == 1


def test_changes_requested_continue_uses_sqlite_cas_for_one_correction(kanban_home, canonical_dispatch):
    results = _parallel_continue(_seed_review_task(changes_requested=True))

    _assert_single_owner(results, "fix-review")
    with kbc.connect() as conn:
        task = kb.get_task(conn, results[0]["task_id"])
        assert task.status == "running"
        assert len(kb.list_runs(conn, task.id)) == 3
        assert len([event for event in kb.list_events(conn, task.id) if event.kind == "routing_selected"]) == 2


def test_review_continue_uses_sqlite_cas_for_one_review(kanban_home, canonical_dispatch):
    results = _parallel_continue(_seed_review_task())

    _assert_single_owner(results, "review")
    with kbc.connect() as conn:
        task = kb.get_task(conn, results[0]["task_id"])
        assert task.status == "running"
        assert len(kb.list_runs(conn, task.id)) == 2
        assert len([event for event in kb.list_events(conn, task.id) if event.kind == "routing_selected"]) == 2
