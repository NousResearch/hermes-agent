"""Durable standing-goal ownership for asynchronous delegations."""

from __future__ import annotations

import queue
import time
import json

import pytest

import hermes_cli.goals as goals
import tools.async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    goals._DB_CACHE.clear()
    ad._reset_for_tests()
    while True:
        try:
            process_registry.completion_queue.get_nowait()
        except queue.Empty:
            break
    yield
    ad._reset_for_tests()
    goals._DB_CACHE.clear()


def _dispatch(goal_id: str, delegation_id: str, result: dict, *, blocker=None):
    def runner():
        if blocker is not None:
            blocker.wait(timeout=5)
        return result

    return ad.dispatch_async_delegation(
        goal=f"work for {delegation_id}",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="session-a",
        parent_session_id="session-a",
        goal_id=goal_id,
        requires_goal_join=True,
        goal_owner_session_id="session-a",
        delegation_id=delegation_id,
        runner=runner,
    )


def _event(delegation_id: str):
    deadline = time.time() + 5
    deferred = []
    try:
        while time.time() < deadline:
            try:
                event = process_registry.completion_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("delegation_id") == delegation_id:
                return event
            deferred.append(event)
    finally:
        for event in deferred:
            process_registry.completion_queue.put(event)
    raise AssertionError(f"no completion event for {delegation_id}")


def test_goal_db_uses_current_profile_after_state_module_import(tmp_path, monkeypatch):
    import hermes_state

    target_home = tmp_path / "active-profile"
    monkeypatch.setenv("HERMES_HOME", str(target_home))
    monkeypatch.setattr(
        hermes_state,
        "DEFAULT_DB_PATH",
        tmp_path / "previous-profile" / "state.db",
    )
    goals._DB_CACHE.clear()

    db = goals._get_session_db()
    assert db is not None
    assert db.db_path == target_home / "state.db"


def test_goal_id_is_append_only_for_positional_constructor_compatibility():
    state = goals.GoalState("legacy positional goal", "paused")

    assert state.status == "paused"
    assert state.goal_id == ""


def test_goal_owner_capture_fails_closed_when_persistence_is_unavailable(monkeypatch):
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)

    with pytest.raises(RuntimeError, match="persistence is unavailable"):
        goals.capture_goal_delegation_owner("session-a")


def test_terminal_results_are_claimed_once_and_block_done_until_reconciled(monkeypatch):
    manager = goals.GoalManager("session-a")
    state = manager.set("Ship the verified change", max_turns=8)

    _dispatch(
        state.goal_id,
        "deleg_one",
        {
            "status": "completed",
            "summary": "first verified result",
            "tokens": {"input": 11, "output": 7, "reasoning": 3},
            "cost_usd": 0.01,
        },
    )
    _dispatch(
        state.goal_id,
        "deleg_two",
        {
            "status": "error",
            "summary": "second worker failed after partial progress",
            "error": "verification command failed",
            "tokens": {"input": 5, "output": 2},
            "cost_usd": 0.02,
        },
    )
    first_event = _event("deleg_one")
    second_event = _event("deleg_two")

    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("done", "done", False, None, False))
    blocked = goals.GoalManager("session-a").evaluate_after_turn("children dispatched")
    assert blocked["verdict"] == "waiting_for_reconciliation"
    current = goals.GoalManager("session-a").state
    assert current is not None
    assert current.status == "active"

    decision = goals.prepare_goal_delegation_delivery(
        first_event,
        "legacy completion",
        current_session_id="session-a",
        consumer="test",
    )
    assert decision["classification"] == "current"
    assert "first verified result" in decision["prompt"]
    assert "second worker failed after partial progress" in decision["prompt"]
    assert "verification command failed" in decision["prompt"]
    assert sorted(decision["delegation_ids"]) == ["deleg_one", "deleg_two"]
    assert decision["reconciliation_attempt"] == 1

    snapshot = ad.get_goal_work_snapshot(state.goal_id)
    assert snapshot["claimed_count"] == 2
    assert snapshot["terminal_unreconciled_count"] == 0
    assert snapshot["tokens"] == {"input": 16, "output": 9, "reasoning": 3}
    assert snapshot["cost_usd"] == pytest.approx(0.03)
    assert snapshot["completed_count"] == 1
    assert snapshot["failed_count"] == 1

    sibling = goals.prepare_goal_delegation_delivery(
        second_event,
        "legacy completion",
        current_session_id="session-a",
        consumer="test",
    )
    assert sibling["prompt"] is None

    assert goals.complete_goal_reconciliation_turn(decision["reconciliation_claim"])
    assert ad.get_goal_work_snapshot(state.goal_id)["reconciled_count"] == 2
    finished = goals.GoalManager("session-a").evaluate_after_turn("results integrated")
    assert finished["status"] == "done"


def test_running_required_work_blocks_done_and_budget_exhaustion_pauses(monkeypatch):
    import threading

    blocker = threading.Event()
    state = goals.GoalManager("session-a").set("Wait for proof", max_turns=1)
    _dispatch(
        state.goal_id,
        "deleg_running",
        {"status": "completed", "summary": "proof"},
        blocker=blocker,
    )
    status_line = goals.GoalManager("session-a").status_line()
    assert "async=1 running" in status_line
    assert "deleg_running" in status_line
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("done", "done", False, None, False))

    verdict = goals.GoalManager("session-a").evaluate_after_turn("dispatched")
    assert verdict["status"] == "paused"
    assert "required async delegation" in verdict["reason"]
    current = goals.GoalManager("session-a").state
    assert current is not None
    assert current.status == "paused"
    blocker.set()
    _event("deleg_running")


def test_direct_mark_done_and_snapshot_failure_are_fail_closed(monkeypatch):
    manager = goals.GoalManager("session-a")
    state = manager.set("cannot finish early")
    _dispatch(
        state.goal_id,
        "deleg_mark_done_guard",
        {"status": "completed", "summary": "awaiting reconciliation"},
    )
    _event("deleg_mark_done_guard")

    manager.mark_done("claimed success")
    guarded = goals.GoalManager("session-a").state
    assert guarded is not None
    assert guarded.status == "active"
    assert guarded.last_verdict == "waiting_for_reconciliation"

    monkeypatch.setattr(
        goals,
        "get_goal_work_snapshot",
        lambda _goal_id: {"available": False, "error": "database offline"},
    )
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("done", "done", False, None, False))
    decision = goals.GoalManager("session-a").evaluate_after_turn("looks complete")
    assert decision["verdict"] == "waiting_for_reconciliation"
    assert decision["status"] == "active"
    assert "unavailable" in decision["reason"]


@pytest.mark.parametrize(
    "flags",
    [
        {"interrupted": True},
        {"failed": True},
        {"partial": True},
        {"error": True},
    ],
)
def test_reconciliation_success_rejects_nonempty_failed_or_partial_turns(flags):
    assert not goals.reconciliation_turn_succeeded("partial but visible output", **flags)
    assert goals.reconciliation_turn_succeeded("fully reconciled")


def test_paused_completion_waits_for_explicit_resume_and_preserves_goal_id():
    manager = goals.GoalManager("session-a")
    state = manager.set("Resume safely", max_turns=8)
    original_goal_id = state.goal_id
    _dispatch(
        state.goal_id,
        "deleg_paused",
        {"status": "completed", "summary": "ready while paused"},
    )
    event = _event("deleg_paused")
    manager.pause("user-paused")

    paused = goals.prepare_goal_delegation_delivery(
        event,
        "legacy completion",
        current_session_id="session-a",
        consumer="test",
    )
    assert paused["classification"] == "paused"
    assert paused["prompt"] is None
    assert ad.get_goal_work_snapshot(original_goal_id)["terminal_unreconciled_count"] == 1

    resumed = goals.GoalManager("session-a").resume()
    assert resumed is not None
    assert resumed.goal_id == original_goal_id
    decision = goals.prepare_goal_resume("session-a", consumer="test-resume")
    assert "ready while paused" in decision["prompt"]
    assert decision["reconciliation_claim"]


def test_three_failed_reconciliation_turns_auto_pause_goal():
    state = goals.GoalManager("session-a").set("Retry boundedly", max_turns=8)
    _dispatch(
        state.goal_id,
        "deleg_retry",
        {"status": "completed", "summary": "needs integration"},
    )
    event = _event("deleg_retry")

    for attempt in range(1, 4):
        decision = goals.prepare_goal_delegation_delivery(
            event,
            "legacy completion",
            current_session_id="session-a",
            consumer=f"attempt-{attempt}",
        )
        assert decision["reconciliation_attempt"] == attempt
        assert goals.release_goal_reconciliation_turn(
            decision["reconciliation_claim"],
            session_id="session-a",
            goal_id=state.goal_id,
            attempt=attempt,
        )
        if attempt < 3:
            event = process_registry.completion_queue.get_nowait()
            assert event.get("semantic_recovery") is True

    paused = goals.GoalManager("session-a").state
    assert paused is not None
    assert paused.status == "paused"
    assert "failed 3 times" in (paused.paused_reason or "")
    assert process_registry.completion_queue.empty()


def test_goal_replacement_abandons_old_work_and_late_result_is_passive():
    import threading

    blocker = threading.Event()
    manager = goals.GoalManager("session-a")
    old = manager.set("Old objective")
    _dispatch(
        old.goal_id,
        "deleg_old",
        {"status": "completed", "summary": "late old result"},
        blocker=blocker,
    )

    new = manager.set("Replacement objective")
    assert new.goal_id != old.goal_id
    assert ad.get_goal_work_snapshot(old.goal_id)["abandoned_count"] == 1

    blocker.set()
    event = _event("deleg_old")
    decision = goals.prepare_goal_delegation_delivery(
        event,
        "passive late result",
        current_session_id="session-a",
        consumer="test",
    )
    assert decision["classification"] == "superseded"
    assert decision["prompt"] == "passive late result"
    assert ad.get_goal_work_snapshot(new.goal_id)["required_count"] == 0


def test_replacement_before_nested_insert_rejects_stale_required_work(monkeypatch):
    manager = goals.GoalManager("session-a")
    old = manager.set("old objective")
    captured = {}
    real_abandon = ad.abandon_goal_work

    def dispatch_then_abandon(goal_id, reason):
        captured.update(
            _dispatch(
                goal_id,
                "deleg_replacement_race",
                {"status": "completed", "summary": "raced replacement"},
            )
        )
        return real_abandon(goal_id, reason)

    monkeypatch.setattr(goals, "_abandon_goal_work", dispatch_then_abandon)
    replacement = manager.set("replacement objective")

    assert replacement.goal_id != old.goal_id
    assert captured["status"] == "rejected"
    assert ad.get_durable_delegation("deleg_replacement_race") is None


def test_clear_before_nested_insert_rejects_stale_required_work():
    manager = goals.GoalManager("session-a")
    old = manager.set("objective to clear")
    manager.clear()

    dispatched = _dispatch(
        old.goal_id,
        "deleg_clear_race",
        {"status": "completed", "summary": "stale clear work"},
    )

    assert dispatched["status"] == "rejected"
    assert ad.get_durable_delegation("deleg_clear_race") is None


def test_reconciliation_retry_keeps_root_owner_across_nested_session():
    manager = goals.GoalManager("root-session")
    state = manager.set("Root objective")
    dispatched = ad.dispatch_async_delegation(
        goal="nested evidence",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="nested-session",
        parent_session_id="nested-session",
        goal_id=state.goal_id,
        requires_goal_join=True,
        goal_owner_session_id="root-session",
        delegation_id="deleg_nested_retry_owner",
        runner=lambda: {"status": "completed", "summary": "nested result"},
    )
    completion = _event(dispatched["delegation_id"])

    retry = ad._goal_reconciliation_retry_event(
        state.goal_id,
        "nested-session",
        completion,
    )
    assert retry["goal_owner_session_id"] == "root-session"

    decision = goals.prepare_goal_delegation_delivery(
        retry,
        "legacy completion",
        current_session_id="nested-session",
        consumer="test-retry",
    )
    assert decision["classification"] == "current"
    assert decision["goal_id"] == state.goal_id


@pytest.mark.parametrize("mutation", ["replace", "clear"])
def test_nested_insert_committing_before_goal_mutation_is_caught_by_sweep(
    monkeypatch, mutation,
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    old = goals.GoalManager("session-a").set("old concurrent objective")
    validation_entered = threading.Event()
    allow_insert = threading.Event()
    replacement_done = threading.Event()
    real_validate = ad._goal_owner_is_current_in_transaction

    def pause_after_validation(conn, owner_session_id, goal_id):
        current = real_validate(conn, owner_session_id, goal_id)
        validation_entered.set()
        assert allow_insert.wait(timeout=5)
        return current

    monkeypatch.setattr(ad, "_goal_owner_is_current_in_transaction", pause_after_validation)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dispatch_future = executor.submit(
            _dispatch,
            old.goal_id,
            "deleg_insert_wins_race",
            {"status": "completed", "summary": "inserted before replacement"},
        )
        assert validation_entered.wait(timeout=5)

        def mutate_goal():
            manager = goals.GoalManager("session-a")
            if mutation == "replace":
                manager.set("replacement concurrent objective")
            else:
                manager.clear()
            replacement_done.set()

        replacement_future = executor.submit(mutate_goal)
        assert not replacement_done.wait(timeout=0.05)
        allow_insert.set()
        assert dispatch_future.result(timeout=5)["status"] == "dispatched"
        replacement_future.result(timeout=5)

    row = ad.get_durable_delegation("deleg_insert_wins_race")
    assert row is not None
    assert row["reconciliation_state"] == "abandoned"


def test_delegation_started_without_goal_remains_passive_after_goal_is_set():
    dispatched = ad.dispatch_async_delegation(
        goal="independent work",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="session-a",
        parent_session_id="session-a",
        runner=lambda: {"status": "completed", "summary": "independent result"},
    )
    event = _event(dispatched["delegation_id"])
    state = goals.GoalManager("session-a").set("Later objective")

    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable is not None
    assert durable["requires_goal_join"] is False
    assert durable["reconciliation_state"] == "not_required"
    decision = goals.prepare_goal_delegation_delivery(
        event,
        "passive independent result",
        current_session_id="session-a",
        consumer="test",
    )
    assert decision == {
        "classification": "unowned",
        "prompt": "passive independent result",
        "status_message": "",
    }
    assert ad.get_goal_work_snapshot(state.goal_id)["required_count"] == 0


def test_legacy_goal_backfill_and_compression_keep_stable_identity():
    db = goals._get_session_db()
    assert db is not None
    db.set_meta(
        goals._meta_key("old-session"),
        json.dumps({"goal": "legacy objective", "status": "active", "max_turns": 5}),
    )

    loaded = goals.load_goal("old-session")
    assert loaded is not None
    assert loaded.goal_id.startswith("goal_")
    loaded_again = goals.load_goal("old-session")
    assert loaded_again is not None
    assert loaded_again.goal_id == loaded.goal_id

    assert goals.migrate_goal_to_session("old-session", "new-session", reason="test")
    migrated = goals.load_goal("new-session")
    assert migrated is not None
    assert migrated.goal_id == loaded.goal_id
    source = goals.load_goal("old-session")
    assert source is not None
    assert source.status == "cleared"
