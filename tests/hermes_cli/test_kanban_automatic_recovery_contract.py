"""RED contracts for automatic recovery from recurring Kanban failures.

Production recurrence evidence: t_7a1a86d4 turned a command-incapable profile
and child-gating deadlock into ``needs_user_action``; t_ca131320 similarly
projected an internal descendant wait as user action; t_039025cd treated a
repository-ineligible assignee as requiring human intervention; t_27655bdf
then recovered that chain with a command-capable worker and verified the F1-F4
integration.  These are internal routing/runtime failures, not requests for a
person to perform work.

These tests deliberately exercise public database/dispatcher/goal-loop
boundaries.  They remain tests-only until the recovery implementation lands.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3

import pytest

from hermes_cli import goals
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_user_action as kua


_RECOVERY_DEADLINE_SECONDS = 120


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conn = kbc.connect(home / "kanban.db")
    try:
        yield conn, home
    finally:
        conn.close()


def _task(conn: sqlite3.Connection, task_id: str) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _events(conn: sqlite3.Connection, task_id: str, kind: str):
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def _latest_recovery(conn: sqlite3.Connection, task_id: str) -> dict:
    events = _events(conn, task_id, "recovery_scheduled")
    assert events, "an internal failure must persist a recovery_scheduled event"
    payload = events[-1].payload
    assert isinstance(payload, dict)
    return payload


def _assert_bounded_recovery(payload: dict, *, now: int, resume_status: str = "ready") -> None:
    assert payload["attempt"] >= 1
    assert payload["resume_status"] == resume_status
    assert payload["deadline_at"] >= now
    assert payload["deadline_at"] - now <= _RECOVERY_DEADLINE_SECONDS


def _record_internal_failure(conn: sqlite3.Connection, task_id: str, error: str) -> None:
    assert kb.claim_task(conn, task_id) is not None
    kbd._record_task_failure(
        conn,
        task_id,
        error,
        outcome="crashed",
        failure_limit=1,
        release_claim=True,
        end_run=True,
    )


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        ("provider API returned 429 rate limit", "rate_limited"),
        ("goal judge API returned BadRequestError", "judge_error"),
    ],
)
def test_internal_api_failures_persist_bounded_retry_not_user_action(
    board, monkeypatch: pytest.MonkeyPatch, error: str, outcome: str
):
    conn, _ = board
    now = 5_000_000
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    task_id = kb.create_task(conn, title=outcome, assignee="worker", max_retries=1)

    _record_internal_failure(conn, task_id, error)

    assert _task(conn, task_id).status == "ready"
    assert kua.get_user_action(conn, task_id) is None
    assert not _events(conn, task_id, "needs_user_action")
    payload = _latest_recovery(conn, task_id)
    assert payload["failure_kind"] == outcome
    _assert_bounded_recovery(payload, now=now)


def test_provider_rate_limit_exit_persists_attempt_deadline_and_resume_state(
    board, monkeypatch: pytest.MonkeyPatch
):
    conn, _ = board
    now = 6_000_000
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_worker_alive", lambda _pid, _started_at: False)
    task_id = kb.create_task(conn, title="quota wall", assignee="worker")
    assert kb.claim_task(conn, task_id) is not None
    pid = 70001
    kbd._set_worker_pid(conn, task_id, pid)
    kbd._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)

    assert task_id not in kbd.detect_crashed_workers(conn)
    assert _task(conn, task_id).status == "ready"
    payload = _latest_recovery(conn, task_id)
    assert payload["failure_kind"] == "rate_limited"
    _assert_bounded_recovery(payload, now=now)


def test_goal_judge_transport_error_does_not_consume_budget_or_block(monkeypatch: pytest.MonkeyPatch):
    statuses = iter(["running"] * 4)
    blocks: list[str] = []
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_a, **_k: ("continue", "judge error: timeout", False, None, True),
    )

    result = goals.run_kanban_goal_loop(
        task_id="t_judge_retry",
        goal_text="finish without human intervention",
        run_turn=lambda _prompt: "still running",
        task_status_fn=lambda: next(statuses),
        block_fn=blocks.append,
        max_turns=1,
        first_response="first",
    )

    assert blocks == []
    assert result["outcome"] == "retry_scheduled"
    assert result["turns_used"] == 1
    assert result["resume_state"]["attempt"] == 1
    assert result["resume_state"]["deadline_seconds"] <= _RECOVERY_DEADLINE_SECONDS


def test_goal_turn_exhaustion_returns_durable_resume_instead_of_human_block(
    monkeypatch: pytest.MonkeyPatch,
):
    blocks: list[str] = []
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_a, **_k: ("continue", "more work remains", False, None, False),
    )

    result = goals.run_kanban_goal_loop(
        task_id="t_budget_resume",
        goal_text="finish all acceptance criteria",
        run_turn=lambda _prompt: "partial",
        task_status_fn=lambda: "running",
        block_fn=blocks.append,
        max_turns=1,
        first_response="partial",
    )

    assert blocks == []
    assert result["outcome"] == "retry_scheduled"
    assert result["resume_state"] == {
        "attempt": 1,
        "reason": "goal_turn_exhausted",
        "resume_status": "ready",
        "deadline_seconds": _RECOVERY_DEADLINE_SECONDS,
    }


def test_nonspawnable_assignee_uses_finite_declared_fallback(
    board, monkeypatch: pytest.MonkeyPatch
):
    conn, _ = board
    task_id = kb.create_task(conn, title="route around missing profile", assignee="missing")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: name == "builder")
    monkeypatch.setattr(kbd, "_dispatch_profile_allowlist", lambda _normalize: ("missing", "builder"))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda name: name)
    spawned: list[str] = []

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, _workspace: spawned.append(task.assignee) or os.getpid(),
        max_spawn=1,
    )

    assert spawned == ["builder"]
    assert result.skipped_nonspawnable == []
    assert _task(conn, task_id).assignee == "builder"
    assignments = _events(conn, task_id, "assigned")
    assert len(assignments) == 1
    assignment_payload = assignments[0].payload
    assert assignment_payload is not None
    assert assignment_payload["source"] == "preclaim_eligibility_routing"
    assert assignment_payload["attempted_profiles"] == ["missing", "builder"]


def test_live_evidence_route_requires_command_and_repository_eligibility(
    board, monkeypatch: pytest.MonkeyPatch
):
    conn, _ = board
    task_id = kb.create_task(
        conn,
        title="command-backed verification",
        body="Run repository tests and capture live command evidence.",
        assignee="web-only",
        workspace_kind="dir",
        workspace_path="/srv/repository",
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(
        kbd,
        "_dispatch_profile_allowlist",
        lambda _normalize: ("web-only", "wrong-repository", "builder"),
    )
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda name: name)
    eligibility = getattr(kbd, "profile_task_eligibility", None)
    assert callable(eligibility), "dispatcher needs a command/repository eligibility predicate"
    monkeypatch.setattr(
        kbd,
        "profile_task_eligibility",
        lambda profile, _task: {
            "web-only": (False, "command_incapable"),
            "wrong-repository": (False, "repository_ineligible"),
            "builder": (True, None),
        }[profile],
    )
    spawned: list[str] = []

    kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, _workspace: spawned.append(task.assignee) or os.getpid(),
        max_spawn=1,
    )

    assert spawned == ["builder"]
    assignment = _events(conn, task_id, "assigned")[-1]
    assignment_payload = assignment.payload
    assert assignment_payload is not None
    assert assignment_payload["attempted_profiles"] == [
        {"profile": "web-only", "reason": "command_incapable"},
        {"profile": "wrong-repository", "reason": "repository_ineligible"},
        {"profile": "builder", "reason": None},
    ]


def test_spawn_without_initial_heartbeat_is_reclaimed_within_finite_grace(
    board, monkeypatch: pytest.MonkeyPatch
):
    conn, _ = board
    now = 7_000_000
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda *_a, **_k: {"terminated": True},
    )
    task_id = kb.create_task(conn, title="silent startup", assignee="worker")
    assert kb.claim_task(conn, task_id) is not None
    kbd._set_worker_pid(conn, task_id, 70101)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at=? WHERE id=?", (now - 121, task_id))
        conn.execute(
            "UPDATE task_runs SET started_at=? WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
            (now - 121, task_id),
        )

    reclaimed = kbd.detect_stale_running(
        conn,
        stale_timeout_seconds=3600,
        signal_fn=lambda *_a: None,
    )

    assert reclaimed == [task_id]
    assert _task(conn, task_id).status == "ready"
    stale = _events(conn, task_id, "stale")[-1].payload
    assert stale is not None
    assert stale["reason"] == "initial_heartbeat_deadline_exceeded"
    assert stale["deadline_seconds"] <= _RECOVERY_DEADLINE_SECONDS


def test_internal_parent_failure_keeps_descendants_on_automatic_path(board):
    conn, _ = board
    parent = kb.create_task(conn, title="internal parent", assignee="worker", max_retries=1)
    child = kb.create_task(conn, title="dependent child", assignee="worker", parents=[parent])

    _record_internal_failure(conn, parent, "worker-local command capability unavailable")

    assert _task(conn, parent).status == "ready"
    assert _task(conn, child).status == "todo"
    assert kua.get_user_action(conn, parent) is None
    assert kb.complete_task(conn, parent, summary="recovered automatically")
    assert _task(conn, child).status == "ready"


@pytest.mark.parametrize(
    "error",
    [
        "provider temporarily unavailable",
        "goal judge API returned 503",
        "assigned profile is not spawnable on this dispatcher",
        "worker did not emit its initial heartbeat",
    ],
)
def test_internal_failures_are_never_mislabeled_needs_user_action(board, error: str):
    conn, _ = board
    task_id = kb.create_task(conn, title="internal retry", assignee="worker", max_retries=1)

    _record_internal_failure(conn, task_id, error)

    assert _task(conn, task_id).status != "needs_user_action"
    assert kua.get_user_action(conn, task_id) is None
    assert not _events(conn, task_id, "needs_user_action")


def test_genuine_user_action_is_fail_closed_and_preserves_exact_contract(board):
    conn, _ = board
    task_id = kb.create_task(conn, title="operator prerequisite", assignee="worker")
    running = kb.claim_task(conn, task_id)
    assert running is not None
    action = {
        "incomplete_status": "Deployment cannot authenticate yet.",
        "reason": "A human-controlled credential is absent.",
        "execution_location": "Operator vault on gateway host",
        "action": "Run `vault put ziva/api-key` and approve the hardware prompt.",
        "expected_success": "`vault probe ziva/api-key` exits 0 and prints `ready`.",
        "automatic_continuation": "The persisted env_present probe resumes the card; do not send a continue message.",
    }
    probe = {"kind": "env_present", "name": "ZIVA_API_KEY"}

    assert kb.block_task(
        conn,
        task_id,
        kind="needs_input",
        reason=action["reason"],
        expected_run_id=running.current_run_id,
        user_action=action,
        readiness_probe=probe,
    )

    state = kua.get_user_action(conn, task_id)
    assert state is not None
    assert state.payload == action
    assert state.readiness_probe == probe
    assert _task(conn, task_id).status == "needs_user_action"


def test_no_status_message_canary_recovers_silent_worker_by_120_seconds(
    board, monkeypatch: pytest.MonkeyPatch
):
    """Production-canary reusable harness: no chat/Status event is injected.

    Time is virtual, so the assertion is deterministic while preserving the
    same 120-second deadline expected from the later real canary.
    """
    conn, _ = board
    start = 8_000_000
    clock = {"now": start}
    monkeypatch.setattr(kbd.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda *_a, **_k: {"terminated": True},
    )
    task_id = kb.create_task(conn, title="no Status message", assignee="worker")
    assert kb.claim_task(conn, task_id) is not None
    kbd._set_worker_pid(conn, task_id, 70201)

    clock["now"] = start + _RECOVERY_DEADLINE_SECONDS
    reclaimed = kbd.detect_stale_running(
        conn,
        stale_timeout_seconds=_RECOVERY_DEADLINE_SECONDS,
        signal_fn=lambda *_a: None,
    )

    assert reclaimed == [task_id]
    assert _task(conn, task_id).status == "ready"
    assert not any(event.kind.lower() == "status" for event in kb.list_events(conn, task_id))
