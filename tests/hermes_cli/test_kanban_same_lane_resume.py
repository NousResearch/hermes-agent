"""Regression coverage for the explicit guarded same-lane resume path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gateway.status import get_process_start_time
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_same_lane_child as gate_child
from hermes_cli import kanban_same_lane_resume as slr

TARGET = "t_dfa23a41"
PR_URL = "https://github.com/NXE-ORG/nxe-helix-alpha/pull/155"
WORKSPACE = "/home/hermes/nxe-helix-alpha/.worktrees/t_dfa23a41"
HEAD_REF = "wt/finance-pr153-producer-output-closure"
HEAD_OID = "1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1"
UPSTREAM = f"origin/{HEAD_REF}"


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def _request(auth: str = "auth-1") -> kbd.SameLaneResumeRequest:
    return kbd.authorised_same_lane_request(auth)


def _observation(**changes) -> kbd.SameLaneResumeObservation:
    value = kbd.SameLaneResumeObservation(
        pr_url=PR_URL,
        pr_number=155,
        pr_state="OPEN",
        pr_base="dev",
        pr_head_ref=HEAD_REF,
        pr_head_oid=HEAD_OID,
        workspace=WORKSPACE,
        workspace_registered=True,
        branch=HEAD_REF,
        clean=True,
        local_oid=HEAD_OID,
        upstream_ref=UPSTREAM,
        upstream_oid=HEAD_OID,
    )
    return replace(value, **changes)


def _insert_target(conn) -> None:
    original = kb.create_task(
        conn,
        title="exact resume target",
        body=f"Remediate {PR_URL}",
        assignee="flynn",
        workspace_kind="worktree",
        workspace_path=WORKSPACE,
        branch_name=HEAD_REF,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET id = ? WHERE id = ?", (TARGET, original))
    kb.add_comment(conn, TARGET, "crash", f"peer-review: changes requested on {PR_URL}")
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (id, task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (1144, ?, 'crash', 'changes_requested', 'changes_requested', 1, 2)",
            (TARGET,),
        )
        kb._append_event(
            conn, TARGET, "changes_requested", {"verdict": "changes_requested"}, run_id=1144
        )
        for run_id, started in ((1146, 3), (1149, 5)):
            conn.execute(
                "INSERT INTO task_runs (id, task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, ?, 'flynn', 'reclaimed', 'reclaimed', ?, ?)",
                (run_id, TARGET, started, started + 1),
            )
            kb._append_event(conn, TARGET, "claimed", {"run_id": run_id}, run_id=run_id)
            kb._append_event(conn, TARGET, "heartbeat", None, run_id=run_id)
            kb._append_event(conn, TARGET, "reclaimed", None, run_id=run_id)
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                (started + 1, run_id),
            )


class FakeChild:
    def __init__(self, conn, *, session="resume-session", cleanup=True, release=True):
        self._conn = conn
        self._cleanup = cleanup
        self._release = release
        self.terminated = 0
        self.released = 0
        self.identity = kbd.GatedChildIdentity(
            pid=os.getpid(),
            process_started_at=get_process_start_time(os.getpid()),
            process_group_id=os.getpgid(os.getpid()),
            session_id=session,
            ready_but_gated=True,
        )

    def release(self, *, task_id: str, run_id: int, authorization_id: str) -> bool:
        self.released += 1
        if not self._release:
            return False
        with kb.write_txn(self._conn):
            kb._append_event(
                self._conn,
                task_id,
                "resume_released",
                {"resume_authorization_id": authorization_id},
                run_id=run_id,
            )
        return True

    def terminate_and_confirm(self) -> bool:
        self.terminated += 1
        return self._cleanup


def _run(conn, *, auth="auth-1", observation=None, child=None, **kwargs):
    request = _request(auth)
    child = child or FakeChild(conn, session=f"session-{auth}")
    outcome = kbd.resume_same_lane(
        conn,
        request,
        observer=lambda: observation or _observation(),
        launcher=lambda _request, _task: child,
        **kwargs,
    )
    return outcome, child


def _task_row(conn):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (TARGET,)).fetchone()


def _events(conn, kind):
    return [event for event in kb.list_events(conn, TARGET) if event.kind == kind]


def test_exact_same_card_resume_publishes_one_coherent_identity(board):
    _insert_target(board)

    outcome, child = _run(board)

    assert outcome.disposition == "published"
    row = _task_row(board)
    assert row["status"] == "running"
    assert row["worker_pid"] == child.identity.pid
    assert row["worker_started_at"] == child.identity.process_started_at
    assert row["session_id"] == child.identity.session_id
    assert row["current_run_id"] == outcome.run_id
    spawned = _events(board, "spawned")
    assert len(spawned) == 1
    assert spawned[0].run_id == outcome.run_id
    assert spawned[0].payload["resume_authorization_id"] == "auth-1"
    assert spawned[0].payload["session_id"] == child.identity.session_id
    kinds = [event.kind for event in kb.list_events(board, TARGET)]
    quarantine_index = kinds.index("resume_quarantined")
    new_kinds = kinds[quarantine_index:]
    assert new_kinds.index("resume_quarantined") < new_kinds.index("claimed") < new_kinds.index("spawned")
    assert new_kinds.index("spawned") < new_kinds.index("resume_released")
    assert "heartbeat" not in new_kinds
    assert child.released == 1


def test_ready_task_with_only_stale_worker_start_is_normalized_before_resume(board):
    _insert_target(board)
    historical_heartbeats = len(_events(board, "heartbeat"))
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
            (85963388, TARGET),
        )

    outcome, _child = _run(board)

    assert outcome.disposition == "published"
    normalized = _events(board, "lifecycle_normalized")
    assert len(normalized) == 1
    assert normalized[0].run_id is None
    assert normalized[0].payload == {
        "field": "worker_started_at",
        "previous_value": 85963388,
        "reason": "terminal_residue",
        "resume_authorization_id": "auth-1",
        "same_lane_resume": True,
    }
    assert len(_events(board, "heartbeat")) == historical_heartbeats
    assert board.execute(
        "SELECT COUNT(*) FROM task_runs WHERE id IN (1146, 1149) AND ended_at IS NOT NULL"
    ).fetchone()[0] == 2


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("status", "running", "task_not_ready"),
        ("claim_lock", "active-claim", "active_task_handle"),
        ("claim_expires", 999, "active_task_handle"),
        ("current_run_id", 1144, "active_task_handle"),
        ("worker_pid", os.getpid(), "active_task_handle"),
        ("session_id", "active-session", "active_task_handle"),
    ],
)
def test_stale_worker_start_is_not_normalized_beside_active_handle(
    board, column, value, reason,
):
    _insert_target(board)
    with kb.write_txn(board):
        board.execute(
            f"UPDATE tasks SET worker_started_at = ?, {column} = ? WHERE id = ?",
            (85963388, value, TARGET),
        )
    launches = []

    def unexpected_launch(*_args):
        launches.append(1)
        return FakeChild(board)

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=unexpected_launch,
    )

    assert outcome.disposition == "refused"
    assert outcome.reason == f"lifecycle_normalization_refused:{reason}"
    assert _task_row(board)["worker_started_at"] == 85963388
    assert not _events(board, "lifecycle_normalized")
    assert launches == []


def test_stale_worker_start_is_not_normalized_while_unpointed_run_is_open(board):
    _insert_target(board)
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
            (85963388, TARGET),
        )
        board.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, 'flynn', 'running', ?)",
            (TARGET, int(time.time())),
        )
    launches = []

    def unexpected_launch(*_args):
        launches.append(1)
        return FakeChild(board)

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=unexpected_launch,
    )

    assert outcome.disposition == "refused"
    assert outcome.reason == "lifecycle_normalization_refused:active_run"
    assert _task_row(board)["worker_started_at"] == 85963388
    assert not _events(board, "lifecycle_normalized")
    assert launches == []


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("status", "running", "task_status_mismatch"),
        ("claim_lock", "someone", "task_not_unclaimed"),
        ("claim_lock", "", "task_not_unclaimed"),
        ("claim_expires", 999, "task_not_unclaimed"),
        ("claim_expires", 0, "task_not_unclaimed"),
        ("current_run_id", 1144, "task_not_unclaimed"),
        ("current_run_id", 0, "task_not_unclaimed"),
        ("worker_pid", 999, "task_not_unclaimed"),
        ("worker_pid", 0, "task_not_unclaimed"),
        ("session_id", "old-session", "task_not_unclaimed"),
        ("session_id", "", "task_not_unclaimed"),
        ("assignee", "crash", "assignee_mismatch"),
        ("workspace_path", "/wrong", "workspace_metadata_mismatch"),
        ("branch_name", "wrong", "branch_metadata_mismatch"),
    ],
)
def test_stale_or_mismatched_card_guard_refuses_before_launch(board, column, value, reason):
    _insert_target(board)
    with kb.write_txn(board):
        board.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, TARGET))
    launches = []

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=lambda *_: launches.append(1),
    )

    assert outcome.disposition == "refused"
    assert outcome.reason == reason
    assert launches == []
    assert not _events(board, "resume_quarantined")


@pytest.mark.parametrize("status", ["todo", "ready", "running", "blocked", "review"])
def test_every_competing_nonterminal_flynn_lane_refuses(board, status):
    _insert_target(board)
    other = kb.create_task(board, title="other Flynn lane", assignee="flynn")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, other))

    outcome, _child = _run(board)

    assert outcome.disposition == "refused"
    assert outcome.reason == f"competing_lane:{other}:{status}"


@pytest.mark.parametrize("status", ["done", "archived"])
def test_terminal_flynn_siblings_do_not_block(board, status):
    _insert_target(board)
    other = kb.create_task(board, title="terminal Flynn lane", assignee="flynn")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, other))

    outcome, _child = _run(board)

    assert outcome.disposition == "published"


def test_another_task_owning_pr_refuses(board):
    _insert_target(board)
    other = kb.create_task(board, title="other owner", assignee="crash")
    kb.add_comment(board, other, "crash", PR_URL)

    outcome, _child = _run(board)

    assert outcome.disposition == "refused"
    assert outcome.reason == "target_pr_owned_by_other_task"


def test_target_conflicting_active_pr_refuses(board):
    _insert_target(board)
    kb.add_comment(board, TARGET, "crash", "https://github.com/NXE-ORG/nxe-helix-alpha/pull/999")

    outcome, _child = _run(board)

    assert outcome.disposition == "refused"
    assert outcome.reason == "target_pr_ownership_conflict"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pr_url", "https://github.com/NXE-ORG/nxe-helix-alpha/pull/999"),
        ("pr_number", 999),
        ("pr_state", "CLOSED"),
        ("pr_base", "main"),
        ("pr_head_ref", "wrong"),
        ("pr_head_oid", "0" * 40),
        ("workspace", "/wrong"),
        ("workspace_registered", False),
        ("branch", "wrong"),
        ("clean", False),
        ("local_oid", "0" * 40),
        ("upstream_ref", "origin/wrong"),
        ("upstream_oid", "0" * 40),
    ],
)
def test_each_typed_pr_worktree_and_upstream_guard_refuses(board, field, value):
    _insert_target(board)

    outcome, _child = _run(board, observation=_observation(**{field: value}))

    assert outcome.disposition == "refused"
    assert outcome.reason == f"observation_{field}_mismatch"


def test_observer_error_refuses_without_quarantine(board):
    _insert_target(board)

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=lambda: (_ for _ in ()).throw(TimeoutError("observer timeout")),
        launcher=lambda *_: pytest.fail("launcher must not run"),
    )

    assert outcome.disposition == "refused"
    assert outcome.reason == "observer_failed:observer timeout"
    assert _task_row(board)["status"] == "ready"


@pytest.mark.parametrize(
    "resume_request",
    [
        replace(_request(), task_id="t_other"),
        replace(_request(), expected_pr_number=999),
        replace(_request(), expected_pr_url="https://github.com/NXE-ORG/nxe-helix-alpha/pull/999"),
    ],
)
def test_any_non_exact_authorization_refuses(board, resume_request):
    _insert_target(board)
    outcome = kbd.resume_same_lane(
        board,
        resume_request,
        observer=_observation,
        launcher=lambda *_: pytest.fail("unauthorized request launched"),
    )
    assert outcome.disposition == "refused"
    assert outcome.reason == "request_not_authorised"


@pytest.mark.parametrize(
    ("identity_change", "reason"),
    [
        ({"pid": -1}, "child_identity_not_live"),
        ({"process_started_at": 1}, "child_identity_not_live"),
        ({"session_id": ""}, "child_session_missing"),
        ({"ready_but_gated": False}, "child_not_gated"),
        ({"process_group_id": 999999}, "child_process_group_mismatch"),
    ],
)
def test_invalid_child_identity_compensates_without_run(board, identity_change, reason):
    _insert_target(board)
    child = FakeChild(board)
    child.identity = replace(child.identity, **identity_change)

    outcome, _ = _run(board, child=child)

    assert outcome.disposition == "compensated"
    assert outcome.reason == reason
    assert child.terminated == 1
    assert _task_row(board)["status"] == "ready"
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE id NOT IN (1144,1146,1149)").fetchone()[0] == 0


def test_duplicate_child_session_compensates_without_run(board):
    _insert_target(board)
    other = kb.create_task(board, title="terminal other", assignee="crash", session_id="duplicate")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (other,))
    child = FakeChild(board, session="duplicate")

    outcome, _ = _run(board, child=child)

    assert outcome.disposition == "compensated"
    assert outcome.reason == "child_session_duplicate"
    assert child.terminated == 1


def test_phantom_reclaims_are_skipped_but_genuine_later_run_blocks(board):
    _insert_target(board)
    with kb.write_txn(board):
        board.execute("UPDATE task_runs SET worker_pid = 123 WHERE id = 1149")
        kb._append_event(board, TARGET, "spawned", {"pid": 123}, run_id=1149)

    outcome, _child = _run(board)

    assert outcome.disposition == "refused"
    assert outcome.reason == "substantive_run_id_mismatch"


def test_launch_failure_compensates_to_ready_without_run_or_heartbeat(board):
    _insert_target(board)

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=lambda *_: (_ for _ in ()).throw(
            slr.SameLaneLaunchError("launch failed", cleanup_confirmed=True)
        ),
    )

    assert outcome.disposition == "compensated"
    row = _task_row(board)
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["current_run_id"] is None
    assert row["worker_pid"] is None
    assert row["session_id"] is None
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE id NOT IN (1144,1146,1149)").fetchone()[0] == 0
    assert _events(board, "spawn_failed")[-1].run_id is None
    assert not _events(board, "heartbeat")[2:]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("claim_lock", "drifted"),
        ("claim_lock", ""),
        ("claim_expires", 999),
        ("claim_expires", 0),
        ("current_run_id", 1144),
        ("current_run_id", 0),
        ("worker_pid", 999),
        ("worker_pid", 0),
        ("worker_started_at", 111),
        ("worker_started_at", 0),
        ("session_id", "drifted-session"),
        ("session_id", ""),
    ],
)
def test_every_task_identity_guard_is_revalidated_while_child_is_gated(board, column, value):
    _insert_target(board)
    child = FakeChild(board)

    def launch(*_args):
        with kb.write_txn(board):
            board.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, TARGET))
        return child

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=launch,
    )

    assert outcome.disposition == "compensated"
    assert outcome.reason == "guard_revalidation_failed:task_not_unclaimed"
    assert child.terminated == 1
    assert _task_row(board)["status"] == "ready"
    assert board.execute(
        "SELECT COUNT(*) FROM task_runs WHERE id NOT IN (1144,1146,1149)"
    ).fetchone()[0] == 0


def test_launcher_cleanup_uncertainty_keeps_quarantine_without_run(board):
    _insert_target(board)

    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=lambda *_: (_ for _ in ()).throw(
            slr.SameLaneLaunchError("kill confirmation failed", cleanup_confirmed=False)
        ),
    )

    assert outcome.disposition == "cleanup_failed"
    row = _task_row(board)
    assert row["status"] == "blocked"
    assert row["block_kind"] == "resume_quarantined"
    assert row["current_run_id"] is None
    assert _events(board, "cleanup_failed")[-1].run_id is None


def test_publication_transaction_failure_kills_child_then_compensates(board):
    _insert_target(board)
    child = FakeChild(board)

    outcome, _ = _run(
        board,
        child=child,
        publication_hook=lambda *_: (_ for _ in ()).throw(RuntimeError("event write failed")),
    )

    assert outcome.disposition == "compensated"
    assert child.terminated == 1
    assert _task_row(board)["status"] == "ready"
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE id NOT IN (1144,1146,1149)").fetchone()[0] == 0
    assert not [e for e in _events(board, "spawned") if e.payload.get("same_lane_resume")]
    assert _events(board, "spawn_failed")[-1].run_id is None


def test_cleanup_failure_keeps_card_quarantined_and_nonrunnable(board):
    _insert_target(board)
    child = FakeChild(board, cleanup=False)

    outcome, _ = _run(
        board,
        child=child,
        publication_hook=lambda *_: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )

    assert outcome.disposition == "cleanup_failed"
    row = _task_row(board)
    assert row["status"] == "blocked"
    assert row["block_kind"] == "resume_quarantined"
    assert row["current_run_id"] is None
    assert _events(board, "cleanup_failed")[-1].run_id is None
    metadata = slr._ResumeFence(board, TARGET).read_metadata()
    assert metadata is not None
    assert metadata["phase"] == "cleanup_failed"
    assert metadata["controller_pid"] == 0
    assert metadata["controller_started_at"] == 0


def test_new_authorization_reconciles_prior_cleanup_failure_before_launch(board):
    _insert_target(board)
    child = FakeChild(board, cleanup=False)
    first, _ = _run(
        board,
        auth="old-cleanup-auth",
        child=child,
        publication_hook=lambda *_: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )
    assert first.disposition == "cleanup_failed"
    fence = slr._ResumeFence(board, TARGET)
    metadata = fence.read_metadata()
    assert metadata is not None
    metadata.update({
        "pid": 99999998,
        "process_started_at": 1,
        "process_group_id": 99999998,
        "controller_pid": 0,
        "controller_started_at": 0,
    })
    fence.write_metadata(metadata)
    launches = []

    def unexpected_launch(*_args):
        launches.append(1)
        return child

    outcome = kbd.resume_same_lane(
        board,
        _request("new-auth"),
        observer=_observation,
        launcher=unexpected_launch,
    )

    assert outcome.disposition == "compensated"
    assert launches == []
    assert _task_row(board)["status"] == "ready"
    assert fence.read_metadata() is None
    assert _events(board, "spawn_failed")[-1].payload["resume_authorization_id"] == "old-cleanup-auth"


def test_gate_release_failure_recovers_real_spawn_without_phantom_heartbeat(board):
    _insert_target(board)
    child = FakeChild(board, release=False)

    outcome, _ = _run(board, child=child)

    assert outcome.disposition == "compensated"
    assert child.terminated == 1
    row = _task_row(board)
    assert row["status"] == "ready"
    assert row["current_run_id"] is None
    run = board.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE id NOT IN (1144,1146,1149)"
    ).fetchone()
    assert tuple(run) == ("spawn_failed", run["ended_at"])
    assert run["ended_at"] is not None
    kinds = [event.kind for event in kb.list_events(board, TARGET)]
    quarantine_index = kinds.index("resume_quarantined")
    assert "heartbeat" not in kinds[quarantine_index:]


def test_release_receipt_wins_ack_race_without_killing_released_worker(board):
    _insert_target(board)

    class AckRaceChild(FakeChild):
        def release(self, *, task_id: str, run_id: int, authorization_id: str) -> bool:
            with kb.write_txn(self._conn):
                kb._append_event(
                    self._conn,
                    task_id,
                    "resume_released",
                    {"resume_authorization_id": authorization_id},
                    run_id=run_id,
                )
            raise TimeoutError("controller missed acknowledgement")

    child = AckRaceChild(board)
    outcome, _ = _run(board, child=child)

    assert outcome.disposition == "published"
    assert outcome.reason == "release_receipt_won_ack_race"
    assert child.terminated == 0
    assert _task_row(board)["status"] == "running"


def test_child_executes_after_durable_release_even_if_ack_pipe_closed(monkeypatch, tmp_path):
    writes = 0
    executed = {}

    monkeypatch.setattr(gate_child, "_install_parent_death", lambda *_: None)
    monkeypatch.setattr(gate_child, "_clear_parent_death", lambda: None)
    monkeypatch.setattr(gate_child, "_process_start", lambda _pid: 123)
    monkeypatch.setattr(gate_child, "_atomic_identity", lambda *_: None)
    monkeypatch.setattr(
        gate_child,
        "_read_line",
        lambda *_: json.dumps({
            "commit": True,
            "task_id": TARGET,
            "run_id": 77,
            "authorization_id": "ack-race",
        }),
    )
    monkeypatch.setattr(gate_child, "_persist_release", lambda *_: (77, "claim-lock"))

    def write(_fd, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise BrokenPipeError("controller exited")
        return len(data)

    def execvpe(program, argv, env):
        executed.update(program=program, argv=argv, env=env)
        raise RuntimeError("exec reached")

    monkeypatch.setattr(gate_child.os, "write", write)
    monkeypatch.setattr(gate_child.os, "execvpe", execvpe)

    with pytest.raises(RuntimeError, match="exec reached"):
        gate_child.main([
            "--gate-fd", "10",
            "--ack-fd", "11",
            "--db", str(tmp_path / "board.db"),
            "--task-id", TARGET,
            "--authorization-id", "ack-race",
            "--session-id", "ack-session",
            "--identity-path", str(tmp_path / "identity.json"),
            "--launch-nonce", "nonce",
            "--parent-pid", str(os.getpid()),
            "--parent-start", "123",
            "--deadline", str(time.time() + 10),
            "--", sys.executable, "-c", "pass",
        ])

    assert writes == 2
    assert executed["program"] == sys.executable
    assert executed["env"]["HERMES_KANBAN_RUN_ID"] == "77"
    assert executed["env"]["HERMES_KANBAN_CLAIM_LOCK"] == "claim-lock"


def test_gate_read_deadline_expires_while_controller_keeps_pipe_open():
    gate_r, gate_w = os.pipe()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="gate deadline expired"):
            gate_child._read_line(gate_r, time.monotonic() + 0.05)
    finally:
        os.close(gate_r)
        os.close(gate_w)
    assert time.monotonic() - started < 1.0


def test_controller_crash_after_publication_before_release_is_compensated(board):
    _insert_target(board)
    request = _request("published-crash")
    identity = kbd.GatedChildIdentity(
        pid=99999998,
        process_started_at=1,
        process_group_id=99999998,
        session_id="published-crash-session",
    )
    with kb.write_txn(board):
        assert slr._quarantine(board, request)
    with kb.write_txn(board):
        run_id = slr._publish(
            board,
            request,
            identity,
            ttl_seconds=None,
            publication_hook=None,
        )
    fence = slr._ResumeFence(board, TARGET)
    fence.write_metadata({
        "resume_authorization_id": request.resume_authorization_id,
        "phase": "published",
        "controller_pid": 99999999,
        "controller_started_at": 1,
        "run_id": run_id,
        "pid": identity.pid,
        "process_started_at": identity.process_started_at,
        "process_group_id": identity.process_group_id,
        "session_id": identity.session_id,
    })

    outcome = kbd.resume_same_lane(
        board,
        request,
        observer=_observation,
        launcher=lambda *_: pytest.fail("crash recovery launched a second child"),
    )

    assert outcome.disposition == "compensated"
    row = _task_row(board)
    assert row["status"] == "ready"
    assert row["current_run_id"] is None
    recovered_run = board.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert recovered_run["status"] == "spawn_failed"
    assert recovered_run["outcome"] == "spawn_failed"
    assert recovered_run["ended_at"] is not None
    assert _events(board, "spawn_failed")[-1].run_id == run_id
    assert not [
        event for event in _events(board, "heartbeat") if event.run_id == run_id
    ]


def test_same_authorization_is_one_use_after_compensation(board):
    _insert_target(board)
    launches = 0

    def fail_launch(*_args):
        nonlocal launches
        launches += 1
        raise slr.SameLaneLaunchError("boom", cleanup_confirmed=True)

    first = kbd.resume_same_lane(board, _request(), observer=_observation, launcher=fail_launch)
    second = kbd.resume_same_lane(board, _request(), observer=_observation, launcher=fail_launch)

    assert first.disposition == "compensated"
    assert second.disposition == "compensated"
    assert second.reason == "authorization_receipt:spawn_failed"
    assert launches == 1


def test_new_authorization_after_compensation_rechecks_all_guards(board):
    _insert_target(board)
    first = kbd.resume_same_lane(
        board,
        _request("auth-old"),
        observer=_observation,
        launcher=lambda *_: (_ for _ in ()).throw(
            slr.SameLaneLaunchError("boom", cleanup_confirmed=True)
        ),
    )
    assert first.disposition == "compensated"

    second, _child = _run(board, auth="auth-new", observation=_observation(pr_state="CLOSED"))

    assert second.disposition == "refused"
    assert second.reason == "observation_pr_state_mismatch"


def test_expired_pre_identity_quarantine_is_crash_recovered_without_run(board):
    _insert_target(board)
    request = _request("crashed-auth")
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = 'resume_quarantined' WHERE id = ?",
            (TARGET,),
        )
        kb._append_event(
            board,
            TARGET,
            "resume_quarantined",
            {"resume_authorization_id": request.resume_authorization_id},
        )
    fence = slr._ResumeFence(board, TARGET)
    fence.write_metadata({
        "resume_authorization_id": request.resume_authorization_id,
        "controller_pid": 99999999,
        "controller_started_at": 1,
        "gate_deadline": int(time.time()) - 1,
        "phase": "launching",
    })

    outcome = kbd.resume_same_lane(
        board,
        request,
        observer=_observation,
        launcher=lambda *_: pytest.fail("crash recovery launched a child"),
    )

    assert outcome.disposition == "compensated"
    assert _task_row(board)["status"] == "ready"
    assert _events(board, "spawn_failed")[-1].payload["phase"] == "crash_recovery"
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE id NOT IN (1144,1146,1149)").fetchone()[0] == 0


def test_contender_cannot_acquire_fence_while_first_attempt_is_gated(board):
    _insert_target(board)
    entered = threading.Event()
    release = threading.Event()
    result = {}
    db_path = Path(board.execute("PRAGMA database_list").fetchone()[2])

    class WaitingChild(FakeChild):
        def __init__(self, conn):
            super().__init__(conn, session="first-session")
            entered.set()
            assert release.wait(5)

    def first_attempt():
        conn = kbc._sqlite_connect(db_path)
        conn.row_factory = board.row_factory
        try:
            result["first"] = kbd.resume_same_lane(
                conn,
                _request(),
                observer=_observation,
                launcher=lambda *_: WaitingChild(conn),
            )
        finally:
            conn.close()

    thread = threading.Thread(target=first_attempt)
    thread.start()
    assert entered.wait(5)
    contender_conn = kbc._sqlite_connect(db_path)
    contender_conn.row_factory = board.row_factory
    try:
        contender = kbd.resume_same_lane(
            contender_conn,
            _request(),
            observer=_observation,
            launcher=lambda *_: pytest.fail("contender launched a second child"),
            fence_timeout_seconds=0.05,
        )
    finally:
        contender_conn.close()
        release.set()
        thread.join(5)

    assert contender.disposition == "refused"
    assert contender.reason.startswith("lease_timeout:")
    assert result["first"].disposition == "published"


def test_ordinary_active_pr_suppression_is_unchanged(board):
    _insert_target(board)
    other = kb.create_task(board, title="unrelated active PR", assignee="crash")
    kb.add_comment(
        board,
        other,
        "crash",
        "https://github.com/NXE-ORG/nxe-helix-alpha/pull/777",
    )

    assert kbd.check_respawn_guard(board, TARGET) == "active_pr"
    assert kbd.check_respawn_guard(board, other) == "active_pr"


def test_nullable_task_level_receipts_round_trip_through_existing_reader(board):
    _insert_target(board)
    outcome = kbd.resume_same_lane(
        board,
        _request(),
        observer=_observation,
        launcher=lambda *_: (_ for _ in ()).throw(
            slr.SameLaneLaunchError("launch failed", cleanup_confirmed=True)
        ),
    )
    assert outcome.disposition == "compensated"

    receipts = [
        event for event in kb.list_events(board, TARGET)
        if event.kind in {"resume_quarantined", "spawn_failed", "cleanup_failed"}
    ]
    assert [event.kind for event in receipts] == ["resume_quarantined", "spawn_failed"]
    assert all(event.run_id is None for event in receipts)


@pytest.mark.linux_only
def test_real_linux_gate_persists_release_before_exec(board, tmp_path, monkeypatch):
    _insert_target(board)
    marker = tmp_path / "worker-ran"
    request = replace(_request(), expected_workspace=str(tmp_path))
    task = kb.get_task(board, TARGET)
    assert task is not None
    worker = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
    ]
    monkeypatch.setattr(
        kbd,
        "_same_lane_worker_material",
        lambda *_args: (worker, dict(os.environ)),
    )
    identity_path = tmp_path / "resume-fence.json"
    controller_started_at = get_process_start_time(os.getpid())
    assert controller_started_at is not None
    child = kbd._launch_same_lane_worker(
        request,
        task,
        identity_path=str(identity_path),
        launch_nonce="nonce",
        deadline=int(time.time()) + 20,
        controller_started_at=controller_started_at,
    )
    try:
        with kb.write_txn(board):
            board.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'resume_quarantined' "
                "WHERE id = ?",
                (TARGET,),
            )
            kb._append_event(
                board,
                TARGET,
                "resume_quarantined",
                {"resume_authorization_id": request.resume_authorization_id},
            )
        with kb.write_txn(board):
            run_id = slr._publish(
                board,
                request,
                child.identity,
                ttl_seconds=None,
                publication_hook=None,
            )
        assert child.release(
            task_id=TARGET,
            run_id=run_id,
            authorization_id=request.resume_authorization_id,
        )
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.read_text() == "ran"
        kinds = [event.kind for event in kb.list_events(board, TARGET) if event.run_id == run_id]
        assert kinds == ["claimed", "spawned", "resume_released"]
    finally:
        close_child = getattr(child, "close", None)
        if callable(close_child):
            close_child()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _launch_crashing_gate_controller(tmp_path: Path, *, wait_for_identity: bool) -> tuple[int, Path, Path]:
    identity_path = tmp_path / ("identity-after.json" if wait_for_identity else "identity-before.json")
    pid_path = tmp_path / ("wrapper-after.pid" if wait_for_identity else "wrapper-before.pid")
    marker = tmp_path / ("worker-after-ran" if wait_for_identity else "worker-before-ran")
    controller = tmp_path / ("controller-after.py" if wait_for_identity else "controller-before.py")
    controller.write_text(
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "from hermes_cli.kanban_same_lane_child import _process_start\n"
        f"identity=Path({str(identity_path)!r})\n"
        f"pid_path=Path({str(pid_path)!r})\n"
        "gate_r,gate_w=os.pipe(); ack_r,ack_w=os.pipe()\n"
        "worker=[sys.executable,'-c',"
        f"\"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\"]\n"
        "argv=[sys.executable,'-m','hermes_cli.kanban_same_lane_child',"
        "'--gate-fd',str(gate_r),'--ack-fd',str(ack_w),'--db',str(identity.parent/'unused.db'),"
        "'--task-id','t_dfa23a41','--authorization-id','crash-test',"
        "'--session-id','crash-session','--identity-path',str(identity),"
        "'--launch-nonce','nonce','--parent-pid',str(os.getpid()),"
        "'--parent-start',str(_process_start(os.getpid())),'--deadline',str(time.time()+10),"
        "'--',*worker]\n"
        "proc=subprocess.Popen(argv,pass_fds=(gate_r,ack_w),start_new_session=True)\n"
        "pid_path.write_text(str(proc.pid))\n"
        + (
            "deadline=time.monotonic()+5\n"
            "while not identity.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
            "if not identity.exists(): raise SystemExit(3)\n"
            if wait_for_identity else ""
        )
        + "os._exit(0)\n",
        encoding="utf-8",
    )
    completed = subprocess.run([sys.executable, str(controller)], check=False, timeout=10)
    assert completed.returncode == 0
    pid = int(pid_path.read_text(encoding="utf-8"))
    return pid, identity_path, marker


def _launch_phase_controlled_controller(tmp_path: Path, phase: str):
    identity = tmp_path / f"{phase}-identity.json"
    wrapper_pid_path = tmp_path / f"{phase}-wrapper.pid"
    ready = tmp_path / f"{phase}-ready"
    release = tmp_path / f"{phase}-release"
    marker = tmp_path / f"{phase}-worker-ran"
    harness = tmp_path / f"{phase}-harness.py"
    harness.write_text(
        "import sys,time\n"
        "from pathlib import Path\n"
        "from hermes_cli import kanban_same_lane_child as child\n"
        f"ready=Path({str(ready)!r}); release=Path({str(release)!r})\n"
        f"phase={phase!r}\n"
        "def pause():\n"
        " ready.write_text('ready')\n"
        " while not release.exists(): time.sleep(0.01)\n"
        "if phase == 'before_parent_death':\n"
        " original=child._install_parent_death\n"
        " def hooked(parent,start): pause(); original(parent,start)\n"
        " child._install_parent_death=hooked\n"
        "else:\n"
        " original=child._atomic_identity\n"
        " def hooked(path,payload): pause(); original(path,payload)\n"
        " child._atomic_identity=hooked\n"
        "raise SystemExit(child.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    controller = tmp_path / f"{phase}-controller.py"
    controller.write_text(
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "from hermes_cli.kanban_same_lane_child import _process_start\n"
        "gate_r,gate_w=os.pipe(); ack_r,ack_w=os.pipe()\n"
        f"identity=Path({str(identity)!r})\n"
        f"worker=[sys.executable,'-c',\"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\"]\n"
        f"argv=[sys.executable,{str(harness)!r},'--gate-fd',str(gate_r),'--ack-fd',str(ack_w),"
        "'--db',str(identity.parent/'unused.db'),'--task-id','t_dfa23a41',"
        "'--authorization-id','phase-test','--session-id','phase-session',"
        "'--identity-path',str(identity),'--launch-nonce','nonce',"
        "'--parent-pid',str(os.getpid()),'--parent-start',str(_process_start(os.getpid())),"
        "'--deadline',str(time.time()+20),'--',*worker]\n"
        "proc=subprocess.Popen(argv,pass_fds=(gate_r,ack_w),start_new_session=True)\n"
        f"Path({str(wrapper_pid_path)!r}).write_text(str(proc.pid))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(controller)])
    assert _wait_until(ready.exists)
    wrapper_pid = int(wrapper_pid_path.read_text(encoding="utf-8"))
    return proc, wrapper_pid, identity, release, marker


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_controller_death_before_parent_death_setup_is_caught_by_parent_recheck(tmp_path):
    controller, wrapper_pid, identity, release, marker = _launch_phase_controlled_controller(
        tmp_path, "before_parent_death"
    )
    controller.kill()
    controller.wait(timeout=5)
    release.write_text("continue", encoding="utf-8")

    assert _wait_until(lambda: not Path(f"/proc/{wrapper_pid}").exists())
    assert not identity.exists()
    assert not marker.exists()


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_controller_death_after_parent_death_setup_before_identity_fsync_kills_wrapper(tmp_path):
    controller, wrapper_pid, identity, _release, marker = _launch_phase_controlled_controller(
        tmp_path, "after_parent_death_before_identity"
    )
    controller.kill()
    controller.wait(timeout=5)

    assert _wait_until(lambda: not Path(f"/proc/{wrapper_pid}").exists())
    assert not identity.exists()
    assert not marker.exists()


@pytest.mark.linux_only
def test_real_controller_death_before_parent_contract_or_identity_cannot_start_work(tmp_path):
    pid, _identity, marker = _launch_crashing_gate_controller(
        tmp_path, wait_for_identity=False
    )

    assert _wait_until(lambda: not Path(f"/proc/{pid}").exists())
    time.sleep(0.1)
    assert not marker.exists()


@pytest.mark.linux_only
def test_real_controller_death_after_identity_fsync_kills_gated_wrapper_before_work(tmp_path):
    pid, identity_path, marker = _launch_crashing_gate_controller(
        tmp_path, wait_for_identity=True
    )

    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    assert payload["pid"] == pid
    assert payload["phase"] == "gated"
    assert _wait_until(lambda: not Path(f"/proc/{pid}").exists())
    time.sleep(0.1)
    assert not marker.exists()


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_cleanup_terminates_real_descendant_that_escaped_the_process_group(tmp_path):
    descendant_pid_path = tmp_path / "escaped.pid"
    parent_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import os,time; os.setsid(); time.sleep(60)'])\n"
        f"Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    gate_r, gate_w = os.pipe()
    ack_r, ack_w = os.pipe()
    try:
        assert _wait_until(descendant_pid_path.exists)
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        started_at = get_process_start_time(proc.pid)
        assert started_at is not None
        identity = kbd.GatedChildIdentity(
            pid=proc.pid,
            process_started_at=started_at,
            process_group_id=os.getpgid(proc.pid),
            session_id="tree-cleanup",
        )
        child = kbd.LinuxGatedChild(proc, gate_w, ack_r, identity)

        assert child.terminate_and_confirm()
        assert _wait_until(lambda: not Path(f"/proc/{proc.pid}").exists())
        assert _wait_until(lambda: not Path(f"/proc/{descendant_pid}").exists())
    finally:
        for pid in (proc.pid, int(descendant_pid_path.read_text()) if descendant_pid_path.exists() else 0):
            if pid > 0:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        for fd in (gate_r, ack_w):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_dead_leader_with_untraceable_escaped_descendant_fails_closed(tmp_path):
    descendant_pid_path = tmp_path / "untraceable-escaped.pid"
    parent_code = (
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import os,time; os.setsid(); time.sleep(60)'])\n"
        f"Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    started_at = get_process_start_time(proc.pid)
    assert started_at is not None
    proc.wait(timeout=5)
    assert descendant_pid_path.exists()
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    identity = kbd.GatedChildIdentity(
        pid=proc.pid,
        process_started_at=started_at,
        process_group_id=proc.pid,
        session_id="dead-leader",
    )
    try:
        assert not slr._terminate_recorded_identity(identity)
        assert Path(f"/proc/{descendant_pid}").exists()
    finally:
        try:
            os.kill(descendant_pid, 9)
        except ProcessLookupError:
            pass


def _crash_after_real_publication(board, tmp_path: Path, *, send_release: bool):
    auth = "release-crash" if send_release else "published-crash-real"
    marker = tmp_path / ("released-worker-ran" if send_release else "unreleased-worker-ran")
    run_path = tmp_path / "published-run-id"
    db_path = Path(board.execute("PRAGMA database_list").fetchone()[2])
    controller = tmp_path / "publication-controller.py"
    release_block = ""
    if send_release:
        release_block = (
            "token=json.dumps({'commit':True,'task_id':TARGET,'run_id':run_id,"
            "'authorization_id':%r},separators=(',',':')).encode()+b'\\n'\n" % auth
            + "os.write(child._gate_fd,token)\n"
            "deadline=time.monotonic()+5\n"
            "while time.monotonic()<deadline:\n"
            " row=conn.execute(\"SELECT 1 FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='resume_released'\",(TARGET,run_id)).fetchone()\n"
            " if row: break\n"
            " time.sleep(0.01)\n"
            "if not row: raise SystemExit(4)\n"
            "time.sleep(0.1)\n"
        )
    controller.write_text(
        "import json,os,sqlite3,sys,time\n"
        "from dataclasses import replace\n"
        "from pathlib import Path\n"
        "from gateway.status import get_process_start_time\n"
        "from hermes_cli import kanban_db as kb\n"
        "from hermes_cli import kanban_db_dispatch as kbd\n"
        "from hermes_cli import kanban_same_lane_resume as slr\n"
        f"TARGET={TARGET!r}\n"
        f"conn=sqlite3.connect({str(db_path)!r},isolation_level=None,timeout=30)\n"
        "conn.row_factory=sqlite3.Row\n"
        f"request=replace(slr.authorised_same_lane_request({auth!r}),expected_workspace={str(tmp_path)!r})\n"
        "task=kb.get_task(conn,TARGET)\n"
        f"worker=[sys.executable,'-c',\"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\"]\n"
        "kbd._same_lane_worker_material=lambda *_args:(worker,dict(os.environ))\n"
        "fence=slr._ResumeFence(conn,TARGET)\n"
        "with kb.write_txn(conn):\n"
        " assert slr._quarantine(conn,request)\n"
        "owner_start=get_process_start_time(os.getpid())\n"
        "child=kbd._launch_same_lane_worker(request,task,identity_path=str(fence.metadata_path),"
        "launch_nonce='real-crash',deadline=int(time.time())+20,controller_started_at=owner_start)\n"
        "with kb.write_txn(conn):\n"
        " run_id=slr._publish(conn,request,child.identity,ttl_seconds=None,publication_hook=None)\n"
        "fence.write_metadata({'resume_authorization_id':request.resume_authorization_id,"
        "'phase':'published','controller_pid':os.getpid(),'controller_started_at':owner_start,"
        "'run_id':run_id,'pid':child.identity.pid,'process_started_at':child.identity.process_started_at,"
        "'process_group_id':child.identity.process_group_id,'session_id':child.identity.session_id})\n"
        f"Path({str(run_path)!r}).write_text(str(run_id))\n"
        + release_block
        + "os._exit(0)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(controller)], capture_output=True, text=True, timeout=20
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return auth, marker, int(run_path.read_text(encoding="utf-8"))


@pytest.mark.linux_only
def test_real_controller_crash_after_commit_before_release_is_recovered_without_work(board, tmp_path):
    _insert_target(board)
    auth, marker, run_id = _crash_after_real_publication(
        board, tmp_path, send_release=False
    )

    outcome = kbd.resume_same_lane(
        board,
        _request(auth),
        observer=_observation,
        launcher=lambda *_: pytest.fail("recovery launched a second child"),
    )

    assert outcome.disposition == "compensated"
    assert outcome.run_id == run_id
    assert not marker.exists()
    assert _task_row(board)["status"] == "ready"
    run = board.execute("SELECT outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["outcome"] == "spawn_failed"
    assert run["ended_at"] is not None
    assert not [event for event in _events(board, "heartbeat") if event.run_id == run_id]


@pytest.mark.linux_only
def test_real_release_survives_controller_exit_only_after_durable_release_receipt(board, tmp_path):
    _insert_target(board)
    _auth, marker, run_id = _crash_after_real_publication(
        board, tmp_path, send_release=True
    )

    assert _wait_until(marker.exists)
    run_events = [event.kind for event in kb.list_events(board, TARGET) if event.run_id == run_id]
    assert run_events == ["claimed", "spawned", "resume_released"]
    assert not [event for event in _events(board, "heartbeat") if event.run_id == run_id]
