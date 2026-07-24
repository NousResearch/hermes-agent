"""Regression coverage for guarded Kanban completion and promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Provide an isolated board for each completion transition test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_blocked_result_cannot_be_recast_as_completion(kanban_home):
    """A blocker remains auditable and cannot promote a dependent task."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.block_task(conn, parent, reason="CHANGES-REQUIRED") is True

        assert kb.complete_task(
            conn,
            parent,
            summary="not approval-ready; missing evidence",
            metadata={"blocker": "CHANGES-REQUIRED"},
        ) is False

        assert kb.get_task(conn, parent).status == "blocked"
        assert kb.get_task(conn, child).status == "todo"
        events = kb.list_events(conn, parent)
        rejected = [event for event in events if event.kind == "completion_rejected"]
        assert len(rejected) == 1
        assert rejected[0].payload == {
            "reason": "blocked_task_requires_explicit_unblock",
            "prior_status": "blocked",
        }
        assert not any(event.kind == "completed" for event in events)


@pytest.mark.parametrize(
    "reason",
    [
        "worker result contains blockers",
        "CHANGES-REQUIRED: update the migration",
        "not approval-ready",
        "missing evidence: focused test output",
    ],
)
def test_blocker_handoff_rejects_every_premature_completion_attempt(
    kanban_home, reason
):
    """Blocker-style handoffs retain their status and audit trail verbatim."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.block_task(conn, parent, reason=reason) is True
        before = kb.list_events(conn, parent)

        assert kb.complete_task(
            conn,
            parent,
            result="accepted",
            summary="premature acceptance attempt",
            metadata={"evidence": []},
        ) is False

        assert kb.get_task(conn, parent).status == "blocked"
        assert kb.get_task(conn, child).status == "todo"
        after = kb.list_events(conn, parent)
        assert [event.kind for event in after[:-1]] == [event.kind for event in before]
        assert after[-1].kind == "completion_rejected"
        assert after[-1].payload == {
            "reason": "blocked_task_requires_explicit_unblock",
            "prior_status": "blocked",
        }
        assert not any(event.kind == "completed" for event in after)


def test_accepted_completion_records_evidence_and_promotes_successor(kanban_home):
    """Only the accepted completion path makes a dependent task ready."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"

        assert kb.claim_task(conn, parent, claimer="test:worker") is not None
        assert kb.complete_task(
            conn,
            parent,
            result="accepted",
            summary="verified delivery",
            metadata={"evidence": ["focused-tests"], "approval_ready": True},
        ) is True

        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        events = kb.list_events(conn, parent)
        completed = [event for event in events if event.kind == "completed"]
        assert len(completed) == 1
        assert completed[0].payload["summary"] == "verified delivery"
        run = kb.latest_run(conn, parent)
        assert run is not None
        assert run.outcome == "completed"
        assert run.metadata == {
            "evidence": ["focused-tests"],
            "approval_ready": True,
        }
        assert any(event.kind == "promoted" for event in kb.list_events(conn, child))


def test_acceptance_is_the_only_successor_promotion_point(kanban_home):
    """Promotion follows done only, never a blocked or premature report."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.claim_task(conn, parent, claimer="test:worker") is not None
        assert kb.block_task(
            conn, parent, reason="needs review", kind="needs_input"
        ) is True
        assert kb.get_task(conn, child).status == "todo"
        assert kb.complete_task(conn, parent, summary="accepted without unblock") is False
        assert kb.get_task(conn, child).status == "todo"

        assert kb.unblock_task(conn, parent) is True
        assert kb.complete_task(
            conn,
            parent,
            summary="accepted after explicit unblock",
            metadata={"evidence": ["review-record"]},
        ) is True
        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        kinds = [event.kind for event in kb.list_events(conn, parent)]
        assert kinds.index("completion_rejected") < kinds.index("completed")


def test_clean_exit_without_completion_receipt_never_promotes_successor(
    kanban_home, monkeypatch,
):
    """A clean worker exit without an explicit receipt is a protocol failure."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("clean_exit", 0))
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, parent, claimer=f"{host}:worker")
        pid = 999991
        kb._set_worker_pid(conn, parent, pid)
        kb._record_worker_exit(pid, 0)

        assert parent in kb.detect_crashed_workers(conn)
        # Upstream gives clean-exit protocol violations a bounded retry budget.
        # The safety invariant here is no acceptance/promotion without a receipt.
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, child).status == "todo"
        kinds = [event.kind for event in kb.list_events(conn, parent)]
        assert "protocol_violation" in kinds
        assert "completed" not in kinds
        assert "gave_up" not in kinds


def test_crash_retry_then_acceptance_promotes_only_after_retry(kanban_home, monkeypatch):
    """A crash requeues work; a later accepted run is the only promotion point."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 1))
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, parent, claimer=f"{host}:worker")
        pid = 999992
        kb._set_worker_pid(conn, parent, pid)
        kb._record_worker_exit(pid, 256)

        assert parent in kb.detect_crashed_workers(conn)
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, child).status == "todo"
        assert any(event.kind == "crashed" for event in kb.list_events(conn, parent))

        kb.claim_task(conn, parent, claimer="test:retry")
        assert kb.complete_task(
            conn,
            parent,
            summary="retry accepted with evidence",
            metadata={"evidence": ["retry-tests"]},
        ) is True
        assert kb.get_task(conn, child).status == "ready"


def test_late_completion_from_reclaimed_run_cannot_promote_successor(kanban_home):
    """A stale run token cannot close the replacement run or release children."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        first = kb.claim_task(conn, parent, claimer="test:first")
        assert first is not None
        first_run = kb.latest_run(conn, parent)
        assert first_run is not None
        assert kb.reclaim_task(conn, parent, reason="worker lost") is True
        second = kb.claim_task(conn, parent, claimer="test:second")
        assert second is not None
        second_run = kb.latest_run(conn, parent)
        assert second_run is not None and second_run.id != first_run.id

        assert kb.complete_task(
            conn, parent, summary="late result", expected_run_id=first_run.id
        ) is False
        assert kb.get_task(conn, parent).status == "running"
        assert kb.get_task(conn, child).status == "todo"
        assert not any(
            event.kind == "completed" for event in kb.list_events(conn, parent)
        )


def test_windows_worker_exit_codes_are_classified_without_posix_wait_helpers(
    monkeypatch,
):
    """Windows Popen return codes still distinguish clean exits from crashes."""
    monkeypatch.setattr(kb.os, "name", "nt")
    kb._record_worker_exit(999993, 0)
    kb._record_worker_exit(999994, 1)

    assert kb._classify_worker_exit(999993) == ("clean_exit", 0)
    assert kb._classify_worker_exit(999994) == ("nonzero_exit", 1)


def test_windows_reaper_polls_registered_workers(monkeypatch):
    """Windows dispatch reaps Popen children without POSIX waitpid."""
    class FinishedProcess:
        pid = 999995

        def poll(self):
            return 7

    monkeypatch.setattr(kb.os, "name", "nt")
    kb._worker_processes.clear()
    kb._recent_worker_exits.pop(FinishedProcess.pid, None)
    kb._worker_processes[FinishedProcess.pid] = FinishedProcess()

    assert kb.reap_worker_zombies() == [FinishedProcess.pid]
    assert FinishedProcess.pid not in kb._worker_processes
    assert kb._classify_worker_exit(FinishedProcess.pid) == ("nonzero_exit", 7)
