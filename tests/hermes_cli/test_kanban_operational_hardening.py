from __future__ import annotations

import json
import signal
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _claim_for_tool(conn, monkeypatch, task_id: str, *, profile: str = "builder", session_id: str = ""):
    task = kb.claim_task(conn, task_id, claimer=f"{profile}:test")
    assert task is not None
    run = kb.latest_run(conn, task_id)
    assert run is not None
    monkeypatch.setenv("HERMES_PROFILE", profile)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    if session_id:
        monkeypatch.setenv("HERMES_SESSION_ID", session_id)
    else:
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    return run


def _make_ready(conn, task_id: str) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
            (task_id,),
        )


def _make_running_again(conn, monkeypatch, task_id: str, *, profile: str = "builder"):
    _make_ready(conn, task_id)
    return _claim_for_tool(conn, monkeypatch, task_id, profile=profile)


def _summary(conn) -> dict[str, int]:
    return dict(Counter(task.status for task in kb.list_tasks(conn)))


def test_integrated_kanban_operational_hardening_board(kanban_home, monkeypatch, tmp_path):
    """Synthetic board covering operational hardening without provider calls."""
    from hermes_cli import active_sessions
    from hermes_cli.scoped_acceptance import evaluate_scoped_acceptance
    from hermes_cli.usage_guard import (
        build_compact_handoff_packet,
        validate_compact_handoff_packet,
    )
    from tools import kanban_tools as kt

    def forbid_live_side_effect(*_args, **_kwargs):
        raise AssertionError("Phase 18 harness must not spawn workers or open network sockets")

    monkeypatch.setattr(subprocess, "Popen", forbid_live_side_effect)
    monkeypatch.setattr(socket, "create_connection", forbid_live_side_effect)

    conn = kb.connect()
    try:
        # Parent/child state: child stays todo until the parent is complete.
        parent = kb.create_task(conn, title="parent artifact", assignee="builder")
        child = kb.create_task(
            conn,
            title="child waits on parent",
            assignee="builder",
            parents=[parent],
        )
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, child).status == "todo"
        assert kb.parent_ids(conn, child) == [parent]

        artifact = tmp_path / "parent-report.txt"
        artifact.write_text("READY\nhost visible report\n", encoding="utf-8")
        _claim_for_tool(conn, monkeypatch, parent)
        complete_payload = json.loads(
            kt._handle_complete(
                {
                    "summary": "parent complete with verified artifact",
                    "artifacts": [str(artifact)],
                    "artifact_markers": {str(artifact): "READY"},
                    "metadata": {"changed_files": [str(artifact)]},
                }
            )
        )
        assert complete_payload["ok"] is True
        assert complete_payload["terminal"] is True
        late_payload = json.loads(kt._handle_comment({"task_id": parent, "body": "late mutation"}))
        assert late_payload.get("ok") is not True
        assert "terminal" in late_payload.get("error", "").lower()
        assert kb.get_task(conn, parent).status == "done"
        assert kb.latest_run(conn, parent).metadata["verified_artifacts"][0]["path"] == str(artifact)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        _claim_for_tool(conn, monkeypatch, child)
        child_payload = json.loads(
            kt._handle_complete({"summary": "child completed after parent evidence"})
        )
        assert child_payload["ok"] is True

        # Missing artifact keeps the run active until the worker blocks with a
        # compact recovery packet and releases its active-session lease.
        missing_artifact_task = kb.create_task(conn, title="missing artifact", assignee="builder")
        missing = tmp_path / "not-created.txt"
        _claim_for_tool(
            conn,
            monkeypatch,
            missing_artifact_task,
            session_id="missing-artifact-session",
        )
        lease, message = active_sessions.try_acquire_active_session(
            session_id="missing-artifact-session",
            surface="kanban-worker",
            config={"max_concurrent_sessions": 10},
        )
        assert lease is not None
        assert message is None
        missing_payload = json.loads(
            kt._handle_complete({"summary": "claimed missing artifact", "artifacts": [str(missing)]})
        )
        assert missing_payload.get("ok") is not True
        assert kb.get_task(conn, missing_artifact_task).status == "running"

        packet = build_compact_handoff_packet(
            task_id=missing_artifact_task,
            phase="artifact verification",
            touched_files=[],
            recent_diff_summary="artifact missing from host-visible path",
            failing_tests=[],
            missing_symbols=[],
            blocked_commands=["kanban_complete rejected missing artifact"],
            declared_artifacts=[str(missing)],
            next_small_step="create host-visible artifact and rerun marker check",
            must_not_repeat=["do not claim scratch-only artifact paths"],
        )
        assert validate_compact_handoff_packet(packet) == []
        blocked_payload = json.loads(
            kt._handle_block(
                {
                    "reason": "review-required: missing host-visible artifact",
                    "kind": "needs_input",
                    "tests_run": ["pytest tests/hermes_cli/test_kanban_operational_hardening.py -q"],
                    "remaining_failures": [packet["next_small_step"]],
                    "metadata": {"usage_guard_handoff": packet},
                }
            )
        )
        assert blocked_payload["ok"] is True
        assert active_sessions.active_session_registry_snapshot() == []
        missing_task = kb.get_task(conn, missing_artifact_task)
        assert missing_task.status == "blocked"
        recovery = kb.latest_run(conn, missing_artifact_task).metadata["block_recovery"]
        assert recovery["exact_blocker"] == "review-required: missing host-visible artifact"

        # Repeated same-cause block routes to triage instead of looping.
        loop_task = kb.create_task(conn, title="same blocker loop", assignee="builder")
        _claim_for_tool(conn, monkeypatch, loop_task)
        assert kb.block_task(conn, loop_task, reason="need credentials", kind="capability")
        assert kb.unblock_task(conn, loop_task)
        _make_running_again(conn, monkeypatch, loop_task)
        assert kb.block_task(conn, loop_task, reason="still need credentials", kind="capability")
        assert kb.get_task(conn, loop_task).status == "triage"
        assert any(event.kind == "block_loop_detected" for event in kb.list_events(conn, loop_task))

        # Reviewer retry reaches verdict-first context on attempt three.
        review_task = kb.create_task(conn, title="review retry", assignee="reviewer")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_task,))
        assert kb.claim_review_task(conn, review_task) is not None
        with kb.write_txn(conn):
            kb._end_run(conn, review_task, outcome="blocked", summary="no verdict")
            conn.execute(
                "UPDATE tasks SET status = 'review', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (review_task,),
            )
        assert kb.claim_review_task(conn, review_task) is not None
        with kb.write_txn(conn):
            kb._end_run(conn, review_task, outcome="blocked", summary="still no verdict")
            conn.execute(
                "UPDATE tasks SET status = 'review', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (review_task,),
            )
        assert kb.claim_review_task(conn, review_task) is not None
        review_context = kb.build_worker_context(conn, review_task)
        assert "## Review verdict required now" in review_context
        assert "review attempt 3" in review_context
        assert "actionable" in review_context
        assert "contract-misread" in review_context

        # Scoped acceptance completes with a caveat when only unrelated full-suite
        # failures remain.
        scoped_task = kb.create_task(conn, title="scoped acceptance", assignee="builder")
        _claim_for_tool(conn, monkeypatch, scoped_task)
        acceptance = {
            "touched_files": ["src/widget.py", "tests/test_widget.py"],
            "task_scope_files": ["src/widget.py", "tests/test_widget.py"],
            "targeted_tests": [{"command": "pytest tests/test_widget.py", "status": "passed"}],
            "required_checks": [{"name": "diff-check", "status": "passed"}],
            "full_suite_failures": [
                {
                    "test": "tests/test_legacy.py::test_baseline",
                    "file": "tests/test_legacy.py",
                    "message": "pre-existing baseline failure",
                }
            ],
        }
        assert evaluate_scoped_acceptance(**acceptance)["decision"] == "complete_with_caveat"
        scoped_payload = json.loads(
            kt._handle_complete(
                {
                    "summary": "scoped widget fix complete",
                    "acceptance": acceptance,
                    "metadata": {"changed_files": ["src/widget.py", "tests/test_widget.py"]},
                }
            )
        )
        assert scoped_payload["ok"] is True
        scoped_run = kb.latest_run(conn, scoped_task)
        assert scoped_run.metadata["scoped_acceptance"]["decision"] == "complete_with_caveat"
        assert "Scoped acceptance caveat" in kb.latest_summary(conn, scoped_task)

        # Synthesizer summary depends on completed parents and cites curated evidence.
        synth_task = kb.create_task(
            conn,
            title="synthesizer summary",
            assignee="synthesizer",
            parents=[parent, scoped_task],
            skills=["kanban-synthesizer", "kanban-worker-operational-discipline"],
        )
        kb.recompute_ready(conn)
        synth = kb.get_task(conn, synth_task)
        assert synth.status == "ready"
        assert "kanban-synthesizer" in synth.skills
        _claim_for_tool(conn, monkeypatch, synth_task, profile="synthesizer")
        synth_payload = json.loads(
            kt._handle_complete(
                {
                    "summary": "synthesized from parent summaries and verified artifacts",
                    "metadata": {
                        "evidence_sources": [kb.latest_summary(conn, parent), str(artifact)],
                        "missing_evidence": [],
                    },
                }
            )
        )
        assert synth_payload["ok"] is True

        # Process reconciliation clears terminal worker pids without touching
        # running tasks.
        terminal_pid_task = kb.create_task(conn, title="terminal pid cleanup", assignee="builder")
        _claim_for_tool(conn, monkeypatch, terminal_pid_task, session_id="terminal-pid-session")
        assert kb.complete_task(
            conn,
            terminal_pid_task,
            summary="done but worker pid remained",
            metadata={"worker_session_id": "terminal-pid-session"},
        )
        run = kb.latest_run(conn, terminal_pid_task)
        lock = f"{kb._claimer_id().split(':', 1)[0]}:phase18"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, claim_lock = ?, claim_expires = ? WHERE id = ?",
                (456789, lock, int(time.time()) + 3600, terminal_pid_task),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ?, claim_lock = ?, metadata = ? WHERE id = ?",
                (
                    456789,
                    lock,
                    json.dumps({"worker_session_id": "terminal-pid-session"}),
                    run.id,
                ),
            )
        signals = []
        alive = {"value": True}
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive["value"])

        def signal_fake_worker(pid, sig):
            signals.append((pid, sig))
            alive["value"] = False

        reconciled = kb.reconcile_terminal_worker_processes(
            conn,
            signal_fn=signal_fake_worker,
        )
        assert reconciled == [terminal_pid_task]
        assert signals == [(456789, signal.SIGTERM)]
        terminal_after = kb.get_task(conn, terminal_pid_task)
        assert terminal_after.worker_pid is None
        assert terminal_after.claim_lock is None

        board_summary = _summary(conn)
        assert board_summary["done"] >= 5
        assert board_summary["blocked"] >= 1
        assert board_summary["triage"] >= 1
        assert board_summary.get("running", 0) == 1  # review retry remains claimed for verdict.
        assert any(
            event.kind == "completed" and event.payload and event.payload.get("verified_artifacts")
            for event in kb.list_events(conn, parent)
        )
    finally:
        conn.close()
