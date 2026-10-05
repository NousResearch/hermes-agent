"""block_task() classifies an untyped breaker block in place (#117363)."""
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _time_out_once(conn, tid: str) -> None:
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - 30
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, tid),
        )
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)


def test_typed_block_classifies_untyped_breaker_block(kanban_home, monkeypatch):
    """Two timeouts trip the breaker untyped; the supervisor's typed block then
    attaches the kind without a status flap, a synthetic run, or lost evidence."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ceiling", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        _time_out_once(conn, tid)
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", None, 0)

        assert kb.block_task(
            conn, tid, reason="needs a human decision", kind="needs_input",
            block_owner="operator", block_evidence="decision record",
            block_unblock_action="record decision", block_followup_review="review outcome",
        ) is True

        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind, after.block_recurrences) == ("blocked", "needs_input", 1)
        assert after.consecutive_failures == parked.consecutive_failures
        assert after.last_failure_error == parked.last_failure_error
        assert after.current_run_id is None
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert (kinds.count("blocked"), kinds.count("gave_up"), kinds.count("timed_out")) == (1, 1, 2)
        assert len(kb.list_runs(conn, tid)) == 2


def test_typed_block_still_refuses_typed_or_live_blocked_cards(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        parked = kb.create_task(conn, title="parked", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, parked)
        _time_out_once(conn, parked)
        assert kb.get_task(conn, parked).status == "blocked"
        stale_run_id = conn.execute(
            "SELECT MAX(id) FROM task_runs WHERE task_id = ?", (parked,)).fetchone()[0]
        assert stale_run_id is not None
        # A worker asserting ownership of its (ended) run cannot classify a parked card.
        assert kb.block_task(
            conn, parked, reason="mine", kind="needs_input", expected_run_id=stale_run_id,
            block_owner="operator", block_evidence="decision record",
            block_unblock_action="record decision", block_followup_review="review outcome",
        ) is False
        assert kb.get_task(conn, parked).block_kind is None

        typed = kb.create_task(conn, title="typed once", assignee="worker")
        kb.claim_task(conn, typed)
        assert kb.block_task(
            conn, typed, reason="decision", kind="needs_input",
            block_owner="operator", block_evidence="decision record",
            block_unblock_action="record decision", block_followup_review="review outcome",
        ) is True
        assert kb.block_task(
            conn, typed, reason="again", kind="capability",
            block_owner="operator", block_evidence="access record",
            block_unblock_action="grant access", block_followup_review="review outcome",
        ) is False
        assert kb.get_task(conn, typed).block_kind == "needs_input"

        live = kb.create_task(conn, title="live run", assignee="worker")
        kb.claim_task(conn, live)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (live,))
        assert kb.block_task(
            conn, live, reason="race", kind="needs_input",
            block_owner="operator", block_evidence="decision record",
            block_unblock_action="record decision", block_followup_review="review outcome",
        ) is False
        assert kb.block_task(conn, live, reason="no kind") is False
        assert kb.get_task(conn, live).block_kind is None


def test_legacy_generic_block_remains_compatible_and_metadata_clears(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy block")
        assert kb.block_task(conn, tid, reason="waiting on upstream") is True
        blocked = kb.get_task(conn, tid)
        assert blocked.status == "blocked"
        assert blocked.block_kind is None

        assert kb.unblock_task(conn, tid) is True
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="needs decision", kind="needs_input",
            block_owner="operator", block_evidence="ticket",
            block_unblock_action="resolve ticket", block_followup_review="review",
        ) is True
        blocked = kb.get_task(conn, tid)
        assert blocked.block_owner == "operator"
        assert blocked.block_evidence == "ticket"
        assert kb.unblock_task(conn, tid) is True
        cleared = kb.get_task(conn, tid)
        assert cleared.block_owner is None
        assert cleared.block_evidence is None


def test_dependency_without_open_parent_requires_human_blocker_metadata(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="orphan dependency", assignee="default")
        with pytest.raises(ValueError, match="human blockers require"):
            kb.block_task(conn, tid, reason="no parent", kind="dependency")
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.block_kind is None

        assert kb.block_task(
            conn, tid, reason="no parent", kind="dependency",
            block_owner="operator", block_evidence="dependency audit",
            block_unblock_action="supply parent", block_followup_review="review linkage",
        ) is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert task.block_owner == "operator"


def test_task_delivery_validation_and_assignment_boundary(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="task_type"):
            kb.create_task(conn, title="bad type", task_type="unknown")
        with pytest.raises(ValueError, match="delivery_type"):
            kb.create_task(conn, title="bad delivery", delivery_type="remote")
        with pytest.raises(ValueError, match="require an assignee"):
            kb.create_task(conn, title="unassigned PR", delivery_type="PR", completion_contract="acme/repo")

        tid = kb.create_task(
            conn, title="typed PR", assignee="default", task_type="implementation",
            delivery_type="PR", completion_contract="acme/repo", initial_status="blocked",
        )
        with pytest.raises(ValueError, match="not a dispatchable profile"):
            kb.assign_task(conn, tid, "ghost")
        assert kb.assign_task(conn, tid, "default") is True


def test_non_pr_delivery_completes_without_pr_acceptance(kanban_home):
    with kbc.connect() as conn:
        for delivery in ("local", "pre_pr"):
            tid = kb.create_task(
                conn, title=delivery, assignee="default", delivery_type=delivery,
                completion_contract="local-only",
            )
            assert kb.complete_task(
                conn, tid, result="done",
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"},
            ) is True
            assert kb.get_task(conn, tid).status == "done"


def test_edit_claim_and_review_boundaries_fail_closed(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="edit", assignee="default")
        with pytest.raises(ValueError, match="delivery_type"):
            kb.edit_task(conn, tid, delivery_type="remote")
        with pytest.raises(ValueError, match="completion_contract"):
            kb.edit_task(conn, tid, delivery_type="PR")

        typed = kb.create_task(
            conn, title="legacy typed", assignee="default", task_type="implementation",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET assignee = NULL WHERE id = ?", (typed,))
        with pytest.raises(ValueError, match="require an assignee"):
            kb.claim_task(conn, typed)

        pr = kb.create_task(
            conn, title="review handoff", assignee="default", delivery_type="PR",
            completion_contract="acme/repo",
        )
        assert kb.request_review(conn, pr, summary="missing acceptance handoff") is False
