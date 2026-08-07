"""Behavioral tests for the upstream-aligned native review lifecycle."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


REVIEW_METADATA = {
    "pr_url": "https://github.com/acme/repo/pull/1",
    "repo": "acme/repo",
    "number": 1,
    "head_sha": "a" * 40,
    "verification_evidence": {"tests_passed": 3},
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return kb.connect()


def test_implementation_handoff_is_claimable_by_reviewer(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="default", summary="PR opened; focused tests pass",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "default"
        assert task.claim_lock is None
        review = kb.claim_review_task(conn, task_id, claimer="worker:default")
        assert review is not None
        assert review.status == "running"
        assert review.assignee == "default"


def test_review_approval_completes_and_changes_create_one_remediation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="default", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:default")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        )
        assert remediation_id
        remediation = kb.get_task(conn, remediation_id)
        assert remediation is not None
        assert remediation.assignee == "dev"
        assert remediation.status == "ready"
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.request_review_changes(conn, task_id, summary="Fix the regression test") is None
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
            (f"review-remediation:{task_id}:%",),
        ).fetchone()
        assert rows["n"] == 1


def test_review_approval_preserves_proof_and_scheduled_is_not_dispatchable(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="default", summary="Evidence attached",
            metadata={**REVIEW_METADATA, "head_sha": "b" * 40,
                      "commit": "abc123", "changed_files": ["src/example.py"]},
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:default")
        assert review is not None
        assert kb.complete_task(
            conn, task_id, summary="Approved after independent review",
            metadata={"approved": True, "commit": "abc123"}, expected_run_id=review.current_run_id,
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.metadata["approved"] is True

        scheduled_id = kb.create_task(conn, title="later", assignee="dev")
        assert kb.schedule_task(conn, scheduled_id, reason="wait for release")
        assert kb.claim_task(conn, scheduled_id) is None
        assert kb.get_task(conn, scheduled_id).status == "scheduled"


def test_unknown_reviewer_and_duplicate_head_do_not_mutate(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with board as conn:
        first_id = kb.create_task(conn, title="first", assignee="dev")
        first_run = kb.claim_task(conn, first_id)
        assert first_run is not None
        with pytest.raises(ValueError, match="does not exist"):
            kb.submit_for_review(conn, first_id, reviewer="missing", summary="ready",
                                 metadata=REVIEW_METADATA, expected_run_id=first_run.current_run_id)
        assert kb.get_task(conn, first_id).status == "running"
        assert kb.submit_for_review(conn, first_id, reviewer="default", summary="ready",
                                    metadata=REVIEW_METADATA,
                                    expected_run_id=first_run.current_run_id)

        second_id = kb.create_task(conn, title="duplicate", assignee="dev")
        second_run = kb.claim_task(conn, second_id)
        assert second_run is not None
        assert not kb.submit_for_review(conn, second_id, reviewer="default", summary="duplicate",
                                        metadata=REVIEW_METADATA,
                                        expected_run_id=second_run.current_run_id)
        assert kb.get_task(conn, second_id).status == "running"


def test_review_handoff_rejects_abbreviated_head_sha_without_mutation(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="immutable head_sha"):
            kb.submit_for_review(
                conn,
                task_id,
                reviewer="default",
                summary="ready",
                metadata={**REVIEW_METADATA, "head_sha": "a" * 12},
                expected_run_id=implementation.current_run_id,
            )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
