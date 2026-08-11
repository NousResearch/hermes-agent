"""Behavioral tests for the upstream-aligned native review lifecycle."""

import json
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
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    kb.init_db()
    return kb.connect()


def test_implementation_handoff_is_claimable_by_reviewer(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="PR opened; focused tests pass",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.claim_lock is None
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert review.status == "running"
        assert review.assignee == "reviewer"


def test_review_changes_returns_same_card_to_implementer(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        )
        assert remediation_id == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "dev"
        assert task.status == "ready"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run["status"] == "ready"
        assert run["outcome"] == "changes_requested"
        assert run["ended_at"] is not None


def test_review_changes_requires_worker_ownership(board, monkeypatch):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", "different-task")
        with pytest.raises(PermissionError, match="refusing to mutate"):
            kb.request_review_changes(
                conn, task_id, summary="must not mutate sibling",
                expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_changes_requested_invalidates_historical_review_submission(board):
    with board as conn:
        task_id = kb.create_task(
            conn, title="implement", assignee="dev",
            metadata={"lane": "dev", "task_type": "implementation"},
        )
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="fix it", expected_run_id=review.current_run_id
        ) == task_id
        fix = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert fix is not None
        with pytest.raises(kb.ReviewRequiredError):
            kb.complete_task(
                conn, task_id, summary="done", metadata=REVIEW_METADATA,
                expected_run_id=fix.current_run_id,
            )


def test_default_review_submission_uses_guaranteed_profile(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"dev", "default"})
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        assert kb.get_task(conn, task_id).assignee == "default"


def test_review_provenance_survives_status_requeue_and_omitted_source_claim(board):
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="reviewer", initial_status="review")
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None

        assert _set_status_direct(conn, task_id, "ready")
        reclaimed = kb.get_task(conn, task_id)
        assert reclaimed is not None
        assert reclaimed.status == "ready"

        claimed = kb.claim_task(conn, task_id, claimer="worker:reviewer")
        assert claimed is not None
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='claimed' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert json.loads(event["payload"])["source_status"] == "review"
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert json.loads(run["metadata"])["source_status"] == "review"


def test_native_review_retry_returns_to_review_dispatch_lane(board):
    with board as conn:
        task_id = kb.create_task(
            conn, title="review retry", assignee="reviewer", initial_status="review"
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.reclaim_task(conn, task_id, reason="review worker exited")
        assert kb.get_task(conn, task_id).status == "review"

        result = kb.dispatch_once(conn, dry_run=True)

        assert result.spawned == [(task_id, "reviewer", "")]


def test_existing_pr_remediation_opt_in_bypasses_only_active_pr_guard(board):
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    with board as conn:
        ordinary_id = kb.create_task(conn, title="ordinary", assignee="dev")
        remediation_id = kb.create_task(
            conn,
            title="remediation",
            assignee="dev",
            metadata={kb.EXISTING_PR_REMEDIATION_METADATA_KEY: True},
        )
        for task_id in (ordinary_id, remediation_id):
            assert kb.add_comment(
                conn, task_id, author="worker", body="PR: https://github.com/acme/repo/pull/1"
            )
            assert _set_status_direct(conn, task_id, "ready")

        assert kb.check_respawn_guard(conn, ordinary_id) == "active_pr"
        assert kb.check_respawn_guard(conn, remediation_id) is None
        assert kb.has_spawnable_ready(conn)


def test_changes_requested_reuses_same_card_for_implementer_and_re_review(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", metadata={"approved": False},
            expected_run_id=review.current_run_id,
        ) == task_id
        fix = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert fix is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="fixed",
            metadata={**REVIEW_METADATA, "head_sha": "b" * 40},
            expected_run_id=fix.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "review"


def test_dev_implementation_rerun_cannot_use_historical_review_submission(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        ) == task_id
        rerun = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert rerun is not None
        assert kb.request_review_changes(
            conn, task_id, summary="I found another issue", expected_run_id=rerun.current_run_id
        ) is None
        assert kb.get_task(conn, task_id).status == "running"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? AND kind='review_changes_requested'",
            (task_id,),
        ).fetchone()["n"] == 1


def test_review_changes_rejects_delegated_child_context(board):
    from agent.delegation_context import delegated_child_context

    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        with delegated_child_context(), pytest.raises(PermissionError, match="delegate_task child"):
            kb.request_review_changes(
                conn, task_id, summary="child must not mutate the board",
                expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_review_handoff_rejects_self_reviewer_without_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="different from the implementer"):
            kb.submit_for_review(
                conn, task_id, reviewer="dev", summary="ready",
                metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_review_handoff_rejects_nonspawnable_reviewer_before_mutation(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "dev")
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="does not resolve to an on-disk Hermes profile"):
            kb.submit_for_review(
                conn, task_id, reviewer="missing", summary="ready",
                metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
            )
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.assignee == "dev"


def test_review_handoff_rejects_abbreviated_sha_before_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="immutable head_sha"):
            kb.submit_for_review(
                conn, task_id, reviewer="reviewer", summary="ready",
                metadata={**REVIEW_METADATA, "head_sha": "abc123"},
                expected_run_id=implementation.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_webhook_rejects_invalid_producer_without_creating_a_card(board):
    with board as conn:
        with pytest.raises(ValueError, match="full immutable 40-character"):
            kb.ingest_pull_request(
                conn, repository="acme/repo", number=9, head_sha="abc123",
                title="invalid producer",
            )
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0


def test_webhook_draft_is_parked_in_triage_with_source_metadata(board):
    with board as conn:
        task_id = kb.ingest_pull_request(
            conn,
            repository="SoLoVisionLLC/SoLoFamilyPlan",
            number=112,
            head_sha="d" * 40,
            title="Draft PR",
            draft=True,
            reviewer="reviewer",
        )
        assert task_id is not None
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.metadata is not None
        assert task.status == "triage"
        assert task.metadata["source"] == "github_pull_request"
        assert task.metadata["draft"] is True


def test_external_webhook_review_routes_changes_to_dev_on_same_pr(board):
    with board as conn:
        task_id = kb.ingest_pull_request(
            conn,
            repository="acme/repo",
            number=77,
            head_sha="e" * 40,
            title="External PR",
            reviewer="reviewer",
            checks_passed=False,
            mergeable=False,
        )
        assert task_id is not None
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn,
            task_id,
            summary="Fix failing check and merge conflict on the existing PR branch",
            metadata={"findings": [{"severity": "high", "path": "src/app.py"}]},
            expected_run_id=review.current_run_id,
        )
        assert remediation_id == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "dev"
        assert task.status == "ready"
        assert task.metadata["original_implementer"] == "dev"
        assert task.metadata["canonical"] is True
        assert task.metadata["lane"] == "DEV"
        assert task.metadata["coding_agent"] == "codex"
        assert task.metadata["existing_pr_remediation"] is True



def test_webhook_replay_preserves_native_review_claim(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="native handoff",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        replay_id = kb.ingest_pull_request(
            conn, repository="ACME/REPO", number=1, head_sha="A" * 40,
            title="webhook replay", reviewer="other", checks_passed=False,
        )
        assert replay_id == task_id
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_webhook_first_native_submission_preserves_native_card(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=1, head_sha="a" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        assert kb.get_task(conn, webhook_id).status == "archived"
        assert kb.get_task(conn, implementation_id).status == "review"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status != 'archived'").fetchone()["n"] == 1


def test_webhook_first_claimed_review_run_is_finalized(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=3, head_sha="c" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        assert webhook_id is not None
        webhook_review = kb.claim_review_task(conn, webhook_id, claimer="worker:reviewer")
        assert webhook_review is not None
        webhook_run_id = webhook_review.current_run_id

        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata={**REVIEW_METADATA, "pr_url": "https://github.com/acme/repo/pull/3",
                      "repo": "acme/repo", "number": 3, "head_sha": "c" * 40},
            expected_run_id=implementation.current_run_id,
        )

        merged_run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=?",
            (webhook_run_id,),
        ).fetchone()
        assert merged_run["status"] == "review"
        assert merged_run["outcome"] == "submitted_for_review"
        assert merged_run["ended_at"] is not None
        assert kb.get_task(conn, webhook_id).status == "archived"
        assert kb.get_task(conn, implementation_id).status == "review"


def test_webhook_first_reconciliation_merges_duplicate_subscriptions(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=2, head_sha="b" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        for task_id, cursor in ((webhook_id, 3), (implementation_id, 7)):
            conn.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
                "VALUES (?, 'telegram', 'same-chat', 'same-thread', 1, ?)",
                (task_id, cursor),
            )
        conn.commit()
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata={**REVIEW_METADATA, "pr_url": "https://github.com/acme/repo/pull/2",
                      "repo": "acme/repo", "number": 2, "head_sha": "b" * 40},
            expected_run_id=implementation.current_run_id,
        )
        row = conn.execute(
            "SELECT task_id, last_event_id FROM kanban_notify_subs "
            "WHERE platform='telegram' AND chat_id='same-chat' AND thread_id='same-thread'"
        ).fetchone()
        assert row["task_id"] == implementation_id
        assert row["last_event_id"] == 7


def test_review_approval_preserves_proof_and_scheduled_is_not_dispatchable(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="Evidence attached",
            metadata={**REVIEW_METADATA, "commit": "abc123", "changed_files": ["src/example.py"]},
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.complete_task(
            conn, task_id, summary="Approved after independent review",
            metadata={"approved": True, "commit": "abc123"},
            expected_run_id=review.current_run_id,
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.metadata["approved"] is True
        scheduled_id = kb.create_task(conn, title="later", assignee="dev")
        assert kb.schedule_task(conn, scheduled_id, reason="wait for release")
        assert kb.claim_task(conn, scheduled_id) is None
        assert kb.get_task(conn, scheduled_id).status == "scheduled"


def test_webhook_same_head_refreshes_live_metadata_and_status(board):
    with board as conn:
        task_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=112, head_sha="d" * 40,
            title="Draft PR", draft=True, reviewer="reviewer",
        )
        assert task_id is not None
        assert kb.get_task(conn, task_id).metadata["draft"] is True

        promoted_id = kb.ingest_pull_request(
            conn, repository="ACME/REPO", number=112, head_sha="D" * 40,
            title="Promoted PR", draft=False, mergeable=True, reviewer="reviewer",
        )
        assert promoted_id == task_id
        promoted = kb.get_task(conn, task_id)
        assert promoted.status == "review"
        assert promoted.title == "Review PR #112: Promoted PR"
        assert promoted.metadata["draft"] is False
        assert promoted.metadata["mergeable"] is True
        assert promoted.metadata["head_sha"] == "d" * 40

        review_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=112, head_sha="d" * 40,
            title="Promoted PR", draft=False, mergeable=False, reviewer="reviewer",
        )
        assert review_id == task_id
        refreshed = kb.get_task(conn, task_id)
        assert refreshed is not None
        assert refreshed.status == "review"
        assert refreshed.metadata["mergeable"] is False


def test_webhook_same_head_replay_is_idempotent_after_metadata_refresh(board):
    with board as conn:
        kwargs = dict(
            repository="acme/repo", number=66, head_sha="e" * 40,
            title="Stable PR", draft=False, mergeable=False, reviewer="reviewer",
        )
        task_id = kb.ingest_pull_request(conn, **kwargs)
        before_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        assert kb.ingest_pull_request(conn, **kwargs) == task_id
        after_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        assert after_events == before_events
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_webhook_closed_and_merged_archive_same_head_cards(board):
    with board as conn:
        task_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=70, head_sha="f" * 40,
            title="Closed PR", reviewer="reviewer",
        )
        assert kb.ingest_pull_request(
            conn, repository="acme/repo", number=70, head_sha="f" * 40,
            title="Closed PR", action="closed", reviewer="reviewer",
        ) == task_id
        assert kb.get_task(conn, task_id).status == "archived"
        assert kb.ingest_pull_request(
            conn, repository="acme/repo", number=70, head_sha="f" * 40,
            title="Closed PR", action="merged", reviewer="reviewer",
        ) == task_id
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
