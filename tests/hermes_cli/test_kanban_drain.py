from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_drain as kd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _review_required_source(
    conn,
    *,
    reason: str = "review-required: implementation needs independent review",
) -> str:
    task_id = kb.create_task(
        conn,
        title="implement narrow backend slice",
        assignee="backend-engineer",
        priority=10,
    )
    kb.claim_task(conn, task_id, claimer="test:impl")
    assert kb.block_task(conn, task_id, reason=reason)
    return task_id


def _completed_review(conn, source_task_id: str, *, approved: bool) -> str:
    review_id = kb.create_task(
        conn,
        title="review implementation packet",
        assignee="reviewer",
        priority=20,
    )
    metadata = {
        "source_task_id": source_task_id,
        "approved": approved,
        "findings": [] if approved else ["missing regression test"],
    }
    assert kb.complete_task(
        conn,
        review_id,
        summary="review packet complete",
        metadata=metadata,
    )
    return review_id


def test_drain_dry_run_reports_review_pass_without_mutation(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        review_id = _completed_review(conn, source_id, approved=True)

        report = kd.drain_review_packets(conn, apply=False)

        assert report["dry_run"] is True
        assert report["summary"]["planned"] == 1
        action = report["actions"][0]
        assert action["action"] == "consume_review_pass"
        assert action["source_task_id"] == source_id
        assert action["review_task_id"] == review_id
        assert action["status"] == "planned"
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []


def test_drain_apply_consumes_review_pass_without_rerunning_source(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        review_id = _completed_review(conn, source_id, approved=True)

        report = kd.drain_review_packets(conn, apply=True)

        assert report["summary"]["applied"] == 1
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "done"
        comments = kb.list_comments(conn, source_id)
        assert len(comments) == 1
        assert "consume_review_pass" in comments[0].body
        assert review_id in comments[0].body
        runs = kb.list_runs(conn, source_id)
        assert [run.outcome for run in runs] == ["blocked", "completed"]
        completed = runs[-1]
        assert completed.metadata is not None
        assert completed.metadata["drain_action"] == "consume_review_pass"
        assert completed.metadata["review_task_id"] == review_id


def test_drain_apply_review_fail_comments_and_creates_idempotent_rework(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        review_id = kb.create_task(conn, title="review packet", assignee="reviewer")
        assert kb.complete_task(
            conn,
            review_id,
            summary="review packet failed",
            metadata={
                "source_card": source_id,
                "verdict": "FAIL",
                "findings": ["missing regression test"],
            },
        )

        first = kd.drain_review_packets(conn, apply=True)
        second = kd.drain_review_packets(conn, apply=True)

        assert first["summary"]["applied"] == 1
        assert second["summary"]["already_applied"] == 1
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        comments = [c for c in kb.list_comments(conn, source_id) if "consume_review_fail" in c.body]
        assert len(comments) == 1
        rework_tasks = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.idempotency_key == f"kanban-drain:review-fail:{source_id}:{review_id}"
        ]
        assert len(rework_tasks) == 1
        assert rework_tasks[0].assignee == "backend-engineer"
        assert rework_tasks[0].status == "ready"
        assert source_id in (rework_tasks[0].body or "")
        assert review_id in (rework_tasks[0].body or "")


def test_drain_apply_is_idempotent_for_review_pass(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        _completed_review(conn, source_id, approved=True)

        first = kd.drain_review_packets(conn, apply=True)
        second = kd.drain_review_packets(conn, apply=True)

        assert first["summary"]["applied"] == 1
        assert second["summary"]["already_applied"] == 1
        assert second["summary"]["skipped"] == 0
        assert len([c for c in kb.list_comments(conn, source_id) if "consume_review_pass" in c.body]) == 1
        assert len([e for e in kb.list_events(conn, source_id) if e.kind == "completed"]) == 1


@pytest.mark.parametrize(
    "reason",
    [
        "review-required: need human product decision before merge",
        "review-required: missing credential for prod smoke test",
        "review-required: production rollout risk requires operator approval",
    ],
)
def test_drain_refuses_human_credential_and_prod_risk_block_text(
    kanban_home: Path,
    reason: str,
) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn, reason=reason)
        _completed_review(conn, source_id, approved=True)

        report = kd.drain_review_packets(conn, apply=True)

        assert report["summary"]["refused"] == 1
        action = report["actions"][0]
        assert action["status"] == "refused"
        assert "hold" in action["reason"]
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []


def test_kanban_drain_cli_json_dry_run(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        review_id = _completed_review(conn, source_id, approved=True)

    raw = kc.run_slash("drain --dry-run --json")
    report = json.loads(raw)

    assert report["dry_run"] is True
    assert report["actions"][0]["source_task_id"] == source_id
    assert report["actions"][0]["review_task_id"] == review_id
    assert report["actions"][0]["status"] == "planned"
    with kb.connect() as conn:
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []
