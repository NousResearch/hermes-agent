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


def _review_packet_metadata(
    source_task_id: str,
    *,
    approved: bool,
    findings: list[str] | None = None,
) -> dict:
    return {
        "review_packet_version": 1,
        "source_task_id": source_task_id,
        "verdict": "PASS" if approved else "FAIL",
        "reviewer": "pr-reviewer",
        "evidence": {
            "diff_path": "workspaces/source-card.patch",
            "tests_run": ["pytest tests/hermes_cli/test_kanban_drain.py -q"],
            "findings": [] if findings is None else findings,
        },
        "authority_boundary": (
            "no merge, deploy, production promotion, public action, credential action, "
            "or external side effect was performed unless separately authorized"
        ),
        "required_followups": [] if approved else ["address review findings"],
    }


def _completed_review(conn, source_task_id: str, *, approved: bool) -> str:
    review_id = kb.create_task(
        conn,
        title="review implementation packet",
        assignee="reviewer",
        priority=20,
    )
    metadata = _review_packet_metadata(
        source_task_id,
        approved=approved,
        findings=[] if approved else ["missing regression test"],
    )
    assert kb.complete_task(
        conn,
        review_id,
        summary="review packet complete",
        metadata=metadata,
    )
    return review_id


def _blocked_task(
    conn,
    *,
    title: str,
    assignee: str = "assistant",
    reason: str | None = None,
    last_failure_error: str | None = None,
    consecutive_failures: int = 0,
) -> str:
    task_id = kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        priority=5,
    )
    kb.claim_task(conn, task_id, claimer="test:worker")
    assert kb.block_task(conn, task_id, reason=reason)
    if last_failure_error is not None or consecutive_failures:
        conn.execute(
            """
            UPDATE tasks
               SET last_failure_error = ?,
                   consecutive_failures = ?
             WHERE id = ?
            """,
            (last_failure_error, consecutive_failures, task_id),
        )
    return task_id


def _timeout_gave_up_source(
    conn,
    *,
    trigger_outcome: str = "timed_out",
    block_class: str = "timeout_gave_up",
    last_failure_error: str = "iteration budget exhausted after 1800s",
    consecutive_failures: int = 3,
) -> str:
    task_id = kb.create_task(
        conn,
        title="implement broad timeout-prone slice",
        body="Original broad task body.",
        assignee="backend-engineer",
        priority=7,
    )
    kb.claim_task(conn, task_id, claimer="test:worker")
    assert kb.block_task(
        conn,
        task_id,
        reason=f"{trigger_outcome}: needs scoped reslice",
        block_class=block_class,
        block_metadata={
            "source": "worker",
            "trigger_outcome": trigger_outcome,
            "evidence": [f"{trigger_outcome}:run"],
            "reslice": {
                "title": "Implement the narrow timeout-safe follow-up",
                "scope": "Only extract the deterministic parser helper and its unit test.",
                "acceptance": [
                    "parser helper is covered by a focused unit test",
                    "source card remains blocked for review after child work",
                ],
            },
        },
    )
    conn.execute(
        """
        UPDATE tasks
           SET last_failure_error = ?,
               consecutive_failures = ?
         WHERE id = ?
        """,
        (last_failure_error, consecutive_failures, task_id),
    )
    return task_id


def _superseded_source(
    conn,
    *,
    block_class: str = "superseded_duplicate",
    canonical_task_id: str | None = None,
    unique_acceptance: list[str] | None = None,
) -> tuple[str, str]:
    canonical_id = canonical_task_id or kb.create_task(
        conn,
        title="canonical implementation card",
        assignee="backend-engineer",
        initial_status="blocked",
    )
    duplicate_id = kb.create_task(
        conn,
        title="duplicate implementation card",
        assignee="backend-engineer",
        priority=3,
    )
    kb.claim_task(conn, duplicate_id, claimer="test:worker")
    assert kb.block_task(
        conn,
        duplicate_id,
        reason="superseded by canonical implementation card",
        block_class=block_class,
        block_metadata={
            "source": "operator",
            "canonical_task_id": canonical_id,
            "canonical_evidence": ["comment:42", "plan:canonical"],
            "duplicate_evidence": ["same acceptance criteria and same target module"],
            "unique_acceptance": [] if unique_acceptance is None else unique_acceptance,
        },
    )
    return duplicate_id, canonical_id


def test_drain_dry_run_includes_board_classification_summary(kanban_home: Path) -> None:
    with kb.connect() as conn:
        review_source_id = _review_required_source(conn)
        _completed_review(conn, review_source_id, approved=True)
        unknown_id = _blocked_task(conn, title="blocked card without current reason")
        parent_id = kb.create_task(
            conn,
            title="parent blocker",
            assignee="product-manager",
            initial_status="blocked",
        )
        child_id = kb.create_task(
            conn,
            title="child waiting on parent",
            assignee="backend-engineer",
            parents=[parent_id],
        )

        report = kd.drain_review_packets(conn, apply=False)

        classification = report["classification"]
        summary = classification["summary"]
        assert summary["by_class"]["review_required"] == 1
        assert summary["by_class"]["classification_debt"] == 2
        assert summary["by_class"]["parent_gated"] == 1
        assert summary["parent_gated_todo"] == 1
        assert summary["review_required"] == 1
        assert summary["classification_debt"] == 2
        tasks = {task["task_id"]: task for task in classification["tasks"]}
        assert tasks[review_source_id]["owner"] == "pr-reviewer"
        assert tasks[review_source_id]["eligible_action"] == "route_to_review"
        assert tasks[unknown_id]["owner"] == "factory-orchestrator"
        assert tasks[child_id]["inferred_class"] == "parent_gated"
        assert tasks[child_id]["nonterminal_parents"] == [parent_id]
        queues = {queue["owner"]: queue for queue in classification["profile_queues"]}
        assert queues["pr-reviewer"]["classes"]["review_required"] == 1
        assert queues["factory-orchestrator"]["classes"]["parent_gated"] == 1


def test_drain_classifies_stale_runtime_failure_for_devops(kanban_home: Path) -> None:
    with kb.connect() as conn:
        runtime_id = _blocked_task(
            conn,
            title="worker vanished during implementation",
            assignee="backend-engineer",
            last_failure_error="pid 12345 not alive",
            consecutive_failures=2,
        )

        classification = kd.classify_board_for_drain(conn)

        tasks = {task["task_id"]: task for task in classification["tasks"]}
        assert tasks[runtime_id]["inferred_class"] == "runtime_infra"
        assert tasks[runtime_id]["owner"] == "devops-engineer"
        assert "last_failure_error" in tasks[runtime_id]["evidence"]
        assert "consecutive_failures:2" in tasks[runtime_id]["evidence"]


def test_structured_block_class_drives_drain_classification(kanban_home: Path) -> None:
    with kb.connect() as conn:
        review_id = kb.create_task(
            conn,
            title="implemented card awaiting review",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            review_id,
            reason="implementation has review packet handoff",
            block_class="review_required",
            block_metadata={
                "source": "worker",
                "owner": "pr-reviewer",
                "evidence": ["comment:1"],
            },
        )

        hold_id = kb.create_task(
            conn,
            title="private repo access needed",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            hold_id,
            reason="needs repo access",
            block_class="credential_hold",
            block_metadata={
                "source": "worker",
                "owner": "human/operator",
                "evidence": ["run:2"],
            },
        )

        classification = kd.classify_board_for_drain(conn)

        tasks = {task["task_id"]: task for task in classification["tasks"]}
        assert tasks[review_id]["inferred_class"] == "review_required"
        assert tasks[review_id]["owner"] == "pr-reviewer"
        assert tasks[review_id]["eligible_action"] == "route_to_review"
        assert tasks[hold_id]["inferred_class"] == "credential_hold"
        assert tasks[hold_id]["owner"] == "human/operator"
        assert tasks[hold_id]["eligible_action"] == "report_only"


def test_cli_block_persists_block_class_and_metadata(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="needs explicit approval",
            assignee="product-manager",
        )

    raw = kc.run_slash(
        "block "
        f"{task_id} "
        "needs approval "
        "--block-class human_hold "
        "--block-metadata '{\"owner\":\"CEO\",\"evidence\":[\"comment:7\"]}'"
    )
    assert raw.startswith(f"Blocked {task_id}")

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.block_class == "human_hold"
        assert task.block_metadata is not None
        assert task.block_metadata["owner"] == "CEO"
        assert task.block_metadata["source"] == "operator"
        assert task.block_metadata["reason"] == "needs approval"
        latest = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(latest["payload"])
        assert payload["block_class"] == "human_hold"
        assert payload["block_metadata"]["evidence"] == ["comment:7"]


def test_drain_review_required_dry_run_reports_without_mutation(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = kb.create_task(
            conn,
            title="implemented card awaiting review",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            source_id,
            reason="implementation has review handoff",
            block_class="review_required",
            block_metadata={"source": "worker", "evidence": ["comment:1"]},
        )

        report = kd.drain_review_required(conn, apply=False)

        assert report["dry_run"] is True
        assert report["summary"]["planned"] == 1
        action = report["actions"][0]
        assert action["action"] == "route_to_review"
        assert action["source_task_id"] == source_id
        assert action["status"] == "planned"
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []


def test_drain_review_required_apply_routes_to_review_idempotently(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = kb.create_task(
            conn,
            title="implemented card awaiting review",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            source_id,
            reason="implementation has review handoff",
            block_class="review_required",
            block_metadata={"source": "worker", "evidence": ["comment:1"]},
        )

        first = kd.drain_review_required(conn, apply=True)
        second = kd.drain_review_required(conn, apply=True)

        assert first["summary"]["applied"] == 1
        assert second["summary"]["already_applied"] == 1
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "review"
        comments = [c for c in kb.list_comments(conn, source_id) if "route_to_review" in c.body]
        assert len(comments) == 1
        events = [e for e in kb.list_events(conn, source_id) if e.kind == "drain_routed_review"]
        assert len(events) == 1
        assert events[0].payload is not None
        assert events[0].payload["action"] == "route_to_review"


def test_drain_review_required_refuses_explicit_holds(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = kb.create_task(
            conn,
            title="needs credential before review",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            source_id,
            reason="needs repo access",
            block_class="credential_hold",
            block_metadata={"source": "worker", "evidence": ["run:7"]},
        )

        report = kd.drain_review_required(conn, apply=True)

        assert report["summary"]["refused"] == 1
        action = report["actions"][0]
        assert action["source_task_id"] == source_id
        assert action["status"] == "refused"
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []


def test_drain_review_required_supports_legacy_prefix_bridge(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)

        report = kd.drain_review_required(conn, apply=True)

        assert report["summary"]["applied"] == 1
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "review"


def test_kanban_drain_cli_review_required_json_apply(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source_id = kb.create_task(
            conn,
            title="implemented card awaiting review",
            assignee="backend-engineer",
        )
        assert kb.block_task(
            conn,
            source_id,
            reason="implementation has review handoff",
            block_class="review_required",
        )

    raw = kc.run_slash("drain --class review_required --apply --json")
    report = json.loads(raw)

    assert report["dry_run"] is False
    assert report["class"] == "review_required"
    assert report["summary"]["applied"] == 1
    assert report["actions"][0]["source_task_id"] == source_id
    with kb.connect() as conn:
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "review"


def test_drain_timeout_gave_up_apply_reslices_timeout_without_clearing_source_failures(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _timeout_gave_up_source(conn, trigger_outcome="timed_out")

        report = kd.drain_timeout_gave_up(conn, apply=True)

        assert report["summary"]["applied"] == 1
        action = report["actions"][0]
        assert action["action"] == "reslice_timeout_gave_up"
        child_id = action["child_task_id"]
        source = kb.get_task(conn, source_id)
        child = kb.get_task(conn, child_id)
        assert source is not None
        assert child is not None
        assert source.status == "blocked"
        assert source.consecutive_failures == 3
        assert source.last_failure_error == "iteration budget exhausted after 1800s"
        assert child.status == "todo"
        assert child.assignee == "backend-engineer"
        assert child.priority == 7
        assert kb.parent_ids(conn, child_id) == [source_id]
        assert child_id in kb.child_ids(conn, source_id)
        assert "Only extract the deterministic parser helper" in (child.body or "")
        assert len([c for c in kb.list_comments(conn, source_id) if "reslice_timeout_gave_up" in c.body]) == 1


def test_drain_timeout_gave_up_apply_reslices_gave_up_source(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _timeout_gave_up_source(
            conn,
            trigger_outcome="gave_up",
            last_failure_error="gave_up after repeated timeout",
            consecutive_failures=4,
        )

        report = kd.drain_timeout_gave_up(conn, apply=True)

        assert report["summary"]["applied"] == 1
        child_id = report["actions"][0]["child_task_id"]
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.idempotency_key == f"kanban-drain:timeout-gave-up-reslice:{source_id}"
        assert kb.parent_ids(conn, child_id) == [source_id]
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert source.consecutive_failures == 4


def test_drain_timeout_gave_up_refuses_credential_hold_auth_blocker(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _timeout_gave_up_source(
            conn,
            block_class="credential_hold",
            last_failure_error="bad credentials while cloning private repo",
        )

        report = kd.drain_timeout_gave_up(conn, apply=True)

        assert report["summary"]["refused"] == 1
        action = report["actions"][0]
        assert action["source_task_id"] == source_id
        assert action["status"] == "refused"
        assert "explicit human/credential/prod-risk block_class" in action["reason"]
        assert kb.child_ids(conn, source_id) == []
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert source.consecutive_failures == 3


def test_drain_timeout_gave_up_apply_is_idempotent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _timeout_gave_up_source(conn)

        first = kd.drain_timeout_gave_up(conn, apply=True)
        second = kd.drain_timeout_gave_up(conn, apply=True)

        assert first["summary"]["applied"] == 1
        assert second["summary"]["already_applied"] == 1
        children = kb.child_ids(conn, source_id)
        assert len(children) == 1
        assert second["actions"][0]["child_task_id"] == children[0]
        assert len([c for c in kb.list_comments(conn, source_id) if "reslice_timeout_gave_up" in c.body]) == 1


def test_drain_superseded_apply_archives_with_canonical_evidence(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        duplicate_id, canonical_id = _superseded_source(conn)

        report = kd.drain_superseded_duplicates(conn, apply=True)

        assert report["summary"]["applied"] == 1
        action = report["actions"][0]
        assert action["action"] == "archive_superseded_duplicate"
        assert action["source_task_id"] == duplicate_id
        assert action["canonical_task_id"] == canonical_id
        duplicate = kb.get_task(conn, duplicate_id)
        assert duplicate is not None
        assert duplicate.status == "archived"
        comments = [
            c for c in kb.list_comments(conn, duplicate_id)
            if "archive_superseded_duplicate" in c.body
        ]
        assert len(comments) == 1
        events = [e for e in kb.list_events(conn, duplicate_id) if e.kind == "archived"]
        assert len(events) == 1
        audit_events = [
            e for e in kb.list_events(conn, duplicate_id)
            if e.kind == "drain_archived_superseded_duplicate"
        ]
        assert len(audit_events) == 1
        assert audit_events[0].payload is not None
        assert audit_events[0].payload["canonical_task_id"] == canonical_id


def test_drain_superseded_refuses_missing_canonical_evidence(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        duplicate_id, _canonical_id = _superseded_source(conn)
        conn.execute(
            "UPDATE tasks SET block_metadata = ? WHERE id = ?",
            (json.dumps({"source": "operator", "canonical_task_id": _canonical_id}), duplicate_id),
        )

        report = kd.drain_superseded_duplicates(conn, apply=True)

        assert report["summary"]["refused"] == 1
        action = report["actions"][0]
        assert action["source_task_id"] == duplicate_id
        assert "canonical evidence missing" in action["reason"]
        duplicate = kb.get_task(conn, duplicate_id)
        assert duplicate is not None
        assert duplicate.status == "blocked"
        assert kb.list_comments(conn, duplicate_id) == []


def test_drain_superseded_refuses_unique_acceptance(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        duplicate_id, _canonical_id = _superseded_source(
            conn,
            unique_acceptance=["also migrate the old webhook endpoint"],
        )

        report = kd.drain_superseded_duplicates(conn, apply=True)

        assert report["summary"]["refused"] == 1
        assert "unique acceptance criteria" in report["actions"][0]["reason"]
        duplicate = kb.get_task(conn, duplicate_id)
        assert duplicate is not None
        assert duplicate.status == "blocked"


def test_drain_superseded_apply_is_idempotent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        duplicate_id, _canonical_id = _superseded_source(conn)

        first = kd.drain_superseded_duplicates(conn, apply=True)
        second = kd.drain_superseded_duplicates(conn, apply=True)

        assert first["summary"]["applied"] == 1
        assert second["summary"]["already_applied"] == 1
        assert len([
            c for c in kb.list_comments(conn, duplicate_id)
            if "archive_superseded_duplicate" in c.body
        ]) == 1


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
            metadata=_review_packet_metadata(
                source_id,
                approved=False,
                findings=["missing regression test"],
            ),
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


def test_drain_refuses_malformed_review_packet_without_mutating_source(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source_id = _review_required_source(conn)
        review_id = kb.create_task(conn, title="review packet", assignee="reviewer")
        assert kb.complete_task(
            conn,
            review_id,
            summary="malformed review packet",
            metadata={
                "source_task_id": source_id,
                "verdict": "PASS",
                "evidence": {"diff_path": "workspaces/source-card.patch"},
                "findings": [],
            },
        )

        report = kd.drain_review_packets(conn, apply=True)

        assert report["summary"]["refused"] == 1
        action = report["actions"][0]
        assert action["action"] == "consume_review_packet"
        assert action["source_task_id"] == source_id
        assert action["review_task_id"] == review_id
        assert action["status"] == "refused"
        assert "reviewer identity missing" in action["reason"]
        assert "tests or checks inspected missing" in action["reason"]
        assert "authority-boundary statement missing" in action["reason"]
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []
        assert [run.outcome for run in kb.list_runs(conn, source_id)] == ["blocked"]


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
    assert report["classification"]["summary"]["review_required"] == 1
    with kb.connect() as conn:
        source = kb.get_task(conn, source_id)
        assert source is not None
        assert source.status == "blocked"
        assert kb.list_comments(conn, source_id) == []
