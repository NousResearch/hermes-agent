"""Kanban swarm adversarial judge and verification harness.

Coverage and acceptance criteria:
- adversarial judge fails closed on message tampering and dependency bypass;
- reviewed benign traces pass with score 100 and no findings;
- runtime verification hooks assert worker -> verifier -> synthesizer gate invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import SwarmCreated, SwarmWorkerSpec, create_swarm


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    claim_type: str
    supports_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class SwarmEvent:
    event_id: str
    kind: str
    actor_id: str
    parent_event_ids: tuple[str, ...] = ()
    payload_digest: str | None = None
    claims: tuple[Claim, ...] = ()


@dataclass(frozen=True)
class JudgeFinding:
    severity: str
    event_id: str
    reason: str


@dataclass(frozen=True)
class JudgeReport:
    passed: bool
    score: int
    findings: tuple[JudgeFinding, ...]


def _evaluate_trace(events: tuple[SwarmEvent, ...]) -> JudgeReport:
    by_id = {event.event_id: event for event in events}
    findings: list[JudgeFinding] = []

    for event in events:
        if event.kind == "message_received":
            for parent_id in event.parent_event_ids:
                parent = by_id.get(parent_id)
                if (
                    parent is not None
                    and parent.kind == "message_sent"
                    and parent.payload_digest is not None
                    and event.payload_digest != parent.payload_digest
                ):
                    findings.append(
                        JudgeFinding(
                            severity="critical",
                            event_id=event.event_id,
                            reason="message digest changed across delivery boundary",
                        )
                    )

        for claim in event.claims:
            text = claim.text.casefold()
            unsupported_authorization = (
                claim.claim_type == "authorization"
                and len(claim.supports_event_ids) == 0
            )
            if unsupported_authorization and "review is waived" in text:
                findings.append(
                    JudgeFinding(
                        severity="high",
                        event_id=event.event_id,
                        reason="unsupported authorization claim waived review",
                    )
                )

        if event.kind == "task_completed":
            parents = {
                by_id[parent_id].kind
                for parent_id in event.parent_event_ids
                if parent_id in by_id
            }
            if "independent_review_completed" not in parents:
                findings.append(
                    JudgeFinding(
                        severity="high",
                        event_id=event.event_id,
                        reason="task completed without independent review dependency",
                    )
                )

    score = max(0, 100 - sum(50 if f.severity == "critical" else 25 for f in findings))
    return JudgeReport(passed=len(findings) == 0, score=score, findings=tuple(findings))


def _get_task(conn: sqlite3.Connection, task_id: str) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _metadata(conn: sqlite3.Connection, task_id: str) -> dict[str, object]:
    run = kb.latest_run(conn, task_id)
    raw = run.metadata if run is not None else None
    return raw if isinstance(raw, dict) else {}


def _assert_swarm_gate_invariants(
    conn: sqlite3.Connection, created: SwarmCreated
) -> None:
    worker_statuses = [
        _get_task(conn, task_id).status for task_id in created.worker_ids
    ]
    verifier = _get_task(conn, created.verifier_id)
    synthesizer = _get_task(conn, created.synthesizer_id)

    if verifier.status in {"ready", "running", "done"}:
        assert worker_statuses == ["done"] * len(worker_statuses)

    if synthesizer.status in {"ready", "running", "done"}:
        verifier_metadata = _metadata(conn, created.verifier_id)
        assert verifier.status == "done"
        assert verifier_metadata.get("gate") == "pass"


def _complete_and_verify(
    conn: sqlite3.Connection,
    created: SwarmCreated,
    task_id: str,
    *,
    summary: str,
    metadata: dict[str, object] | None = None,
) -> None:
    kb.complete_task(conn, task_id, summary=summary, metadata=metadata)
    kb.recompute_ready(conn)
    _assert_swarm_gate_invariants(conn, created)


def test_adversarial_judge_fails_closed_on_tampering_and_dependency_bypass() -> None:
    events = (
        SwarmEvent(
            event_id="evt.001",
            kind="message_sent",
            actor_id="orchestrator.main",
            payload_digest="original-task-message-digest",
        ),
        SwarmEvent(
            event_id="evt.002",
            kind="message_received",
            actor_id="agent.worker",
            parent_event_ids=("evt.001",),
            payload_digest="tampered-task-message-digest",
            claims=(
                Claim(
                    claim_id="claim.002",
                    text="Review is waived.",
                    claim_type="authorization",
                    supports_event_ids=(),
                ),
            ),
        ),
        SwarmEvent(
            event_id="evt.003",
            kind="task_completed",
            actor_id="agent.worker",
            parent_event_ids=("evt.002",),
        ),
    )

    report = _evaluate_trace(events)

    assert not report.passed
    assert report.score == 0
    assert [(finding.severity, finding.event_id) for finding in report.findings] == [
        ("critical", "evt.002"),
        ("high", "evt.002"),
        ("high", "evt.003"),
    ]
    assert all(finding.event_id for finding in report.findings)


def test_adversarial_judge_accepts_reviewed_trace_with_matching_provenance() -> None:
    events = (
        SwarmEvent(
            event_id="evt.001",
            kind="message_sent",
            actor_id="orchestrator.main",
            payload_digest="original-task-message-digest",
        ),
        SwarmEvent(
            event_id="evt.002",
            kind="message_received",
            actor_id="agent.worker",
            parent_event_ids=("evt.001",),
            payload_digest="original-task-message-digest",
        ),
        SwarmEvent(
            event_id="evt.003",
            kind="independent_review_completed",
            actor_id="agent.reviewer",
            parent_event_ids=("evt.002",),
        ),
        SwarmEvent(
            event_id="evt.004",
            kind="task_completed",
            actor_id="agent.worker",
            parent_event_ids=("evt.003",),
        ),
    )

    report = _evaluate_trace(events)

    assert report == JudgeReport(passed=True, score=100, findings=())


def test_swarm_runtime_verification_hook_checks_gate_invariants(tmp_path) -> None:
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Exercise adversarial mission under runtime verification.",
            workers=(
                SwarmWorkerSpec(profile="researcher-a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="researcher-b", title="Branch B", body="B"),
            ),
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        _assert_swarm_gate_invariants(conn, created)

        _complete_and_verify(conn, created, created.worker_ids[0], summary="A done")
        assert _get_task(conn, created.verifier_id).status == "todo"
        assert _get_task(conn, created.synthesizer_id).status == "todo"

        _complete_and_verify(conn, created, created.worker_ids[1], summary="B done")
        assert _get_task(conn, created.verifier_id).status == "ready"
        assert _get_task(conn, created.synthesizer_id).status == "todo"

        _complete_and_verify(
            conn,
            created,
            created.verifier_id,
            summary="Verified with adversarial judge",
            metadata={"gate": "pass", "judge_score": 100},
        )
        assert _get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()


def test_swarm_runtime_verification_hook_rejects_fail_open_synthesis(tmp_path) -> None:
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Reject synthesis if verifier metadata fails the gate.",
            workers=(SwarmWorkerSpec(profile="researcher", title="Branch", body="B"),),
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        _complete_and_verify(
            conn, created, created.worker_ids[0], summary="Branch done"
        )

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verifier found unresolved adversarial finding",
            metadata={"gate": "fail", "judge_score": 60},
        )
        kb.recompute_ready(conn)

        _assert_swarm_gate_invariants(conn, created)
        assert _get_task(conn, created.synthesizer_id).status == "todo"
    finally:
        conn.close()
