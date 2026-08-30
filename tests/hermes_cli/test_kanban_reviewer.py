from pathlib import Path
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_reviewer import submit_reviewer_result


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def valid(verdict="CHANGES_REQUESTED"):
    payload = {"schema_version": 1, "verdict": verdict, "summary": "Concrete review"}
    payload["findings"] = [] if verdict == "APPROVED" else [{
        "finding_id": "F1", "severity": "high", "affected_files_or_areas": ["src/a.py"],
        "required_changes": ["Add the missing guard"], "verification_evidence": ["Run the focused test"],
    }]
    if verdict == "BLOCKED":
        payload["findings"] = []
        payload["ambiguity_or_blocker_reason"] = "Need product decision"
    return payload


def test_approved_persists_and_completes(conn):
    tid = kb.create_task(conn, title="approve", assignee="builder")
    result = submit_reviewer_result(conn, tid, valid("APPROVED"))
    assert result["accepted"] and kb.get_task(conn, tid).status == "done"
    assert any(e.kind == "reviewer_result_approved" for e in kb.list_events(conn, tid))


def test_valid_correction_is_idempotent_and_linked(conn):
    tid = kb.create_task(conn, title="fix", assignee="builder")
    first = submit_reviewer_result(conn, tid, valid())
    second = submit_reviewer_result(conn, tid, valid())
    assert first["correction_task_id"] == second["correction_task_id"]
    child = first["correction_task_id"]
    assert kb.parent_ids(conn, child) == [tid]
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.get_task(conn, child).status == "ready"
    handoff = json.loads(kb.get_task(conn, child).body)
    assert handoff["findings"][0]["required_changes"] == ["Add the missing guard"]
    assert handoff["verification_targets"] == ["Run the focused test"]
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "reviewer_correction_created"]) == 1


def test_replay_does_not_consume_cycle_before_distinct_correction(conn):
    tid = kb.create_task(conn, title="replay", assignee="builder")
    first = submit_reviewer_result(conn, tid, valid())
    replays = [submit_reviewer_result(conn, tid, valid()) for _ in range(3)]
    distinct = submit_reviewer_result(conn, tid, {**valid(), "summary": "A distinct correction"})

    assert all(replay["correction_task_id"] == first["correction_task_id"] for replay in replays)
    assert all(replay["cycle"] == 1 for replay in replays)
    assert distinct["correction_task_id"] != first["correction_task_id"]
    assert distinct["cycle"] == 2
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "reviewer_correction_created"]) == 2


def test_invalid_result_only_audits_and_does_not_mutate_graph(conn):
    tid = kb.create_task(conn, title="safe", assignee="builder")
    before = (kb.get_task(conn, tid).status, kb.child_ids(conn, tid))
    result = submit_reviewer_result(conn, tid, {"verdict": "CHANGES_REQUESTED", "summary": "vague", "findings": []})
    after = (kb.get_task(conn, tid).status, kb.child_ids(conn, tid))
    assert not result["accepted"] and before == after
    assert any(e.kind == "reviewer_result_rejected" for e in kb.list_events(conn, tid))


@pytest.mark.parametrize(
    "payload, message",
    [
        ({**valid(), "ambiguity_or_blocker_reason": "conflicting scope"}, "cannot contain"),
        ({**valid("BLOCKED"), "findings": [{
            "finding_id": "F1", "severity": "high", "affected_files_or_areas": ["x"],
            "required_changes": ["y"], "verification_evidence": ["z"],
        }]}, "cannot contain"),
    ],
)
def test_contradictory_result_is_rejected_without_graph_mutation(conn, payload, message):
    tid = kb.create_task(conn, title="contradiction", assignee="builder")
    result = submit_reviewer_result(conn, tid, payload)
    assert result["accepted"] is False
    assert message in result["reason"]
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.child_ids(conn, tid) == []
    assert any(e.kind == "reviewer_result_rejected" for e in kb.list_events(conn, tid))


def test_blocked_routes_to_native_block(conn):
    tid = kb.create_task(conn, title="blocked", assignee="builder")
    result = submit_reviewer_result(conn, tid, valid("BLOCKED"))
    assert result["accepted"] and kb.get_task(conn, tid).status == "blocked"


def test_reviewer_payload_is_redacted_at_every_durable_boundary(conn):
    tid = kb.create_task(conn, title="redact", assignee="builder")
    secret = "api_key=super-secret-value"
    payload = {**valid("BLOCKED"), "ambiguity_or_blocker_reason": secret}
    result = submit_reviewer_result(conn, tid, payload, reviewer=secret)
    assert result["accepted"]
    assert secret not in json.dumps([e.payload for e in kb.list_events(conn, tid)])
    assert secret not in (kb.get_task(conn, tid).result or "")


def test_three_corrections_escalate_without_fourth_child(conn):
    tid = kb.create_task(conn, title="cap", assignee="builder")
    ids = [submit_reviewer_result(conn, tid, {**valid(), "summary": f"pass {i}"})["correction_task_id"] for i in range(3)]
    fourth = submit_reviewer_result(conn, tid, {**valid(), "summary": "pass 4"})
    assert all(ids) and fourth["escalated"] and fourth["correction_task_id"] is None
    assert kb.get_task(conn, tid).status == "blocked"


def test_injected_audit_failure_recovers_idempotently(conn, monkeypatch):
    tid = kb.create_task(conn, title="retry", assignee="builder")
    original = __import__("hermes_cli.kanban_reviewer", fromlist=["_audit"])._audit
    calls = {"n": 0}

    def fail_once(*args, **kwargs):
        if calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("injected audit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr("hermes_cli.kanban_reviewer._audit", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        submit_reviewer_result(conn, tid, valid())
    assert kb.child_ids(conn, tid) == []
    retry = submit_reviewer_result(conn, tid, valid())
    assert retry["correction_task_id"] is not None
    assert len(kb.child_ids(conn, tid)) == 1
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "reviewer_correction_created"]) == 1
