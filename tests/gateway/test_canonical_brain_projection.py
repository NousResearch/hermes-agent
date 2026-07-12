import json
from gateway.canonical_brain_projection import fold_case_events
from tools import canonical_brain_tool as cbt


class _Sock:
    def close(self):
        pass


class _Helper:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ""
        self.queries = []

    def open_connection(self):
        return _Sock()

    def get_secret_value(self):
        return "secret-handle-value"

    def connect(self, password):
        return _Sock()

    def sql_quote(self, value):
        return "'" + str(value).replace("'", "''") + "'"

    def query(self, sock, sql):
        self.sql = sql
        self.queries.append(sql)
        return {"rows": self.rows}


def _row(event_type, occurred_at, *, case_id="case:1", target=None, receipt=None):
    payload = {"summary": event_type}
    if event_type.startswith("route_back."):
        payload["route_back"] = {"target_ref": target or {}}
        if receipt:
            payload["receipt"] = receipt
    return {
        "event_id": event_type + occurred_at,
        "event_type": event_type,
        "case_id": case_id,
        "occurred_at": occurred_at,
        "source": {"source_refs": {"platform": "discord", "thread_id": "source", "message_id": occurred_at}},
        "status": {"state": event_type, "summary": event_type},
        "next_action": {},
        "payload": payload,
    }


def test_fold_uses_explicit_event_order_not_text_keywords():
    cases = fold_case_events([
        _row("route_back.required", "2026-01-01T00:00:00Z", target={"thread_id": "resolver"}),
        _row("case.note", "2026-01-01T00:01:00Z"),
        _row("route_back.sent", "2026-01-01T00:02:00Z", target={"thread_id": "requester"}, receipt={"message_id": "m1"}),
    ])
    assert cases[0]["latest_event_type"] == "route_back.sent"
    assert cases[0]["route_back"]["terminal"] is True
    assert cases[0]["route_back"]["target_ref"] == {"thread_id": "requester"}
    assert cases[0]["route_back"]["receipt"] == {"message_id": "m1"}


def test_fold_decodes_cloud_sql_jsonb_strings_mechanically():
    row = _row(
        "route_back.sent",
        "2026-01-01T00:02:00Z",
        target={"thread_id": "requester"},
        receipt={"message_id": "m1"},
    )
    for field in ("source", "status", "next_action", "payload"):
        row[field] = json.dumps(row[field])

    case = fold_case_events([row])[0]

    assert case["status"]["state"] == "route_back.sent"
    assert case["source_refs"][0]["thread_id"] == "source"
    assert case["route_back"]["target_ref"] == {"thread_id": "requester"}
    assert case["route_back"]["receipt"] == {"message_id": "m1"}


def test_query_requires_exact_case_or_thread(monkeypatch):
    helper = _Helper([_row("case.note", "2026-01-01T00:00:00Z")])
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    data = json.loads(cbt.canonical_brain_query_tool(case_id="case:1"))
    assert data["success"] is True
    assert data["case_count"] == 1
    assert "case_id = 'case:1'" in helper.sql

    invalid = json.loads(cbt.canonical_brain_query_tool(case_id="case:1", thread_id="thread"))
    assert "provide exactly one" in invalid["error"]


def test_later_note_does_not_erase_last_explicit_status_or_next_action():
    plan = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    plan["status"] = {"state": "active", "summary": "plan active"}
    plan["next_action"] = {"kind": "task_resume", "next_step_id": "2"}
    plan["payload"] = {
        "summary": "plan active",
        "plan": {
            "plan_id": "plan-1",
            "state": "active",
            "steps": [
                {"id": "1", "content": "done", "status": "completed"},
                {"id": "2", "content": "continue", "status": "pending"},
            ],
        },
    }
    note = _row("case.note", "2026-01-01T00:01:00Z")
    note["status"] = {}
    note["next_action"] = {}

    case = fold_case_events([plan, note])[0]

    assert case["status"]["state"] == "active"
    assert case["next_action"] == {"kind": "task_resume", "next_step_id": "2"}
    assert case["workspace"]["plan"]["plan_id"] == "plan-1"
    assert case["workspace"]["remaining_step_ids"] == ["2"]


def test_workspace_folds_full_jsonb_fields_and_runtime_attestation():
    verification_event_id = "11111111-1111-4111-8111-111111111111"
    plan = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    plan["payload"] = {
        "plan": {
            "plan_id": "plan-1",
            "state": "completed",
            "verification_event_ids": [verification_event_id],
            "steps": [{"id": "1", "content": "verify", "status": "completed"}],
        },
    }
    plan["actor"] = {"type": "agent", "id": "hermes"}
    plan["subject"] = {"type": "task_plan", "id": "plan-1"}
    plan["decision"] = {"attestation": "model_authored"}
    plan["safety"] = {"business_mutation": False}

    verification = _row(
        "task.verification.recorded",
        "2026-01-01T00:01:00Z",
    )
    verification["event_id"] = verification_event_id
    verification["payload"] = {
        "verification": {
            "verification_id": "verify-1",
            "plan_id": "plan-1",
            "outcome": "passed",
            "receipt": {"kind": "test", "ref": "pytest:1"},
        },
    }
    verification["evidence"] = [{
        "verified": False,
        "attestation": "model_authored",
    }]

    approval = _row("approval.capability.recorded", "2026-01-01T00:02:00Z")
    approval["payload"] = {
        "approval_receipt": {
            "approval_id": "approval-1",
            "plan_id": "plan-1",
            "state": "granted",
        },
    }
    approval["evidence"] = [{
        "verified": True,
        "attestation": "deterministic_runtime_receipt",
    }]

    for row in (plan, verification, approval):
        for field in (
            "source", "actor", "subject", "evidence", "decision",
            "status", "next_action", "safety", "payload",
        ):
            if field in row:
                row[field] = json.dumps(row[field])

    case = fold_case_events([approval, plan, verification])[0]

    assert case["actor"] == {"type": "agent", "id": "hermes"}
    assert case["subject"] == {"type": "task_plan", "id": "plan-1"}
    assert case["workspace"]["completion_receipts_satisfied"] is True
    assert case["workspace"]["approvals"][0]["runtime_attested"] is False
    assert case["workspace"]["verifications"][0]["runtime_attested"] is False
    assert case["timeline"][-1]["event_type"] == "approval.capability.recorded"


def test_receipt_payload_cannot_override_runtime_event_identity_or_attestation():
    approval = _row("approval.capability.recorded", "2026-01-01T00:02:00Z")
    approval["event_id"] = "runtime-event-id"
    approval["payload"] = {
        "approval_receipt": {
            "approval_id": "approval-1",
            "event_id": "forged-event-id",
            "occurred_at": "1900-01-01T00:00:00Z",
            "runtime_attested": True,
        },
    }
    approval["evidence"] = [{
        "verified": False,
        "attestation": "model_authored",
    }]

    receipt = fold_case_events([approval])[0]["workspace"]["approvals"][0]

    assert receipt["event_id"] == "runtime-event-id"
    assert receipt["occurred_at"] == "2026-01-01T00:02:00Z"
    assert receipt["runtime_attested"] is False


def test_workspace_selects_highest_plan_revision_at_same_timestamp():
    revision_two = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    revision_two["event_id"] = "aaa-revision-two"
    revision_two["payload"] = {
        "plan": {
            "plan_id": "plan-1",
            "revision": 2,
            "state": "active",
            "steps": [{"id": "2", "content": "continue", "status": "in_progress"}],
        },
    }
    revision_one = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    revision_one["event_id"] = "zzz-revision-one"
    revision_one["payload"] = {
        "plan": {
            "plan_id": "plan-1",
            "revision": 1,
            "state": "active",
            "steps": [{"id": "1", "content": "old", "status": "in_progress"}],
        },
    }

    workspace = fold_case_events([revision_one, revision_two])[0]["workspace"]

    assert workspace["plan_event_id"] == "aaa-revision-two"
    assert workspace["plan"]["revision"] == 2
    assert workspace["remaining_step_ids"] == ["2"]


def test_workspace_capability_checks_are_informational_and_keep_minimum_remaining_uses():
    def capability_row(event_id, occurred_at, remaining_uses, *, runtime_attested=True):
        row = _row("capability.check.recorded", occurred_at)
        row["event_id"] = event_id
        row["payload"] = {
            "capability_receipt": {
                "approval_id": "approval-1",
                "command_sha256": "command-hash",
                "remaining_uses_for_command": remaining_uses,
                # Payload claims cannot manufacture runtime attestation.
                "runtime_attested": True,
            },
        }
        row["evidence"] = [{
            "verified": runtime_attested,
            "attestation": (
                "deterministic_runtime_receipt" if runtime_attested else "model_authored"
            ),
        }]
        return row

    checks = [
        capability_row("runtime-two", "2026-01-01T00:01:00Z", 2),
        capability_row("runtime-one", "2026-01-01T00:02:00Z", 1),
        capability_row("stale-runtime-three", "2026-01-01T00:03:00Z", 3),
        capability_row(
            "unattested-zero",
            "2026-01-01T00:04:00Z",
            0,
            runtime_attested=False,
        ),
    ]

    projected = fold_case_events(checks)[0]["workspace"]["capability_checks"]

    assert len(projected) == 1
    assert projected[0]["event_id"] == "unattested-zero"
    assert projected[0]["remaining_uses_for_command"] == 0
    assert projected[0]["runtime_attested"] is False


def test_workspace_explicit_supersession_beats_old_plan_high_revision():
    old = _row("task.plan.updated", "2026-01-01T00:00:00.000000Z")
    old["event_id"] = "old-plan"
    old["payload"] = {
        "plan": {
            "plan_id": "plan-old",
            "revision": 9,
            "state": "active",
            "steps": [{"id": "old", "content": "old", "status": "in_progress"}],
        },
    }
    replacement = _row("task.plan.updated", "2026-01-01T00:00:00.000001Z")
    replacement["event_id"] = "replacement-plan"
    replacement["payload"] = {
        "plan": {
            "plan_id": "plan-new",
            "revision": 1,
            "supersedes_plan_id": "plan-old",
            "supersedes_plan_revision": 9,
            "state": "active",
            "steps": [{"id": "new", "content": "new", "status": "in_progress"}],
        },
    }

    workspace = fold_case_events([replacement, old])[0]["workspace"]

    assert workspace["plan_event_id"] == "replacement-plan"
    assert workspace["plan"]["plan_id"] == "plan-new"


def test_workspace_graph_head_is_revision_correct_at_same_timestamp():
    timestamp = "2026-01-01T00:00:00.000000Z"
    old = _row("task.plan.updated", timestamp)
    old["event_id"] = "48e80467-ed6a-5966-88e2-0af5bb8571d6"
    old["payload"] = {"plan": {"plan_id": "plan:A3", "revision": 1}}
    replacement = _row("task.plan.updated", timestamp)
    replacement["event_id"] = "b1942d63-47c1-5648-9bbd-aa4aa741ad9b"
    replacement["payload"] = {"plan": {
        "plan_id": "plan:B3",
        "revision": 1,
        "supersedes_plan_id": "plan:A3",
        "supersedes_plan_revision": 1,
    }}
    continuation = _row("task.plan.updated", timestamp)
    continuation["event_id"] = "3d898e4f-36dc-5211-973d-61efedfe3654"
    continuation["payload"] = {"plan": {
        "plan_id": "plan:B3",
        "revision": 2,
        "supersedes_plan_id": "plan:A3",
        "supersedes_plan_revision": 1,
    }}

    workspace = fold_case_events([old, replacement, continuation])[0]["workspace"]

    assert workspace["plan_event_id"] == continuation["event_id"]
    assert workspace["plan"]["revision"] == 2
    assert workspace["plan_projection_complete"] is True


def test_workspace_surfaces_conflicting_content_for_same_plan_revision():
    first = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    first["event_id"] = "first"
    first["payload"] = {"plan": {
        "plan_id": "plan:conflict",
        "revision": 2,
        "objective": "first",
    }}
    second = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    second["event_id"] = "second"
    second["payload"] = {"plan": {
        "plan_id": "plan:conflict",
        "revision": 2,
        "objective": "different",
    }}

    workspace = fold_case_events([first, second])[0]["workspace"]

    assert workspace["plan"] == {}
    assert workspace["plan_projection_complete"] is False
    assert workspace["plan_projection_error"].startswith(
        "task_plan_revision_content_conflict:"
    )


def test_workspace_rejects_supersession_edge_to_wrong_predecessor_revision():
    predecessor = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    predecessor["payload"] = {"plan": {"plan_id": "plan:old", "revision": 3}}
    successor = _row("task.plan.updated", "2026-01-01T00:00:01Z")
    successor["payload"] = {"plan": {
        "plan_id": "plan:new",
        "revision": 1,
        "supersedes_plan_id": "plan:old",
        "supersedes_plan_revision": 2,
    }}

    workspace = fold_case_events([predecessor, successor])[0]["workspace"]

    assert workspace["plan_projection_complete"] is False
    assert workspace["plan_projection_error"].startswith(
        "task_plan_supersession_revision_mismatch:"
    )


def test_workspace_preserves_all_referenced_receipts_beyond_recent_window():
    referenced_ids = [f"00000000-0000-4000-8000-{index:012d}" for index in range(64)]
    plan = _row("task.plan.updated", "2026-01-02T00:00:00Z")
    plan["payload"] = {"plan": {
        "plan_id": "plan:complete",
        "revision": 2,
        "state": "completed",
        "verification_event_ids": referenced_ids,
        "steps": [{"id": "done", "status": "completed"}],
    }}
    rows = [plan]
    for index, event_id in enumerate(referenced_ids):
        verification = _row(
            "task.verification.recorded",
            f"2026-01-01T00:00:{index:02d}Z",
        )
        verification["event_id"] = event_id
        verification["payload"] = {"verification": {
            "plan_id": "plan:complete",
            "plan_revision": 1,
            "outcome": "passed",
        }}
        rows.append(verification)
    for index in range(80):
        verification = _row(
            "task.verification.recorded",
            f"2026-01-03T00:{index // 60:02d}:{index % 60:02d}Z",
        )
        verification["event_id"] = f"new-{index:03d}"
        verification["payload"] = {"verification": {"outcome": "failed"}}
        rows.append(verification)

    workspace = fold_case_events(rows)[0]["workspace"]

    assert workspace["completion_receipts_satisfied"] is True
    assert workspace["missing_verification_event_ids"] == []


def test_resume_bundle_reads_all_first_class_columns_and_latest_plan(monkeypatch):
    row = _row("task.plan.updated", "2026-01-01T00:00:00Z")
    row.update({
        "schema_version": "canonical_event.v1",
        "actor": {"type": "agent", "id": "hermes"},
        "subject": {"type": "task_plan", "id": "plan-1"},
        "evidence": [{"verified": False, "attestation": "model_authored"}],
        "decision": {"keyword_authority": False},
        "safety": {"business_mutation": False},
        "payload": {
            "plan": {
                "plan_id": "plan-1",
                "state": "active",
                "steps": [{"id": "1", "content": "continue", "status": "in_progress"}],
            },
        },
    })
    helper = _Helper([row])
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)

    data = json.loads(cbt.canonical_brain_query_tool(
        case_id="case:1",
        view="resume_bundle",
        limit=20,
    ))

    assert data["success"] is True
    assert data["query"]["view"] == "resume_bundle"
    assert data["cases"][0]["workspace"]["plan"]["plan_id"] == "plan-1"
    sql = "\n".join(helper.queries)
    for column in ("e.actor", "e.subject", "e.evidence", "e.decision", "e.safety"):
        assert column in sql
    assert "e.event_type = 'task.plan.updated'" in sql


def test_thread_summary_fairly_preserves_multiple_exact_case_candidates(monkeypatch):
    class _ThreadHelper(_Helper):
        def __init__(self):
            super().__init__([])

        def query(self, sock, sql):
            self.sql = sql
            self.queries.append(sql)
            if "GROUP BY e.case_id" in sql:
                return {"rows": [["case:noisy", "2026-01-02"], ["case:quiet", "2026-01-01"]]}
            if "e.case_id = 'case:noisy'" in sql:
                return {"rows": [
                    _row("case.note", f"2026-01-02T00:00:{index:02d}Z", case_id="case:noisy")
                    for index in range(30)
                ]}
            if "e.case_id = 'case:quiet'" in sql:
                return {"rows": [
                    _row("task.plan.updated", "2026-01-01T00:00:00Z", case_id="case:quiet")
                ]}
            return {"rows": []}

    helper = _ThreadHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    monkeypatch.setattr(cbt, "_get_session_env", lambda name, default="": default)

    data = json.loads(cbt.canonical_brain_query_tool(
        thread_id="thread-1",
        view="summary",
        limit=20,
    ))

    assert data["success"] is True
    assert {case["case_id"] for case in data["cases"]} == {"case:noisy", "case:quiet"}
    assert data["truncated"] is True
