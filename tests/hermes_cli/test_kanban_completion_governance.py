"""Provider-free tests for the transactional Kanban completion guard."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_completion_broker as broker
from hermes_cli import kanban_completion_guard as guard


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "task_id",
        "run_id",
        "agent",
        "prompt_version",
        "status",
        "summary",
        "deliverables",
        "evidence",
        "open_questions",
        "approval",
        "qa_gate",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "task_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "agent": {"type": "string", "minLength": 1},
        "prompt_version": {"type": "string", "minLength": 1},
        "status": {"const": "completed"},
        "summary": {"type": "string", "minLength": 1},
        "deliverables": {"type": "array", "minItems": 1},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["status", "ref"],
                "properties": {
                    "status": {"const": "verified"},
                    "ref": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
        },
        "open_questions": {"type": "array", "maxItems": 0},
        "approval": {"type": "object"},
        "qa_gate": {"type": "object"},
    },
}

APPROVAL = {
    "state": "approved",
    "required_actions": ["synthetic completion fixture"],
    "approval_ref": "human:test-approval",
}
QA_GATE = {
    "required": True,
    "status": "pass",
    "review_task_id": "qa-test-1",
}
QA_RESULT = {
    "schema_version": "1.0.0",
    "task_id": "qa-test-1",
    "run_id": "qa-result-run-test-1",
    "agent": "QA_Tester_Agent",
    "prompt_version": "1.0.0",
    "status": "completed",
    "summary": "Synthetic independent QA passed.",
    "deliverables": [{"path": "qa-report.json"}],
    "evidence": [{"status": "verified", "ref": "pytest:synthetic-qa"}],
    "open_questions": [],
    "approval": {
        "state": "not_required",
        "required_actions": [],
        "approval_ref": None,
    },
    "qa_gate": {
        "required": False,
        "status": "not_required",
        "review_task_id": None,
    },
}


def _db_path(conn) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return str(Path(row["file"]).resolve())


def _policy(conn, **overrides):
    value = {
        "schema_version": "1.0.0",
        "policy_id": "test-policy-v1",
        "board": "synthetic-governed-board",
        "database_path": _db_path(conn),
        "result_schema": RESULT_SCHEMA,
        "result_schema_sha256": guard.canonical_sha256(RESULT_SCHEMA),
        "allowed_profiles": ["profile-test"],
        "allowed_completion_sources": ["worker-tool", "broker", "test"],
        "require_deliverables": True,
        "worker_isolation": {
            "mode": "docker",
            "network": False,
            "toolsets": ["terminal"],
            "mount_hermes_resources": False,
            "broker_socket": "/run/hermes-kanban-broker/completion.sock",
        },
    }
    value.update(overrides)
    return value


def _activation(policy, **overrides):
    value = {
        "schema_version": "1.0.0",
        "enabled": True,
        "kill_switch": False,
        "policy_sha256": guard.canonical_sha256(policy),
    }
    value.update(overrides)
    return value


def _binding(task_id: str, **overrides):
    value = {
        "schema_version": "1.0.0",
        "native_task_id": task_id,
        "external_task_id": "task-envelope-test-1",
        "assigned_agent": "Backend_Developer_Agent",
        "runtime_profile": "profile-test",
        "prompt_version": "1.0.0",
        "task_envelope_sha256": "a" * 64,
        "workflow_type": "major_development",
        "approval": APPROVAL,
        "qa_gate": QA_GATE,
        "qa_result": QA_RESULT,
        "qa_result_sha256": guard.canonical_sha256(QA_RESULT),
    }
    value.update(overrides)
    return value


def _envelope(**overrides):
    value = {
        "schema_version": "1.0.0",
        "task_id": "task-envelope-test-1",
        "run_id": "result-run-test-1",
        "agent": "Backend_Developer_Agent",
        "prompt_version": "1.0.0",
        "status": "completed",
        "summary": "Synthetic guarded completion passed.",
        "deliverables": [{"path": "artifact.txt"}],
        "evidence": [{"status": "verified", "ref": "pytest:synthetic"}],
        "open_questions": [],
        "approval": APPROVAL,
        "qa_gate": QA_GATE,
    }
    value.update(overrides)
    return value


def _insert_json_row(conn, table: str, columns: tuple[str, str], value: dict, *, extra=()):
    raw = guard.canonical_json(value)
    digest = guard.canonical_sha256(value)
    names = ", ".join((*[item[0] for item in extra], *columns))
    placeholders = ", ".join("?" for _ in range(len(extra) + 2))
    values = (*[item[1] for item in extra], raw, digest)
    conn.execute(f"INSERT INTO {table}({names}) VALUES ({placeholders})", values)


def _install_governance(
    conn,
    task_id: str,
    *,
    policy_overrides=None,
    activation_overrides=None,
    binding_overrides=None,
    omit: str | None = None,
):
    policy = _policy(conn, **(policy_overrides or {}))
    activation = _activation(policy, **(activation_overrides or {}))
    binding = _binding(task_id, **(binding_overrides or {}))
    if omit != "policy":
        _insert_json_row(
            conn,
            "completion_governance_policy",
            ("policy_json", "policy_sha256"),
            policy,
            extra=(("policy_version", policy["policy_id"]), ("created_at", 1)),
        )
    if omit != "activation":
        _insert_json_row(
            conn,
            "completion_governance_activation",
            ("activation_json", "activation_sha256"),
            activation,
            extra=(("singleton_id", 1), ("updated_at", 1)),
        )
    if omit != "binding":
        _insert_json_row(
            conn,
            "completion_governance_bindings",
            ("binding_json", "binding_sha256"),
            binding,
            extra=(("native_task_id", task_id), ("created_at", 1)),
        )
    conn.commit()
    return policy, activation, binding


def _running_task(conn):
    task_id = kb.create_task(conn, title="governed synthetic", assignee="profile-test")
    assert kb.claim_task(conn, task_id) is not None
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    return task_id, int(task.current_run_id)


def _complete(conn, task_id: str, run_id: int, envelope=None):
    return kb.complete_task(
        conn,
        task_id,
        result=json.dumps(envelope or _envelope(), ensure_ascii=False),
        summary="Synthetic guarded completion passed.",
        expected_run_id=run_id,
        completion_context=guard.CompletionContext(
            caller_profile="profile-test",
            native_task_id=task_id,
            native_run_id=run_id,
            source="test",
        ),
    )


def _broker_payload(task_id: str, run_id: int, **overrides):
    value = {
        "version": "1.0.0",
        "operation": "complete",
        "request_id": "request-test-1",
        "profile": "profile-test",
        "task_id": task_id,
        "run_id": run_id,
        "result": json.dumps(_envelope()),
        "summary": "Synthetic broker completion passed.",
        "metadata": None,
        "created_cards": None,
    }
    value.update(overrides)
    return value


def _call_broker(conn, task_id: str, run_id: int, **overrides):
    config = broker.BrokerConfig(
        db_path=Path(_db_path(conn)),
        socket_path=Path(_db_path(conn)).with_suffix(".sock"),
        uid_profiles={os.getuid(): frozenset({"profile-test"})},
    )
    client, server = socket.socketpair()
    try:
        raw = guard.canonical_json(_broker_payload(task_id, run_id, **overrides))
        client.sendall((raw + "\n").encode("utf-8"))
        broker.handle_connection(server, config)
        response = client.recv(65536)
        return json.loads(response.decode("utf-8"))
    finally:
        client.close()
        server.close()


def test_legacy_board_completion_remains_compatible(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy")
        assert kb.complete_task(conn, task_id, result="legacy result") is True
        assert kb.get_task(conn, task_id).status == "done"


def test_plain_sqlite_legacy_completion_needs_no_application_udf(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="plain sqlite legacy")
        database_path = _db_path(conn)

    with sqlite3.connect(database_path) as plain_conn:
        governance_rows = plain_conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM completion_governance_policy) +
                (SELECT COUNT(*) FROM completion_governance_activation) +
                (SELECT COUNT(*) FROM completion_governance_bindings)
            """
        ).fetchone()[0]
        trigger_sql = plain_conn.execute(
            """
            SELECT sql FROM sqlite_master
             WHERE type = 'trigger'
               AND name = 'trg_governed_completion_requires_permit'
            """
        ).fetchone()

        assert governance_rows == 0
        assert trigger_sql is not None
        updated = plain_conn.execute(
            "UPDATE tasks SET status = 'done', result = ? WHERE id = ?",
            ("plain sqlite legacy result", task_id),
        )
        assert updated.rowcount == 1
        assert plain_conn.execute(
            "SELECT status, result FROM tasks WHERE id = ?", (task_id,)
        ).fetchone() == ("done", "plain sqlite legacy result")


def test_governed_completion_atomically_writes_receipt(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)

        assert _complete(conn, task_id, run_id) is True
        task = kb.get_task(conn, task_id)
        receipt = conn.execute(
            "SELECT * FROM completion_governance_receipts WHERE native_task_id = ?",
            (task_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()

        assert task.status == "done"
        assert receipt is not None
        assert json.loads(event["payload"])["governance_receipt_sha256"] == receipt["receipt_sha256"]
        assert conn.execute("SELECT COUNT(*) FROM completion_governance_permits").fetchone()[0] == 0


def test_result_receipt_digest_binds_original_utf8_text(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        raw_result = json.dumps(_envelope(summary="Árvíztűrő tükörfúrógép"), ensure_ascii=False, indent=2)

        assert kb.complete_task(
            conn,
            task_id,
            result=raw_result,
            summary="Unicode digest fixture.",
            expected_run_id=run_id,
            completion_context=guard.CompletionContext(
                caller_profile="profile-test",
                native_task_id=task_id,
                native_run_id=run_id,
                source="test",
            ),
        )
        receipt = conn.execute(
            "SELECT result_sha256 FROM completion_governance_receipts WHERE native_task_id=?",
            (task_id,),
        ).fetchone()
        assert receipt["result_sha256"] == guard.text_sha256(raw_result)
        assert receipt["result_sha256"] != guard.canonical_sha256(json.loads(raw_result))


def test_cas_ineligible_completion_leaves_no_orphan_receipt_or_permit(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        conn.execute(
            """
            CREATE TRIGGER synthetic_cas_ineligible
            BEFORE UPDATE OF status ON tasks
            WHEN NEW.id = OLD.id AND NEW.status = 'done'
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )
        conn.commit()

        assert _complete(conn, task_id, run_id) is False
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running"
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_governance_receipts"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_governance_permits"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
            (task_id,),
        ).fetchone()[0] == 0
        run = conn.execute("SELECT status, outcome FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert run["status"] == "running"
        assert run["outcome"] is None


@pytest.mark.parametrize(
    ("activation_overrides", "expected_reason"),
    [
        ({"enabled": False}, "activation is disabled"),
        ({"kill_switch": True}, "kill switch is engaged"),
    ],
)
def test_activation_controls_fail_closed(
    kanban_home, activation_overrides, expected_reason
):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(
            conn,
            task_id,
            activation_overrides=activation_overrides,
        )
        with pytest.raises(guard.CompletionGovernanceDenied, match=expected_reason):
            _complete(conn, task_id, run_id)
        assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize("missing", ["policy", "activation", "binding"])
def test_partial_governance_state_denies(kanban_home, missing):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id, omit=missing)
        with pytest.raises(guard.CompletionGovernanceDenied):
            _complete(conn, task_id, run_id)
        assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize(
    "envelope_overrides",
    [
        {"task_id": "wrong-task"},
        {"agent": "Wrong_Agent"},
        {"prompt_version": "stale"},
        {"approval": {"required": True, "status": "missing"}},
        {"qa_gate": {"required": True, "status": "fail", "review_task_id": "qa-test-1"}},
        {"evidence": [{"status": "not_run", "ref": "none"}]},
        {"open_questions": ["still blocked"]},
    ],
)
def test_result_binding_semantics_deny_mismatch(kanban_home, envelope_overrides):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        envelope = _envelope(**envelope_overrides)
        with pytest.raises(guard.CompletionGovernanceDenied):
            _complete(conn, task_id, run_id, envelope)
        assert kb.get_task(conn, task_id).status == "running"


def test_expected_run_id_is_mandatory_and_bound(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        result = json.dumps(_envelope())
        with pytest.raises(guard.CompletionGovernanceDenied, match="expected_run_id"):
            kb.complete_task(conn, task_id, result=result)
        with pytest.raises(guard.CompletionGovernanceDenied, match="run binding"):
            kb.complete_task(
                conn,
                task_id,
                result=result,
                expected_run_id=run_id + 1,
                completion_context=guard.CompletionContext(
                    caller_profile="profile-test",
                    native_task_id=task_id,
                    native_run_id=run_id + 1,
                    source="test",
                ),
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_missing_or_mismatched_completion_context_denies(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        result = json.dumps(_envelope())
        with pytest.raises(guard.CompletionGovernanceDenied, match="context is mandatory"):
            kb.complete_task(conn, task_id, result=result, expected_run_id=run_id)
        with pytest.raises(guard.CompletionGovernanceDenied, match="caller profile"):
            kb.complete_task(
                conn,
                task_id,
                result=result,
                expected_run_id=run_id,
                completion_context=guard.CompletionContext(
                    caller_profile="wrong-profile",
                    native_task_id=task_id,
                    native_run_id=run_id,
                    source="test",
                ),
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_raw_sql_completion_and_reopen_are_blocked_by_trigger(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        with pytest.raises(Exception, match="kernel permit"):
            conn.execute(
                "UPDATE tasks SET status='done', result='raw bypass' WHERE id=?",
                (task_id,),
            )
        conn.rollback()
        assert kb.get_task(conn, task_id).status == "running"

        assert _complete(conn, task_id, run_id)
        with pytest.raises(Exception, match="cannot be reopened"):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.rollback()
        assert kb.get_task(conn, task_id).status == "done"


def test_completed_result_is_immutable_through_sql_and_helper(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        assert _complete(conn, task_id, run_id)

        with pytest.raises(Exception, match="immutable"):
            conn.execute("UPDATE tasks SET result='tampered' WHERE id=?", (task_id,))
        conn.rollback()
        with pytest.raises(guard.CompletionGovernanceDenied, match="immutable"):
            kb.edit_completed_task_result(conn, task_id, result="tampered")
        assert json.loads(kb.get_task(conn, task_id).result)["task_id"] == "task-envelope-test-1"


def test_receipt_write_failure_rolls_back_task_run_event_and_permit(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        artifact = workspace / "synthetic.txt"
        artifact.write_text("synthetic", encoding="utf-8")
        staged_copy = kb.task_attachments_dir(task_id) / artifact.name
        _install_governance(conn, task_id)
        conn.execute(
            """
            CREATE TRIGGER synthetic_receipt_failure
            BEFORE INSERT ON completion_governance_receipts
            BEGIN
                SELECT RAISE(ABORT, 'synthetic receipt failure');
            END
            """
        )
        conn.commit()

        with pytest.raises(Exception, match="synthetic receipt failure"):
            kb.complete_task(
                conn,
                task_id,
                result=json.dumps(_envelope()),
                summary="Synthetic guarded completion passed.",
                metadata={"artifacts": [str(artifact)]},
                expected_run_id=run_id,
                completion_context=guard.CompletionContext(
                    caller_profile="profile-test",
                    native_task_id=task_id,
                    native_run_id=run_id,
                    source="test",
                ),
            )

        task = kb.get_task(conn, task_id)
        run = conn.execute("SELECT status, outcome FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert task.status == "running"
        assert run["status"] == "running"
        assert run["outcome"] is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
            (task_id,),
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM completion_governance_receipts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM completion_governance_permits").fetchone()[0] == 0
        assert not staged_copy.exists()
        assert artifact.exists(), "rollback must preserve the original workspace artifact"


def test_malformed_policy_json_denies_without_mutation(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        conn.execute(
            "UPDATE completion_governance_policy SET policy_json='{', policy_sha256=?",
            ("0" * 64,),
        )
        conn.commit()
        with pytest.raises(guard.CompletionGovernanceDenied, match="malformed"):
            _complete(conn, task_id, run_id)
        assert kb.get_task(conn, task_id).status == "running"


def test_unix_peer_authenticated_broker_completes_through_same_kernel(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        response = _call_broker(conn, task_id, run_id)

        assert response["ok"] is True
        assert response["completed"] is True
        assert isinstance(response["receipt_sha256"], str)
        assert kb.get_task(conn, task_id).status == "done"


def test_broker_rejects_profile_not_bound_to_peer_uid(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        response = _call_broker(conn, task_id, run_id, profile="other-profile")

        assert response["ok"] is False
        assert response["code"] == "invalid_request"
        assert kb.get_task(conn, task_id).status == "running"


def test_broker_rejects_host_artifact_copy_oracle(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        _install_governance(conn, task_id)
        response = _call_broker(
            conn,
            task_id,
            run_id,
            metadata={"artifacts": ["/etc/passwd"]},
        )

        assert response["ok"] is False
        assert response["code"] == "invalid_request"
        assert kb.get_task(conn, task_id).status == "running"
