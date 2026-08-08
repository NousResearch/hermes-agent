"""Transactional, opt-in governance for Kanban completion.

The guard is intentionally implemented beside the Kanban persistence layer,
not as a model-tool plugin.  A board with no governance rows keeps legacy
behaviour.  Presence of any governance row opts the database into fail-closed
mode: partial configuration, malformed JSON, a missing schema validator, or a
semantic mismatch denies completion.

This module is deployment-agnostic.  Privileged installation of policy,
activation, and binding rows belongs to the DB owner/broker boundary; ordinary
worker-facing APIs never expose an installer or mutable evaluator registry.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional


_POLICY_TABLE = "completion_governance_policy"
_ACTIVATION_TABLE = "completion_governance_activation"
_BINDING_TABLE = "completion_governance_bindings"
_RECEIPT_TABLE = "completion_governance_receipts"
_PERMIT_TABLE = "completion_governance_permits"
_GOVERNANCE_TABLES = (_POLICY_TABLE, _ACTIVATION_TABLE, _BINDING_TABLE)


class CompletionGovernanceDenied(ValueError):
    """Raised when a governed completion or result mutation is not authorized."""


@dataclass(frozen=True)
class CompletionContext:
    """Caller identity supplied by an OS-isolated adapter or DB broker."""

    caller_profile: str
    native_task_id: str
    native_run_id: int
    source: str
    peer_uid: Optional[int] = None


@dataclass(frozen=True)
class CompletionAuthorization:
    """Trusted policy decision retained only for the current DB transaction."""

    native_task_id: str
    external_task_id: str
    result_run_id: str
    policy_version: str
    policy_sha256: str
    activation_sha256: str
    binding_sha256: str
    task_envelope_sha256: str
    result_sha256: str
    runtime_profile: str
    result_envelope: Mapping[str, Any]

    def receipt(self, *, native_run_id: int, created_at: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "native_task_id": self.native_task_id,
            "external_task_id": self.external_task_id,
            "native_run_id": int(native_run_id),
            "result_run_id": self.result_run_id,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "activation_sha256": self.activation_sha256,
            "binding_sha256": self.binding_sha256,
            "task_envelope_sha256": self.task_envelope_sha256,
            "result_sha256": self.result_sha256,
            "runtime_profile": self.runtime_profile,
            "created_at": int(created_at),
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used by all hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _deny(reason: str) -> NoReturn:
    raise CompletionGovernanceDenied(f"governed completion denied: {reason}")


def _table_has_rows(conn: sqlite3.Connection, table: str) -> bool:
    try:
        return conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
    except sqlite3.OperationalError:
        # A partially installed database is governed-but-broken, never legacy.
        return True


def governance_rows_present(conn: sqlite3.Connection) -> bool:
    """Return True when any opt-in governance state is present or malformed."""

    return any(_table_has_rows(conn, table) for table in _GOVERNANCE_TABLES)


def _one_row(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row:
    rows = conn.execute(sql, params).fetchall()
    if len(rows) != 1:
        _deny("governance state is missing or ambiguous")
    return rows[0]


def _verified_json(raw: Any, expected_sha256: Any, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, str) or not isinstance(expected_sha256, str):
        _deny(f"{label} JSON/hash is missing")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        _deny(f"{label} JSON is malformed")
    if not isinstance(value, dict):
        _deny(f"{label} must be an object")
    actual = canonical_sha256(value)
    if actual != expected_sha256:
        _deny(f"{label} hash mismatch")
    return value, actual


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _deny(f"{label} fields do not match the 1.0.0 contract")


def _database_path(conn: sqlite3.Connection) -> str:
    rows = conn.execute("PRAGMA database_list").fetchall()
    for row in rows:
        name = row[1] if not isinstance(row, sqlite3.Row) else row["name"]
        if name == "main":
            raw = row[2] if not isinstance(row, sqlite3.Row) else row["file"]
            if not raw:
                _deny("in-memory databases cannot use path-bound governance")
            return str(Path(raw).resolve())
    _deny("main database path is unavailable")
    raise AssertionError("unreachable")


def _load_policy_activation(
    conn: sqlite3.Connection,
) -> tuple[dict[str, Any], str, str, dict[str, Any], str]:
    policy_row = _one_row(
        conn,
        f"SELECT policy_version, policy_json, policy_sha256 FROM {_POLICY_TABLE}",
    )
    policy, policy_sha = _verified_json(
        policy_row["policy_json"], policy_row["policy_sha256"], "policy"
    )
    _require_exact_keys(
        policy,
        {
            "schema_version",
            "policy_id",
            "board",
            "database_path",
            "result_schema",
            "result_schema_sha256",
            "allowed_profiles",
            "allowed_completion_sources",
            "require_deliverables",
            "worker_isolation",
        },
        "policy",
    )
    if policy.get("schema_version") != "1.0.0":
        _deny("unsupported policy schema version")
    if policy_row["policy_version"] != policy.get("policy_id"):
        _deny("policy identity mismatch")
    if policy.get("database_path") != _database_path(conn):
        _deny("database path is outside the activated board scope")
    profiles = policy.get("allowed_profiles")
    if not isinstance(profiles, list) or not profiles or any(
        not isinstance(item, str) or not item for item in profiles
    ):
        _deny("allowed_profiles is invalid")
    sources = policy.get("allowed_completion_sources")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(item, str) or not item for item in sources
    ):
        _deny("allowed_completion_sources is invalid")
    schema = policy.get("result_schema")
    if not isinstance(schema, dict):
        _deny("result_schema is missing")
    if canonical_sha256(schema) != policy.get("result_schema_sha256"):
        _deny("result_schema hash mismatch")
    isolation = policy.get("worker_isolation")
    _validate_worker_isolation(isolation)

    activation_row = _one_row(
        conn,
        f"SELECT activation_json, activation_sha256 FROM {_ACTIVATION_TABLE}",
    )
    activation, activation_sha = _verified_json(
        activation_row["activation_json"],
        activation_row["activation_sha256"],
        "activation",
    )
    _require_exact_keys(
        activation,
        {"schema_version", "enabled", "kill_switch", "policy_sha256"},
        "activation",
    )
    if activation.get("schema_version") != "1.0.0":
        _deny("unsupported activation schema version")
    if activation.get("enabled") is not True:
        _deny("activation is disabled")
    if activation.get("kill_switch") is not False:
        _deny("kill switch is engaged")
    if activation.get("policy_sha256") != policy_sha:
        _deny("activation is not bound to the active policy")
    return policy, str(policy_row["policy_version"]), policy_sha, activation, activation_sha


def _validate_worker_isolation(value: Any) -> None:
    if not isinstance(value, dict):
        _deny("worker isolation policy is missing")
    _require_exact_keys(
        value,
        {
            "mode",
            "network",
            "toolsets",
            "mount_hermes_resources",
            "broker_socket",
        },
        "worker isolation",
    )
    if value.get("mode") != "docker":
        _deny("worker isolation mode must be docker")
    if value.get("network") is not False:
        _deny("governed worker network must be disabled")
    if value.get("mount_hermes_resources") is not False:
        _deny("governed workers must not mount Hermes resources")
    toolsets = value.get("toolsets")
    if toolsets != ["terminal"]:
        _deny("governed worker toolsets must be exactly ['terminal']")
    broker_socket = value.get("broker_socket")
    if not isinstance(broker_socket, str) or not Path(broker_socket).is_absolute():
        _deny("worker broker socket must be an absolute path")


def _validate_result_schema(
    policy: Mapping[str, Any],
    envelope: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        _deny(f"{label} must be an object")
    try:
        import jsonschema
    except ImportError:
        _deny("JSON Schema validator is unavailable")
    try:
        jsonschema.Draft202012Validator(
            policy["result_schema"],
            format_checker=jsonschema.FormatChecker(),
        ).validate(envelope)
    except Exception:
        _deny(f"{label} schema validation failed")
    return envelope


def _validate_approval_and_qa(
    policy: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    approval = binding.get("approval")
    if not isinstance(approval, dict):
        _deny("binding approval is missing")
    approval_state = approval.get("state")
    if approval_state not in {"approved", "not_required"}:
        _deny("required human approval is not effective")
    if approval_state == "approved" and not approval.get("approval_ref"):
        _deny("approved scope has no approval reference")

    workflow_type = binding.get("workflow_type")
    if workflow_type not in {
        "simple",
        "major_development",
        "publishing",
        "legal_financial",
        "research",
        "vault_maintenance",
    }:
        _deny("binding workflow type is invalid")
    qa_gate = binding.get("qa_gate")
    if not isinstance(qa_gate, dict):
        _deny("binding QA gate is missing")
    qa_required = qa_gate.get("required") is True
    if workflow_type in {"major_development", "publishing"} and not qa_required:
        _deny("workflow requires independent QA")
    if not qa_required:
        if qa_gate != {
            "required": False,
            "status": "not_required",
            "review_task_id": None,
        }:
            _deny("non-required QA gate is inconsistent")
        if binding.get("qa_result") is not None or binding.get("qa_result_sha256") is not None:
            _deny("unexpected QA result on a non-required gate")
        return
    if qa_gate.get("status") != "pass" or not isinstance(
        qa_gate.get("review_task_id"), str
    ):
        _deny("required QA gate is not pass")
    qa_result = binding.get("qa_result")
    qa_hash = binding.get("qa_result_sha256")
    if not isinstance(qa_result, dict) or not isinstance(qa_hash, str):
        _deny("required QA result is missing")
    if canonical_sha256(qa_result) != qa_hash:
        _deny("QA result hash mismatch")
    qa_result = _validate_result_schema(policy, qa_result, "QA result envelope")
    if qa_result.get("task_id") != qa_gate.get("review_task_id"):
        _deny("QA result task identity mismatch")
    if qa_result.get("agent") != "QA_Tester_Agent":
        _deny("QA result was not issued by QA_Tester_Agent")
    if qa_result.get("status") != "completed":
        _deny("QA result status is not completed")
    if qa_result.get("open_questions") != []:
        _deny("QA result has open questions")
    qa_evidence = qa_result.get("evidence")
    if not isinstance(qa_evidence, list) or not qa_evidence or any(
        not isinstance(item, dict) or item.get("status") != "verified"
        for item in qa_evidence
    ):
        _deny("QA result evidence is not fully verified")
    if qa_result.get("qa_gate") != {
        "required": False,
        "status": "not_required",
        "review_task_id": None,
    }:
        _deny("QA review result cannot recursively require QA")


def load_worker_isolation(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    """Return the active fail-closed worker isolation contract for a board."""

    if not governance_rows_present(conn):
        return None
    policy, _version, _policy_sha, _activation, _activation_sha = (
        _load_policy_activation(conn)
    )
    return dict(policy["worker_isolation"])


def authorize_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str],
    expected_run_id: Optional[int],
    completion_context: Optional[CompletionContext],
) -> Optional[CompletionAuthorization]:
    """Authorize a completion from the same write transaction as its CAS.

    ``None`` means the database has no governance rows and retains legacy
    behaviour.  Every configured, partial, or corrupt state raises instead.
    """

    if not governance_rows_present(conn):
        return None
    policy, policy_version, policy_sha, _activation, activation_sha = (
        _load_policy_activation(conn)
    )

    if expected_run_id is None:
        _deny("expected_run_id is mandatory")
    if completion_context is None:
        _deny("completion context is mandatory")
    if completion_context.native_task_id != task_id:
        _deny("completion context task mismatch")
    if int(completion_context.native_run_id) != int(expected_run_id):
        _deny("completion context run mismatch")
    if completion_context.source not in policy["allowed_completion_sources"]:
        _deny("completion source is not allowlisted")
    if not isinstance(result, str) or not result.strip():
        _deny("result-envelope JSON is mandatory")

    task = _one_row(
        conn,
        "SELECT id, assignee, status, current_run_id FROM tasks WHERE id = ?",
        (task_id,),
    )
    if task["status"] != "running":
        _deny("governed tasks may complete only from running")
    if task["current_run_id"] is None or int(task["current_run_id"]) != int(expected_run_id):
        _deny("native run binding mismatch")

    binding_row = _one_row(
        conn,
        f"SELECT binding_json, binding_sha256 FROM {_BINDING_TABLE} WHERE native_task_id = ?",
        (task_id,),
    )
    binding, binding_sha = _verified_json(
        binding_row["binding_json"], binding_row["binding_sha256"], "binding"
    )
    _require_exact_keys(
        binding,
        {
            "schema_version",
            "native_task_id",
            "external_task_id",
            "assigned_agent",
            "runtime_profile",
            "prompt_version",
            "task_envelope_sha256",
            "workflow_type",
            "approval",
            "qa_gate",
            "qa_result",
            "qa_result_sha256",
        },
        "binding",
    )
    if binding.get("schema_version") != "1.0.0":
        _deny("unsupported binding schema version")
    if binding.get("native_task_id") != task_id:
        _deny("native task binding mismatch")
    runtime_profile = binding.get("runtime_profile")
    if task["assignee"] != runtime_profile:
        _deny("current assignee/profile binding mismatch")
    if completion_context.caller_profile != runtime_profile:
        _deny("caller profile binding mismatch")
    if runtime_profile not in policy["allowed_profiles"]:
        _deny("runtime profile is not allowlisted")
    if not isinstance(binding.get("task_envelope_sha256"), str):
        _deny("task envelope hash is missing")
    _validate_approval_and_qa(policy, binding)

    try:
        envelope = json.loads(result)
    except (TypeError, ValueError):
        _deny("result is not valid JSON")
    envelope = _validate_result_schema(policy, envelope, "result envelope")

    if envelope.get("schema_version") != "1.0.0":
        _deny("unsupported result envelope version")
    if envelope.get("status") != "completed":
        _deny("result status is not completed")
    if envelope.get("task_id") != binding.get("external_task_id"):
        _deny("external task identity mismatch")
    if envelope.get("agent") != binding.get("assigned_agent"):
        _deny("agent identity mismatch")
    if envelope.get("prompt_version") != binding.get("prompt_version"):
        _deny("prompt version mismatch")
    if envelope.get("approval") != binding.get("approval"):
        _deny("approval scope mismatch")
    if envelope.get("qa_gate") != binding.get("qa_gate"):
        _deny("QA gate mismatch")
    if envelope.get("open_questions") != []:
        _deny("open questions remain")
    deliverables = envelope.get("deliverables")
    if policy.get("require_deliverables") is True and (
        not isinstance(deliverables, list) or not deliverables
    ):
        _deny("deliverables are required")
    evidence = envelope.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        _deny("verified evidence is required")
    if any(not isinstance(item, dict) or item.get("status") != "verified" for item in evidence):
        _deny("all evidence must be verified")
    result_run_id = envelope.get("run_id")
    if not isinstance(result_run_id, str) or not result_run_id:
        _deny("result run identity is missing")

    return CompletionAuthorization(
        native_task_id=task_id,
        external_task_id=str(binding["external_task_id"]),
        result_run_id=result_run_id,
        policy_version=policy_version,
        policy_sha256=policy_sha,
        activation_sha256=activation_sha,
        binding_sha256=binding_sha,
        task_envelope_sha256=str(binding["task_envelope_sha256"]),
        result_sha256=text_sha256(result),
        runtime_profile=str(runtime_profile),
        result_envelope=envelope,
    )


def insert_completion_permit(
    conn: sqlite3.Connection,
    authorization: CompletionAuthorization,
    *,
    result: Optional[str],
    created_at: int,
) -> None:
    conn.execute(
        f"INSERT INTO {_PERMIT_TABLE}(native_task_id, result, created_at) VALUES (?, ?, ?)",
        (authorization.native_task_id, result, int(created_at)),
    )


def remove_completion_permit(conn: sqlite3.Connection, task_id: str) -> None:
    conn.execute(
        f"DELETE FROM {_PERMIT_TABLE} WHERE native_task_id = ?",
        (task_id,),
    )


def insert_completion_receipt(
    conn: sqlite3.Connection,
    authorization: CompletionAuthorization,
    *,
    native_run_id: int,
    created_at: int,
) -> dict[str, Any]:
    receipt = authorization.receipt(
        native_run_id=native_run_id,
        created_at=created_at,
    )
    conn.execute(
        f"""
        INSERT INTO {_RECEIPT_TABLE}(
            receipt_sha256, native_task_id, external_task_id,
            native_run_id, result_run_id, policy_version,
            policy_sha256, activation_sha256, binding_sha256,
            task_envelope_sha256, result_sha256, runtime_profile,
            receipt_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            receipt["receipt_sha256"],
            receipt["native_task_id"],
            receipt["external_task_id"],
            receipt["native_run_id"],
            receipt["result_run_id"],
            receipt["policy_version"],
            receipt["policy_sha256"],
            receipt["activation_sha256"],
            receipt["binding_sha256"],
            receipt["task_envelope_sha256"],
            receipt["result_sha256"],
            receipt["runtime_profile"],
            canonical_json(receipt),
            receipt["created_at"],
        ),
    )
    return receipt


def assert_completed_result_mutable(conn: sqlite3.Connection, task_id: str) -> None:
    """Deny post-completion result replacement on governed boards."""

    del task_id
    if governance_rows_present(conn):
        _deny("completed governed results are immutable")
