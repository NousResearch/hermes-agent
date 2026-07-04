"""SQLite runtime ledger for autonomous contract PM execution.

The ledger is the authoritative runtime state for contract-driven PM profiles.
JSON/YAML files, Kanban cards, Discord messages, and handoffs are projections or
audit artifacts. This module keeps state transitions mechanical and inspectable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, get_args

from .compiler import compute_contract_sha256
from .models import CleanupRecord, CleanupState, CleanupType, Contract, LedgerSeed, SprintState, WorkerCloseoutEnvelope

SCHEMA_VERSION = 1
VALID_SPRINT_STATES = set(get_args(SprintState))
VALID_CLEANUP_STATES = set(get_args(CleanupState))
VALID_CLEANUP_TYPES = set(get_args(CleanupType))
SprintEventType = Literal[
    "ledger_initialized",
    "sprint_state_changed",
    "gate_resolved",
    "cleanup_recorded",
    "checkpoint_recorded",
    "worker_closeout_imported",
]


class LedgerError(ValueError):
    """Raised when a ledger operation would violate contract execution rules."""


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_jsonable(value: Any) -> Any:
    try:
        from hermes_cli.control_db import redact_jsonable

        return redact_jsonable(value)
    except Exception:
        return value


def _resolve_artifact_path(db_path: str | Path, artifact: str | Path) -> Path:
    candidate = Path(artifact).expanduser()
    if candidate.is_absolute():
        return candidate
    db_parent = Path(db_path).expanduser().resolve().parent
    return (db_parent / candidate).resolve()


def _assert_transition_allowed(conn: sqlite3.Connection, row: sqlite3.Row, sprint_id: str, new_state: SprintState) -> None:
    dispatch_states = {"packet_generated", "dispatched", "in_progress", "review_required", "verification_required", "completed", "completed_with_warnings"}
    terminal_success = {"completed", "completed_with_warnings"}
    if new_state in dispatch_states:
        deps = json.loads(row["depends_on_json"])
        incomplete = [
            dep
            for dep in deps
            if not conn.execute(
                "SELECT 1 FROM sprints WHERE sprint_id=? AND state IN ('completed', 'completed_with_warnings', 'skipped_by_galt_decision')",
                (dep,),
            ).fetchone()
        ]
        if incomplete:
            raise LedgerError(f"sprint {sprint_id} has incomplete dependencies: {', '.join(incomplete)}")
        blockers = unresolved_blocking_gates_for_sprint(conn, sprint_id)
        if blockers:
            raise LedgerError(f"sprint {sprint_id} has unresolved blocking gates: {', '.join(blockers)}")
    if new_state in terminal_success:
        open_cleanup = list(
            conn.execute(
                "SELECT id FROM cleanup WHERE sprint_id=? AND state IN ('active_needed', 'open', 'orphaned_blocker')",
                (sprint_id,),
            )
        )
        if open_cleanup:
            raise LedgerError(f"sprint {sprint_id} has unresolved cleanup records: {', '.join(r['id'] for r in open_cleanup)}")


def _transition_sprint_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    sprint_id: str,
    new_state: SprintState,
    *,
    actor: str,
    evidence: dict[str, Any] | None = None,
    artifact_path: str | None = None,
) -> None:
    old_state = row["state"]
    _assert_transition_allowed(conn, row, sprint_id, new_state)
    now = _now()
    started_at = row["started_at"]
    completed_at = row["completed_at"]
    if new_state == "in_progress" and not started_at:
        started_at = now
    if new_state in {"completed", "completed_with_warnings", "failed", "skipped_by_galt_decision", "superseded"}:
        completed_at = now
    conn.execute(
        "UPDATE sprints SET state=?, started_at=?, completed_at=?, updated_at=? WHERE sprint_id=?",
        (new_state, started_at, completed_at, now, sprint_id),
    )
    append_event(
        conn,
        actor=actor,
        event_type="sprint_state_changed",
        sprint_id=sprint_id,
        payload={"oldState": old_state, "newState": new_state, "evidence": evidence or {}},
        artifact_path=artifact_path,
    )



def create_schema(conn: sqlite3.Connection) -> None:
    """Create the v1 ledger schema if it does not exist."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sprints (
            sprint_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            section TEXT NOT NULL,
            title TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            parallel_safe INTEGER NOT NULL,
            material_type TEXT NOT NULL,
            objective TEXT NOT NULL,
            depends_on_json TEXT NOT NULL,
            gates_json TEXT NOT NULL,
            required_inputs_json TEXT NOT NULL,
            required_context_json TEXT NOT NULL,
            implementation_requirements_json TEXT NOT NULL,
            acceptance_json TEXT NOT NULL,
            stop_conditions_json TEXT NOT NULL,
            evidence_required_json TEXT NOT NULL,
            allowed_paths_json TEXT NOT NULL,
            forbidden_paths_json TEXT NOT NULL,
            mcp_grants_json TEXT NOT NULL DEFAULT '[]',
            worker_packet TEXT,
            handoff_artifact TEXT,
            verification_evidence_json TEXT NOT NULL DEFAULT '[]',
            completed_by TEXT,
            source_completed_at TEXT,
            review_json TEXT NOT NULL,
            closeout_json TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gates (
            gate_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            owner TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            blocks_sprint_ids_json TEXT NOT NULL,
            resolution_condition TEXT NOT NULL,
            evidence_required_json TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            resolved_by TEXT,
            resolved_at TEXT,
            evidence_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            event_type TEXT NOT NULL,
            sprint_id TEXT,
            payload_json TEXT NOT NULL,
            artifact_path TEXT
        );

        CREATE TABLE IF NOT EXISTS cleanup (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            sprint_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            state TEXT NOT NULL,
            identifier TEXT NOT NULL,
            owner TEXT NOT NULL,
            close_condition TEXT NOT NULL,
            closed_at TEXT,
            notes TEXT
        );
        """
    )
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(sprints)")}
    migrations = {
        "mcp_grants_json": "ALTER TABLE sprints ADD COLUMN mcp_grants_json TEXT NOT NULL DEFAULT '[]'",
        "worker_packet": "ALTER TABLE sprints ADD COLUMN worker_packet TEXT",
        "handoff_artifact": "ALTER TABLE sprints ADD COLUMN handoff_artifact TEXT",
        "verification_evidence_json": "ALTER TABLE sprints ADD COLUMN verification_evidence_json TEXT NOT NULL DEFAULT '[]'",
        "completed_by": "ALTER TABLE sprints ADD COLUMN completed_by TEXT",
        "source_completed_at": "ALTER TABLE sprints ADD COLUMN source_completed_at TEXT",
    }
    for column, ddl in migrations.items():
        if column not in existing_columns:
            conn.execute(ddl)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))


def append_event(
    conn: sqlite3.Connection,
    *,
    actor: str,
    event_type: SprintEventType | str,
    sprint_id: str | None = None,
    payload: dict[str, Any] | None = None,
    artifact_path: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO events(timestamp, actor, event_type, sprint_id, payload_json, artifact_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_now(), actor, event_type, sprint_id, _json(payload or {}), artifact_path),
    )
    return int(cur.lastrowid or 0)


def initialize_ledger(path: str | Path, seed: LedgerSeed, *, actor: str = "galt", force: bool = False) -> None:
    """Initialize a SQLite ledger from a validated seed.

    Existing ledgers are fail-closed unless ``force`` is explicitly true. The
    operation is transactional and records a ledger_initialized event.
    """

    with _connect(path) as conn:
        create_schema(conn)
        existing = conn.execute("SELECT value FROM meta WHERE key='contract_sha256'").fetchone()
        if existing and existing[0] != seed.contractSha256 and not force:
            raise LedgerError("ledger already initialized with a different contract hash")
        if existing and not force:
            raise LedgerError("ledger already initialized; pass force=True to rebuild")

        conn.execute("DELETE FROM sprints")
        conn.execute("DELETE FROM gates")
        conn.execute("DELETE FROM cleanup")
        conn.execute("DELETE FROM events")
        meta = {
            "schema_version": str(SCHEMA_VERSION),
            "contract_id": seed.contractId,
            "contract_version": seed.contractVersion,
            "contract_sha256": seed.contractSha256,
            "project_id": seed.projectId,
            "contract_lock_json": _json(seed.contractLock.model_dump(mode="json") if seed.contractLock else None),
            "mcp_runtime_json": _json(seed.mcpRuntime.model_dump(mode="json") if seed.mcpRuntime else None),
            "amendments_json": _json(seed.amendments),
            "initialized_at": _now(),
        }
        conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", meta.items())

        now = _now()
        for sprint in seed.sprints:
            conn.execute(
                """
                INSERT INTO sprints(
                    sprint_id, state, section, title, order_index, priority, parallel_safe,
                    material_type, objective, depends_on_json, gates_json,
                    required_inputs_json, required_context_json, implementation_requirements_json,
                    acceptance_json, stop_conditions_json, evidence_required_json,
                    allowed_paths_json, forbidden_paths_json, mcp_grants_json,
                    worker_packet, handoff_artifact, verification_evidence_json,
                    completed_by, source_completed_at, review_json, closeout_json,
                    started_at, completed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sprint.sprintId,
                    sprint.state,
                    sprint.section,
                    sprint.title,
                    sprint.order,
                    sprint.priority,
                    1 if sprint.parallelSafe else 0,
                    sprint.materialType,
                    sprint.objective,
                    _json(sprint.dependsOn),
                    _json(sprint.gates),
                    _json(sprint.requiredInputs),
                    _json(sprint.requiredContext),
                    _json(sprint.implementationRequirements),
                    _json([item.model_dump(mode="json", exclude_none=True) for item in sprint.acceptance]),
                    _json(sprint.stopConditions),
                    _json(sprint.evidenceRequired),
                    _json(sprint.allowedPaths),
                    _json(sprint.forbiddenPaths),
                    _json([grant.model_dump(mode="json", exclude_none=True) for grant in sprint.mcpGrants]),
                    sprint.workerPacket,
                    sprint.handoffArtifact,
                    _json(sprint.verificationEvidence),
                    sprint.completedBy,
                    sprint.completedAt,
                    _json(sprint.review.model_dump(mode="json")),
                    _json(sprint.closeout.model_dump(mode="json")),
                    sprint.startedAt,
                    sprint.completedAt,
                    now,
                ),
            )

        for gate in seed.gates:
            conn.execute(
                """
                INSERT INTO gates(
                    gate_id, type, owner, severity, description, blocks_sprint_ids_json,
                    resolution_condition, evidence_required_json, resolved, resolved_by, resolved_at, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate.gateId,
                    gate.type,
                    gate.owner,
                    gate.severity,
                    gate.description,
                    _json(gate.blocksSprintIds),
                    gate.resolutionCondition,
                    _json(gate.evidenceRequired),
                    1 if gate.resolved else 0,
                    gate.resolvedBy,
                    gate.resolvedAt,
                    _json(gate.evidence),
                ),
            )

        for record in seed.cleanupRegistry.records:
            record_cleanup(conn, record, actor=actor, commit=False)
        append_event(conn, actor=actor, event_type="ledger_initialized", payload={"contractSha256": seed.contractSha256})
        conn.commit()


def _load_json_cell(row: sqlite3.Row, key: str) -> Any:
    return json.loads(row[key])


def unresolved_blocking_gates_for_sprint(conn: sqlite3.Connection, sprint_id: str) -> list[str]:
    gates = []
    for row in conn.execute("SELECT gate_id, severity, blocks_sprint_ids_json, resolved FROM gates"):
        blocks = json.loads(row["blocks_sprint_ids_json"])
        if row["severity"] == "blocking" and not row["resolved"] and sprint_id in blocks:
            gates.append(row["gate_id"])
    sprint = conn.execute("SELECT gates_json FROM sprints WHERE sprint_id=?", (sprint_id,)).fetchone()
    if sprint:
        explicit = json.loads(sprint["gates_json"])
        for gate_id in explicit:
            row = conn.execute("SELECT severity, resolved FROM gates WHERE gate_id=?", (gate_id,)).fetchone()
            if row and row["severity"] == "blocking" and not row["resolved"] and gate_id not in gates:
                gates.append(gate_id)
    return sorted(gates)


def ready_sprints(path: str | Path) -> list[str]:
    """Return sprint ids whose dependencies and blocking gates are satisfied."""

    with _connect(path) as conn:
        rows = list(conn.execute("SELECT * FROM sprints WHERE state IN ('not_started', 'ready') ORDER BY order_index, sprint_id"))
        completed = {
            row["sprint_id"]
            for row in conn.execute("SELECT sprint_id FROM sprints WHERE state IN ('completed', 'completed_with_warnings', 'skipped_by_galt_decision')")
        }
        result: list[str] = []
        for row in rows:
            deps = _load_json_cell(row, "depends_on_json")
            if any(dep not in completed for dep in deps):
                continue
            if unresolved_blocking_gates_for_sprint(conn, row["sprint_id"]):
                continue
            result.append(row["sprint_id"])
        return result


def transition_sprint(
    path: str | Path,
    sprint_id: str,
    new_state: SprintState,
    *,
    actor: str,
    evidence: dict[str, Any] | None = None,
    artifact_path: str | None = None,
) -> None:
    """Transition a sprint state with gate/dependency checks for dispatch states."""

    if new_state not in VALID_SPRINT_STATES:
        raise LedgerError(f"invalid sprint state: {new_state}")
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM sprints WHERE sprint_id=?", (sprint_id,)).fetchone()
        if row is None:
            raise LedgerError(f"unknown sprint id: {sprint_id}")
        _transition_sprint_row(conn, row, sprint_id, new_state, actor=actor, evidence=evidence, artifact_path=artifact_path)
        conn.commit()


def resolve_gate(
    path: str | Path,
    gate_id: str,
    *,
    actor: str,
    evidence: Iterable[str],
    artifact_path: str | None = None,
) -> None:
    """Mark a gate resolved and record evidence."""

    evidence_list = list(evidence)
    if not evidence_list:
        raise LedgerError("gate resolution requires at least one evidence item")
    with _connect(path) as conn:
        row = conn.execute("SELECT gate_id FROM gates WHERE gate_id=?", (gate_id,)).fetchone()
        if row is None:
            raise LedgerError(f"unknown gate id: {gate_id}")
        conn.execute(
            "UPDATE gates SET resolved=1, resolved_by=?, resolved_at=?, evidence_json=? WHERE gate_id=?",
            (actor, _now(), _json(evidence_list), gate_id),
        )
        append_event(conn, actor=actor, event_type="gate_resolved", payload={"gateId": gate_id, "evidence": evidence_list}, artifact_path=artifact_path)
        conn.commit()


def record_cleanup(conn: sqlite3.Connection, record: CleanupRecord, *, actor: str, commit: bool = True) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO cleanup(id, type, created_by, sprint_id, created_at, state, identifier, owner, close_condition, closed_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.type,
            record.createdBy,
            record.sprintId,
            record.createdAt,
            record.state,
            record.identifier,
            record.owner,
            record.closeCondition,
            record.closedAt,
            record.notes,
        ),
    )
    append_event(conn, actor=actor, event_type="cleanup_recorded", sprint_id=record.sprintId, payload=record.model_dump(mode="json"))
    if commit:
        conn.commit()


def record_cleanup_entry(path: str | Path, record: CleanupRecord, *, actor: str) -> None:
    """Record or replace a cleanup entry in the authoritative ledger."""

    with _connect(path) as conn:
        if not conn.execute("SELECT 1 FROM sprints WHERE sprint_id=?", (record.sprintId,)).fetchone():
            raise LedgerError(f"unknown sprint id for cleanup record: {record.sprintId}")
        record_cleanup(conn, record, actor=actor)


def update_cleanup_state(
    path: str | Path,
    cleanup_id: str,
    state: CleanupState,
    *,
    actor: str,
    notes: str | None = None,
) -> None:
    """Update cleanup state so closeout blockers can be resolved through the ledger."""

    if state not in VALID_CLEANUP_STATES:
        raise LedgerError(f"invalid cleanup state: {state}")
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM cleanup WHERE id=?", (cleanup_id,)).fetchone()
        if row is None:
            raise LedgerError(f"unknown cleanup id: {cleanup_id}")
        now = _now()
        closed_at = row["closed_at"]
        if state in {"closed", "archived", "retained_with_reason"} and not closed_at:
            closed_at = now
        conn.execute(
            "UPDATE cleanup SET state=?, closed_at=?, notes=? WHERE id=?",
            (state, closed_at, notes if notes is not None else row["notes"], cleanup_id),
        )
        append_event(
            conn,
            actor=actor,
            event_type="cleanup_recorded",
            sprint_id=row["sprint_id"],
            payload={"id": cleanup_id, "oldState": row["state"], "newState": state, "notes": notes},
        )
        conn.commit()


def export_state(path: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable current-state projection from SQLite."""

    with _connect(path) as conn:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta ORDER BY key")}
        sprints = []
        for row in conn.execute("SELECT * FROM sprints ORDER BY order_index, sprint_id"):
            sprints.append(
                {
                    "sprintId": row["sprint_id"],
                    "state": row["state"],
                    "section": row["section"],
                    "title": row["title"],
                    "order": row["order_index"],
                    "priority": row["priority"],
                    "parallelSafe": bool(row["parallel_safe"]),
                    "materialType": row["material_type"],
                    "objective": row["objective"],
                    "dependsOn": json.loads(row["depends_on_json"]),
                    "gates": json.loads(row["gates_json"]),
                    "allowedPaths": json.loads(row["allowed_paths_json"]),
                    "forbiddenPaths": json.loads(row["forbidden_paths_json"]),
                    "mcpGrants": json.loads(row["mcp_grants_json"]),
                    "workerPacket": row["worker_packet"],
                    "handoffArtifact": row["handoff_artifact"],
                    "verificationEvidence": json.loads(row["verification_evidence_json"]),
                    "completedBy": row["completed_by"],
                    "sourceCompletedAt": row["source_completed_at"],
                    "startedAt": row["started_at"],
                    "completedAt": row["completed_at"],
                    "updatedAt": row["updated_at"],
                    "unresolvedBlockingGates": unresolved_blocking_gates_for_sprint(conn, row["sprint_id"]),
                }
            )
        gates = [dict(row) | {"resolved": bool(row["resolved"]), "evidence": json.loads(row["evidence_json"])} for row in conn.execute("SELECT * FROM gates ORDER BY gate_id")]
        cleanup = [dict(row) for row in conn.execute("SELECT * FROM cleanup ORDER BY sprint_id, id")]
        events = [dict(row) | {"payload": json.loads(row["payload_json"])} for row in conn.execute("SELECT * FROM events ORDER BY id")]
        return {"meta": meta, "readySprints": ready_sprints(path), "sprints": sprints, "gates": gates, "cleanup": cleanup, "events": events}



TRUSTED_NO_LIVE_DB_PROOF_PROVENANCE = {"control_plane_captured", "sqlite_derived", "git_derived"}
TRUSTED_COMMAND_EVIDENCE_PROVENANCE = {"control_plane_captured", "pm_rerun"}
WAIVER_WARNING_CLASSES = {"deferred_non_blocking", "requires_next_sprint_ticket"}


def _load_packet_from_row(db_path: str | Path, row: sqlite3.Row, issues: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    packet_ref = row["worker_packet"]
    if not packet_ref:
        issues.append({"severity": "error", "code": "missing_worker_packet", "sprintId": row["sprint_id"]})
        return None, None
    packet_path = _resolve_artifact_path(db_path, packet_ref)
    if not packet_path.exists():
        issues.append({"severity": "error", "code": "missing_worker_packet_file", "sprintId": row["sprint_id"], "path": str(packet_path)})
        return None, None
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append({"severity": "error", "code": "invalid_worker_packet_json", "sprintId": row["sprint_id"], "path": str(packet_path), "error": str(exc)})
        return None, None
    return packet, sha256_file(packet_path)


def _packet_command_requirements(packet: dict[str, Any]) -> list[dict[str, str]]:
    acceptance = packet.get("acceptanceCriteria") or []
    commands = packet.get("verificationCommands") or []
    requirements: list[dict[str, str]] = []
    for index, command in enumerate(commands):
        matches = [item.get("id") for item in acceptance if item.get("verification") == "command" and item.get("command") == command and item.get("id")]
        command_id = matches[0] if len(matches) == 1 else f"verificationCommands[{index}]"
        requirements.append({"commandId": command_id, "command": command})
    return requirements


def _artifact_issue(db_path: str | Path, artifact: Any, sprint_id: str) -> dict[str, Any] | None:
    if not artifact:
        return None
    path_value = artifact.get("path") if isinstance(artifact, dict) else None
    if not path_value:
        return {"severity": "error", "code": "handoff_artifact_missing_path", "sprintId": sprint_id}
    artifact_path = _resolve_artifact_path(db_path, path_value)
    if not artifact_path.exists():
        return {"severity": "error", "code": "missing_handoff_artifact", "sprintId": sprint_id, "path": str(artifact_path)}
    expected_hash = artifact.get("sha256") if isinstance(artifact, dict) else None
    if expected_hash:
        observed = sha256_file(artifact_path)
        if observed != expected_hash:
            return {"severity": "error", "code": "handoff_artifact_hash_mismatch", "sprintId": sprint_id, "path": str(artifact_path), "expected": expected_hash, "observed": observed}
    return None


def validate_worker_closeout(
    db_path: str | Path,
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    envelope: WorkerCloseoutEnvelope,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    sprint_id = row["sprint_id"]
    if envelope.sprintId != sprint_id:
        issues.append({"severity": "error", "code": "sprint_id_mismatch", "expected": sprint_id, "observed": envelope.sprintId})
    packet, packet_hash = _load_packet_from_row(db_path, row, issues)
    if packet_hash and packet_hash != envelope.workerPacketSha256:
        issues.append({"severity": "error", "code": "worker_packet_hash_mismatch", "sprintId": sprint_id, "expected": packet_hash, "observed": envelope.workerPacketSha256})
    if envelope.resultStatus in {"completed", "completed_with_warnings"}:
        proof = envelope.noLiveDbMutationProof
        if proof is None:
            issues.append({"severity": "error", "code": "missing_no_live_db_mutation_proof", "sprintId": sprint_id})
        elif proof.provenance not in TRUSTED_NO_LIVE_DB_PROOF_PROVENANCE:
            issues.append({"severity": "error", "code": "untrusted_no_live_db_mutation_proof", "sprintId": sprint_id, "provenance": proof.provenance})
        for warning in envelope.warnings:
            if warning.blocker:
                issues.append({"severity": "error", "code": "terminal_success_with_blocker", "sprintId": sprint_id, "warningId": warning.id})
            if warning.warningClass == "requires_benjamin_acceptance":
                issues.append({"severity": "error", "code": "benjamin_acceptance_warning_blocks_terminal_success", "sprintId": sprint_id, "warningId": warning.id})
    if packet:
        evidence_by_key = {
            (item.commandId, item.command): item
            for item in envelope.verificationEvidence
            if item.type == "command" and item.commandId and item.command
        }
        waived_command_ids = {warning.commandId for warning in envelope.warnings if warning.commandId and warning.warningClass in WAIVER_WARNING_CLASSES}
        for requirement in _packet_command_requirements(packet):
            key = (requirement["commandId"], requirement["command"])
            evidence = evidence_by_key.get(key)
            if evidence is None:
                if requirement["commandId"] in waived_command_ids:
                    continue
                issues.append({"severity": "error", "code": "missing_command_evidence", "sprintId": sprint_id, **requirement})
                continue
            if not evidence.passed:
                issues.append({"severity": "error", "code": "command_evidence_failed", "sprintId": sprint_id, **requirement})
            if evidence.provenance not in TRUSTED_COMMAND_EVIDENCE_PROVENANCE:
                issues.append({"severity": "error", "code": "untrusted_command_evidence", "sprintId": sprint_id, **requirement, "provenance": evidence.provenance})
    handoff_issue = _artifact_issue(db_path, envelope.handoffArtifact.model_dump(mode="json") if envelope.handoffArtifact else None, sprint_id)
    if handoff_issue:
        issues.append(handoff_issue)
    return {"ok": not any(issue["severity"] == "error" for issue in issues), "sprintId": sprint_id, "issues": issues}


def import_worker_closeout(
    path: str | Path,
    sprint_id: str,
    envelope: WorkerCloseoutEnvelope | dict[str, Any],
    *,
    actor: str,
    apply: bool = False,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    parsed = envelope if isinstance(envelope, WorkerCloseoutEnvelope) else WorkerCloseoutEnvelope.model_validate(envelope)
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM sprints WHERE sprint_id=?", (sprint_id,)).fetchone()
        if row is None:
            raise LedgerError(f"unknown sprint id: {sprint_id}")
        report = validate_worker_closeout(path, conn, row, parsed)
        report["apply"] = apply
        if not apply:
            return report
        if not report["ok"]:
            raise LedgerError("worker closeout validation failed: " + ", ".join(issue["code"] for issue in report["issues"] if issue["severity"] == "error"))
        target_state: SprintState
        if parsed.resultStatus == "completed":
            target_state = "completed"
        elif parsed.resultStatus == "completed_with_warnings":
            target_state = "completed_with_warnings"
        elif parsed.resultStatus == "failed":
            target_state = "failed"
        else:
            target_state = "blocked_galt"
        redacted_closeout = _redact_jsonable(parsed.model_dump(mode="json", exclude_none=True))
        redacted_evidence = _redact_jsonable([item.model_dump(mode="json", exclude_none=True) for item in parsed.verificationEvidence])
        _transition_sprint_row(
            conn,
            row,
            sprint_id,
            target_state,
            actor=actor,
            evidence={"workerCloseout": {"schemaVersion": parsed.schemaVersion, "resultStatus": parsed.resultStatus}},
            artifact_path=artifact_path,
        )
        conn.execute(
            """
            UPDATE sprints
            SET verification_evidence_json=?, handoff_artifact=?, completed_by=?, source_completed_at=?, closeout_json=?, updated_at=?
            WHERE sprint_id=?
            """,
            (
                _json(redacted_evidence),
                parsed.handoffArtifact.path if parsed.handoffArtifact else row["handoff_artifact"],
                actor,
                parsed.sourceCompletedAt or _now(),
                _json(redacted_closeout),
                _now(),
                sprint_id,
            ),
        )
        append_event(
            conn,
            actor=actor,
            event_type="worker_closeout_imported",
            sprint_id=sprint_id,
            payload={"resultStatus": parsed.resultStatus, "issues": report["issues"]},
            artifact_path=artifact_path,
        )
        conn.commit()
        return report | {"targetState": target_state}


def ledger_check(path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    with _connect(path) as conn:
        create_schema(conn)
        for row in conn.execute("SELECT * FROM sprints ORDER BY order_index, sprint_id"):
            sprint_id = row["sprint_id"]
            packet, packet_hash = _load_packet_from_row(path, row, issues) if row["worker_packet"] else (None, None)
            closeout_raw = json.loads(row["closeout_json"] or "{}")
            terminal_success = row["state"] in {"completed", "completed_with_warnings"}
            if terminal_success:
                if closeout_raw.get("schemaVersion") != "worker-closeout/v1":
                    issues.append({"severity": "error", "code": "terminal_sprint_missing_structured_closeout", "sprintId": sprint_id})
                else:
                    try:
                        envelope = WorkerCloseoutEnvelope.model_validate(closeout_raw)
                        issues.extend(validate_worker_closeout(path, conn, row, envelope)["issues"])
                        if packet_hash and envelope.workerPacketSha256 != packet_hash:
                            issues.append({"severity": "error", "code": "worker_packet_db_file_hash_drift", "sprintId": sprint_id})
                    except Exception as exc:
                        issues.append({"severity": "error", "code": "invalid_structured_closeout", "sprintId": sprint_id, "error": str(exc)})
                open_cleanup = [r["id"] for r in conn.execute("SELECT id FROM cleanup WHERE sprint_id=? AND state IN ('active_needed', 'open', 'orphaned_blocker')", (sprint_id,))]
                if open_cleanup:
                    issues.append({"severity": "error", "code": "terminal_sprint_open_cleanup", "sprintId": sprint_id, "cleanupIds": open_cleanup})
    ok = not any(issue["severity"] == "error" for issue in issues)
    return {"ok": ok, "strict": strict, "issues": issues}


def write_projection_files(path: str | Path, output_dir: str | Path) -> list[Path]:
    """Write current-state, sprint-ledger, and events projections next to a repo."""

    state = export_state(path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "current-state.json": state,
        "sprint-ledger.json": {"contract": state["meta"], "sprints": state["sprints"], "gates": state["gates"], "cleanup": state["cleanup"]},
        "events.jsonl": None,
    }
    written: list[Path] = []
    for name, payload in files.items():
        target = destination / name
        if name.endswith(".jsonl"):
            target.write_text("".join(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n" for event in state["events"]), encoding="utf-8")
        else:
            target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(target)
    return written


def verify_contract_lock(path: str | Path, contract_payload: Contract | dict[str, Any] | str | bytes) -> bool:
    """Return true if the payload matches the ledger contract hash."""

    observed = compute_contract_sha256(contract_payload)
    with _connect(path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='contract_sha256'").fetchone()
        if row is None:
            raise LedgerError("ledger has no contract_sha256")
        return row["value"] == observed
