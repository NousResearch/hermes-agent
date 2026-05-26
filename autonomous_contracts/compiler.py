"""Compiler utilities for contract-ledger autonomous PM execution."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import Contract, ContractLock, LedgerGateRecord, LedgerSeed, LedgerSprintRecord, WorkerPacket
from .validation import validate_contract


def load_contract(path: str | Path) -> Contract:
    """Load and validate a YAML or JSON executable contract."""

    contract_path = Path(path)
    raw = contract_path.read_text(encoding="utf-8")
    if contract_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"contract file must contain a mapping: {contract_path}")
    return validate_contract(data)


def _canonical_contract_bytes(contract: Contract) -> bytes:
    payload = contract.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_contract_sha256(contract: Contract | dict[str, Any] | str | bytes) -> str:
    """Compute a stable SHA-256 for a contract.

    Contract/dict input hashes canonical JSON so equivalent YAML formatting does
    not perturb the lock. Raw str/bytes input hashes the exact content.
    """

    if isinstance(contract, Contract):
        raw = _canonical_contract_bytes(contract)
    elif isinstance(contract, dict):
        raw = _canonical_contract_bytes(validate_contract(contract))
    elif isinstance(contract, str):
        raw = contract.encode("utf-8")
    else:
        raw = contract
    return hashlib.sha256(raw).hexdigest()


def create_contract_lock(contract: Contract | dict[str, Any], approved_by: str, approved_at: datetime | None = None) -> ContractLock:
    """Create a contract lock bound to the canonical validated contract hash."""

    validated = validate_contract(contract)
    return ContractLock.new(
        validated,
        compute_contract_sha256(validated),
        approvedBy=approved_by,
        approvedAt=approved_at,
    )


def compile_ledger_seed(contract: Contract | dict[str, Any], *, approved_by: str | None = None) -> LedgerSeed:
    """Compile deterministic initial ledger state from a validated contract."""

    validated = validate_contract(contract)
    contract_hash = compute_contract_sha256(validated)
    lock = create_contract_lock(validated, approved_by) if approved_by else None
    records = [
        LedgerSprintRecord(
            sprintId=sprint.id,
            state="not_started",
            section=sprint.section,
            title=sprint.title,
            order=sprint.order,
            priority=sprint.priority,
            parallelSafe=sprint.parallelSafe,
            materialType=sprint.materialType,
            objective=sprint.objective,
            dependsOn=list(sprint.dependsOn),
            gates=list(sprint.gates),
            requiredInputs=list(sprint.requiredInputs),
            requiredContext=list(sprint.requiredContext),
            implementationRequirements=list(sprint.implementationRequirements),
            acceptance=list(sprint.acceptance),
            stopConditions=list(sprint.stopConditions),
            evidenceRequired=list(sprint.evidenceRequired),
            allowedPaths=list(sprint.allowedPaths),
            forbiddenPaths=list(sprint.forbiddenPaths),
            review=sprint.review,
            closeout=sprint.closeout,
        )
        for sprint in sorted(validated.sprints, key=lambda s: (s.order, s.id))
    ]
    gate_records = [
        LedgerGateRecord(
            gateId=gate.id,
            type=gate.type,
            owner=gate.owner,
            severity=gate.severity,
            description=gate.description,
            blocksSprintIds=list(gate.blocksSprintIds),
            resolutionCondition=gate.resolutionCondition,
            evidenceRequired=list(gate.evidenceRequired),
            resolved=False,
        )
        for gate in sorted(validated.gates, key=lambda g: g.id)
    ]
    return LedgerSeed(
        contractId=validated.contractId,
        contractVersion=validated.contractVersion,
        contractSha256=contract_hash,
        projectId=validated.project.id,
        contractLock=lock,
        sprints=records,
        gates=gate_records,
    )


def generate_worker_packet(
    contract: Contract | dict[str, Any],
    sprint_id: str,
    *,
    worker_role: str,
    assigned_worker: str,
    packet_id: str | None = None,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> WorkerPacket:
    """Generate a scoped worker packet for a single sprint.

    Workers receive only sprint-bound mission, paths, acceptance ids, commands,
    stop conditions, and output requirements. The master contract is not copied
    into the packet.
    """

    validated = validate_contract(contract)
    sprint = next((item for item in validated.sprints if item.id == sprint_id), None)
    if sprint is None:
        raise ValueError(f"unknown sprint id: {sprint_id}")

    commands = [criterion.command for criterion in sprint.acceptance if criterion.command]
    output_requirements = ["summary", "changed_files", "commands_run", "test_results", "unresolved_blockers"]
    if sprint.review.required:
        output_requirements.append("review_request_materials")

    return WorkerPacket(
        packetId=packet_id or f"{sprint.id}.{worker_role}.001",
        projectId=validated.project.id,
        sprintId=sprint.id,
        workerRole=worker_role,  # type: ignore[arg-type]
        assignedWorker=assigned_worker,
        sessionId=session_id or str(uuid.uuid4()),
        allowedPaths=list(sprint.allowedPaths),
        forbiddenPaths=list(sprint.forbiddenPaths),
        mission=sprint.objective,
        context=context or {},
        acceptanceCriteria=list(sprint.acceptance),
        verificationCommands=commands,
        stopConditions=list(sprint.stopConditions),
        outputRequirements=output_requirements,
    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
