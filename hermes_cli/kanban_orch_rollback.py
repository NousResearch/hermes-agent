"""ORCH V4 Rollback — forward phase state machine with CAS.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §17 (Durable rollback contract).
§13 T43 test contract.

Runtime-independent pure model. No live writer switch, no live DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RollbackError(ValueError):
    """Rollback protocol violation."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# §17 Rollback phases — forward transition, never DB-file replacement
ROLLBACK_PHASES = (
    "gate_submits",
    "fence_draining",
    "workers_stopped",
    "leases_revoked",
    "snapshot_sealed",
    "code_switched",
    "old_writer_receipts_verified",
    "verified",
    "reopened",
)

# Fence stays closed from fence_draining through verified
FENCE_CLOSED_PHASES = frozenset({
    "fence_draining", "workers_stopped", "leases_revoked",
    "snapshot_sealed", "code_switched", "old_writer_receipts_verified", "verified",
})


@dataclass
class RollbackOperation:
    """Durable rollback operation state."""
    operation_id: str
    owner_token_hash: str
    source_live_hash: str
    source_code_hash: str
    target_live_hash: str
    target_code_hash: str
    fence_generation: int = 0
    phase_revision: int = 0
    current_phase: str = "gate_submits"
    completed_phases: list[str] = field(default_factory=list)
    is_active: bool = False  # reopened = True

    # Phase conditions (tracked for CAS)
    phase_conditions: dict[str, dict[str, Any]] = field(default_factory=dict)


def create_rollback(
    *,
    operation_id: str,
    owner_token_hash: str,
    source_live_hash: str,
    source_code_hash: str,
    target_live_hash: str,
    target_code_hash: str,
) -> RollbackOperation:
    """Create a new rollback operation at gate_submits."""
    # §17.2: Compatible preimage must be M0 fence-aware
    # Pre-M0 code can never be a rollback target
    if not source_code_hash:
        raise RollbackError("preimage_missing")
    if not target_code_hash:
        raise RollbackError("target_missing")

    return RollbackOperation(
        operation_id=operation_id,
        owner_token_hash=owner_token_hash,
        source_live_hash=source_live_hash,
        source_code_hash=source_code_hash,
        target_live_hash=target_live_hash,
        target_code_hash=target_code_hash,
        fence_generation=1,  # starts at 1 (fence closed)
        current_phase="gate_submits",
    )


def advance_phase(
    op: RollbackOperation,
    *,
    target_phase: str,
    owner_token_hash: str,
    expected_revision: int,
) -> str:
    """Advance rollback to target_phase with owner + revision CAS.

    Rules:
    - Cannot skip phases (must be next in sequence)
    - Owner token must match
    - Phase revision must match
    - Fence stays closed from fence_draining through verified
    - reopened requires additional David-level token (is_active=True)
    """
    if owner_token_hash != op.owner_token_hash:
        raise RollbackError("owner_token_mismatch")

    if expected_revision != op.phase_revision:
        raise RollbackError("phase_revision_mismatch")

    # Find current and target phase indices
    try:
        current_idx = ROLLBACK_PHASES.index(op.current_phase)
        target_idx = ROLLBACK_PHASES.index(target_phase)
    except ValueError:
        raise RollbackError("invalid_phase")

    # Cannot skip: must advance by exactly 1
    if target_idx != current_idx + 1:
        if target_idx <= current_idx:
            raise RollbackError("phase_rollback_not_allowed")
        raise RollbackError("phase_skip_not_allowed")

    # Fence must stay closed from fence_draining through verified
    if target_phase in FENCE_CLOSED_PHASES and op.fence_generation < 1:
        raise RollbackError("fence_not_closed")

    # reopened requires additional David-level token
    if target_phase == "reopened":
        if op.current_phase != "verified":
            raise RollbackError("reopen_before_verified")
        # In pure model, caller must set is_active=True before calling
        # This is the "additional David-level token" gate
        if not op.is_active:
            raise RollbackError("reopen_requires_additional_token")

    # Phase-specific conditions (§17)
    if target_phase == "fence_draining":
        # Write fence must be closed; dirty requests listed
        if op.fence_generation < 1:
            raise RollbackError("fence_not_closed_for_draining")
    elif target_phase == "workers_stopped":
        # Tracked processes stopped; no new claims
        pass  # Model: caller ensures
    elif target_phase == "leases_revoked":
        # All leases list empty across all boards
        pass  # Model: caller ensures
    elif target_phase == "snapshot_sealed":
        # Sealed board range; board identity verified
        pass  # Model: caller ensures
    elif target_phase == "code_switched":
        # New code is reading old code bytes
        pass  # Model: caller ensures
    elif target_phase == "old_writer_receipts_verified":
        # Verification receipts from all previous writer roles
        pass  # Model: caller ensures
    elif target_phase == "verified":
        # All submitted gates closed and zero unexpected writers
        # Late ACK/event after cutoff is appended, never discarded
        pass  # Model: caller ensures

    # Advance
    op.completed_phases.append(op.current_phase)
    op.current_phase = target_phase
    op.phase_revision += 1

    # reopened releases the fence
    if target_phase == "reopened":
        op.fence_generation = 0  # fence released
        op.is_active = True

    return op.current_phase


def resume_rollback(
    op: RollbackOperation,
    *,
    owner_token_hash: str,
) -> str:
    """§17.1: Crash resumes from durable phase/revision.

    Returns current phase without advancing.
    """
    if owner_token_hash != op.owner_token_hash:
        raise RollbackError("resume_owner_mismatch")
    return op.current_phase


def authorize_reopen(op: RollbackOperation, *, david_token: str) -> None:
    """§17.6: Reopen requires additional David-level token.

    Observer, manifest, transition-write and unknown-delivery invariants
    must pass before reopen is authorized.
    """
    if op.current_phase != "verified":
        raise RollbackError("reopen_before_verified")
    if not david_token:
        raise RollbackError("david_token_missing")
    op.is_active = True  # Mark as authorized


def check_no_early_reopen(op: RollbackOperation) -> bool:
    """§17.1: Fence stays closed from fence_draining through verified.

    Returns True if fence is properly closed (no early reopen).
    """
    if op.current_phase in FENCE_CLOSED_PHASES:
        return op.fence_generation >= 1  # fence must be closed
    if op.current_phase == "gate_submits":
        return True  # not yet started
    if op.current_phase == "reopened":
        return op.fence_generation == 0  # properly released
    return False


def check_preimage_compatible(
    source_code_hash: str,
    target_code_hash: str,
    *,
    is_m0_aware: bool = True,
) -> bool:
    """§17.2: Compatible preimage means M0 fence-aware source hash.

    Pre-M0 code can never be a rollback target.
    """
    if not is_m0_aware:
        return False
    if not source_code_hash or not target_code_hash:
        return False
    return source_code_hash != target_code_hash  # must be different (rollback)


__all__ = [
    "RollbackError", "ROLLBACK_PHASES", "FENCE_CLOSED_PHASES",
    "RollbackOperation", "create_rollback", "advance_phase",
    "resume_rollback", "authorize_reopen", "check_no_early_reopen",
    "check_preimage_compatible",
]
