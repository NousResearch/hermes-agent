"""Public transactional seam for external workers driving the Kanban board.

This module is the public lifecycle surface for an out-of-process worker or
supervisor driving a Kanban task:

* :func:`submit` — validate and lock an existing task attachment named
  ``mas-task-spec.v1.json`` as the immutable spec for a task;
* :func:`connect` — open an initialized board connection without requiring
  consumers to import Kanban's implementation module;
* :func:`read_submitted_attachment` — reload the accepted spec attachment's
  exact bytes;
* :func:`list_ready` — enumerate ready tasks waiting for an external
  claim;
* :func:`claim_external` — atomically claim a ready task for an
  external worker, returning a typed :class:`Lease`;
* :func:`bind_process` — bind a host/pid/pgid/start-token quad before any
  child is started;
* :func:`heartbeat` / :func:`still_owns` — refresh / check the lease;
* :func:`hold_for_recovery` — atomically extend the lease and bump the
  inconclusive-hold counter (uncertainty is NOT absence);
* :func:`put_result` — validate and persist a typed ExecutionResult on the
  run BEFORE finalize;
* :func:`finalize` — close the run with COMPLETE / REQUEUE / BLOCK,
  returning a typed COMMITTED / ALREADY_COMMITTED_SAME_HASH / REJECTED
  outcome;
* :func:`recover_expired` — recover one explicitly-expected run whose
  lease has lapsed, with affirmative process-absence or no-start proof;
* :func:`list_active`, :func:`get_run` — typed lease inspection;
* :func:`read_result` — reload exact result bytes staged before a restart;
* :func:`put_artifact` / :func:`read_artifact` — idempotent named-output
  persistence bound to the active lease.

Design invariants:

* **Immutable spec identity.** The canonical input is exactly one task
  attachment named ``mas-task-spec.v1.json``. Its identity is the triple
  ``(attachment_id, SHA-256 of the exact raw bytes, schema_version =
  "mas-task-spec.v1")``. We never hash normalized title/body or attachment
  lists — the raw bytes ARE the spec.

* **Strict JSON.** Spec and result bytes are decoded with a strict parser
  that rejects invalid UTF-8, BOM, trailing data, duplicate keys,
  NaN/Infinity, a missing or wrong ``schema_version``, extra or missing
  top-level keys, and wrong container types. Callers can NEVER supply a
  hash; the module always computes SHA-256 itself.

* **Per-run typed identity.** Every mutating call carries a :class:`Lease`
  that bundles the expected run id, task id, copied spec identity, opaque
  lease token, and expected lease state. Inside ONE ``BEGIN IMMEDIATE``
  transaction the call validates ALL of them against the run row before
  touching anything. Authorization is never by run id alone.

* **Durable substates.** Bind records a durable ``bound`` substate BEFORE
  any child start ACK. Hold records durable ``recovery_required`` and
  increments the recovery counter. Finalize / recover record ``committed``
  plus the typed result metadata (exact bytes, hash, disposition, block
  kind).

* **Affirmative recovery.** :func:`recover_expired` operates on ONE
  explicitly expected run, requires the lease to be expired, and accepts
  EITHER a typed ``ProcessAbsenceProof`` whose identity matches the bound
  process OR a typed ``NoStartAckProof`` for an unbound run. It NEVER
  performs ``os.kill``-based absence inference, NEVER releases/requeues
  without an affirmative proof, and MAY NOT create a new requeue decision once
  two inconclusive holds are durable. Exact result bytes already staged before
  a crash remain replayable. Uncertainty leaves the run active.

* **Uniform result persistence.** :func:`finalize` stores the SAME result
  metadata (exact bytes, hash, disposition, block kind) for COMPLETE,
  REQUEUE, and BLOCK. :func:`put_result` allows the complete terminal result to
  be persisted AFTER process absence is proven but BEFORE the task transition.
  A same-hash replay also requires an exact disposition + block_kind match;
  anything else is REJECTED.

* **Bridge has no private surface.** The bridge uses only this module.
  It never opens Hermes SQL or touches the filesystem directly; named
  artifacts flow through :func:`put_artifact` / :func:`read_artifact`.

The module has no dependency on a specific external supervisor. It talks to
``kanban_db`` through the same SQLite connection the rest of Hermes uses.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Public API version. Bumped only if the public surface of this module
#: changes shape (new required argument, removed function, new typed return
#: field). Additive optional changes do NOT bump this.
EXTERNAL_WORKER_API_VERSION = 1

#: The worker kind stamped on every run this module creates. Native
#: dispatcher runs leave ``worker_kind`` NULL (treated as ``native``); the
#: distinction lets native reclaim/crash/runtime iteration filter external
#: runs out and lets the public module find its own runs cheaply.
WORKER_KIND = kb.WORKER_KIND_EXTERNAL_MAS
StaleBoardConnection = kb.StaleBoardConnection

#: Canonical schema_version strings. Folded into the spec identity triple
#: and validated by the strict parser.
SPEC_SCHEMA_VERSION = "mas-task-spec.v1"
RESULT_SCHEMA_VERSION = "mas-execution-result.v1"

#: The exact filename of the canonical spec attachment. ``submit`` refuses
#: any other name; ``read_submitted_attachment`` reloads by this name.
SPEC_ATTACHMENT_NAME = kb.EXTERNAL_SPEC_ATTACHMENT_NAME

#: Ceiling on inconclusive holds. Once ``external_recovery_count`` reaches
#: this value, :func:`recover_expired` refuses a newly supplied REQUEUE — the
#: supervisor has lost contact too many times without progress. Exact result
#: bytes durably staged before a crash remain replayable.
RECOVERY_REQUEUE_LIMIT = 2

#: Default lease TTL when callers don't pass an explicit expiry. Mirrors
#: the native dispatcher default so external workers inherit the same
#: lease window.
DEFAULT_LEASE_TTL_SECONDS = kb.DEFAULT_CLAIM_TTL_SECONDS
MAX_RESULT_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = kb.KANBAN_ATTACHMENT_MAX_BYTES

_TASKSPEC_KEYS = frozenset(
    {
        "schema_version",
        "board",
        "repo_key",
        "objective",
        "acceptance_criteria",
        "scope",
        "base_sha",
        "risk",
        "workflow",
        "execution",
        "verification",
        "delivery",
    }
)
_TASKSPEC_NESTED_KEYS = {
    "scope": frozenset({"include", "exclude"}),
    "execution": frozenset(
        {"timeout_seconds", "max_attempts", "max_tokens", "max_cost_usd"}
    ),
    "verification": frozenset(
        {"check_ids", "fresh_reviewer", "security_review"}
    ),
    "delivery": frozenset({"mode", "push", "deploy"}),
}

_RESULT_TOP_KEYS = frozenset({"mas"})
_RESULT_MAS_KEYS = frozenset(
    {
        "schema_version",
        "spec_sha256",
        "submitted_attachment_id",
        "run_id",
        "attempt",
        "outcome",
        "disposition",
        "block_kind",
        "writer",
        "reviewer",
        "process",
        "git",
        "deliverable",
        "checks",
        "usage",
        "failure_signature",
        "summary",
        "artifacts",
    }
)
_RESULT_NESTED_KEYS = {
    "writer": frozenset({"backend", "model"}),
    "reviewer": frozenset({"backend", "model", "verdict"}),
    "process": frozenset({"identity", "termination_status", "absence_proven"}),
    "git": frozenset(
        {
            "base_sha",
            "head_sha",
            "branch",
            "diff_sha256",
            "changed_files",
            "primary_unchanged",
        }
    ),
    "deliverable": frozenset({"kind", "attachment", "sha256"}),
    "usage": frozenset(
        {
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cost_usd",
            "cost_status",
            "source",
        }
    ),
}
_RESULT_IDENTITY_KEYS = frozenset({"host", "pid", "pgid", "start_token"})
_RESULT_CHECK_KEYS = frozenset(
    {"id", "status", "argv", "exit_code", "duration_ms"}
)
_RESULT_ARTIFACT_KEYS = frozenset({"name", "sha256"})
_RESULT_DISPOSITIONS = {
    "COMPLETE": "complete",
    "REQUEUE": "requeue",
    "BLOCK": "block",
}


# ---------------------------------------------------------------------------
# Lease states (durable external state-machine positions on the run)
# ---------------------------------------------------------------------------

#: Just claimed — no process bound yet. ``bind_process`` requires this.
LEASE_ACTIVE = "active"
#: Process identity bound; ``bind_process`` transitioned the run here.
LEASE_BOUND = "bound"
#: Held for recovery; ``hold_for_recovery`` transitioned the run here.
LEASE_HOLDING = "holding"
#: Terminal — ``finalize`` or ``recover_expired`` closed the run.
LEASE_COMMITTED = "committed"

#: Internal durable substate labels (recorded on ``external_substate``).
#: These mirror the lease states plus a typed ``claimed`` value for the
#: post-claim, pre-bind window.
SUBSTATE_CLAIMED = "claimed"
SUBSTATE_BOUND = "bound"
SUBSTATE_HOLDING = "recovery_required"
SUBSTATE_COMMITTED = "committed"

#: Dispositions accepted by finalize / recover.
DISPOSITION_COMPLETE = "COMPLETE"
DISPOSITION_REQUEUE = "REQUEUE"
DISPOSITION_BLOCK = "BLOCK"
VALID_DISPOSITIONS = frozenset({
    DISPOSITION_COMPLETE,
    DISPOSITION_REQUEUE,
    DISPOSITION_BLOCK,
})

#: Finalize outcomes.
FINALIZE_COMMITTED = "COMMITTED"
FINALIZE_ALREADY_COMMITTED_SAME_HASH = "ALREADY_COMMITTED_SAME_HASH"
FINALIZE_REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class ExternalWorkerError(RuntimeError):
    """Base class for all external-worker seam errors."""


class SpecParseError(ExternalWorkerError):
    """Raised when strict JSON parsing or TaskSpec/Result schema validation fails."""


class SpecMutationError(ExternalWorkerError):
    """Raised when ``submit`` is called on a task that is already locked."""


class AttachmentMismatchError(ExternalWorkerError):
    """Raised when an attachment's on-disk bytes drift from the locked hash,
    when its filename is not the canonical spec name, or when its recorded
    size disagrees with the bytes on disk."""


class AttachmentNotOwnedError(ExternalWorkerError):
    """Raised when an attachment id belongs to a different task."""


class ClaimRejected(ExternalWorkerError):
    """Raised when ``claim_external`` cannot transition the task to running."""


class BindMismatch(ExternalWorkerError):
    """Raised when ``bind_process`` is called with a divergent identity quad."""


class NotOwner(ExternalWorkerError):
    """Raised when a mutating caller's lease does not match the run row."""


class LeaseStateError(ExternalWorkerError):
    """Raised when the run's durable lease state disagrees with the caller's
    expected state, or the operation does not accept the caller's state."""


class RecoveryRejected(ExternalWorkerError):
    """Raised when ``recover_expired`` refuses to recover (uncertainty,
    threshold reached, proof mismatch)."""


class ResultRejected(ExternalWorkerError):
    """Raised when a result fails strict JSON / schema / identity validation."""


class ArtifactCollision(ExternalWorkerError):
    """Raised when ``put_artifact`` is called with a name already used for
    different bytes on the same run."""


class ArtifactNotFound(ExternalWorkerError):
    """Raised when ``read_artifact`` is called for a name that doesn't exist."""


# ---------------------------------------------------------------------------
# Typed dataclasses — public API surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecIdentity:
    """Immutable identity triple of a locked spec.

    ``attachment_id`` is the row id of the accepted spec attachment;
    ``spec_hash`` is the SHA-256 of its exact raw bytes at lock time;
    ``schema_version`` is the ``schema_version`` field from inside the
    spec (``mas-task-spec.v1``). All three are recorded on the task row
    by :func:`submit` and copied into ``task_runs`` by :func:`claim_external`.
    """

    attachment_id: int
    spec_hash: str
    schema_version: str


@dataclass(frozen=True)
class Lease:
    """Typed lease / run handle.

    Carries everything a mutating call must validate inside its
    ``BEGIN IMMEDIATE`` transaction: the run id, task id, copied spec
    identity, opaque lease token, the caller's expected lease state,
    the lease expiry, and the 1-based attempt index.
    """

    run_id: int
    task_id: str
    spec: SpecIdentity
    lease_token: str
    lease_state: str
    lease_expires_at: int
    attempt: int
    recovery_count: int = 0
    process: Optional[BoundProcess] = None
    result_hash: Optional[str] = None
    disposition: Optional[str] = None
    block_kind: Optional[str] = None


@dataclass(frozen=True)
class BoundProcess:
    """Per-run external process identity. Bound before any child starts.

    The quad (host, pid, pgid, start_token) is the identity recovery
    matches against. ``start_token`` is an opaque caller-chosen nonce
    (UUID / random hex) that lets the caller prove the bind corresponds
    to *this* launch attempt and not a racing sibling.
    """

    host: str
    pid: int
    pgid: int
    start_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("BoundProcess.host is required")
        if not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("BoundProcess.pid must be a positive int")
        if not isinstance(self.pgid, int) or self.pgid <= 0:
            raise ValueError("BoundProcess.pgid must be a positive int")
        if not isinstance(self.start_token, str) or not self.start_token.strip():
            raise ValueError("BoundProcess.start_token is required")


@dataclass(frozen=True)
class ProcessAbsenceProof:
    """Affirmative, caller-supplied proof that the bound process is gone.

    The identity quad MUST match the run's bound :class:`BoundProcess`.
    ``evidence`` is a free-form short string (e.g. ``"waitpid:9"``,
    ``"container:exited"``) the caller uses to justify the claim. This
    module NEVER performs ``os.kill``-based absence inference — the
    caller must supply this proof affirmatively.
    """

    host: str
    pid: int
    pgid: int
    start_token: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("ProcessAbsenceProof.host is required")
        if not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("ProcessAbsenceProof.pid must be a positive int")
        if not isinstance(self.pgid, int) or self.pgid <= 0:
            raise ValueError("ProcessAbsenceProof.pgid must be a positive int")
        if not isinstance(self.start_token, str) or not self.start_token.strip():
            raise ValueError("ProcessAbsenceProof.start_token is required")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("ProcessAbsenceProof.evidence is required")


@dataclass(frozen=True)
class NoStartAckProof:
    """Affirmative proof that no child was ever started for this run.

    Used by :func:`recover_expired` for a run that was claimed but never
    had :func:`bind_process` called (so there is no process to prove
    absent). The supervisor asserts it never ACK'd a child start.
    """

    run_id: int
    task_id: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, int) or self.run_id <= 0:
            raise ValueError("NoStartAckProof.run_id must be a positive int")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("NoStartAckProof.task_id is required")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("NoStartAckProof.evidence is required")


@dataclass(frozen=True)
class RecoveryHoldProof:
    """Typed bounded proof accepted by :func:`hold_for_recovery`.

    The supervisor has lost contact with the worker but is NOT yet
    declaring it dead — it just needs a bounded window to investigate.
    Carries the run/task identity and a short evidence string. If a
    process was bound, ``bound`` must match the run's bound identity.
    """

    run_id: int
    task_id: str
    bound: Optional[BoundProcess]
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, int) or self.run_id <= 0:
            raise ValueError("RecoveryHoldProof.run_id must be a positive int")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("RecoveryHoldProof.task_id is required")
        if self.bound is not None and not isinstance(self.bound, BoundProcess):
            raise TypeError("RecoveryHoldProof.bound must be BoundProcess or None")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("RecoveryHoldProof.evidence is required")


@dataclass(frozen=True)
class ExecutionResult:
    """Typed ExecutionResult. ``result_bytes`` are canonical strict JSON
    (schema ``mas-execution-result.v1``). The module computes the SHA-256
    itself; callers must never supply a hash.

    ``disposition`` / ``block_kind`` duplicate the JSON fields so the
    public API has them as first-class typed values. They MUST match the
    JSON exactly on :func:`finalize` / :func:`put_result`.
    """

    disposition: str
    block_kind: Optional[str]
    result_bytes: bytes
    run_id: int
    task_id: str
    spec: SpecIdentity

    def __post_init__(self) -> None:
        if self.disposition not in VALID_DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {sorted(VALID_DISPOSITIONS)}, "
                f"got {self.disposition!r}"
            )
        if self.disposition == DISPOSITION_BLOCK:
            if self.block_kind not in kb.VALID_BLOCK_KINDS:
                raise ValueError(
                    f"block_kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} "
                    f"for BLOCK, got {self.block_kind!r}"
                )
        else:
            if self.block_kind is not None:
                raise ValueError(
                    f"block_kind must be None for disposition={self.disposition!r}"
                )
        if not isinstance(self.result_bytes, (bytes, bytearray)):
            raise TypeError("result_bytes must be bytes")
        if not isinstance(self.run_id, int) or self.run_id <= 0:
            raise ValueError("run_id must be a positive int")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id is required")
        if not isinstance(self.spec, SpecIdentity):
            raise TypeError("spec must be SpecIdentity")


@dataclass(frozen=True)
class PersistedResult:
    """Exact durable result bytes and their terminal identity tuple."""

    run_id: int
    task_id: str
    result_bytes: bytes
    result_hash: str
    disposition: str
    block_kind: Optional[str]


@dataclass(frozen=True)
class FinalizeOutcome:
    """Typed return value of :func:`finalize`."""

    status: str
    run_id: int
    task_id: str
    result_hash: str
    prior_hash: Optional[str] = None
    disposition: Optional[str] = None
    block_kind: Optional[str] = None

    @property
    def committed(self) -> bool:
        return self.status == FINALIZE_COMMITTED

    @property
    def already_committed(self) -> bool:
        return self.status == FINALIZE_ALREADY_COMMITTED_SAME_HASH


@dataclass(frozen=True)
class RecoveredRun:
    """Typed return value of :func:`recover_expired` for one reclaimed run."""

    run_id: int
    task_id: str
    spec: SpecIdentity
    disposition: str
    block_kind: Optional[str]
    recovery_count: int
    requeued: bool
    blocked: bool


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a named external artifact persisted on a run."""

    run_id: int
    name: str
    sha256: str
    size: int


# ---------------------------------------------------------------------------
# Strict JSON parsing + schema validation
# ---------------------------------------------------------------------------


def _reject_constant(value: str) -> None:
    """``parse_constant`` hook: refuse NaN / Infinity / -Infinity."""
    raise SpecParseError(f"numeric constant not allowed: {value!r}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    """``object_pairs_hook``: build a dict but reject duplicate keys."""
    seen: set[str] = set()
    out: dict = {}
    for k, v in pairs:
        if not isinstance(k, str):
            raise SpecParseError(f"non-string JSON key: {k!r}")
        if k in seen:
            raise SpecParseError(f"duplicate key: {k!r}")
        seen.add(k)
        out[k] = v
    return out


def _decode_strict_json(raw: bytes) -> Any:
    """Decode bytes as strict JSON.

    Rejects: non-bytes input, empty input, UTF-8 BOM (both byte-mark and
    decoded codepoint), invalid UTF-8, trailing non-whitespace data,
    duplicate keys, NaN / Infinity / -Infinity, and any ``JSONDecodeError``.
    Returns the parsed top-level value (caller validates container type).
    """
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if not isinstance(raw, bytes):
        raise SpecParseError("input must be bytes")
    if not raw:
        raise SpecParseError("empty input")
    # Reject UTF-8 / UTF-16 BOMs at the byte level.
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SpecParseError("UTF-8 BOM not allowed")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        raise SpecParseError("UTF-16 BOM not allowed")
    # Strict UTF-8 decode (rejects invalid UTF-8 sequences).
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecParseError(f"invalid UTF-8: {exc}") from exc
    # Reject a BOM that slipped through as the U+FEFF codepoint.
    if text and text[0] == "\ufeff":
        raise SpecParseError("UTF-8 BOM not allowed")
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise SpecParseError(f"invalid JSON: {exc}") from exc


def _ensure_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise SpecParseError(f"{label} must be a string, got {type(value).__name__}")
    return value


def _ensure_nullable_str(value: Any, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _ensure_str(value, label=label)


def _ensure_int(value: Any, *, label: str) -> int:
    # JSON floats that happen to be integer-valued are NOT accepted —
    # an integer field means an integer token in the source.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecParseError(f"{label} must be an integer, got {type(value).__name__}")
    return value


def _ensure_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SpecParseError(f"{label} must be a boolean")
    return value


def _ensure_object(
    value: Any, *, label: str, keys: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecParseError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise SpecParseError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _ensure_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise SpecParseError(f"{label} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise SpecParseError(f"{label} must contain only strings")
    return value


def _validate_taskspec(obj: Any) -> str:
    spec = _ensure_object(obj, label="TaskSpec", keys=_TASKSPEC_KEYS)
    schema_version = _ensure_str(spec["schema_version"], label="schema_version")
    if schema_version != SPEC_SCHEMA_VERSION:
        raise SpecParseError(
            f"TaskSpec schema_version must be {SPEC_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    for field_name in ("board", "repo_key", "objective", "base_sha", "risk", "workflow"):
        if not _ensure_str(spec[field_name], label=field_name):
            raise SpecParseError(f"TaskSpec.{field_name} must not be empty")
    _ensure_str_list(spec["acceptance_criteria"], label="acceptance_criteria")

    scope = _ensure_object(
        spec["scope"], label="scope", keys=_TASKSPEC_NESTED_KEYS["scope"]
    )
    _ensure_str_list(scope["include"], label="scope.include")
    _ensure_str_list(scope["exclude"], label="scope.exclude")

    execution = _ensure_object(
        spec["execution"],
        label="execution",
        keys=_TASKSPEC_NESTED_KEYS["execution"],
    )
    for field_name in ("timeout_seconds", "max_attempts", "max_tokens"):
        _ensure_int(execution[field_name], label=f"execution.{field_name}")
    cost = execution["max_cost_usd"]
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
    ):
        raise SpecParseError("execution.max_cost_usd must be finite numeric or null")

    verification = _ensure_object(
        spec["verification"],
        label="verification",
        keys=_TASKSPEC_NESTED_KEYS["verification"],
    )
    _ensure_str_list(verification["check_ids"], label="verification.check_ids")
    _ensure_bool(verification["fresh_reviewer"], label="verification.fresh_reviewer")
    _ensure_bool(verification["security_review"], label="verification.security_review")

    delivery = _ensure_object(
        spec["delivery"], label="delivery", keys=_TASKSPEC_NESTED_KEYS["delivery"]
    )
    _ensure_str(delivery["mode"], label="delivery.mode")
    _ensure_bool(delivery["push"], label="delivery.push")
    _ensure_bool(delivery["deploy"], label="delivery.deploy")
    return schema_version


def _validate_execution_result(
    obj: Any,
    *,
    expected_run_id: int,
    expected_spec: SpecIdentity,
) -> dict[str, Any]:
    """Validate a parsed mas-execution-result.v1 object.

    Returns ``(disposition, block_kind)``. Raises on any contract violation.
    """
    top = _ensure_object(obj, label="ExecutionResult", keys=_RESULT_TOP_KEYS)
    mas = _ensure_object(top["mas"], label="ExecutionResult.mas", keys=_RESULT_MAS_KEYS)
    schema_version = _ensure_str(mas["schema_version"], label="mas.schema_version")
    if schema_version != RESULT_SCHEMA_VERSION:
        raise SpecParseError(
            f"ExecutionResult schema_version must be {RESULT_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    run_id = _ensure_int(mas["run_id"], label="mas.run_id")
    if run_id != expected_run_id:
        raise SpecParseError(
            f"ExecutionResult.run_id {run_id} != expected {expected_run_id}"
        )
    spec_attachment_id = _ensure_int(
        mas["submitted_attachment_id"], label="mas.submitted_attachment_id"
    )
    if spec_attachment_id != expected_spec.attachment_id:
        raise SpecParseError(
            f"ExecutionResult.spec_attachment_id {spec_attachment_id} != "
            f"expected {expected_spec.attachment_id}"
        )
    spec_hash = _ensure_str(mas["spec_sha256"], label="mas.spec_sha256")
    if spec_hash != expected_spec.spec_hash:
        raise SpecParseError(f"ExecutionResult.spec_hash {spec_hash!r} != expected")
    attempt = _ensure_int(mas["attempt"], label="mas.attempt")
    if attempt <= 0:
        raise SpecParseError("mas.attempt must be positive")
    outcome = _ensure_str(mas["outcome"], label="mas.outcome")
    if outcome not in {
        "completed",
        "failed",
        "aborted_before_start",
        "recovered_after_crash",
    }:
        raise SpecParseError(f"unsupported mas.outcome: {outcome!r}")
    raw_disposition = _ensure_str(mas["disposition"], label="mas.disposition")
    reverse_dispositions = {value: key for key, value in _RESULT_DISPOSITIONS.items()}
    disposition = reverse_dispositions.get(raw_disposition)
    if disposition is None:
        raise SpecParseError(
            f"ExecutionResult.mas.disposition must be one of "
            f"{sorted(reverse_dispositions)}, got {raw_disposition!r}"
        )
    raw_block_kind = mas["block_kind"]
    if raw_block_kind is None:
        block_kind: Optional[str] = None
    elif isinstance(raw_block_kind, str):
        block_kind = raw_block_kind
    else:
        raise SpecParseError(
            f"ExecutionResult.block_kind must be a string or null, "
            f"got {type(raw_block_kind).__name__}"
        )
    if disposition == DISPOSITION_BLOCK:
        if block_kind not in kb.VALID_BLOCK_KINDS:
            raise SpecParseError(
                f"ExecutionResult.block_kind must be one of "
                f"{sorted(kb.VALID_BLOCK_KINDS)} for BLOCK, got {block_kind!r}"
            )
    else:
        if block_kind is not None:
            raise SpecParseError(
                f"ExecutionResult.block_kind must be null for "
                f"disposition={disposition!r}"
            )
    writer = _ensure_object(
        mas["writer"], label="mas.writer", keys=_RESULT_NESTED_KEYS["writer"]
    )
    _ensure_nullable_str(writer["backend"], label="mas.writer.backend")
    _ensure_nullable_str(writer["model"], label="mas.writer.model")
    reviewer = _ensure_object(
        mas["reviewer"], label="mas.reviewer", keys=_RESULT_NESTED_KEYS["reviewer"]
    )
    _ensure_nullable_str(reviewer["backend"], label="mas.reviewer.backend")
    _ensure_nullable_str(reviewer["model"], label="mas.reviewer.model")
    _ensure_nullable_str(reviewer["verdict"], label="mas.reviewer.verdict")

    process = _ensure_object(
        mas["process"], label="mas.process", keys=_RESULT_NESTED_KEYS["process"]
    )
    identity = process["identity"]
    if identity is not None:
        identity = _ensure_object(
            identity, label="mas.process.identity", keys=_RESULT_IDENTITY_KEYS
        )
        _ensure_str(identity["host"], label="mas.process.identity.host")
        if _ensure_int(identity["pid"], label="mas.process.identity.pid") <= 0:
            raise SpecParseError("mas.process.identity.pid must be positive")
        if _ensure_int(identity["pgid"], label="mas.process.identity.pgid") <= 0:
            raise SpecParseError("mas.process.identity.pgid must be positive")
        _ensure_str(identity["start_token"], label="mas.process.identity.start_token")
    termination_status = _ensure_str(
        process["termination_status"], label="mas.process.termination_status"
    )
    absence_proven = _ensure_bool(
        process["absence_proven"], label="mas.process.absence_proven"
    )

    git = _ensure_object(mas["git"], label="mas.git", keys=_RESULT_NESTED_KEYS["git"])
    _ensure_str(git["base_sha"], label="mas.git.base_sha")
    for field_name in ("head_sha", "branch", "diff_sha256"):
        _ensure_nullable_str(git[field_name], label=f"mas.git.{field_name}")
    _ensure_str_list(git["changed_files"], label="mas.git.changed_files")
    _ensure_bool(git["primary_unchanged"], label="mas.git.primary_unchanged")

    deliverable = _ensure_object(
        mas["deliverable"],
        label="mas.deliverable",
        keys=_RESULT_NESTED_KEYS["deliverable"],
    )
    for field_name in ("kind", "attachment", "sha256"):
        _ensure_nullable_str(deliverable[field_name], label=f"mas.deliverable.{field_name}")

    checks = mas["checks"]
    if not isinstance(checks, list):
        raise SpecParseError("mas.checks must be an array")
    for index, check in enumerate(checks):
        check = _ensure_object(
            check, label=f"mas.checks[{index}]", keys=_RESULT_CHECK_KEYS
        )
        _ensure_str(check["id"], label=f"mas.checks[{index}].id")
        status = _ensure_str(check["status"], label=f"mas.checks[{index}].status")
        if status not in {"passed", "failed", "infra_error"}:
            raise SpecParseError(f"unsupported check status: {status!r}")
        _ensure_str_list(check["argv"], label=f"mas.checks[{index}].argv")
        if check["exit_code"] is not None:
            _ensure_int(check["exit_code"], label=f"mas.checks[{index}].exit_code")
        if _ensure_int(check["duration_ms"], label=f"mas.checks[{index}].duration_ms") < 0:
            raise SpecParseError("check duration_ms must be non-negative")

    usage = _ensure_object(
        mas["usage"], label="mas.usage", keys=_RESULT_NESTED_KEYS["usage"]
    )
    for field_name in ("input_tokens", "output_tokens", "reasoning_tokens"):
        value = usage[field_name]
        if value is not None and _ensure_int(value, label=f"mas.usage.{field_name}") < 0:
            raise SpecParseError(f"mas.usage.{field_name} must be non-negative")
    cost = usage["cost_usd"]
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
    ):
        raise SpecParseError("mas.usage.cost_usd must be finite numeric or null")
    _ensure_str(usage["cost_status"], label="mas.usage.cost_status")
    _ensure_str(usage["source"], label="mas.usage.source")

    _ensure_nullable_str(mas["failure_signature"], label="mas.failure_signature")
    summary = _ensure_str(mas["summary"], label="mas.summary")
    artifacts = mas["artifacts"]
    if not isinstance(artifacts, list):
        raise SpecParseError("mas.artifacts must be an array")
    for index, artifact in enumerate(artifacts):
        artifact = _ensure_object(
            artifact, label=f"mas.artifacts[{index}]", keys=_RESULT_ARTIFACT_KEYS
        )
        _ensure_str(artifact["name"], label=f"mas.artifacts[{index}].name")
        _ensure_str(artifact["sha256"], label=f"mas.artifacts[{index}].sha256")
    return {
        "disposition": disposition,
        "block_kind": block_kind,
        "summary": summary,
        "attempt": attempt,
        "outcome": outcome,
        "process_identity": identity,
        "termination_status": termination_status,
        "absence_proven": absence_proven,
        "parsed": top,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_attachment_row(conn: sqlite3.Connection, attachment_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, task_id, filename, stored_path, size, spec_hash "
        "FROM task_attachments WHERE id = ?",
        (int(attachment_id),),
    ).fetchone()
    if row is None:
        raise AttachmentMismatchError(f"attachment {attachment_id} does not exist")
    return row


def _read_attachment_bytes(row: sqlite3.Row) -> bytes:
    p = Path(row["stored_path"])
    if not p.is_file():
        raise AttachmentMismatchError(
            f"attachment {row['id']}: blob missing at {row['stored_path']!r}"
        )
    file_size = p.stat().st_size
    if file_size > kb.KANBAN_ATTACHMENT_MAX_BYTES:
        raise AttachmentMismatchError(
            f"attachment {row['id']}: exceeds the attachment size limit"
        )
    if int(row["size"] or 0) != file_size:
        raise AttachmentMismatchError(
            f"attachment {row['id']}: size drifted "
            f"(row={row['size']}, disk={file_size})"
        )
    raw = p.read_bytes()
    if file_size != len(raw):
        raise AttachmentMismatchError(
            f"attachment {row['id']}: size changed while reading"
        )
    return raw


def _lease_from_run_row(row: sqlite3.Row) -> Lease:
    """Build a :class:`Lease` from a ``task_runs`` row using its CURRENT
    lease state. Used by read paths (:func:`list_active`, :func:`get_run`)."""
    spec = SpecIdentity(
        attachment_id=int(row["external_spec_attachment_id"]),
        spec_hash=row["external_spec_hash"],
        schema_version=row["external_spec_schema"],
    )
    process = None
    if row["external_host"] is not None:
        process = BoundProcess(
            host=row["external_host"],
            pid=int(row["external_pid"]),
            pgid=int(row["external_pgid"]),
            start_token=row["external_start_token"],
        )
    return Lease(
        run_id=int(row["id"]),
        task_id=row["task_id"],
        spec=spec,
        lease_token=row["claim_lock"] or "",
        lease_state=row["external_lease_state"] or LEASE_ACTIVE,
        lease_expires_at=int(row["claim_expires"]) if row["claim_expires"] else 0,
        attempt=int(row["external_attempt"]),
        recovery_count=int(row["external_recovery_count"] or 0),
        process=process,
        result_hash=row["external_result_hash"],
        disposition=row["external_terminal_disposition"],
        block_kind=row["external_block_kind"],
    )


def _check_lease_identity(
    conn: sqlite3.Connection,
    lease: Lease,
    *,
    check_active_task: bool = True,
) -> sqlite3.Row:
    """Validate the identity parts of a lease (run id, task id, worker_kind,
    copied spec identity, lease token). Does NOT check lease_state or
    ended_at — callers that need an idempotency path on already-terminal
    runs use this then route on ``external_substate`` themselves.
    """
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?",
        (int(lease.run_id),),
    ).fetchone()
    if row is None:
        raise NotOwner(f"unknown run {lease.run_id}")
    if row["task_id"] != lease.task_id:
        raise NotOwner(
            f"run {lease.run_id} task_id mismatch: caller={lease.task_id!r} "
            f"db={row['task_id']!r}"
        )
    if row["worker_kind"] != WORKER_KIND:
        raise NotOwner(
            f"run {lease.run_id} is not an external-mas run "
            f"(worker_kind={row['worker_kind']!r})"
        )
    if int(row["external_spec_attachment_id"] or 0) != lease.spec.attachment_id:
        raise NotOwner(f"run {lease.run_id} spec attachment_id mismatch")
    if row["external_spec_hash"] != lease.spec.spec_hash:
        raise NotOwner(f"run {lease.run_id} spec_hash mismatch")
    if row["external_spec_schema"] != lease.spec.schema_version:
        raise NotOwner(f"run {lease.run_id} spec_schema mismatch")
    if int(row["external_attempt"] or 0) != lease.attempt:
        raise NotOwner(f"run {lease.run_id} attempt mismatch")
    if row["claim_lock"] != lease.lease_token:
        raise NotOwner(f"run {lease.run_id} lease_token mismatch")
    if check_active_task:
        if row["claim_expires"] is None or int(row["claim_expires"]) != int(
            lease.lease_expires_at
        ):
            raise NotOwner(f"run {lease.run_id} lease expiry mismatch")
        task = conn.execute(
            "SELECT status, current_run_id, claim_lock, claim_expires "
            "FROM tasks WHERE id = ?",
            (lease.task_id,),
        ).fetchone()
        if (
            task is None
            or task["status"] != "running"
            or int(task["current_run_id"] or 0) != lease.run_id
            or task["claim_lock"] != lease.lease_token
            or int(task["claim_expires"] or 0) != lease.lease_expires_at
        ):
            raise NotOwner(f"task {lease.task_id} no longer matches run lease")
    return row


def _validate_lease(
    conn: sqlite3.Connection,
    lease: Lease,
    *,
    allowed_states: frozenset[str],
    require_unexpired: bool = True,
) -> sqlite3.Row:
    """Validate every lease field against the DB inside the caller's txn.

    Single source of truth for "does this lease authorise this op?":
    run id, task id, worker_kind, ended_at IS NULL, copied spec identity
    (attachment id + hash + schema), opaque lease token (== ``claim_lock``),
    and the caller's expected lease state (must match the durable
    ``external_lease_state`` AND be in this op's allowed set).

    Returns the run row for the caller to use.
    """
    row = _check_lease_identity(conn, lease)
    if row["ended_at"] is not None:
        raise NotOwner(f"run {lease.run_id} already ended")
    if require_unexpired and lease.lease_expires_at <= int(time.time()):
        raise NotOwner(f"run {lease.run_id} lease expired")
    if lease.lease_state not in allowed_states:
        raise LeaseStateError(
            f"operation requires lease state in {sorted(allowed_states)}, "
            f"caller expected {lease.lease_state!r}"
        )
    db_state = row["external_lease_state"] or LEASE_ACTIVE
    if db_state != lease.lease_state:
        raise LeaseStateError(
            f"run {lease.run_id} lease state mismatch: "
            f"caller expected {lease.lease_state!r} db={db_state!r}"
        )
    return row


def _safe_artifact_name(raw: str) -> str:
    """Validate a single safe basename without silently rewriting it."""
    if not isinstance(raw, str) or raw != raw.strip():
        raise ValueError("artifact name must be a non-empty basename")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", raw):
        raise ValueError("artifact name contains an unsafe character")
    return raw


def _external_artifacts_dir(conn: sqlite3.Connection, run_id: int) -> Path:
    """Directory for named artifacts, using only an integer run id."""
    base = (
        kb.attachments_root_for_connection(conn)
        / "external_artifacts"
        / f"run-{int(run_id)}"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _validate_result_artifacts(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    facts: dict[str, Any],
) -> None:
    """Verify every result-declared artifact is durable and byte-identical."""
    mas = facts["parsed"]["mas"]
    declared: dict[str, str] = {}
    for artifact in mas["artifacts"]:
        try:
            name = _safe_artifact_name(artifact["name"])
        except ValueError as exc:
            raise ResultRejected(f"invalid declared artifact name: {exc}") from exc
        sha = artifact["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ResultRejected(
                f"declared artifact {name!r} has an invalid SHA-256"
            )
        if name in declared:
            raise ResultRejected(f"artifact {name!r} is declared more than once")
        declared[name] = sha

    deliverable = mas["deliverable"]
    attachment = deliverable["attachment"]
    deliverable_sha = deliverable["sha256"]
    if (attachment is None) != (deliverable_sha is None):
        raise ResultRejected(
            "deliverable attachment and sha256 must both be set or both be null"
        )
    if attachment is not None:
        try:
            attachment = _safe_artifact_name(attachment)
        except ValueError as exc:
            raise ResultRejected(f"invalid deliverable artifact name: {exc}") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", deliverable_sha):
            raise ResultRejected("deliverable has an invalid SHA-256")
        prior = declared.get(attachment)
        if prior is not None and prior != deliverable_sha:
            raise ResultRejected(
                f"deliverable {attachment!r} conflicts with its artifact declaration"
            )
        declared[attachment] = deliverable_sha

    artifact_root = (
        kb.attachments_root_for_connection(conn) / "external_artifacts"
    ).resolve()
    for name, expected_sha in declared.items():
        row = conn.execute(
            "SELECT stored_path, sha256, size FROM task_external_artifacts "
            "WHERE run_id = ? AND name = ?",
            (int(run_id), name),
        ).fetchone()
        if row is None:
            raise ResultRejected(
                f"declared artifact {name!r} is not persisted on run {run_id}"
            )
        if row["sha256"] != expected_sha:
            raise ResultRejected(
                f"declared artifact {name!r} hash does not match its durable row"
            )
        path = Path(row["stored_path"])
        try:
            path.resolve().relative_to(artifact_root)
        except (ValueError, OSError) as exc:
            raise ResultRejected(
                f"declared artifact {name!r} escaped the artifact root"
            ) from exc
        if not path.is_file():
            raise ResultRejected(f"declared artifact {name!r} blob is missing")
        raw = path.read_bytes()
        if len(raw) != int(row["size"]) or _sha256_hex(raw) != expected_sha:
            raise ResultRejected(
                f"declared artifact {name!r} bytes do not match its SHA-256"
            )


def _outcome_for_disposition(disposition: str) -> str:
    return {
        DISPOSITION_COMPLETE: "completed",
        DISPOSITION_REQUEUE: "requeued",
        DISPOSITION_BLOCK: "blocked",
    }[disposition]


def _transition_task_for_terminal(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    disposition: str,
    block_kind: Optional[str],
    summary: str,
    now: int,
) -> None:
    """Transition the task row to match the run's terminal disposition.

    Called inside the enclosing write_txn so the run-row update, task-row
    transition, and event append are one atomic unit. COMPLETE -> done,
    REQUEUE -> ready, BLOCK -> blocked / triage with typed block kind.
    """
    if disposition == DISPOSITION_COMPLETE:
        conn.execute(
            "UPDATE tasks "
            "SET status = 'done', result = ?, completed_at = ?, "
            "    claim_lock = NULL, claim_expires = NULL, "
            "    worker_pid = NULL, current_run_id = NULL, "
            "    block_kind = NULL, block_recurrences = 0 "
            "WHERE id = ?",
            (summary, now, task_id),
        )
    elif disposition == DISPOSITION_REQUEUE:
        conn.execute(
            "UPDATE tasks "
            "SET status = 'ready', claim_lock = NULL, "
            "    claim_expires = NULL, worker_pid = NULL, "
            "    current_run_id = NULL "
            "WHERE id = ?",
            (task_id,),
        )
    else:  # DISPOSITION_BLOCK
        prev = conn.execute(
            "SELECT block_kind, block_recurrences FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        prev_kind = prev["block_kind"] if prev and prev["block_kind"] else None
        prev_rec = (
            int(prev["block_recurrences"])
            if prev and prev["block_recurrences"] is not None
            else 0
        )
        same_cause = prev_kind == block_kind
        new_rec = prev_rec + 1 if same_cause else 1
        routed = "triage" if new_rec >= kb.BLOCK_RECURRENCE_LIMIT else "blocked"
        conn.execute(
            "UPDATE tasks "
            "SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "    worker_pid = NULL, current_run_id = NULL, "
            "    block_kind = ?, block_recurrences = ? "
            "WHERE id = ?",
            (routed, block_kind, new_rec, task_id),
        )


# ---------------------------------------------------------------------------
# Public API — submit / read_submitted_attachment / list_ready
# ---------------------------------------------------------------------------


def connect(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> sqlite3.Connection:
    """Open an initialized Kanban board through the public worker seam.

    External-worker integrations should import this module only. The optional
    ``db_path`` is useful for isolated tests; production callers normally pass
    an allowlisted ``board`` or rely on Hermes' standard board resolution.
    """
    return kb.connect(db_path, board=board)


def submit(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    spec_attachment_id: int,
) -> SpecIdentity:
    """Validate and lock the spec attachment onto ``task_id``.

    The canonical input is exactly one existing task attachment named
    :data:`SPEC_ATTACHMENT_NAME` (``mas-task-spec.v1.json``). ``submit``
    validates the bytes as strict JSON, parses them as a TaskSpec, computes
    the SHA-256 of the EXACT raw bytes (never a normalized form), and
    atomically records on the task row: the accepted attachment id, the
    spec hash, the schema version, and the lock time, then routes the task
    to Hermes-native ``ready`` or dependency-gated ``todo``. The task /
    attachment may already exist (the caller is
    expected to have created them via :func:`kanban_db.create_task` +
    :func:`kanban_db.store_attachment_bytes`).

    Refuses an already-locked task (``external_spec_hash`` IS NOT NULL)
    with :class:`SpecMutationError`. Does NOT touch task title/body — the
    spec attachment is the sole authoritative input from this point on.

    Returns the immutable :class:`SpecIdentity`.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id is required")
    if not isinstance(spec_attachment_id, int) or spec_attachment_id <= 0:
        raise ValueError("spec_attachment_id must be a positive int")

    with kb._board_lifecycle_write_txn(conn):
        task_row = conn.execute(
            "SELECT id, status, claim_lock, current_run_id, "
            "       external_spec_hash, external_spec_attachment_id, "
            "       external_spec_schema "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None:
            raise ExternalWorkerError(f"task {task_id} does not exist")
        if task_row["external_spec_hash"] is not None:
            raise SpecMutationError(
                f"task {task_id} already has an immutable external spec "
                f"(hash={task_row['external_spec_hash'][:12]}…); "
                "re-submit is refused"
            )
        if task_row["status"] not in {"triage", "todo"}:
            raise ExternalWorkerError(
                f"task {task_id} must be in triage or todo before submit "
                f"(status={task_row['status']!r})"
            )
        if task_row["claim_lock"] is not None or task_row["current_run_id"] is not None:
            raise ExternalWorkerError(f"task {task_id} already has execution state")
        # Resolve the requested row before checking the canonical-count
        # invariant so cross-task ids retain their precise ownership error.
        att = _read_attachment_row(conn, spec_attachment_id)
        if att["task_id"] != task_id:
            raise AttachmentNotOwnedError(
                f"attachment {spec_attachment_id} belongs to task "
                f"{att['task_id']}, not {task_id}"
            )
        canonical = conn.execute(
            "SELECT id FROM task_attachments "
            "WHERE task_id = ? AND filename = ? ORDER BY id",
            (task_id, SPEC_ATTACHMENT_NAME),
        ).fetchall()
        if len(canonical) != 1 or int(canonical[0]["id"]) != spec_attachment_id:
            raise AttachmentMismatchError(
                f"task {task_id} must have exactly one canonical spec attachment"
            )
        if att["filename"] != SPEC_ATTACHMENT_NAME:
            raise AttachmentMismatchError(
                f"attachment {spec_attachment_id}: filename must be "
                f"{SPEC_ATTACHMENT_NAME!r}, got {att['filename']!r}"
            )
        raw = _read_attachment_bytes(att)
        # Strict JSON + TaskSpec schema validation. Any violation raises
        # and the enclosing write_txn rolls back, leaving the task row
        # untouched (atomic submit rollback).
        obj = _decode_strict_json(raw)
        schema_version = _validate_taskspec(obj)
        # Compute the spec hash from the exact raw bytes.
        spec_hash = _sha256_hex(raw)
        unfinished_parent = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived') "
            "LIMIT 1",
            (task_id,),
        ).fetchone()
        next_status = "todo" if unfinished_parent is not None else "ready"
        # Atomically record identity and route through Hermes-native lanes.
        now = int(time.time())
        cur = conn.execute(
            "UPDATE tasks "
            "SET external_spec_attachment_id = ?, "
            "    external_spec_hash = ?, "
            "    external_spec_schema = ?, "
            "    external_spec_locked_at = ?, status = ? "
            "WHERE id = ? AND status IN ('triage', 'todo') "
            "  AND claim_lock IS NULL AND current_run_id IS NULL "
            "  AND external_spec_hash IS NULL",
            (
                int(spec_attachment_id),
                spec_hash,
                schema_version,
                now,
                next_status,
                task_id,
            ),
        )
        if cur.rowcount != 1:
            raise ExternalWorkerError(f"task {task_id} changed during submit")
        # Stamp the attachment's spec_hash column for immutability —
        # delete_attachment refuses to remove it once locked.
        conn.execute(
            "UPDATE task_attachments SET spec_hash = ? WHERE id = ?",
            (spec_hash, int(spec_attachment_id)),
        )
        kb._append_event(
            conn,
            task_id,
            "mas_spec_submitted",
            {
                "attachment_id": int(spec_attachment_id),
                "spec_hash": spec_hash,
                "schema": schema_version,
                "size": len(raw),
                "status": next_status,
            },
        )
        return SpecIdentity(
            attachment_id=int(spec_attachment_id),
            spec_hash=spec_hash,
            schema_version=schema_version,
        )


def read_submitted_attachment(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    attachment_id: int,
) -> bytes:
    """Reload the accepted spec attachment and return its exact bytes.

    Verifies ownership (attachment belongs to ``task_id``), filename
    (== ``mas-task-spec.v1.json``), on-disk size (== recorded size), and
    SHA-256 (== locked ``external_spec_hash`` AND == attachment row's
    ``spec_hash``). Returns the exact raw bytes.

    Raises :class:`AttachmentMismatchError` on any drift.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id is required")
    if not isinstance(attachment_id, int) or attachment_id <= 0:
        raise ValueError("attachment_id must be a positive int")

    att = _read_attachment_row(conn, attachment_id)
    if att["task_id"] != task_id:
        raise AttachmentNotOwnedError(
            f"attachment {attachment_id} belongs to task {att['task_id']}, "
            f"not {task_id}"
        )
    if att["filename"] != SPEC_ATTACHMENT_NAME:
        raise AttachmentMismatchError(
            f"attachment {attachment_id}: filename must be "
            f"{SPEC_ATTACHMENT_NAME!r}, got {att['filename']!r}"
        )
    task_row = conn.execute(
        "SELECT external_spec_attachment_id, external_spec_hash, "
        "       external_spec_schema "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if task_row is None:
        raise ExternalWorkerError(f"task {task_id} does not exist")
    if (
        task_row["external_spec_attachment_id"] is None
        or int(task_row["external_spec_attachment_id"]) != attachment_id
    ):
        raise AttachmentMismatchError(
            f"attachment {attachment_id} is not the accepted spec for task {task_id}"
        )
    raw = _read_attachment_bytes(att)
    actual = _sha256_hex(raw)
    locked_hash = task_row["external_spec_hash"]
    if locked_hash is None or actual != locked_hash:
        raise AttachmentMismatchError(
            f"attachment {attachment_id}: sha256 drift "
            f"(locked={locked_hash!r}, disk={actual!r})"
        )
    if att["spec_hash"] is None or att["spec_hash"] != actual:
        raise AttachmentMismatchError(
            f"attachment {attachment_id}: attachment spec_hash drift "
            f"(row={att['spec_hash']!r}, disk={actual!r})"
        )
    return raw


def list_ready(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    tenant: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[kb.Task]:
    """Return external tasks waiting for a claim.

    Returns tasks in ``ready`` status whose spec is fully locked
    (``external_spec_hash`` + ``external_spec_attachment_id`` +
    ``external_spec_schema`` all set), in created-at ascending order.
    Native ``claim_task`` excludes these.
    """
    q = (
        "SELECT * FROM tasks "
        "WHERE status = 'ready' "
        "  AND external_spec_hash IS NOT NULL "
        "  AND external_spec_attachment_id IS NOT NULL "
        "  AND external_spec_schema IS NOT NULL "
        "  AND NOT EXISTS ("
        "      SELECT 1 FROM task_runs r WHERE r.task_id = tasks.id "
        "      AND r.worker_kind = ? AND r.ended_at IS NULL"
        "  )"
    )
    params: list[Any] = [WORKER_KIND]
    if assignee is not None:
        q += " AND assignee = ?"
        params.append(kb._canonical_assignee(assignee))
    if tenant is not None:
        q += " AND tenant = ?"
        params.append(tenant)
    q += " ORDER BY created_at ASC, id ASC"
    if limit is not None and int(limit) > 0:
        q += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(q, params).fetchall()
    return [kb.Task.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Public API — claim / bind / heartbeat / still_owns / hold
# ---------------------------------------------------------------------------


def claim_external(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    expected_spec: SpecIdentity,
    lease_token: str,
    lease_expires_at: int,
) -> Lease:
    """Atomically transition ``ready → running`` for an external worker.

    The caller passes the expected :class:`SpecIdentity`, an opaque
    ``lease_token`` (any string the caller wants to use as the auth
    secret for subsequent calls), and the ``lease_expires_at`` epoch
    second. Inside ONE ``BEGIN IMMEDIATE`` transaction the call re-reads
    the accepted attachment, recomputes its SHA-256, verifies it matches
    the locked hash AND the caller's ``expected_spec``, then CAS-transitions
    the task to running. A new ``task_runs`` row is inserted with the
    copied spec identity, attempt index, durable ``claimed`` substate,
    and ``active`` lease state.

    Returns the typed :class:`Lease`.
    """
    if not isinstance(expected_spec, SpecIdentity):
        raise TypeError(
            f"expected_spec must be SpecIdentity, got {type(expected_spec)!r}"
        )
    if not isinstance(lease_token, str) or not lease_token.strip():
        raise ValueError("lease_token is required")
    if not isinstance(lease_expires_at, int) or isinstance(lease_expires_at, bool):
        raise ValueError("lease_expires_at must be an int epoch second")
    now = int(time.time())
    if lease_expires_at <= now:
        raise ExternalWorkerError("lease_expires_at must be a future epoch second")

    with kb._board_lifecycle_write_txn(conn):
        task_row = conn.execute(
            "SELECT id, status, external_spec_attachment_id, "
            "       external_spec_hash, external_spec_schema, "
            "       current_run_id, claim_lock "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None:
            raise ClaimRejected(f"unknown task {task_id}")
        open_run_id = kb._open_external_run_id(conn, task_id)
        if open_run_id is not None:
            raise ClaimRejected(
                f"task {task_id} already has open external run {open_run_id}"
            )
        if task_row["status"] != "ready":
            raise ClaimRejected(
                f"task {task_id} not in 'ready' (status={task_row['status']!r})"
            )
        if task_row["external_spec_hash"] is None:
            raise ClaimRejected(f"task {task_id} has no locked external spec")
        # Spec identity cross-check.
        if (
            int(task_row["external_spec_attachment_id"] or 0)
            != expected_spec.attachment_id
        ):
            raise ClaimRejected(f"task {task_id} spec attachment_id mismatch")
        if task_row["external_spec_hash"] != expected_spec.spec_hash:
            raise ClaimRejected(f"task {task_id} spec_hash mismatch")
        if task_row["external_spec_schema"] != expected_spec.schema_version:
            raise ClaimRejected(f"task {task_id} spec_schema mismatch")
        unfinished_parent = conn.execute(
            "SELECT p.id, p.status FROM tasks p "
            "JOIN task_links l ON l.parent_id = p.id "
            "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived') "
            "ORDER BY p.id LIMIT 1",
            (task_id,),
        ).fetchone()
        if unfinished_parent is not None:
            raise ClaimRejected(
                f"task {task_id} has unfinished parent "
                f"{unfinished_parent['id']} ({unfinished_parent['status']})"
            )
        # Re-read & rehash the accepted attachment inside the same txn.
        att = _read_attachment_row(conn, expected_spec.attachment_id)
        if att["task_id"] != task_id:
            raise ClaimRejected(
                f"attachment {expected_spec.attachment_id} not owned by task {task_id}"
            )
        if att["filename"] != SPEC_ATTACHMENT_NAME:
            raise ClaimRejected(
                f"attachment {expected_spec.attachment_id} filename mismatch"
            )
        raw = _read_attachment_bytes(att)
        actual_hash = _sha256_hex(raw)
        if actual_hash != expected_spec.spec_hash:
            raise ClaimRejected(
                f"attachment {expected_spec.attachment_id} hash drift on re-read"
            )
        if att["spec_hash"] != expected_spec.spec_hash:
            raise ClaimRejected(
                f"attachment {expected_spec.attachment_id} lock hash drift"
            )
        # CAS: only transition ready → running if the task is still
        # ready and unclaimed. A concurrent writer (dashboard, native
        # claim, another external worker) flips status and we refuse.
        cur = conn.execute(
            "UPDATE tasks "
            "SET status = 'running', claim_lock = ?, claim_expires = ?, "
            "    started_at = COALESCE(started_at, ?) "
            "WHERE id = ? AND status = 'ready' AND claim_lock IS NULL "
            "  AND current_run_id IS NULL",
            (lease_token, int(lease_expires_at), now, task_id),
        )
        if cur.rowcount != 1:
            raise ClaimRejected(
                f"task {task_id} not claimable (racing writer changed state)"
            )
        # Attempts are scoped to closed external runs of this exact spec.
        prior = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs "
            "WHERE task_id = ? AND worker_kind = ? "
            "AND external_spec_hash = ? AND ended_at IS NOT NULL",
            (task_id, WORKER_KIND, expected_spec.spec_hash),
        ).fetchone()
        attempt = int(prior["n"]) + 1
        run_cur = conn.execute(
            "INSERT INTO task_runs ("
            "    task_id, status, claim_lock, claim_expires, started_at, "
            "    worker_kind, external_spec_attachment_id, "
            "    external_spec_hash, external_spec_schema, "
            "    external_attempt, external_substate, "
            "    external_lease_state, external_recovery_count"
            ") VALUES (?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                task_id,
                lease_token,
                int(lease_expires_at),
                now,
                WORKER_KIND,
                expected_spec.attachment_id,
                expected_spec.spec_hash,
                expected_spec.schema_version,
                attempt,
                SUBSTATE_CLAIMED,
                LEASE_ACTIVE,
            ),
        )
        run_id = int(run_cur.lastrowid or 0)
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        kb._append_event(
            conn,
            task_id,
            "external_claimed",
            {
                "run_id": run_id,
                "claim_expires": int(lease_expires_at),
                "attempt": attempt,
                "spec_attachment_id": expected_spec.attachment_id,
                "spec_hash": expected_spec.spec_hash,
            },
            run_id=run_id,
        )
        return Lease(
            run_id=run_id,
            task_id=task_id,
            spec=expected_spec,
            lease_token=lease_token,
            lease_state=LEASE_ACTIVE,
            lease_expires_at=int(lease_expires_at),
            attempt=attempt,
        )


def bind_process(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    process: BoundProcess,
) -> Lease:
    """Bind the host/pid/pgid/start_token quad to an external run.

    Must be called BEFORE any external child is started, so a later
    :func:`recover_expired` can prove the bound process is absent. Records
    a durable ``bound`` substate BEFORE any child start ACK.

    Idempotent: a second call with the SAME identity quad returns the
    current lease (state == ``bound``). A second call with a DIVERGENT
    quad raises :class:`BindMismatch`.
    """
    if not isinstance(process, BoundProcess):
        raise TypeError(f"process must be BoundProcess, got {type(process)!r}")

    with kb.write_txn(conn):
        row = _check_lease_identity(conn, lease)
        if row["ended_at"] is not None or lease.lease_expires_at <= int(time.time()):
            raise NotOwner(f"run {lease.run_id} is not an active lease")
        run_id = int(row["id"])
        if row["external_host"] is not None:
            # Already bound — idempotent iff every field matches exactly.
            same = (
                row["external_host"] == process.host
                and int(row["external_pid"]) == process.pid
                and int(row["external_pgid"]) == process.pgid
                and row["external_start_token"] == process.start_token
            )
            if not same:
                raise BindMismatch(
                    f"run {run_id} already bound to a different process "
                    f"(host={row['external_host']!r}, "
                    f"pid={row['external_pid']!r}, "
                    f"pgid={row['external_pgid']!r})"
                )
            if row["external_lease_state"] != LEASE_BOUND or lease.lease_state not in {
                LEASE_ACTIVE,
                LEASE_BOUND,
            }:
                raise LeaseStateError(f"run {run_id} is not in bound state")
            return _lease_from_run_row(row)
        if (
            lease.lease_state != LEASE_ACTIVE
            or (row["external_lease_state"] or LEASE_ACTIVE) != LEASE_ACTIVE
        ):
            raise LeaseStateError(f"run {run_id} is not claim-active")
        # Not yet bound — record durable BOUND substate. The CAS clauses
        # (claim_lock, external_lease_state) close the race with a
        # concurrent bind / hold / finalize.
        cur = conn.execute(
            "UPDATE task_runs "
            "SET external_host = ?, external_pid = ?, "
            "    external_pgid = ?, external_start_token = ?, "
            "    external_substate = ?, external_lease_state = ? "
            "WHERE id = ? AND ended_at IS NULL "
            "  AND claim_lock = ? AND external_lease_state = ?",
            (
                process.host,
                int(process.pid),
                int(process.pgid),
                process.start_token,
                SUBSTATE_BOUND,
                LEASE_BOUND,
                run_id,
                lease.lease_token,
                LEASE_ACTIVE,
            ),
        )
        if cur.rowcount != 1:
            raise LeaseStateError(
                f"run {run_id} racing writer changed state during bind"
            )
        kb._append_event(
            conn,
            lease.task_id,
            "external_process_bound",
            {
                "run_id": run_id,
                "host": process.host,
                "pid": int(process.pid),
                "pgid": int(process.pgid),
            },
            run_id=run_id,
        )
        updated = conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _lease_from_run_row(updated)


def heartbeat(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    lease_expires_at: int,
) -> Lease:
    """Extend the lease. Returns the updated :class:`Lease` (same state).

    Accepts ACTIVE, BOUND, and HOLDING lease states. The caller's expected
    state must match the DB exactly. Raises :class:`NotOwner` /
    :class:`LeaseStateError` on any mismatch.
    """
    if not isinstance(lease_expires_at, int) or isinstance(lease_expires_at, bool):
        raise ValueError("lease_expires_at must be an int epoch second")
    if lease_expires_at <= int(time.time()):
        raise ExternalWorkerError("lease_expires_at must be a future epoch second")

    with kb.write_txn(conn):
        row = _validate_lease(
            conn,
            lease,
            allowed_states=frozenset({LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}),
        )
        run_id = int(row["id"])
        cur = conn.execute(
            "UPDATE task_runs "
            "SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ? AND ended_at IS NULL AND claim_lock = ? "
            "  AND claim_expires = ? AND external_lease_state = ?",
            (
                int(lease_expires_at),
                int(time.time()),
                run_id,
                lease.lease_token,
                lease.lease_expires_at,
                lease.lease_state,
            ),
        )
        if cur.rowcount != 1:
            raise NotOwner(f"run {run_id} no longer owned by this lease")
        task_cur = conn.execute(
            "UPDATE tasks "
            "SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ? "
            "  AND claim_expires = ? AND current_run_id = ?",
            (
                int(lease_expires_at),
                lease.task_id,
                lease.lease_token,
                lease.lease_expires_at,
                run_id,
            ),
        )
        if task_cur.rowcount != 1:
            raise NotOwner(f"task {lease.task_id} no longer matches run lease")
        updated = conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _lease_from_run_row(updated)


def still_owns(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
) -> bool:
    """Return True iff the complete expected lease is current and unexpired.

    Read-only; safe to poll. Checks EXACT ownership (run id + task id +
    copied spec identity + lease token), expiry, and ended_at.
    """
    if not isinstance(lease, Lease):
        raise TypeError("lease must be Lease")
    try:
        row = _check_lease_identity(conn, lease)
    except ExternalWorkerError:
        return False
    return (
        row["ended_at"] is None
        and (row["external_lease_state"] or LEASE_ACTIVE) == lease.lease_state
        and lease.lease_expires_at > int(time.time())
    )


def hold_for_recovery(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    proof: RecoveryHoldProof,
    extension_seconds: Optional[int] = None,
) -> Lease:
    """Atomically extend the lease, append a bounded event, and bump the
    inconclusive-hold counter.

    Used by a supervisor that has lost contact with the worker but is not
    yet ready to declare it dead: it holds the run for a bounded recovery
    window while it investigates. Each hold increments
    ``external_recovery_count``; when that counter reaches
    :data:`RECOVERY_REQUEUE_LIMIT`, :func:`recover_expired` will refuse a new
    REQUEUE decision. An exact result staged before a crash is still replayed.

    Never treats uncertainty as absence: this call only extends the lease
    and record the hold. The actual recovery decision is in
    :func:`recover_expired`, which requires an affirmative proof.
    """
    if not isinstance(proof, RecoveryHoldProof):
        raise TypeError(f"proof must be RecoveryHoldProof, got {type(proof)!r}")
    if proof.run_id != lease.run_id or proof.task_id != lease.task_id:
        raise RecoveryRejected("hold proof identity does not match lease")

    ttl = (
        int(extension_seconds)
        if extension_seconds is not None
        else kb._resolve_claim_ttl_seconds(None)
    )
    if ttl <= 0:
        raise ValueError("extension_seconds must be positive")
    new_expires = int(time.time()) + ttl

    with kb.write_txn(conn):
        row = _validate_lease(
            conn,
            lease,
            allowed_states=frozenset({LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}),
            require_unexpired=False,
        )
        run_id = int(row["id"])
        if row["external_result_hash"] is not None:
            raise RecoveryRejected(
                f"run {run_id} already has a persisted result; finalize or "
                "recover those exact bytes instead of adding a recovery hold"
            )
        # If a process is bound, the proof's bound identity must match.
        if row["external_host"] is not None:
            if proof.bound is None:
                raise RecoveryRejected(
                    f"run {run_id} is bound but hold proof carries no bound identity"
                )
            if not (
                proof.bound.host == row["external_host"]
                and int(proof.bound.pid) == int(row["external_pid"])
                and int(proof.bound.pgid) == int(row["external_pgid"])
                and proof.bound.start_token == row["external_start_token"]
            ):
                raise RecoveryRejected(
                    f"run {run_id} hold proof bound identity mismatch"
                )
        elif proof.bound is not None:
            # Proof asserts a bound identity for an unbound run — refuse.
            raise RecoveryRejected(
                f"run {run_id} is unbound but hold proof carries a bound identity"
            )
        prior_holds = int(row["external_recovery_count"])
        if prior_holds >= RECOVERY_REQUEUE_LIMIT:
            raise RecoveryRejected(
                f"run {run_id} already requires manual recovery"
            )
        new_holds = prior_holds + 1
        cur = conn.execute(
            "UPDATE task_runs "
            "SET claim_expires = ?, external_recovery_count = ?, "
            "    external_substate = ?, external_lease_state = ? "
            "WHERE id = ? AND ended_at IS NULL AND claim_lock = ? "
            "  AND claim_expires = ? AND external_lease_state = ?",
            (
                new_expires,
                new_holds,
                SUBSTATE_HOLDING,
                LEASE_HOLDING,
                run_id,
                lease.lease_token,
                lease.lease_expires_at,
                lease.lease_state,
            ),
        )
        if cur.rowcount != 1:
            raise NotOwner(f"run {run_id} no longer owned by this lease")
        task_cur = conn.execute(
            "UPDATE tasks "
            "SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ? "
            "  AND claim_expires = ? AND current_run_id = ?",
            (
                new_expires,
                lease.task_id,
                lease.lease_token,
                lease.lease_expires_at,
                run_id,
            ),
        )
        if task_cur.rowcount != 1:
            raise NotOwner(f"task {lease.task_id} no longer matches run lease")
        kb._append_event(
            conn,
            lease.task_id,
            "external_hold_for_recovery",
            {
                "run_id": run_id,
                "reason": (proof.evidence or "").strip()[:200] or None,
                "inconclusive_holds": new_holds,
                "claim_expires": new_expires,
            },
            run_id=run_id,
        )
        if new_holds == RECOVERY_REQUEUE_LIMIT:
            kb._append_event(
                conn,
                lease.task_id,
                "external_manual_recovery_required",
                {"run_id": run_id, "inconclusive_holds": new_holds},
                run_id=run_id,
            )
        updated = conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _lease_from_run_row(updated)


# ---------------------------------------------------------------------------
# Public API — put_result / finalize
# ---------------------------------------------------------------------------


def _validate_and_hash_result(
    raw: bytes,
    *,
    expected_run_id: int,
    expected_spec: SpecIdentity,
) -> tuple[str, dict[str, Any]]:
    """Decode and validate result bytes; return their hash and typed facts.

    The hash is ALWAYS computed by this module, never trusted from the
    caller. Raises :class:`ResultRejected` on any contract violation.
    """
    if len(raw) > MAX_RESULT_BYTES:
        raise ResultRejected(
            f"ExecutionResult exceeds {MAX_RESULT_BYTES} byte limit"
        )
    try:
        obj = _decode_strict_json(raw)
    except SpecParseError as exc:
        raise ResultRejected(f"invalid ExecutionResult JSON: {exc}") from exc
    try:
        facts = _validate_execution_result(
            obj,
            expected_run_id=expected_run_id,
            expected_spec=expected_spec,
        )
    except SpecParseError as exc:
        raise ResultRejected(f"invalid ExecutionResult schema: {exc}") from exc
    return _sha256_hex(bytes(raw)), facts


def _require_result_absence_proof(row: sqlite3.Row, facts: dict[str, Any]) -> None:
    if not facts["absence_proven"]:
        raise ResultRejected("ExecutionResult does not prove process-group absence")
    identity = facts["process_identity"]
    if row["external_host"] is None:
        if (
            identity is not None
            or facts["outcome"] != "aborted_before_start"
            or facts["termination_status"] != "not_started"
        ):
            raise ResultRejected(
                "unbound run requires aborted_before_start, not_started, "
                "and null process identity"
            )
        return
    if identity is None:
        raise ResultRejected("bound run result is missing process identity")
    if not (
        identity["host"] == row["external_host"]
        and int(identity["pid"]) == int(row["external_pid"])
        and int(identity["pgid"]) == int(row["external_pgid"])
        and identity["start_token"] == row["external_start_token"]
    ):
        raise ResultRejected("ExecutionResult process identity does not match the run")


def put_result(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    disposition: str,
    result_bytes: bytes,
    block_kind: Optional[str] = None,
) -> str:
    """Validate and persist a result on the run WITHOUT finalizing.

    The result is validated (strict JSON, schema, identity) and its hash,
    disposition, block_kind, and exact bytes are stored on the run row.
    The run remains active — a subsequent :func:`finalize` with the same
    bytes is the commit step. Returns the computed SHA-256.

    This lets a supervisor durably stage the complete terminal result after it
    has proved process absence; :func:`finalize` then commits those same bytes.
    """
    if disposition not in VALID_DISPOSITIONS:
        raise ValueError(f"disposition must be one of {sorted(VALID_DISPOSITIONS)}")
    if disposition == DISPOSITION_BLOCK:
        if block_kind not in kb.VALID_BLOCK_KINDS:
            raise ValueError(
                f"block_kind must be one of {sorted(kb.VALID_BLOCK_KINDS)}"
            )
    else:
        block_kind = None
    if not isinstance(result_bytes, (bytes, bytearray)):
        raise TypeError("result_bytes must be bytes")

    with kb.write_txn(conn):
        row = _validate_lease(
            conn,
            lease,
            allowed_states=frozenset({LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}),
        )
        run_id = int(row["id"])
        h, facts = _validate_and_hash_result(
            bytes(result_bytes),
            expected_run_id=run_id,
            expected_spec=lease.spec,
        )
        json_disposition = facts["disposition"]
        json_block_kind = facts["block_kind"]
        if facts["attempt"] != lease.attempt:
            raise ResultRejected("result attempt does not match the run lease")
        if json_disposition != disposition:
            raise ResultRejected(
                f"result JSON disposition {json_disposition!r} != "
                f"argument {disposition!r}"
            )
        if (json_block_kind or None) != (block_kind or None):
            raise ResultRejected(
                f"result JSON block_kind {json_block_kind!r} != argument {block_kind!r}"
            )
        _require_result_absence_proof(row, facts)
        _validate_result_artifacts(conn, run_id=run_id, facts=facts)
        if row["external_result_hash"] is not None:
            if (
                row["external_result_hash"] == h
                and row["external_terminal_disposition"] == disposition
                and (row["external_block_kind"] or None) == (block_kind or None)
                and row["external_result_json"] == bytes(result_bytes).decode("utf-8")
            ):
                return h
            raise ResultRejected(f"run {run_id} already has a divergent staged result")
        cur = conn.execute(
            "UPDATE task_runs "
            "SET external_terminal_disposition = ?, "
            "    external_block_kind = ?, "
            "    external_result_hash = ?, "
            "    external_result_json = ?, metadata = ?, summary = ? "
            "WHERE id = ? AND ended_at IS NULL AND claim_lock = ? "
            "  AND claim_expires = ? AND external_lease_state = ? "
            "  AND external_result_hash IS NULL",
            (
                disposition,
                block_kind,
                h,
                bytes(result_bytes).decode("utf-8"),
                bytes(result_bytes).decode("utf-8"),
                facts["summary"],
                run_id,
                lease.lease_token,
                lease.lease_expires_at,
                lease.lease_state,
            ),
        )
        if cur.rowcount != 1:
            raise ResultRejected(f"run {run_id} changed while staging result")
        kb._append_event(
            conn,
            lease.task_id,
            "external_result_persisted",
            {
                "run_id": run_id,
                "disposition": disposition,
                "block_kind": block_kind,
                "result_hash": h,
            },
            run_id=run_id,
        )
        return h


def finalize(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    disposition: str,
    result_bytes: bytes,
    block_kind: Optional[str] = None,
) -> FinalizeOutcome:
    """Finalize an external run with COMPLETE / REQUEUE / BLOCK.

    Atomically stores the same result metadata (exact bytes, hash,
    disposition, block_kind) for all three dispositions, transitions the
    run to ``committed`` and the task to ``done`` / ``ready`` /
    ``blocked``, and returns a typed outcome:

    * :data:`FINALIZE_COMMITTED` — first finalize for this run+hash+tuple;
      the run is closed.
    * :data:`FINALIZE_ALREADY_COMMITTED_SAME_HASH` — idempotent re-finalize
      with the same hash AND exact disposition AND exact block_kind.
    * :data:`FINALIZE_REJECTED` — the run was already finalized with a
      divergent hash or terminal tuple.

    The strict result must contain an affirmative absence proof whose process
    identity matches the bound run. An unbound run requires a null identity
    and ``aborted_before_start`` outcome. No task transition can release the
    claim based on an absent or mismatched proof.
    """
    if disposition not in VALID_DISPOSITIONS:
        raise ValueError(f"disposition must be one of {sorted(VALID_DISPOSITIONS)}")
    if disposition == DISPOSITION_BLOCK:
        if block_kind not in kb.VALID_BLOCK_KINDS:
            raise ValueError(
                f"block_kind must be one of {sorted(kb.VALID_BLOCK_KINDS)}"
            )
    else:
        block_kind = None
    if not isinstance(result_bytes, (bytes, bytearray)):
        raise TypeError("result_bytes must be bytes")
    with kb.write_txn(conn):
        row = _check_lease_identity(conn, lease, check_active_task=False)
        run_id = int(row["id"])
        new_hash, facts = _validate_and_hash_result(
            bytes(result_bytes),
            expected_run_id=run_id,
            expected_spec=lease.spec,
        )
        json_disposition = facts["disposition"]
        json_block_kind = facts["block_kind"]
        if facts["attempt"] != lease.attempt:
            raise ResultRejected("result attempt does not match the run lease")
        if json_disposition != disposition:
            raise ResultRejected(
                f"result JSON disposition {json_disposition!r} != "
                f"argument {disposition!r}"
            )
        if (json_block_kind or None) != (block_kind or None):
            raise ResultRejected(
                f"result JSON block_kind {json_block_kind!r} != argument {block_kind!r}"
            )
        _validate_result_artifacts(conn, run_id=run_id, facts=facts)
        # Idempotency: if this run already reached COMMITTED, the replay
        # must match hash + disposition + block_kind exactly. The caller's
        # prior lease is still accepted for identity; the lease_state field
        # is whatever it was at first finalize time.
        if row["external_substate"] == SUBSTATE_COMMITTED:
            prior_hash = row["external_result_hash"]
            prior_disp = row["external_terminal_disposition"]
            prior_kind = row["external_block_kind"]
            if (
                prior_hash == new_hash
                and prior_disp == disposition
                and (prior_kind or None) == (block_kind or None)
            ):
                return FinalizeOutcome(
                    status=FINALIZE_ALREADY_COMMITTED_SAME_HASH,
                    run_id=run_id,
                    task_id=lease.task_id,
                    result_hash=new_hash,
                    disposition=disposition,
                    block_kind=block_kind,
                )
            return FinalizeOutcome(
                status=FINALIZE_REJECTED,
                run_id=run_id,
                task_id=lease.task_id,
                result_hash=new_hash,
                prior_hash=prior_hash,
            )
        row = _validate_lease(
            conn,
            lease,
            allowed_states=frozenset({LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}),
        )
        _require_result_absence_proof(row, facts)
        if row["external_result_hash"] is not None and not (
            row["external_result_hash"] == new_hash
            and row["external_terminal_disposition"] == disposition
            and (row["external_block_kind"] or None) == (block_kind or None)
            and row["external_result_json"] == bytes(result_bytes).decode("utf-8")
        ):
            return FinalizeOutcome(
                status=FINALIZE_REJECTED,
                run_id=run_id,
                task_id=lease.task_id,
                result_hash=new_hash,
                prior_hash=row["external_result_hash"],
            )
        # Persist + transition atomically. The CAS clauses close the race
        # with a concurrent finalize / hold. We do NOT null ``claim_lock``
        # on the run row: the lease token remains the auth secret for the
        # idempotent replay path (the row's ``external_lease_state`` /
        # ``ended_at`` flag prevents any further mutation).
        now = int(time.time())
        cur = conn.execute(
            "UPDATE task_runs "
            "SET status = ?, outcome = ?, ended_at = ?, "
            "    external_terminal_disposition = ?, "
            "    external_block_kind = ?, "
            "    external_result_hash = ?, "
            "    external_result_json = ?, "
            "    metadata = ?, summary = ?, "
            "    external_substate = ?, external_lease_state = ?, "
            "    claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND ended_at IS NULL "
            "  AND claim_lock = ? AND claim_expires = ? "
            "  AND external_lease_state = ?",
            (
                _outcome_for_disposition(disposition),
                _outcome_for_disposition(disposition),
                now,
                disposition,
                block_kind,
                new_hash,
                bytes(result_bytes).decode("utf-8"),
                bytes(result_bytes).decode("utf-8"),
                facts["summary"],
                SUBSTATE_COMMITTED,
                LEASE_COMMITTED,
                run_id,
                lease.lease_token,
                lease.lease_expires_at,
                lease.lease_state,
            ),
        )
        if cur.rowcount != 1:
            raise NotOwner(f"run {run_id} racing writer changed state")
        _transition_task_for_terminal(
            conn,
            task_id=lease.task_id,
            disposition=disposition,
            block_kind=block_kind,
            summary=facts["summary"],
            now=now,
        )
        kb._append_event(
            conn,
            lease.task_id,
            f"external_finalized_{disposition.lower()}",
            {
                "run_id": run_id,
                "disposition": disposition,
                "block_kind": block_kind,
                "hash": new_hash,
                "termination_status": facts["termination_status"],
                "absence_proven": facts["absence_proven"],
            },
            run_id=run_id,
        )
        if disposition == DISPOSITION_COMPLETE:
            kb._recompute_ready_in_txn(conn)
        return FinalizeOutcome(
            status=FINALIZE_COMMITTED,
            run_id=run_id,
            task_id=lease.task_id,
            result_hash=new_hash,
            disposition=disposition,
            block_kind=block_kind,
        )


# ---------------------------------------------------------------------------
# Public API — recover_expired
# ---------------------------------------------------------------------------


def recover_expired(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    proof: ProcessAbsenceProof | NoStartAckProof,
    result: ExecutionResult,
) -> RecoveredRun:
    """Recover ONE explicitly-expected expired run.

    Requires EITHER:

    * a typed :class:`ProcessAbsenceProof` whose identity matches the
      bound process, AND the lease has expired; OR
    * a typed :class:`NoStartAckProof` for an unbound run (the supervisor
      asserts it never ACK'd a child start), AND the lease has expired.

    Accepts a complete :class:`ExecutionResult` and disposition, which
    is validated, hashed, and persisted exactly as :func:`finalize` does.

    NEVER performs ``os.kill``-based absence inference. NEVER releases or
    requeues without an affirmative proof. MAY NOT introduce a new REQUEUE
    once two inconclusive holds are durable (``external_recovery_count >=
    RECOVERY_REQUEUE_LIMIT``). Exact bytes already staged by :func:`put_result`
    remain authoritative and replayable after a crash; otherwise a REQUEUE
    raises :class:`RecoveryRejected` and leaves the run active.

    Uncertainty (any failed identity / proof / state check) raises
    :class:`RecoveryRejected` and leaves the run active.
    """
    if not isinstance(lease, Lease):
        raise TypeError("lease must be Lease")
    if not isinstance(proof, (ProcessAbsenceProof, NoStartAckProof)):
        raise TypeError("proof must be ProcessAbsenceProof or NoStartAckProof")
    if not isinstance(result, ExecutionResult):
        raise TypeError("result must be ExecutionResult")
    now_ts = int(time.time())

    with kb.write_txn(conn):
        try:
            row = _check_lease_identity(conn, lease, check_active_task=False)
        except ExternalWorkerError as exc:
            raise RecoveryRejected(str(exc)) from exc
        run_id = lease.run_id
        task_id = lease.task_id
        spec = lease.spec
        result_hash, facts = _validate_and_hash_result(
            result.result_bytes,
            expected_run_id=run_id,
            expected_spec=spec,
        )
        if (
            result.run_id != run_id
            or result.task_id != task_id
            or result.spec != spec
            or result.disposition != facts["disposition"]
            or (result.block_kind or None) != (facts["block_kind"] or None)
            or facts["attempt"] != lease.attempt
        ):
            raise ResultRejected("ExecutionResult identity does not match the lease")
        _validate_result_artifacts(conn, run_id=run_id, facts=facts)
        if row["external_substate"] == SUBSTATE_COMMITTED:
            if (
                row["external_result_hash"] == result_hash
                and row["external_terminal_disposition"] == result.disposition
                and (row["external_block_kind"] or None) == (result.block_kind or None)
            ):
                holds = int(row["external_recovery_count"])
                return RecoveredRun(
                    run_id=run_id,
                    task_id=task_id,
                    spec=spec,
                    disposition=result.disposition,
                    block_kind=result.block_kind,
                    recovery_count=holds,
                    requeued=(result.disposition == DISPOSITION_REQUEUE),
                    blocked=(result.disposition == DISPOSITION_BLOCK),
                )
            raise RecoveryRejected(f"run {run_id} already has a divergent result")
        try:
            row = _check_lease_identity(conn, lease)
        except ExternalWorkerError as exc:
            raise RecoveryRejected(str(exc)) from exc
        if row["ended_at"] is not None:
            raise RecoveryRejected(f"run {run_id} already ended")
        db_lease_state = row["external_lease_state"] or LEASE_ACTIVE
        if (
            lease.lease_state
            not in {LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}
            or db_lease_state != lease.lease_state
        ):
            raise RecoveryRejected(
                f"run {run_id} lease state mismatch: "
                f"caller={lease.lease_state!r} db={db_lease_state!r}"
            )
        # Lease must be expired.
        if row["claim_expires"] is None or int(row["claim_expires"]) > now_ts:
            raise RecoveryRejected(f"run {run_id} lease not expired")
        # Proof shape vs bound state.
        if isinstance(proof, NoStartAckProof):
            if proof.run_id != run_id or proof.task_id != task_id:
                raise RecoveryRejected(f"run {run_id} no-start proof identity mismatch")
            if row["external_host"] is not None:
                raise RecoveryRejected(
                    f"run {run_id} no-start proof supplied for a bound run"
                )
        else:  # ProcessAbsenceProof
            if row["external_host"] is None:
                raise RecoveryRejected(
                    f"run {run_id} absence proof supplied for an unbound run"
                )
            if not (
                proof.host == row["external_host"]
                and int(proof.pid) == int(row["external_pid"])
                and int(proof.pgid) == int(row["external_pgid"])
                and proof.start_token == row["external_start_token"]
            ):
                raise RecoveryRejected(f"run {run_id} absence proof identity mismatch")
        _require_result_absence_proof(row, facts)
        holds = int(row["external_recovery_count"])
        staged_same = (
            row["external_result_hash"] == result_hash
            and row["external_terminal_disposition"] == result.disposition
            and (row["external_block_kind"] or None) == (result.block_kind or None)
            and row["external_result_json"]
            == bytes(result.result_bytes).decode("utf-8")
        )
        # Requeue suppression at the recovery limit. Uncertainty leaves
        # the run ACTIVE. Exact bytes already staged before a crash are
        # replayed, however: recovery may not replace that durable decision.
        if (
            result.disposition == DISPOSITION_REQUEUE
            and holds >= RECOVERY_REQUEUE_LIMIT
            and not staged_same
        ):
            raise RecoveryRejected(
                f"run {run_id} requeue suppressed at "
                f"{RECOVERY_REQUEUE_LIMIT} inconclusive holds"
            )
        if row["external_result_hash"] is not None and not staged_same:
            raise RecoveryRejected(
                f"run {run_id} already has a divergent staged result"
            )
        # Persist + transition atomically. See ``finalize`` for why
        # ``claim_lock`` is NOT nulled (idempotent replay auth).
        ts_now = int(time.time())
        outcome = _outcome_for_disposition(result.disposition)
        cur = conn.execute(
            "UPDATE task_runs "
            "SET status = ?, outcome = ?, ended_at = ?, "
            "    external_terminal_disposition = ?, "
            "    external_block_kind = ?, "
            "    external_result_hash = ?, "
            "    external_result_json = ?, "
            "    metadata = ?, summary = ?, "
            "    external_substate = ?, external_lease_state = ?, "
            "    claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND ended_at IS NULL "
            "  AND claim_lock = ? AND claim_expires = ? "
            "  AND external_lease_state = ?",
            (
                outcome,
                outcome,
                ts_now,
                result.disposition,
                result.block_kind,
                result_hash,
                bytes(result.result_bytes).decode("utf-8"),
                bytes(result.result_bytes).decode("utf-8"),
                facts["summary"],
                SUBSTATE_COMMITTED,
                LEASE_COMMITTED,
                int(run_id),
                lease.lease_token,
                lease.lease_expires_at,
                lease.lease_state,
            ),
        )
        if cur.rowcount != 1:
            raise RecoveryRejected(
                f"run {run_id} racing writer changed state during recovery"
            )
        _transition_task_for_terminal(
            conn,
            task_id=task_id,
            disposition=result.disposition,
            block_kind=result.block_kind,
            summary=facts["summary"],
            now=ts_now,
        )
        kb._append_event(
            conn,
            task_id,
            "external_recovered",
            {
                "run_id": int(run_id),
                "disposition": result.disposition,
                "block_kind": result.block_kind,
                "hash": result_hash,
                "inconclusive_holds": holds,
                "proof_kind": type(proof).__name__,
                "proof_evidence": getattr(proof, "evidence", "")[:200],
            },
            run_id=int(run_id),
        )
        if result.disposition == DISPOSITION_COMPLETE:
            kb._recompute_ready_in_txn(conn)
        return RecoveredRun(
            run_id=int(run_id),
            task_id=task_id,
            spec=spec,
            disposition=result.disposition,
            block_kind=result.block_kind,
            recovery_count=holds,
            requeued=(result.disposition == DISPOSITION_REQUEUE),
            blocked=(result.disposition == DISPOSITION_BLOCK),
        )


# ---------------------------------------------------------------------------
# Public API — list_active / get_run / put_artifact / read_artifact
# ---------------------------------------------------------------------------


def list_active(
    conn: sqlite3.Connection,
    *,
    host: Optional[str] = None,
) -> list[Lease]:
    """Return all currently-active external-mas runs as typed leases.

    ``host`` filters by the bound process host. Without ``host``, returns
    every active external run on the board.
    """
    q = (
        "SELECT r.* FROM task_runs r "
        "WHERE r.worker_kind = ? AND r.ended_at IS NULL"
    )
    params: list[Any] = [WORKER_KIND]
    if host is not None:
        q += " AND r.external_host = ?"
        params.append(host)
    q += " ORDER BY r.started_at ASC, r.id ASC"
    rows = conn.execute(q, params).fetchall()
    return [_lease_from_run_row(r) for r in rows]


def get_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
) -> Lease:
    """Fetch one external run, including terminal result identity."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ? AND worker_kind = ?",
        (int(run_id), WORKER_KIND),
    ).fetchone()
    if row is None:
        raise ExternalWorkerError(f"unknown external run {run_id}")
    return _lease_from_run_row(row)


def read_result(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
) -> Optional[PersistedResult]:
    """Return exact staged/terminal result bytes for crash-safe replay.

    The complete run/task/spec/token/attempt identity must match. ``None``
    means no result has been persisted for this run yet.
    """
    if not isinstance(lease, Lease):
        raise TypeError("lease must be Lease")
    row = _check_lease_identity(conn, lease, check_active_task=False)
    values = (
        row["external_result_json"],
        row["external_result_hash"],
        row["external_terminal_disposition"],
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ResultRejected(f"run {lease.run_id} has an incomplete durable result")
    raw = row["external_result_json"].encode("utf-8")
    actual_hash, facts = _validate_and_hash_result(
        raw,
        expected_run_id=lease.run_id,
        expected_spec=lease.spec,
    )
    if actual_hash != row["external_result_hash"]:
        raise ResultRejected(f"run {lease.run_id} durable result hash drift")
    if facts["attempt"] != lease.attempt:
        raise ResultRejected(f"run {lease.run_id} durable result attempt mismatch")
    if (
        facts["disposition"] != row["external_terminal_disposition"]
        or (facts["block_kind"] or None) != (row["external_block_kind"] or None)
    ):
        raise ResultRejected(f"run {lease.run_id} durable result tuple drift")
    _validate_result_artifacts(conn, run_id=lease.run_id, facts=facts)
    return PersistedResult(
        run_id=lease.run_id,
        task_id=lease.task_id,
        result_bytes=raw,
        result_hash=actual_hash,
        disposition=facts["disposition"],
        block_kind=facts["block_kind"],
    )


def put_artifact(
    conn: sqlite3.Connection,
    *,
    lease: Lease,
    name: str,
    data: bytes,
    content_type: Optional[str] = None,
) -> ArtifactRef:
    """Persist a named artifact bound to the active lease.

    Keyed by ``(run_id, name)`` with a safe name. Same name + same bytes
    is idempotent (returns the existing :class:`ArtifactRef`). Same name
    + divergent bytes raises :class:`ArtifactCollision`. The write is
    authorized by the lease inside the same ``write_txn`` that inserts
    the row.
    """
    safe = _safe_artifact_name(name)
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} byte limit")

    with kb.write_txn(conn):
        row = _validate_lease(
            conn,
            lease,
            allowed_states=frozenset({LEASE_ACTIVE, LEASE_BOUND, LEASE_HOLDING}),
        )
        run_id = int(row["id"])
        sha = _sha256_hex(data)
        existing = conn.execute(
            "SELECT sha256, size, stored_path FROM task_external_artifacts "
            "WHERE run_id = ? AND name = ?",
            (run_id, safe),
        ).fetchone()
        if existing is not None:
            existing_path = Path(existing["stored_path"])
            existing_bytes = (
                existing_path.read_bytes() if existing_path.is_file() else None
            )
            if (
                existing["sha256"] != sha
                or existing_bytes is None
                or len(existing_bytes) != int(existing["size"])
                or _sha256_hex(existing_bytes) != sha
            ):
                raise ArtifactCollision(
                    f"artifact {safe!r} already exists on run {run_id} "
                    "with divergent bytes"
                )
            return ArtifactRef(
                run_id=run_id,
                name=safe,
                sha256=sha,
                size=int(existing["size"]),
            )
        dest_dir = _external_artifacts_dir(conn, run_id)
        dest_path = dest_dir / f"{run_id}-{safe}"
        # Don't clobber an existing blob from a prior run with the same id
        # (impossible because AUTOINCREMENT, but be safe).
        if dest_path.exists():
            dest_path = dest_dir / f"{run_id}-{safe}-{sha[:8]}"
        dest_path.write_bytes(data)
        try:
            conn.execute(
                "INSERT INTO task_external_artifacts "
                "(run_id, task_id, name, stored_path, content_type, size, "
                " sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    lease.task_id,
                    safe,
                    str(dest_path.resolve()),
                    content_type,
                    len(data),
                    sha,
                    int(time.time()),
                ),
            )
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise
        return ArtifactRef(
            run_id=run_id,
            name=safe,
            sha256=sha,
            size=len(data),
        )


def read_artifact(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    name: str,
) -> bytes:
    """Read and rehash an artifact from an active or terminal external run."""
    safe = _safe_artifact_name(name)
    run = conn.execute(
        "SELECT 1 FROM task_runs WHERE id = ? AND worker_kind = ?",
        (int(run_id), WORKER_KIND),
    ).fetchone()
    if run is None:
        raise ArtifactNotFound(f"unknown external run {run_id}")
    art = conn.execute(
        "SELECT stored_path, sha256, size FROM task_external_artifacts "
        "WHERE run_id = ? AND name = ?",
        (int(run_id), safe),
    ).fetchone()
    if art is None:
        raise ArtifactNotFound(f"artifact {safe!r} not found on run {run_id}")
    p = Path(art["stored_path"])
    artifact_root = (
        kb.attachments_root_for_connection(conn) / "external_artifacts"
    ).resolve()
    try:
        p.resolve().relative_to(artifact_root)
    except (ValueError, OSError) as exc:
        raise ExternalWorkerError(
            f"artifact {safe!r} path escaped the artifact root"
        ) from exc
    if not p.is_file():
        raise ExternalWorkerError(f"artifact {safe!r} blob missing on run {run_id}")
    raw = p.read_bytes()
    if len(raw) != int(art["size"]) or _sha256_hex(raw) != art["sha256"]:
        raise ExternalWorkerError(f"artifact {safe!r} content drift on read")
    return raw


__all__ = [
    # Constants
    "EXTERNAL_WORKER_API_VERSION",
    "WORKER_KIND",
    "SPEC_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SPEC_ATTACHMENT_NAME",
    "RECOVERY_REQUEUE_LIMIT",
    "DEFAULT_LEASE_TTL_SECONDS",
    "MAX_RESULT_BYTES",
    "MAX_ARTIFACT_BYTES",
    # Lease states
    "LEASE_ACTIVE",
    "LEASE_BOUND",
    "LEASE_HOLDING",
    "LEASE_COMMITTED",
    # Substates
    "SUBSTATE_CLAIMED",
    "SUBSTATE_BOUND",
    "SUBSTATE_HOLDING",
    "SUBSTATE_COMMITTED",
    # Dispositions
    "DISPOSITION_COMPLETE",
    "DISPOSITION_REQUEUE",
    "DISPOSITION_BLOCK",
    "VALID_DISPOSITIONS",
    # Finalize outcomes
    "FINALIZE_COMMITTED",
    "FINALIZE_ALREADY_COMMITTED_SAME_HASH",
    "FINALIZE_REJECTED",
    # Dataclasses
    "SpecIdentity",
    "Lease",
    "BoundProcess",
    "ProcessAbsenceProof",
    "NoStartAckProof",
    "RecoveryHoldProof",
    "ExecutionResult",
    "PersistedResult",
    "FinalizeOutcome",
    "RecoveredRun",
    "ArtifactRef",
    # Errors
    "ExternalWorkerError",
    "SpecParseError",
    "SpecMutationError",
    "AttachmentMismatchError",
    "AttachmentNotOwnedError",
    "ClaimRejected",
    "BindMismatch",
    "NotOwner",
    "LeaseStateError",
    "RecoveryRejected",
    "ResultRejected",
    "ArtifactCollision",
    "ArtifactNotFound",
    "StaleBoardConnection",
    # Public API
    "connect",
    "submit",
    "read_submitted_attachment",
    "list_ready",
    "claim_external",
    "bind_process",
    "heartbeat",
    "still_owns",
    "hold_for_recovery",
    "put_result",
    "finalize",
    "recover_expired",
    "list_active",
    "get_run",
    "read_result",
    "put_artifact",
    "read_artifact",
]
