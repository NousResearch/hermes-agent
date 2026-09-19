"""Sealed task-envelope v2 validation, canonical values, and context binding.

Self-contained: no dependency on phase2_enforcement, the durable store, or any
runtime/session wiring.  The store imports from here; never the reverse.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from agent.phase2_errors import AuthorityError
from agent.phase2_sqlite import _SQLITE_INT_MAX, _SQLITE_INT_MIN

_HASH_KEYS = ("idempotency_key", "input_hash")
_ALLOWED_SURFACES = {
    "direct_model",
    "combo",
    "acp_worker",
    "a2a",
    "cloud",
    "local_tool",
}
_SHA256_RE_LEN = 64
# Bounded audit metadata for a rejected caller's malformed fence claim.
_FENCE_REPR_MAX = 64
# Bounded per-invariant sample size in a migration diagnostic.
_MIGRATION_SAMPLE_MAX = 5


def _freeze_envelope_value(value: Any) -> Any:
    """Return an immutable snapshot of JSON-compatible envelope data."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuthorityError("sealed envelope object keys must be strings")
            frozen[key] = _freeze_envelope_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_envelope_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise AuthorityError("sealed envelope numbers must be finite")
    raise AuthorityError(
        f"sealed envelope contains unsupported value type: {type(value).__name__}"
    )


# ── Envelope validation constants ─────────────────────────────────────────────
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = (
    "graph_id",
    "node_id",
    "attempt_id",
    "idempotency_key",
    "objective",
    "execution_surface",
    "lane",
    "roots",
    "permissions",
    "network_policy",
    "exec_policy",
    "destructive_policy",
    "budgets",
    "deadline_utc",
    "retry_policy",
    "cancellation",
    "evidence_policy",
    "verifier_policy",
    "planner_hash",
    "policy_hash",
    "input_hash",
    "lease",
    "fence",
)
_ALLOWED_LANES = {"hermes", "fable", "claude_opus", "omx_codex", "omp", "orca"}
_ALLOWED_NETWORK = {"none", "loopback_only", "allowlist"}
_ALLOWED_EXEC = {"none", "sandboxed", "host"}
_ALLOWED_DESTRUCTIVE = {"forbid", "require_confirmation", "allow_within_roots"}
_STATELESS_SURFACES = {"direct_model", "combo"}

# ── Context-var envelope binding ──────────────────────────────────────────────
_CURRENT_ENVELOPE: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "phase2_current_sealed_envelope", default=None
)
_CURRENT_FENCE: ContextVar[int | None] = ContextVar(
    "phase2_current_fence", default=None
)


# ── Canonical value helpers ───────────────────────────────────────────────────


def _json_tree(value: Any) -> Any:
    """Copy mappings/sequences into a strict JSON-compatible value tree."""

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_tree(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_tree(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sealed_json(value: Any) -> str:
    """Canonical JSON for an immutable sealed row, refusing non-finite numbers.

    ``json.dumps`` emits the non-standard tokens ``Infinity``/``NaN`` by
    default. ``sealed_plans`` and ``sealed_nodes`` are immutable and undeletable
    by trigger, and a graph_id can only ever be sealed once, so a single
    ``usd_max: inf`` would permanently wedge that graph behind a plan whose JSON
    no strict reader can parse and whose ceiling can never be breached
    (contract §9 requires budget breach to fail closed; ``tokens_max: inf``
    additionally aborts accounting at ``int(inf)`` with an untyped
    ``OverflowError``). Refusing at seal time keeps the defect out of the
    immutable table instead of leaving a permanently ungrantable graph behind.
    """

    try:
        return _canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuthorityError("sealed plan must contain finite JSON values") from exc


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise AuthorityError("authority timestamps must be timezone-aware")
    return current.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuthorityError("invalid authority timestamp") from exc
    if parsed.tzinfo is None:
        raise AuthorityError("authority timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise AuthorityError(f"{field} must be a non-negative number")
    # Convert once under a typed boundary. Arbitrarily large JSON integers are
    # valid Python values but cannot participate in finite budget arithmetic.
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise AuthorityError(f"{field} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise AuthorityError(f"{field} must be a finite number")
    return normalized


def _expiry_from_ttl(granted_at: datetime, ttl_s: Any) -> datetime:
    """Add a finite TTL under the authority store's typed error contract."""

    ttl = _nonnegative_number(ttl_s, "ttl_s")
    try:
        return granted_at + timedelta(seconds=ttl)
    except OverflowError as exc:
        raise AuthorityError("ttl_s exceeds the representable timestamp range") from exc


def _budget_decimal(value: Any, field: str) -> Decimal:
    """Return the exact decimal represented by one validated JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise AuthorityError(f"{field} must be a non-negative number")
    if isinstance(value, int):
        return Decimal(value)
    if not math.isfinite(value):
        raise AuthorityError(f"{field} must be a finite number")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:  # pragma: no cover - value is finite
        raise AuthorityError(f"{field} must be a finite number") from exc


def _tokens_max_integer(value: Any, field: str) -> int:
    """Return an exact non-negative signed-64-bit tokens_max integer.

    Token accounting is exact integer arithmetic. Accepting a float here would
    silently round values above 2**53, while accepting an integer beyond the
    SQLite range would make persistence fail later with an untyped driver
    ``OverflowError``. Enforce the durable representation at every boundary.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _SQLITE_INT_MAX
    ):
        raise AuthorityError(f"{field} must be a non-negative signed 64-bit integer")
    return value


def _normalized_reservation_metadata(
    value: Any, field: str = "budget reservation metadata"
) -> dict[str, Any] | None:
    """Canonicalize optional reservation metadata, or fail closed typed."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AuthorityError(f"{field} must be an object")
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuthorityError(f"{field} is not canonically serializable") from exc
    return normalized or None


def _bounded_repr(value: Any) -> str:
    """Render an untrusted value as bounded, JSON-safe audit text."""

    try:
        text = repr(value)
    except Exception:  # pragma: no cover - defensive: hostile __repr__
        return f"<unrepresentable {type(value).__name__}>"
    if len(text) > _FENCE_REPR_MAX:
        text = text[: _FENCE_REPR_MAX - 3] + "..."
    return text


def _bindable_fence(value: int) -> bool:
    """True if a non-``bool`` ``int`` fits the signed 64-bit ``fence`` column."""

    return _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX


def _normalize_claimed_fence(value: Any) -> tuple[int | None, dict[str, Any]]:
    """Normalize a caller-supplied fence to a column- and hash-stable value.

    Returns ``(fence, audit)``. ``fence`` is both bound to the INTEGER ``fence``
    column and folded into the event hash, so it must be exactly the type *and*
    value SQLite hands back on read: a non-``bool`` ``int`` inside the signed
    64-bit range, or ``None``. Every other value is malformed and normalizes to
    ``None``.
    """

    if value is None:
        return None, {"claimed_fence": None, "claimed_fence_malformed": False}
    if isinstance(value, int) and not isinstance(value, bool):
        if _bindable_fence(value):
            return value, {"claimed_fence": value, "claimed_fence_malformed": False}
        reason = "out_of_range"
    else:
        reason = "not_an_integer"
    return None, {
        "claimed_fence": None,
        "claimed_fence_malformed": True,
        "claimed_fence_reason": reason,
        "claimed_fence_type": type(value).__name__,
        "claimed_fence_repr": _bounded_repr(value),
    }


def _canonical_identity(value: Any) -> str | None:
    """Canonical JSON of an untrusted envelope, or ``None`` if unrepresentable."""

    try:
        return _canonical_json(dict(value))
    except (TypeError, ValueError, RecursionError):
        return None


def _valid_sha256(value: Any, field: str) -> str:
    """Return a validated lowercase SHA-256 hex digest or fail closed."""

    if not isinstance(value, str) or len(value) != _SHA256_RE_LEN:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except (TypeError, ValueError) as exc:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest") from exc
    if value.lower() != value:
        raise AuthorityError(f"{field} must be a SHA-256 hex digest")
    return value


# ── Envelope validation (self-contained, no enforcement dependency) ────────────


def _env_aware_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _env_nonnegative_number(value: Any) -> bool:
    """True for a finite, non-negative, non-``bool`` number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _env_nonempty_strings(value: Any, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def validate_sealed_envelope(envelope: Mapping[str, Any]) -> list[str]:
    """Return a sorted list of field-level errors in a sealed task envelope v2.

    An empty list means the envelope is structurally valid. The store calls this
    before persisting any grant; callers may call it independently for pre-flight
    checks. No I/O is performed.
    """

    errors = [field for field in _REQUIRED_FIELDS if envelope.get(field) is None]
    if envelope.get("envelope_version") != 2:
        errors.append("envelope_version")
    for field in ("graph_id", "node_id", "attempt_id", "objective"):
        if (
            not isinstance(envelope.get(field), str)
            or not envelope.get(field, "").strip()
        ):
            errors.append(field)
    if len(str(envelope.get("objective") or "")) > 2000:
        errors.append("objective")
    if envelope.get("execution_surface") not in _ALLOWED_SURFACES:
        errors.append("execution_surface")
    if envelope.get("lane") not in _ALLOWED_LANES:
        errors.append("lane")
    if not _env_nonempty_strings(envelope.get("roots"), allow_empty=False):
        errors.append("roots")
    elif any(not Path(root).expanduser().is_absolute() for root in envelope["roots"]):
        errors.append("roots")

    permissions = envelope.get("permissions")
    if not isinstance(permissions, Mapping):
        errors.append("permissions")
        permissions = {}
    for field in ("read", "write", "spawn"):
        values = permissions.get(field)
        if not _env_nonempty_strings(values):
            errors.append(f"permissions.{field}")
        elif field in {"read", "write"} and any(
            not Path(value).expanduser().is_absolute() for value in (values or [])
        ):
            errors.append(f"permissions.{field}")

    network_policy = envelope.get("network_policy")
    if network_policy not in _ALLOWED_NETWORK:
        errors.append("network_policy")
    if network_policy == "allowlist" and not _env_nonempty_strings(
        envelope.get("network_allowlist"), allow_empty=False
    ):
        errors.append("network_allowlist")
    if envelope.get("exec_policy") not in _ALLOWED_EXEC:
        errors.append("exec_policy")
    if envelope.get("destructive_policy") not in _ALLOWED_DESTRUCTIVE:
        errors.append("destructive_policy")

    budgets = envelope.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets")
        budgets = {}
    for field in ("usd_max", "wall_clock_s_max"):
        if not _env_nonnegative_number(budgets.get(field)):
            errors.append(f"budgets.{field}")
    tokens_max = budgets.get("tokens_max")
    if (
        isinstance(tokens_max, bool)
        or not isinstance(tokens_max, int)
        or not 0 <= tokens_max <= _SQLITE_INT_MAX
    ):
        errors.append("budgets.tokens_max")
    if _env_aware_datetime(envelope.get("deadline_utc")) is None:
        errors.append("deadline_utc")

    retry = envelope.get("retry_policy")
    if not isinstance(retry, Mapping):
        errors.append("retry_policy")
        retry = {}
    max_attempts = retry.get("max_attempts")
    retry_eligible = retry.get("retry_eligible")
    valid_max_attempts = (
        isinstance(max_attempts, int)
        and not isinstance(max_attempts, bool)
        and max_attempts >= 1
    )
    if not valid_max_attempts:
        errors.append("retry_policy.max_attempts")
    if not isinstance(retry_eligible, bool):
        errors.append("retry_policy.retry_eligible")
    if not isinstance(retry.get("backoff_s"), list) or not all(
        _env_nonnegative_number(item) for item in retry.get("backoff_s", [])
    ):
        errors.append("retry_policy.backoff_s")
    writes = permissions.get("write") if isinstance(permissions, Mapping) else []
    if writes and (max_attempts != 1 or retry_eligible is not False):
        errors.append("retry_policy.mutation")
    if (
        valid_max_attempts
        and max_attempts > 1
        and envelope.get("execution_surface") not in _STATELESS_SURFACES
    ):
        errors.append("retry_policy.execution_surface")

    cancellation = envelope.get("cancellation")
    if not isinstance(cancellation, Mapping):
        errors.append("cancellation")
        cancellation = {}
    if cancellation.get("mode") != "cooperative":
        errors.append("cancellation.mode")
    if not _env_nonnegative_number(cancellation.get("grace_s")):
        errors.append("cancellation.grace_s")

    evidence = envelope.get("evidence_policy")
    if not isinstance(evidence, Mapping):
        errors.append("evidence_policy")
        evidence = {}
    if not _env_nonempty_strings(
        evidence.get("required_receipt_kinds"), allow_empty=False
    ):
        errors.append("evidence_policy.required_receipt_kinds")
    min_receipts = evidence.get("min_receipts")
    if (
        not isinstance(min_receipts, int)
        or isinstance(min_receipts, bool)
        or min_receipts < 1
    ):
        errors.append("evidence_policy.min_receipts")

    verifier = envelope.get("verifier_policy")
    if not isinstance(verifier, Mapping):
        errors.append("verifier_policy")
        verifier = {}
    for field in ("required", "verifier_lane_must_differ", "clean_workspace_required"):
        if not isinstance(verifier.get(field), bool):
            errors.append(f"verifier_policy.{field}")
    if writes and not all(
        verifier.get(field) is True
        for field in (
            "required",
            "verifier_lane_must_differ",
            "clean_workspace_required",
        )
    ):
        errors.append("verifier_policy.mutation")

    for field in ("idempotency_key", "planner_hash", "policy_hash", "input_hash"):
        if not _HASH_RE.fullmatch(str(envelope.get(field) or "")):
            errors.append(field)

    lease = envelope.get("lease")
    if not isinstance(lease, Mapping):
        errors.append("lease")
        lease = {}
    if not isinstance(lease.get("holder"), str) or not lease.get("holder", "").strip():
        errors.append("lease.holder")
    granted = _env_aware_datetime(lease.get("granted_utc"))
    if granted is None:
        errors.append("lease.granted_utc")
    if not _env_nonnegative_number(lease.get("ttl_s")) or lease.get("ttl_s") == 0:
        errors.append("lease.ttl_s")
    if not isinstance(lease.get("renewable"), bool):
        errors.append("lease.renewable")
    fence = envelope.get("fence")
    if (
        not isinstance(fence, int)
        or isinstance(fence, bool)
        or fence < 1
        or not _bindable_fence(fence)
    ):
        errors.append("fence")
    return sorted(set(errors))


# ── Context-scoped envelope binding ───────────────────────────────────────────


@contextmanager
def bind_sealed_envelope(
    envelope: Mapping[str, Any], *, current_fence: int | None = None
) -> Iterator[None]:
    """Bind a defensive immutable copy to the current execution context.

    The bound envelope and fence are scoped to the calling async task or thread
    via ``ContextVar``; they reset unconditionally on exit, even on exception.
    """

    frozen = _freeze_envelope_value(envelope)
    envelope_token = _CURRENT_ENVELOPE.set(frozen)
    fence_token = _CURRENT_FENCE.set(current_fence)
    try:
        yield
    finally:
        _CURRENT_FENCE.reset(fence_token)
        _CURRENT_ENVELOPE.reset(envelope_token)


def current_sealed_envelope() -> Mapping[str, Any] | None:
    """Return the envelope bound to the current execution context, or ``None``."""

    return _CURRENT_ENVELOPE.get()


def current_authoritative_fence() -> int | None:
    """Return the fence bound by the Hermes control plane for this context."""

    return _CURRENT_FENCE.get()
