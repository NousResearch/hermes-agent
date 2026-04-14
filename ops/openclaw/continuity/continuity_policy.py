#!/usr/bin/env python3
"""Shared continuity freshness/taxonomy policy constants.

Keep this helper intentionally tiny so shell-embedded Python surfaces can import
it without adding runtime risk.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
from typing import Any, Final, Iterable, Mapping

# Freshness defaults (seconds)
DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC: Final[int] = 6 * 60 * 60  # 21600
DEFAULT_HANDOVER_FRESHNESS_MAX_AGE_SEC: Final[int] = 30 * 60  # 1800
DEFAULT_CHECKPOINT_FRESHNESS_MAX_AGE_SEC: Final[int] = 30 * 60  # 1800
DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC: Final[int] = 6 * 60 * 60  # 21600

# Queue-remediation threshold defaults (seconds)
DEFAULT_CONTINUITY_ORPHANED_RUNNING_MIN_SEC: Final[int] = 30 * 60  # 1800
DEFAULT_CONTINUITY_QUEUE_STALE_WAVE_READY_IDLE_SEC: Final[int] = 30 * 60  # 1800

# Not-ready taxonomy partition used across continuity/operator surfaces.
DRIFT_REASON_SET: Final[frozenset[str]] = frozenset(
    {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }
)

# Auto-reconcile is intentionally narrower than general drift partition.
AUTO_RECONCILE_DRIFT_REASON_SET: Final[frozenset[str]] = frozenset(
    {
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }
)

# Verify-gate preflight predicted blocker severity taxonomy.
# Keep this centralized so downstream continuity/operator surfaces do not drift
# when new blocker families are introduced.
VERIFY_GATE_PREFLIGHT_SEVERE_EXACT_REASONS: Final[frozenset[str]] = frozenset(
    {
        "strict_autonomy_required_override_denied",
    }
)

VERIFY_GATE_PREFLIGHT_SEVERE_REASON_PREFIXES: Final[tuple[str, ...]] = (
    "layered_health_gate:",
    "failover_stress_runtime_evidence_gate:",
    "execution_supervisor_launch_readiness_severity_gate:",
    "execution_supervisor_probe_execution_gate:",
    "execution_supervisor_worker_health_canary_gate:",
)


def is_severe_verify_gate_preflight_blocker(reason: Any) -> bool:
    """True when verify-gate preflight predicted blocker is blocker-grade.

    This is intentionally narrow/whitelist-driven so warn-vs-blocker posture
    stays explicit and deterministic across all wrapper surfaces.
    """

    blocker = str(reason or "").strip()
    if not blocker:
        return False
    if blocker in VERIFY_GATE_PREFLIGHT_SEVERE_EXACT_REASONS:
        return True
    return any(blocker.startswith(prefix) for prefix in VERIFY_GATE_PREFLIGHT_SEVERE_REASON_PREFIXES)

# Generation-pointer fail-close reason taxonomy used by operator surfaces.
GENERATION_POINTER_READ_PHASE_PIN_SUPPRESSIBLE_REASON_SET: Final[frozenset[str]] = frozenset(
    {
        "generation_pointer_current_sha_mismatch",
        "generation_pointer_current_generated_at_mismatch",
        "generation_pointer_generation_mismatch",
    }
)

CURRENT_PUBLISH_LOCK_OWNER_REL: Final[str] = "state/continuity/latest/current_publish.lock.owner.json"
PUBLISH_LOCK_DERIVATIVE_SURFACE_SET: Final[frozenset[str]] = frozenset(
    {
        "blocker_registry",
        "operator_mission_control",
    }
)
PUBLISH_LOCK_CANONICAL_SOURCE_SURFACE_SET: Final[frozenset[str]] = frozenset(
    {
        "continuity_current",
        "continuity_now",
    }
)

PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON: Final[str] = (
    "continuity_current_publish_lock_wait_budget_exceeded"
)
PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON: Final[str] = (
    "continuity_current_publish_lock_hold_budget_exceeded"
)
PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON: Final[str] = (
    "continuity_current_publish_lock_owner_not_alive"
)
PUBLISH_LOCK_WARNING_REASON_SET: Final[frozenset[str]] = frozenset(
    {
        PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON,
        PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON,
        PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON,
    }
)
PUBLISH_LOCK_SOURCE_CURRENT_STALE_REASON: Final[str] = (
    "blocker_registry_source_current_stale"
)
PUBLISH_LOCK_SOURCE_CURRENT_GENERATED_AT_MISMATCH_REASON: Final[str] = (
    "blocker_registry_source_current_generated_at_mismatch"
)
PUBLISH_LOCK_SOURCE_DEGRADED_REASON_SET: Final[frozenset[str]] = frozenset(
    {
        PUBLISH_LOCK_SOURCE_CURRENT_STALE_REASON,
        PUBLISH_LOCK_SOURCE_CURRENT_GENERATED_AT_MISMATCH_REASON,
    }
)
# Keep mission-control escalation narrower than raw presence: only actionable
# or budget-breached registry statuses should surface operator rows by default.
PUBLISH_LOCK_ACTIONABLE_STATUS_SET: Final[frozenset[str]] = frozenset(
    {
        "invalid",
        "unreadable",
        "owner_not_alive",
        "wait_budget_exceeded",
        "hold_budget_exceeded",
    }
)


def unique_reason_rows(rows: Iterable[Any]) -> list[str]:
    """Normalize reason-like rows to unique, trimmed, stable-order strings."""

    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        txt = str(row or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


def publish_lock_surface_declared(surface_name: Any) -> bool:
    """True when a surface is explicitly allowed to derive publish-lock posture."""

    return str(surface_name or "").strip() in PUBLISH_LOCK_DERIVATIVE_SURFACE_SET


def project_blocker_registry_publish_lock_signal(
    *,
    blocker_registry: Any = None,
    publish_lock: Any = None,
    blockers: Any = None,
    current_generated_at: Any = None,
) -> dict[str, Any]:
    """Project actionable publish-lock surfacing from blocker-registry.

    This keeps mission-control consumption explicitly downstream of the
    blocker_registry derivative contract and within the frozen
    blocker_registry/operator_mission_control boundary instead of reaching
    deeper into continuity_now/current for additional publish-lock semantics.
    """

    registry_map = blocker_registry if isinstance(blocker_registry, Mapping) else {}
    publish_lock_map = publish_lock if isinstance(publish_lock, Mapping) else {}
    if not publish_lock_map and isinstance(registry_map.get("publish_lock"), Mapping):
        publish_lock_map = registry_map.get("publish_lock")

    blocker_rows = blockers if isinstance(blockers, list) else registry_map.get("blockers")
    warning_reasons: list[str] = []
    if isinstance(blocker_rows, list):
        for row in blocker_rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("severity") or "").strip() != "warn":
                continue
            reason = str(row.get("reason") or "").strip()
            if reason in PUBLISH_LOCK_WARNING_REASON_SET:
                warning_reasons.append(reason)

    registry_freshness = registry_map.get("freshness") if isinstance(registry_map.get("freshness"), Mapping) else {}
    source_current = registry_map.get("source_current") if isinstance(registry_map.get("source_current"), Mapping) else {}
    source_current_path = (
        str(source_current.get("path") or registry_freshness.get("source") or "").strip() or None
    )
    source_current_generated = (
        str(source_current.get("generated_at") or registry_freshness.get("source_generated_at") or "").strip() or None
    )
    expected_current_generated = str(current_generated_at or "").strip() or None
    source_current_matches_current_generated_at: bool | None = None
    if source_current_generated and expected_current_generated:
        source_current_matches_current_generated_at = (
            source_current_generated == expected_current_generated
        )

    source_current_fresh = registry_freshness.get("fresh")
    if not isinstance(source_current_fresh, bool):
        source_current_fresh = None

    source_degraded_reasons: list[str] = []
    if source_current_fresh is False:
        source_degraded_reasons.append(PUBLISH_LOCK_SOURCE_CURRENT_STALE_REASON)
    if source_current_matches_current_generated_at is False:
        source_degraded_reasons.append(PUBLISH_LOCK_SOURCE_CURRENT_GENERATED_AT_MISMATCH_REASON)
    source_degraded_reasons = unique_reason_rows(source_degraded_reasons)

    warning_reasons = unique_reason_rows(warning_reasons)
    status = str(publish_lock_map.get("status") or "").strip() or None
    present = publish_lock_map.get("present") is True
    action_required = bool(publish_lock_map.get("action_required") is True or warning_reasons)
    surface_active = bool(
        action_required
        or (status in PUBLISH_LOCK_ACTIONABLE_STATUS_SET if status else False)
        or warning_reasons
    )

    source_current_age_sec = registry_freshness.get("source_age_sec")
    if not isinstance(source_current_age_sec, (int, float)):
        source_current_age_sec = None
    source_current_max_age_sec = registry_freshness.get("max_age_sec")
    if not isinstance(source_current_max_age_sec, (int, float)):
        source_current_max_age_sec = None

    return {
        "generated_at": str(registry_map.get("generated_at") or "").strip() or None,
        "present": present,
        "path": str(publish_lock_map.get("path") or "").strip() or None,
        "status": status,
        "owner_pid": publish_lock_map.get("owner_pid"),
        "owner_alive": publish_lock_map.get("owner_alive"),
        "owner_age_sec": publish_lock_map.get("owner_age_sec"),
        "lock_wait_sec": publish_lock_map.get("lock_wait_sec"),
        "lock_hold_warn_sec": publish_lock_map.get("lock_hold_warn_sec"),
        "owner_exceeds_wait_budget": publish_lock_map.get("owner_exceeds_wait_budget"),
        "owner_exceeds_lock_hold_warn": publish_lock_map.get("owner_exceeds_lock_hold_warn"),
        "owner_host": publish_lock_map.get("owner_host"),
        "owner_command": publish_lock_map.get("owner_command"),
        "recommended_action": str(publish_lock_map.get("recommended_action") or "").strip() or None,
        "inspect_command": str(publish_lock_map.get("inspect_command") or "").strip() or None,
        "warning_reasons": warning_reasons,
        "action_required": action_required,
        "surface_active": surface_active,
        "source_current_path": source_current_path,
        "source_current_generated_at": source_current_generated,
        "source_current_age_sec": source_current_age_sec,
        "source_current_max_age_sec": source_current_max_age_sec,
        "source_current_fresh": source_current_fresh,
        "source_current_matches_current_generated_at": source_current_matches_current_generated_at,
        "source_degraded": bool(source_degraded_reasons),
        "source_degraded_reasons": source_degraded_reasons,
    }


def parse_iso_utc(raw: Any) -> dt.datetime | None:
    """Best-effort parse for ISO timestamps, treating naive values as UTC."""

    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out


def project_reset_ready_refresh_posture(
    *,
    surface: Any = None,
    latest_payload: Any = None,
    path: Any = None,
    sha256: Any = None,
    present: Any = None,
    now_ts: Any = None,
    freshness_max_age_sec: Any = None,
) -> dict[str, Any]:
    """Project canonical reset-ready-refresh posture from current/latest surfaces.

    This helper is intentionally side-effect free so shell-embedded Python
    surfaces can share the same normalization logic while resolving paths/IO
    locally.
    """

    surface_map = surface if isinstance(surface, Mapping) else {}
    latest_map = latest_payload if isinstance(latest_payload, Mapping) else {}

    path_text = str(path or surface_map.get("path") or "").strip()
    sha_text = str(sha256 or surface_map.get("sha256") or "").strip() or None

    present_value: bool
    if isinstance(present, bool):
        present_value = present
    else:
        present_value = bool(surface_map.get("present") is True or bool(latest_map))

    ok = surface_map.get("ok") if isinstance(surface_map.get("ok"), bool) else None
    if ok is None and isinstance(latest_map.get("ok"), bool):
        ok = latest_map.get("ok")

    phase = str(surface_map.get("phase") or latest_map.get("phase") or "").strip() or None
    if phase is None and ok is True:
        phase = "complete"

    partial_refresh = surface_map.get("partial_refresh") if isinstance(surface_map.get("partial_refresh"), Mapping) else {}
    if not partial_refresh and isinstance(latest_map.get("partial_refresh"), Mapping):
        partial_refresh = latest_map.get("partial_refresh")

    def _partial_flag(name: str) -> bool | None:
        raw_value = partial_refresh.get(name)
        return raw_value if isinstance(raw_value, bool) else None

    partial_current = _partial_flag("current_refreshed")
    partial_proof = _partial_flag("proof_refreshed")
    partial_handover = _partial_flag("handover_refreshed")

    explicit_partial_failure = surface_map.get("partial_failure")
    if isinstance(explicit_partial_failure, bool):
        partial_failure = explicit_partial_failure
    else:
        partial_failure = bool(
            present_value
            and any(value is False for value in [partial_current, partial_proof, partial_handover])
        )

    error_code = str(
        surface_map.get("error_code")
        or (((latest_map.get("error") or {}).get("code")) if isinstance(latest_map.get("error"), Mapping) else "")
        or ""
    ).strip() or None

    explicit_degraded = surface_map.get("degraded")
    if isinstance(explicit_degraded, bool):
        degraded = explicit_degraded
    else:
        degraded = bool(present_value and (ok is False or partial_failure))

    generated_at = str(surface_map.get("generated_at") or latest_map.get("generated_at") or "").strip() or None

    def _coerce_nonnegative_int(raw: Any) -> int | None:
        if isinstance(raw, bool):
            return None
        try:
            return max(0, int(raw))
        except Exception:
            return None

    freshness_limit_sec = _coerce_nonnegative_int(surface_map.get("freshness_limit_sec"))
    if freshness_limit_sec is None:
        freshness_limit_sec = _coerce_nonnegative_int(freshness_max_age_sec)
    if freshness_limit_sec is None:
        freshness_limit_sec = int(DEFAULT_RESET_READY_REFRESH_FRESHNESS_MAX_AGE_SEC)

    age_sec = _coerce_nonnegative_int(surface_map.get("age_sec"))
    fresh = surface_map.get("fresh") if isinstance(surface_map.get("fresh"), bool) else None
    stale = surface_map.get("stale") if isinstance(surface_map.get("stale"), bool) else None

    if fresh is None and isinstance(stale, bool):
        fresh = not stale
    if stale is None and isinstance(fresh, bool):
        stale = not fresh

    if freshness_limit_sec > 0 and (age_sec is None or fresh is None):
        generated_dt = parse_iso_utc(generated_at)
        now_ts_int = _coerce_nonnegative_int(now_ts)
        if generated_dt is not None and now_ts_int is not None:
            derived_age_sec = max(0, int(now_ts_int - int(generated_dt.timestamp())))
            age_sec = derived_age_sec
            fresh = derived_age_sec <= freshness_limit_sec
            stale = not fresh

    if stale is None:
        stale = fresh is False

    status = "missing"
    if present_value:
        if degraded:
            status = "degraded"
        elif ok is True:
            status = "ok"
        else:
            status = "present"

    recommended_action = None
    if degraded or stale:
        recommended_action = "rerun_reset_ready_refresh"
    elif present_value:
        recommended_action = "inspect_reset_ready_refresh_result"

    return {
        "path": path_text,
        "sha256": sha_text,
        "generated_at": generated_at,
        "present": present_value,
        "status": status,
        "ok": ok,
        "phase": phase,
        "error_code": error_code,
        "freshness_limit_sec": freshness_limit_sec,
        "age_sec": age_sec,
        "fresh": fresh,
        "stale": stale,
        "partial_refresh": {
            "current_refreshed": partial_current,
            "proof_refreshed": partial_proof,
            "handover_refreshed": partial_handover,
        },
        "degraded": degraded,
        "partial_failure": partial_failure,
        "action_required": bool(degraded or stale),
        "recommended_action": recommended_action,
    }


def project_reset_ready_refresh_blocker_warning_metadata(
    *,
    posture: Any = None,
) -> dict[str, Any] | None:
    """Project reset-ready-refresh warn-row metadata for blocker-registry.

    Keep the warn metadata contract explicit and rooted in canonical posture so
    blocker_registry does not duplicate field-level shaping logic.
    """

    posture_map = posture if isinstance(posture, Mapping) else {}
    if posture_map.get("present") is not True:
        return None

    return {
        "recommended_action": str(posture_map.get("recommended_action") or "").strip() or None,
        "action_required": posture_map.get("action_required") is True,
        "context": {
            "status": posture_map.get("status"),
            "ok": posture_map.get("ok"),
            "phase": posture_map.get("phase"),
            "error_code": posture_map.get("error_code"),
            "partial_failure": posture_map.get("partial_failure"),
            "generated_at": posture_map.get("generated_at"),
            "freshness_limit_sec": posture_map.get("freshness_limit_sec"),
            "age_sec": posture_map.get("age_sec"),
            "fresh": posture_map.get("fresh"),
            "stale": posture_map.get("stale"),
        },
    }


# Keep reset-ready-refresh escalation intentionally narrow until additional
# operators/evidence justify expansion. Adjacent degraded phases remain warn-level,
# and stale-only posture remains warning-level by contract.
RESET_READY_REFRESH_EXPLICIT_BLOCKER_RULES: Final[tuple[tuple[str, str, str], ...]] = (
    ("alignment_check", "proof_alignment_mismatch", "reset_ready_refresh_alignment_mismatch"),
)


def project_reset_ready_refresh_escalation_reason(
    *,
    posture: Any = None,
    degraded: Any = None,
    phase: Any = None,
    error_code: Any = None,
) -> str | None:
    """Project explicit blocker reason for reset-ready-refresh posture.

    Contract is deliberately whitelist-only: only exact phase/code tuples in
    RESET_READY_REFRESH_EXPLICIT_BLOCKER_RULES can escalate from warning to
    blocker semantics.
    """

    posture_map = posture if isinstance(posture, Mapping) else {}

    degraded_value: bool
    if isinstance(degraded, bool):
        degraded_value = degraded
    else:
        degraded_value = posture_map.get("degraded") is True

    # Explicit boundary: freshness staleness alone stays warning-level and must
    # not enter blocker taxonomy unless an explicit degraded escalation rule
    # matches below.
    if posture_map.get("stale") is True and not degraded_value:
        return None

    if not degraded_value:
        return None

    phase_value = str(phase if phase is not None else posture_map.get("phase") or "").strip()
    error_code_value = str(
        error_code if error_code is not None else posture_map.get("error_code") or ""
    ).strip()

    for expected_phase, expected_error_code, reason in RESET_READY_REFRESH_EXPLICIT_BLOCKER_RULES:
        if phase_value == expected_phase and error_code_value == expected_error_code:
            return reason
    return None


def generation_pointer_core_failclose_reasons(
    *,
    pointer_current_sha256: Any,
    current_sha256: Any,
    pointer_current_generated_at: Any,
    current_generated_at: Any,
    pointer_generation_id: Any,
    current_generation_id: Any,
) -> list[str]:
    """Canonical generation-pointer fail-close reason shaping.

    Keep this helper side-effect free so shell-embedded Python surfaces can call
    it without coupling to their local payload structure.
    """

    reasons: list[str] = []

    pointer_current_sha = str(pointer_current_sha256 or "").strip()
    current_sha = str(current_sha256 or "").strip()
    pointer_current_ts = str(pointer_current_generated_at or "").strip()
    current_ts = str(current_generated_at or "").strip()
    pointer_generation = str(pointer_generation_id or "").strip()
    current_generation = str(current_generation_id or "").strip()

    if not pointer_current_sha:
        reasons.append("generation_pointer_missing_current_sha256")
    elif current_sha and pointer_current_sha != current_sha:
        reasons.append("generation_pointer_current_sha_mismatch")

    if not pointer_current_ts:
        reasons.append("generation_pointer_missing_current_generated_at")
    elif current_ts and pointer_current_ts != current_ts:
        reasons.append("generation_pointer_current_generated_at_mismatch")

    current_dt = parse_iso_utc(current_ts)
    pointer_dt = parse_iso_utc(pointer_current_ts)
    if current_dt is not None and pointer_dt is not None and pointer_dt < current_dt:
        reasons.append("generation_pointer_stale")

    if current_generation and not pointer_generation:
        reasons.append("generation_pointer_missing_generation_id")
    elif current_generation and pointer_generation and current_generation != pointer_generation:
        reasons.append("generation_pointer_generation_mismatch")

    return unique_reason_rows(reasons)


def continuity_now_contract_expected_fields(
    *,
    contract_obj: Any,
    source_refs: Any,
) -> tuple[str, str, str]:
    """Extract normalized expected continuity_now contract pin fields."""

    contract_map = contract_obj if isinstance(contract_obj, Mapping) else {}
    source_map = source_refs if isinstance(source_refs, Mapping) else {}

    expected_sha = str(contract_map.get("sha256") or source_map.get("continuity_now_sha256") or "").strip()
    expected_generated_at = str(contract_map.get("generated_at") or "").strip()
    expected_generation = str(contract_map.get("coherence_build_generation_id") or "").strip()
    return expected_sha, expected_generated_at, expected_generation


def continuity_now_contract_declared(
    *,
    contract_obj: Any,
    source_refs: Any,
    require_sha_pin: bool,
) -> bool:
    """Resolve whether continuity_now contract validation is fail-close declared."""

    if not require_sha_pin:
        return isinstance(contract_obj, Mapping)

    expected_sha, _, _ = continuity_now_contract_expected_fields(
        contract_obj=contract_obj,
        source_refs=source_refs,
    )
    return bool(expected_sha)


def continuity_now_contract_failclose_reasons(
    *,
    contract_declared: Any,
    contract_path: pathlib.Path,
    expected_sha256: Any,
    expected_generated_at: Any,
    expected_coherence_build_generation_id: Any,
) -> tuple[list[str], str | None, dict[str, Any] | None]:
    """Validate continuity_now contract payload and shape fail-close reasons.

    Returns `(reasons, actual_sha256, payload)` where payload is populated only
    when a readable JSON object was loaded.
    """

    declared = bool(contract_declared)
    if not declared:
        return [], None, None

    expected_sha = str(expected_sha256 or "").strip()
    expected_generated = str(expected_generated_at or "").strip()
    expected_generation = str(expected_coherence_build_generation_id or "").strip()

    path = pathlib.Path(contract_path)
    if not path.exists():
        return ["continuity_now_contract_missing"], None, None

    try:
        raw = path.read_text(encoding="utf-8")
        actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("continuity_now_contract_not_object")

        reasons: list[str] = []
        if expected_sha and actual_sha != expected_sha:
            reasons.append("continuity_now_contract_sha_mismatch")

        actual_generated = str(payload.get("generated_at") or "").strip()
        if expected_generated and actual_generated != expected_generated:
            reasons.append("continuity_now_contract_generated_at_mismatch")

        actual_generation = str((((payload.get("coherence") or {}).get("build_generation_id") or "")).strip())
        if expected_generation and actual_generation != expected_generation:
            reasons.append("continuity_now_contract_generation_mismatch")

        return unique_reason_rows(reasons), actual_sha, payload
    except Exception:
        return ["continuity_now_contract_unreadable"], None, None


def read_nonnegative_int_env(name: str, *, default: int) -> int:
    """Read an integer env var, clamped to >=0, with deterministic fallback."""

    try:
        return max(0, int(os.environ.get(name, str(int(default)))))
    except Exception:
        return int(default)
