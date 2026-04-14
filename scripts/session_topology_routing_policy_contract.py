#!/usr/bin/env python3
"""Session-topology routing policy contract helpers.

Centralizes schema loading + deterministic policy projection for task-class
model-family preferences and coding qualification thresholds.
"""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


DEFAULT_TASK_CLASS_MODEL_FAMILY_DEFAULTS: Dict[str, str] = {
    "reading": "DeepSeek",
    "triage": "DeepSeek",
    "audit_compression": "DeepSeek",
    "research": "Gemini",
    "planning": "Gemini",
    "comparison": "Kimi",
    "implementation": "Codex",
    "code:generate": "Codex",
    "code:edit": "Codex",
    "code:review": "DeepSeek",
    "code:test": "Codex",
    "code:docs": "DeepSeek",
}

DEFAULT_TASK_CLASS_MODEL_FAMILY_FALLBACKS: Dict[str, List[str]] = {
    "reading": ["Gemini", "Kimi", "Codex"],
    "triage": ["Gemini", "Codex"],
    "audit_compression": ["Gemini", "Kimi", "Codex"],
    "research": ["Kimi", "Codex"],
    "planning": ["Kimi", "Codex"],
    "comparison": ["Gemini", "Codex"],
    "implementation": ["Gemini", "DeepSeek"],
    "code:generate": ["Gemini", "Kimi", "DeepSeek"],
    "code:edit": ["Gemini", "DeepSeek", "Kimi"],
    "code:review": ["Gemini", "Codex", "Kimi"],
    "code:test": ["Gemini", "DeepSeek", "Kimi"],
    "code:docs": ["Gemini", "Codex", "Kimi"],
}

DEFAULT_UNKNOWN_TASK_CLASS_DEFAULT_FAMILY = "Gemini"
DEFAULT_UNKNOWN_TASK_CLASS_FALLBACK_FAMILIES = ["Codex"]

DEFAULT_CODING_HIGH_RISK_MIN_SCORE_0_100: Dict[str, float] = {
    "high": 85.0,
    "critical": 90.0,
}

DEFAULT_CODING_READINESS_REQUIREMENTS: Dict[str, Set[str]] = {
    "high": {"qualified", "provisional"},
    "critical": {"qualified"},
}
DEFAULT_CODING_REQUIRE_QUALIFICATION_SIGNAL = True
DEFAULT_CODING_STRICT_READINESS_TRIGGER_VERIFICATION_CLASSES: Set[str] = {"validator_plus_human"}
DEFAULT_CODING_STRICT_READINESS_TRIGGER_COMPLEXITY_TIERS: Set[str] = {"high"}
DEFAULT_CODING_STRICT_READINESS_ALLOWED_READINESS: Set[str] = {"qualified", "provisional"}

DEFAULT_CODEX_QUOTA_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES: Set[str] = {
    "probationary",
    "quarantined",
}
DEFAULT_CODEX_QUOTA_EXHAUSTION_REASON_PREFIXES: Tuple[str, ...] = (
    "quota_exhausted",
    "quota_exhausted_additional",
    "runtime_bodyless_usage_limit",
)
DEFAULT_TELEGRAM_DIRECT_WORKER_TARGET_DISALLOWED_LANE_TOKENS: Set[str] = {
    "inbox",
    "main",
    "main_session",
    "telegram_direct",
    "direct",
    "cockpit",
}


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_schema(path: Path) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}, None
    if not path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(path)}, None
    try:
        payload = load_json_file(path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "schema_path": str(path), "detail": str(exc)}, None
    if not isinstance(payload, dict):
        return False, "gate_unavailable", {"error": "schema_not_object", "schema_path": str(path)}, None
    return True, None, {"schema_path": str(path)}, payload


def load_routing_policy(policy_path: Path, policy_schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    try:
        policy_doc = load_json_file(policy_path)
    except Exception as exc:
        return (
            False,
            "routing_policy_invalid",
            {"error": "routing_policy_unreadable", "path": str(policy_path), "detail": str(exc)},
            None,
        )

    if not isinstance(policy_doc, dict):
        return False, "routing_policy_invalid", {"error": "routing_policy_not_object", "path": str(policy_path)}, None

    ok, reason, details, schema_doc = _load_json_schema(policy_schema_path)
    if not ok:
        return False, reason, details, None

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(policy_doc),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return (
            False,
            "routing_policy_invalid",
            {
                "error": "routing_policy_schema_validation_failed",
                "path": str(policy_path),
                "data_path": json_ptr(err.absolute_path),
                "schema_path": json_ptr(err.absolute_schema_path),
                "message": str(err.message),
            },
            None,
        )

    return True, None, {"path": str(policy_path), "schema_path": str(policy_schema_path), "policy_id": policy_doc.get("policy_id")}, policy_doc


def _task_class_entries(policy_doc: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    matrix = policy_doc.get("task_class_model_family") if isinstance(policy_doc.get("task_class_model_family"), Mapping) else {}
    rows = matrix.get("task_classes") if isinstance(matrix.get("task_classes"), list) else []

    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task_class = str(row.get("task_class") or "").strip()
        if not task_class:
            continue
        out[task_class] = row
    return out


def routing_policy_known_task_classes(policy_doc: Mapping[str, Any]) -> Set[str]:
    return set(_task_class_entries(policy_doc).keys())


def routing_policy_task_family(policy_doc: Mapping[str, Any], task_class: str) -> Tuple[str, List[str]]:
    matrix = policy_doc.get("task_class_model_family") if isinstance(policy_doc.get("task_class_model_family"), Mapping) else {}
    defaults = matrix.get("defaults") if isinstance(matrix.get("defaults"), Mapping) else {}

    default_family = str(defaults.get("default_model_family") or DEFAULT_UNKNOWN_TASK_CLASS_DEFAULT_FAMILY).strip() or DEFAULT_UNKNOWN_TASK_CLASS_DEFAULT_FAMILY
    default_fallbacks = [
        str(item or "").strip()
        for item in (defaults.get("fallback_model_families") if isinstance(defaults.get("fallback_model_families"), list) else DEFAULT_UNKNOWN_TASK_CLASS_FALLBACK_FAMILIES)
        if str(item or "").strip()
    ]

    entry = _task_class_entries(policy_doc).get(str(task_class or "").strip())
    if not isinstance(entry, Mapping):
        return default_family, default_fallbacks

    task_default_family = str(entry.get("default_model_family") or default_family).strip() or default_family
    task_fallbacks = [
        str(item or "").strip()
        for item in (entry.get("fallback_model_families") if isinstance(entry.get("fallback_model_families"), list) else default_fallbacks)
        if str(item or "").strip()
    ]
    return task_default_family, task_fallbacks


def routing_policy_coding_min_score(policy_doc: Mapping[str, Any], risk_tier: str) -> Optional[float]:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    thresholds = coding.get("high_risk_min_score_0_100") if isinstance(coding.get("high_risk_min_score_0_100"), Mapping) else {}

    token = str(risk_tier or "").strip().lower()
    raw = thresholds.get(token)
    if raw is None:
        raw = DEFAULT_CODING_HIGH_RISK_MIN_SCORE_0_100.get(token)
    if raw is None:
        return None

    try:
        return float(raw)
    except Exception:
        return None


def routing_policy_coding_allowed_readiness(policy_doc: Mapping[str, Any], risk_tier: str) -> Set[str]:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    readiness = coding.get("readiness_requirements") if isinstance(coding.get("readiness_requirements"), Mapping) else {}

    token = str(risk_tier or "").strip().lower()
    raw = readiness.get(token)

    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.add(text)

    if values:
        return values

    return set(DEFAULT_CODING_READINESS_REQUIREMENTS.get(token, set()))


def routing_policy_coding_require_qualification_signal(policy_doc: Mapping[str, Any]) -> bool:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    raw = coding.get("require_qualification_signal_for_coding")
    if isinstance(raw, bool):
        return raw
    return bool(DEFAULT_CODING_REQUIRE_QUALIFICATION_SIGNAL)


def routing_policy_coding_strict_readiness_trigger_verification_classes(policy_doc: Mapping[str, Any]) -> Set[str]:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    strict = coding.get("strict_readiness_profile") if isinstance(coding.get("strict_readiness_profile"), Mapping) else {}
    raw = strict.get("trigger_verification_classes")

    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip()
            if token:
                values.add(token)

    if values or isinstance(raw, list):
        return values

    return set(DEFAULT_CODING_STRICT_READINESS_TRIGGER_VERIFICATION_CLASSES)


def routing_policy_coding_strict_readiness_trigger_complexity_tiers(policy_doc: Mapping[str, Any]) -> Set[str]:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    strict = coding.get("strict_readiness_profile") if isinstance(coding.get("strict_readiness_profile"), Mapping) else {}
    raw = strict.get("trigger_complexity_tiers")

    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip().lower()
            if token:
                values.add(token)

    if values or isinstance(raw, list):
        return values

    return set(DEFAULT_CODING_STRICT_READINESS_TRIGGER_COMPLEXITY_TIERS)


def routing_policy_coding_strict_readiness_allowed_readiness(policy_doc: Mapping[str, Any]) -> Set[str]:
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    strict = coding.get("strict_readiness_profile") if isinstance(coding.get("strict_readiness_profile"), Mapping) else {}
    raw = strict.get("allowed_readiness_states")

    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip()
            if token:
                values.add(token)

    if values:
        return values

    return set(DEFAULT_CODING_STRICT_READINESS_ALLOWED_READINESS)


def _routing_policy_codex_quota(policy_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = policy_doc.get("codex_quota_routing")
    if isinstance(raw, Mapping):
        return raw
    return {}


def routing_policy_codex_quota_routine_dispatch_lane_disable_statuses(policy_doc: Mapping[str, Any]) -> Set[str]:
    raw = _routing_policy_codex_quota(policy_doc).get("routine_dispatch_lane_disable_statuses")
    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip().lower()
            if token:
                values.add(token)
    if values:
        return values
    return set(DEFAULT_CODEX_QUOTA_ROUTINE_DISPATCH_LANE_DISABLE_STATUSES)


def routing_policy_codex_quota_exhaustion_reason_prefixes(policy_doc: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = _routing_policy_codex_quota(policy_doc).get("quota_exhaustion_reason_prefixes")
    values: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip().lower()
            if token and token not in values:
                values.append(token)
    if values:
        return tuple(values)
    return tuple(DEFAULT_CODEX_QUOTA_EXHAUSTION_REASON_PREFIXES)


def routing_policy_telegram_direct_worker_target_disallowed_lane_tokens(policy_doc: Mapping[str, Any]) -> Set[str]:
    raw = _routing_policy_codex_quota(policy_doc).get("telegram_direct_worker_target_disallowed_lane_tokens")
    values: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item or "").strip().lower().replace("-", "_")
            if token:
                values.add(token)
    if values:
        return values
    return set(DEFAULT_TELEGRAM_DIRECT_WORKER_TARGET_DISALLOWED_LANE_TOKENS)


def routing_policy_qualification_signal_max_age_seconds(policy_doc: Mapping[str, Any]) -> int:
    """Get the maximum age in seconds for qualification signals."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    raw = coding.get("qualification_signal_max_age_seconds")
    if isinstance(raw, int) and raw >= 3600:
        return raw
    # Default: 24 hours
    return 86400


def _risk_tier_token(risk_tier: str) -> str:
    return str(risk_tier or "").strip().lower()


def _parse_utc_timestamp(raw: Any) -> Optional[dt.datetime]:
    token = str(raw or "").strip()
    if not token:
        return None
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(token)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def routing_policy_qualification_signal_max_age_seconds_by_risk_tier(policy_doc: Mapping[str, Any], risk_tier: str) -> int:
    """Get the maximum age in seconds for qualification signals for a specific risk tier."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    token = _risk_tier_token(risk_tier)

    # First check tier-specific configuration
    by_tier = coding.get("qualification_signal_max_age_seconds_by_risk_tier")
    if isinstance(by_tier, Mapping):
        tier_value = by_tier.get(token)
        if isinstance(tier_value, int) and tier_value >= 3600:
            return tier_value

    # Fall back to global max age
    return routing_policy_qualification_signal_max_age_seconds(policy_doc)


def routing_policy_provider_evidence_max_age_seconds(policy_doc: Mapping[str, Any]) -> int:
    """Get the maximum age in seconds for provider evidence."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    raw = coding.get("provider_evidence_max_age_seconds")
    if isinstance(raw, int) and raw >= 3600:
        return raw
    # Default: 48 hours
    return 172800


def routing_policy_provider_evidence_max_age_seconds_by_risk_tier(policy_doc: Mapping[str, Any], risk_tier: str) -> int:
    """Get the maximum age in seconds for provider evidence for a specific risk tier."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    token = _risk_tier_token(risk_tier)

    # First check tier-specific configuration
    by_tier = coding.get("provider_evidence_max_age_seconds_by_risk_tier")
    if isinstance(by_tier, Mapping):
        tier_value = by_tier.get(token)
        if isinstance(tier_value, int) and tier_value >= 3600:
            return tier_value

    # Fall back to global max age
    return routing_policy_provider_evidence_max_age_seconds(policy_doc)


def routing_policy_legacy_missing_timestamp_grace_period_seconds(policy_doc: Mapping[str, Any]) -> Optional[int]:
    """Get the grace period in seconds for legacy packets missing timestamps."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    
    raw = coding.get("legacy_missing_timestamp_grace_period_seconds")
    if raw is None:
        return None
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def routing_policy_legacy_missing_timestamp_grace_period_seconds_by_risk_tier(policy_doc: Mapping[str, Any], risk_tier: str) -> Optional[int]:
    """Get the grace period in seconds for legacy packets missing timestamps for a specific risk tier.

    Fail-closed doctrine: tier overrides are only active when a global grace window is explicitly enabled.
    If the global switch is unset/null, legacy grace is disabled even if tier values exist.
    When globally enabled, grace is bounded to `policy.generated_at + grace_period_seconds`.
    """
    status = routing_policy_legacy_missing_timestamp_grace_status_by_risk_tier(policy_doc, risk_tier)
    if status.get("grace_window_active") is True:
        value = status.get("grace_period_seconds")
        if isinstance(value, int) and value >= 0:
            return value
    return None


def routing_policy_legacy_missing_timestamp_grace_status_by_risk_tier(
    policy_doc: Mapping[str, Any],
    risk_tier: str,
    *,
    now_utc: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """Return bounded grace-window status for legacy timestamp migration compatibility."""
    coding = policy_doc.get("coding_qualification") if isinstance(policy_doc.get("coding_qualification"), Mapping) else {}
    token = _risk_tier_token(risk_tier)
    now = now_utc.astimezone(dt.timezone.utc) if isinstance(now_utc, dt.datetime) else dt.datetime.now(dt.timezone.utc)

    global_grace = routing_policy_legacy_missing_timestamp_grace_period_seconds(policy_doc)
    status: Dict[str, Any] = {
        "risk_tier": token or None,
        "grace_window_active": False,
        "global_grace_enabled": isinstance(global_grace, int) and global_grace >= 0,
        "configured_global_grace_period_seconds": global_grace,
        "grace_period_seconds": None,
        "inactive_reason": None,
    }

    if global_grace is None:
        status["inactive_reason"] = "global_grace_disabled"
        return status

    # First check tier-specific configuration
    by_tier = coding.get("legacy_missing_timestamp_grace_period_seconds_by_risk_tier")
    configured_grace = global_grace
    if isinstance(by_tier, Mapping):
        tier_value = by_tier.get(token)
        if isinstance(tier_value, int) and tier_value >= 0:
            configured_grace = tier_value

    status["grace_period_seconds"] = configured_grace

    generated_at = _parse_utc_timestamp(policy_doc.get("generated_at"))
    if generated_at is None:
        status["inactive_reason"] = "policy_generated_at_missing_or_invalid"
        return status

    if generated_at > now:
        status["policy_generated_at"] = generated_at.isoformat().replace("+00:00", "Z")
        status["inactive_reason"] = "policy_generated_at_in_future"
        return status

    expires_at = generated_at + dt.timedelta(seconds=configured_grace)
    remaining_seconds = int((expires_at - now).total_seconds())

    status["policy_generated_at"] = generated_at.isoformat().replace("+00:00", "Z")
    status["grace_window_expires_at"] = expires_at.isoformat().replace("+00:00", "Z")
    status["grace_window_remaining_seconds"] = max(0, remaining_seconds)

    if remaining_seconds >= 0:
        status["grace_window_active"] = True
        status["inactive_reason"] = None
        return status

    status["inactive_reason"] = "grace_window_expired"
    return status
