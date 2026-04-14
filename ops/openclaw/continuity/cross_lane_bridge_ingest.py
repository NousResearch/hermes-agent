#!/usr/bin/env python3
"""Deterministic validator/ingest runner for Controlled Cross-Lane Bridge Contract v1."""

from __future__ import annotations

import argparse
import datetime as dt
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Optional

try:  # pragma: no cover - dependency wiring is tested in contract tests
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "cross_lane_bridge_object.schema.json"
DEFAULT_DECISION_LOG = Path("state/continuity/latest/cross_lane_bridge_ingest_decisions.jsonl")

SCHEMA_VERSION = "lane.bridge_ingest_decision.v1"
BRIDGE_SCHEMA_VERSION = "lane.bridge_object.v1"

ALLOWED_OBJECT_CLASSES = {
    "doctrine_object",
    "promotion_candidate",
    "lane_crossover_packet",
    "evidence_closeout",
    "approved_artifact_ref",
}

EXPECTED_SCHEMA_REF_BY_CLASS = {
    "doctrine_object": "docs/ops/schemas/doctrine_object.schema.json",
    "promotion_candidate": "docs/ops/schemas/promotion_candidate.schema.json",
    "lane_crossover_packet": "docs/ops/schemas/lane_crossover_packet.schema.json",
    "evidence_closeout": "docs/ops/schemas/evidence_closeout.schema.json",
}

CANONICAL_REJECTION_CODES = {
    "schema_invalid",
    "schema_version_unsupported",
    "unknown_object_class",
    "lane_identity_mismatch",
    "lane_epoch_mismatch",
    "object_not_found",
    "content_hash_missing",
    "content_hash_mismatch",
    "object_schema_mismatch",
    "promotion_missing",
    "promotion_state_invalid",
    "promotion_gate_not_satisfied",
    "review_not_approved",
    "leakage_risk",
    "classification_forbidden",
    "redaction_required",
    "inline_context_over_limit",
    "unverified_content_blocked",
    "cross_lane_write_scope_violation",
    "expired_bridge_object",
    "gate_unavailable",
}

_PROMOTED_STATES = {"APPROVED", "PROMOTED"}
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_PROMOTION_ID_RE = re.compile(r"^prom_[a-z0-9._-]+$")
_LANE_CROSSOVER_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "lane_crossover_packet.schema.json"
_LANE_CROSSOVER_GUARD_PATH = Path(__file__).with_name("lane_crossover_ingress_guard.py")

_LANE_CROSSOVER_REJECTION_MAP = {
    "packet_not_object": "schema_invalid",
    "contamination_guard_missing": "schema_invalid",
    "contamination_guard_inline_context_invalid": "schema_invalid",
    "contamination_guard_inline_context_negative": "schema_invalid",
    "contamination_guard_inline_context_exceeds_max": "inline_context_over_limit",
    "receiver_lane_mismatch": "lane_identity_mismatch",
    "receiver_epoch_mismatch": "lane_epoch_mismatch",
    "self_crossover_forbidden": "lane_identity_mismatch",
    "sender_lane_policy_missing": "lane_identity_mismatch",
    "sender_lane_not_authorized": "lane_identity_mismatch",
    "sender_epoch_mismatch": "lane_epoch_mismatch",
    "packet_type_unknown": "schema_invalid",
    "required_evidence_refs_missing": "schema_invalid",
    "expires_at_invalid": "schema_invalid",
    "packet_expired": "expired_bridge_object",
    "schema_validation_failed": "schema_invalid",
    "cross_lane_ticket_required": "cross_lane_write_scope_violation",
    "cross_lane_ticket_payload_invalid": "cross_lane_write_scope_violation",
    "cross_lane_operation_missing": "cross_lane_write_scope_violation",
    "cross_lane_lease_mode_invalid": "cross_lane_write_scope_violation",
    "cross_lane_risk_tier_invalid": "cross_lane_write_scope_violation",
    "cross_lane_attestation_missing": "cross_lane_write_scope_violation",
    "cross_lane_fencing_term_invalid": "cross_lane_write_scope_violation",
    "cross_lane_fencing_term_stale": "cross_lane_write_scope_violation",
    "cross_lane_unknown_callsite": "cross_lane_write_scope_violation",
    "cross_lane_fast_path_not_allowlisted": "cross_lane_write_scope_violation",
    "cross_lane_risk_tier_forbidden": "cross_lane_write_scope_violation",
    "cross_lane_fast_path_risk_tier_not_allowed": "cross_lane_write_scope_violation",
    "cross_lane_fast_path_ttl_invalid": "cross_lane_write_scope_violation",
    "cross_lane_fast_path_ttl_exceeded": "cross_lane_write_scope_violation",
    "lane_topology_policy_unavailable": "gate_unavailable",
    "lane_authority_contract_unavailable": "gate_unavailable",
}


@lru_cache(maxsize=1)
def _load_lane_crossover_evaluator() -> tuple[Callable[..., dict[str, Any]] | None, str | None]:
    try:
        spec = importlib.util.spec_from_file_location("lane_crossover_ingress_guard", _LANE_CROSSOVER_GUARD_PATH)
        if spec is None or spec.loader is None:
            return None, "lane_crossover_guard_spec_unavailable"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evaluator = getattr(module, "evaluate_lane_crossover_ingress", None)
        if not callable(evaluator):
            return None, "lane_crossover_guard_evaluator_unavailable"
        return evaluator, None
    except Exception as exc:
        return None, f"lane_crossover_guard_load_failed:{exc}"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_dt(value: Any) -> Optional[dt.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.timezone.utc)


def _resolve_now(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    parsed = _parse_dt(value)
    if parsed is not None:
        return parsed
    return _utc_now()


def _resolve_inside(root: Path, rel: str) -> Path | None:
    text = str(rel or "").strip()
    if not text:
        return None
    try:
        candidate = (root / text).resolve()
        if not candidate.is_relative_to(root.resolve()):
            return None
    except Exception:
        return None
    return candidate


def _normalize_sha256(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.split(":", 1)[1]
    if _SHA256_RE.fullmatch(raw):
        return raw
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_reason(code: str, *, stage: str) -> str:
    text = str(code or "").strip()
    if text in CANONICAL_REJECTION_CODES:
        return text
    if stage == "schema":
        return "schema_invalid"
    return "gate_unavailable"


def _add_reason(
    reasons: list[dict[str, str]],
    seen: set[str],
    code: str,
    *,
    detail: str = "",
    stage: str = "runtime",
) -> None:
    normalized = _normalize_reason(code, stage=stage)
    if normalized in seen:
        return
    seen.add(normalized)
    row = {"code": normalized}
    if detail:
        row["detail"] = detail
    reasons.append(row)


def _schema_validate(payload: Mapping[str, Any], schema_path: Path) -> tuple[bool, str | None, str | None]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", "schema_validator_unavailable"
    if not schema_path.exists():
        return False, "gate_unavailable", f"schema_missing:{schema_path}"

    try:
        schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "gate_unavailable", f"schema_parse_failed:{exc}"
    if not isinstance(schema_doc, Mapping):
        return False, "gate_unavailable", "schema_not_object"

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, None

    err = errors[0]
    data_ptr = "$" if not err.absolute_path else "$/" + "/".join(str(p) for p in err.absolute_path)
    schema_ptr = "$" if not err.absolute_schema_path else "$/" + "/".join(str(p) for p in err.absolute_schema_path)
    detail = f"schema_validation_failed:data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    return False, "schema_invalid", detail


def _parse_sender_policy(entries: Mapping[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(entries, Mapping):
        return out
    for raw_lane, raw_epoch in entries.items():
        lane = str(raw_lane or "").strip()
        epoch = str(raw_epoch or "").strip()
        if lane and epoch:
            out[lane] = epoch
    return out


def _apply_lane_crossover_packet_guard(
    *,
    object_class: str,
    object_ref: Mapping[str, Any] | None,
    resolved_object_path: Path | None,
    receiver_lane_id: str,
    receiver_lane_epoch: str,
    sender_policy: Mapping[str, str],
    workspace: Path,
    now_utc: dt.datetime,
    reasons: list[dict[str, str]],
    seen_codes: set[str],
) -> None:
    if object_class != "lane_crossover_packet":
        return
    if resolved_object_path is None or not resolved_object_path.exists() or not resolved_object_path.is_file():
        return

    evaluator, load_error = _load_lane_crossover_evaluator()
    if evaluator is None:
        _add_reason(reasons, seen_codes, "gate_unavailable", detail=str(load_error or "lane_crossover_guard_missing"))
        return

    try:
        lane_packet = json.loads(resolved_object_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _add_reason(
            reasons,
            seen_codes,
            "schema_invalid",
            stage="schema",
            detail=f"lane_crossover_packet_load_failed:{exc}",
        )
        return

    schema_ref = str(((object_ref or {}).get("schema_ref") or "")).strip()
    resolved_schema_path = _resolve_inside(workspace, schema_ref) if schema_ref else None
    lane_schema_path = resolved_schema_path if resolved_schema_path is not None else _LANE_CROSSOVER_SCHEMA_PATH

    try:
        lane_decision = evaluator(
            lane_packet,
            receiver_lane_id=receiver_lane_id,
            receiver_lane_epoch=receiver_lane_epoch,
            sender_lane_epochs=sender_policy,
            now=now_utc,
            schema_path=lane_schema_path,
        )
    except Exception as exc:
        _add_reason(reasons, seen_codes, "gate_unavailable", detail=f"lane_crossover_guard_eval_failed:{exc}")
        return

    if lane_decision.get("accepted") is True:
        return

    lane_reasons = lane_decision.get("rejection_reasons") if isinstance(lane_decision.get("rejection_reasons"), list) else []
    if not lane_reasons:
        lane_codes = lane_decision.get("rejection_codes") if isinstance(lane_decision.get("rejection_codes"), list) else []
        lane_reasons = [{"code": str(code)} for code in lane_codes]

    for row in lane_reasons:
        if not isinstance(row, Mapping):
            continue
        lane_code = str(row.get("code") or "").strip()
        if not lane_code:
            continue
        mapped_code = _LANE_CROSSOVER_REJECTION_MAP.get(lane_code, "gate_unavailable")
        lane_detail = str(row.get("detail") or "").strip()
        detail = f"lane_crossover_guard:{lane_code}"
        if lane_detail:
            detail = f"{detail}:{lane_detail}"
        stage = "schema" if mapped_code == "schema_invalid" else "runtime"
        _add_reason(reasons, seen_codes, mapped_code, detail=detail, stage=stage)


def evaluate_cross_lane_bridge_ingress(
    packet: Any,
    *,
    receiver_lane_id: str,
    receiver_lane_epoch: str,
    sender_lane_epochs: Mapping[str, str] | None = None,
    allow_cross_lane_write: bool = False,
    now: Any = None,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    workspace_root: Path | str = ROOT,
) -> dict[str, Any]:
    """Fail-closed evaluation for cross-lane bridge objects."""

    now_utc = _resolve_now(now)
    workspace = Path(workspace_root).resolve()
    reasons: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    checks = {
        "schema_valid": False,
        "schema_version_supported": False,
        "object_class_allowed": False,
        "lane_identity_match": False,
        "lane_epoch_match": False,
        "object_ref_resolved": False,
        "content_hash_match": False,
        "class_gates_passed": False,
        "contamination_guard_passed": False,
        "expiry_valid": False,
        "write_scope_allowed": False,
    }

    sender_policy = _parse_sender_policy(sender_lane_epochs)

    if not isinstance(packet, Mapping):
        _add_reason(
            reasons,
            seen_codes,
            "schema_invalid",
            stage="schema",
            detail="packet must be a JSON object",
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "evaluated_at": _iso_utc(now_utc),
            "accepted": False,
            "gate_outcome": "REJECTED_INVALID",
            "checks": checks,
            "rejection_reasons": reasons,
            "rejection_codes": [item["code"] for item in reasons],
            "bridge": {},
        }

    bridge = dict(packet)

    # 1) Schema + required fields + enum checks.
    packet_schema_version = str(bridge.get("schema_version") or "").strip()
    if packet_schema_version == BRIDGE_SCHEMA_VERSION:
        checks["schema_version_supported"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "schema_version_unsupported",
            stage="schema",
            detail=f"expected={BRIDGE_SCHEMA_VERSION} actual={packet_schema_version or 'unset'}",
        )

    object_class = str(bridge.get("object_class") or "").strip()
    if object_class in ALLOWED_OBJECT_CLASSES:
        checks["object_class_allowed"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "unknown_object_class",
            stage="schema",
            detail=f"object_class={object_class or 'unset'}",
        )

    schema_ok, schema_code, schema_detail = _schema_validate(bridge, Path(schema_path))
    if schema_ok:
        checks["schema_valid"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            schema_code or "schema_invalid",
            stage="schema",
            detail=str(schema_detail or "schema_validation_failed"),
        )

    # 2) Lane tuple validation.
    expected_to_lane = str(receiver_lane_id or "").strip()
    expected_to_epoch = str(receiver_lane_epoch or "").strip()

    to_lane_id = str(bridge.get("to_lane_id") or "").strip()
    to_lane_epoch = str(bridge.get("to_lane_epoch") or "").strip()
    from_lane_id = str(bridge.get("from_lane_id") or "").strip()
    from_lane_epoch = str(bridge.get("from_lane_epoch") or "").strip()

    lane_match = to_lane_id and to_lane_id == expected_to_lane
    epoch_match = to_lane_epoch and to_lane_epoch == expected_to_epoch

    if lane_match:
        checks["lane_identity_match"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "lane_identity_mismatch",
            detail=f"to_lane expected={expected_to_lane or 'unset'} actual={to_lane_id or 'unset'}",
        )

    if epoch_match:
        checks["lane_epoch_match"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "lane_epoch_mismatch",
            detail=f"to_epoch expected={expected_to_epoch or 'unset'} actual={to_lane_epoch or 'unset'}",
        )

    if not sender_policy:
        _add_reason(reasons, seen_codes, "lane_identity_mismatch", detail="sender lane policy missing")
    else:
        expected_from_epoch = sender_policy.get(from_lane_id)
        if expected_from_epoch is None:
            _add_reason(
                reasons,
                seen_codes,
                "lane_identity_mismatch",
                detail=f"sender lane not authorized: {from_lane_id or 'unset'}",
            )
        elif from_lane_epoch != expected_from_epoch:
            _add_reason(
                reasons,
                seen_codes,
                "lane_epoch_mismatch",
                detail=f"sender epoch expected={expected_from_epoch} actual={from_lane_epoch or 'unset'}",
            )

    # 3) Object-ref resolution (path/hash/schema-ref).
    object_ref = bridge.get("object_ref") if isinstance(bridge.get("object_ref"), Mapping) else None
    resolved_object_path: Path | None = None
    normalized_hash: str | None = None

    if object_ref is None:
        _add_reason(reasons, seen_codes, "schema_invalid", stage="schema", detail="object_ref must be an object")
    else:
        normalized_hash = _normalize_sha256(object_ref.get("content_hash"))
        if normalized_hash is None:
            _add_reason(reasons, seen_codes, "content_hash_missing", detail="object_ref.content_hash missing or invalid")

        object_path = str(object_ref.get("path") or "").strip()
        if not object_path:
            _add_reason(reasons, seen_codes, "object_not_found", detail="object_ref.path missing")
        else:
            resolved_object_path = _resolve_inside(workspace, object_path)
            if resolved_object_path is None:
                _add_reason(reasons, seen_codes, "gate_unavailable", detail=f"unsafe object_ref.path: {object_path}")
            elif not resolved_object_path.exists() or not resolved_object_path.is_file():
                _add_reason(reasons, seen_codes, "object_not_found", detail=f"missing object_ref.path: {object_path}")
            else:
                checks["object_ref_resolved"] = True
                if normalized_hash is not None:
                    actual_hash = _sha256_file(resolved_object_path)
                    if actual_hash != normalized_hash:
                        _add_reason(reasons, seen_codes, "content_hash_mismatch", detail=f"path={object_path}")
                    else:
                        checks["content_hash_match"] = True

                expected_bytes = object_ref.get("bytes")
                if isinstance(expected_bytes, int) and expected_bytes >= 0:
                    actual_bytes = resolved_object_path.stat().st_size
                    if actual_bytes != expected_bytes:
                        _add_reason(
                            reasons,
                            seen_codes,
                            "content_hash_mismatch",
                            detail=f"bytes mismatch expected={expected_bytes} actual={actual_bytes}",
                        )

        schema_ref = str(object_ref.get("schema_ref") or "").strip()
        expected_schema_ref = EXPECTED_SCHEMA_REF_BY_CLASS.get(object_class)
        if expected_schema_ref is not None and schema_ref != expected_schema_ref:
            _add_reason(
                reasons,
                seen_codes,
                "object_schema_mismatch",
                detail=f"expected schema_ref={expected_schema_ref} actual={schema_ref or 'unset'}",
            )

        if object_class == "approved_artifact_ref" and not schema_ref:
            _add_reason(reasons, seen_codes, "object_schema_mismatch", detail="approved_artifact_ref requires schema_ref")

        if schema_ref:
            resolved_schema_ref = _resolve_inside(workspace, schema_ref)
            if resolved_schema_ref is None:
                _add_reason(reasons, seen_codes, "gate_unavailable", detail=f"unsafe schema_ref path: {schema_ref}")
            elif not resolved_schema_ref.exists() or not resolved_schema_ref.is_file():
                _add_reason(reasons, seen_codes, "gate_unavailable", detail=f"missing schema_ref file: {schema_ref}")

    # 3b) Lane crossover packet ingress guard (real receiver handoff path).
    _apply_lane_crossover_packet_guard(
        object_class=object_class,
        object_ref=object_ref,
        resolved_object_path=resolved_object_path,
        receiver_lane_id=expected_to_lane,
        receiver_lane_epoch=expected_to_epoch,
        sender_policy=sender_policy,
        workspace=workspace,
        now_utc=now_utc,
        reasons=reasons,
        seen_codes=seen_codes,
    )

    # 4) Class-specific promotion/review gates.
    approval = bridge.get("approval") if isinstance(bridge.get("approval"), Mapping) else None
    promotion = bridge.get("promotion") if isinstance(bridge.get("promotion"), Mapping) else None

    approval_state = str((approval or {}).get("approval_state") or "").strip()
    reviewer_role = str((approval or {}).get("reviewer_role") or "").strip()
    approval_decision_ref = str((approval or {}).get("decision_ref") or "").strip()

    if promotion is None:
        _add_reason(reasons, seen_codes, "promotion_missing", detail="promotion object missing")
    if approval is None:
        _add_reason(reasons, seen_codes, "review_not_approved", detail="approval object missing")
    elif approval_state not in {"approved", "promoted"}:
        _add_reason(reasons, seen_codes, "review_not_approved", detail=f"approval_state={approval_state or 'unset'}")

    if isinstance(promotion, Mapping):
        promotion_required = bool(promotion.get("promotion_required") is True)
        promotion_id = str(promotion.get("promotion_id") or "").strip()
        promotion_state = str(promotion.get("promotion_state") or "").strip()
        leakage_check = str(promotion.get("leakage_check") or "").strip()

        if object_class == "doctrine_object":
            if not promotion_required:
                _add_reason(reasons, seen_codes, "promotion_gate_not_satisfied", detail="doctrine_object requires promotion_required=true")
            if not _PROMOTION_ID_RE.fullmatch(promotion_id):
                _add_reason(reasons, seen_codes, "promotion_missing", detail="doctrine_object requires prom_* promotion_id")
            if promotion_state not in _PROMOTED_STATES:
                _add_reason(reasons, seen_codes, "promotion_state_invalid", detail=f"promotion_state={promotion_state or 'unset'}")
            if leakage_check != "pass":
                _add_reason(reasons, seen_codes, "leakage_risk", detail=f"leakage_check={leakage_check or 'unset'}")

        elif object_class == "promotion_candidate":
            if promotion_state in {"", "LOCAL_ONLY"}:
                _add_reason(reasons, seen_codes, "promotion_state_invalid", detail=f"promotion_state={promotion_state or 'unset'}")
            if leakage_check != "pass":
                _add_reason(reasons, seen_codes, "leakage_risk", detail=f"leakage_check={leakage_check or 'unset'}")

        elif object_class == "approved_artifact_ref":
            if not promotion_required:
                _add_reason(reasons, seen_codes, "promotion_gate_not_satisfied", detail="approved_artifact_ref requires promotion_required=true")
            if not _PROMOTION_ID_RE.fullmatch(promotion_id):
                _add_reason(reasons, seen_codes, "promotion_missing", detail="approved_artifact_ref requires prom_* promotion_id")
            if promotion_state not in _PROMOTED_STATES:
                _add_reason(reasons, seen_codes, "promotion_state_invalid", detail=f"promotion_state={promotion_state or 'unset'}")
            if leakage_check != "pass":
                _add_reason(reasons, seen_codes, "leakage_risk", detail=f"leakage_check={leakage_check or 'unset'}")
            if approval_state != "promoted":
                _add_reason(reasons, seen_codes, "review_not_approved", detail="approved_artifact_ref requires approval_state=promoted")

        if object_class == "evidence_closeout":
            source_refs = bridge.get("source_refs") if isinstance(bridge.get("source_refs"), list) else []
            if not any(str(item or "").strip() for item in source_refs):
                _add_reason(reasons, seen_codes, "promotion_gate_not_satisfied", detail="evidence_closeout requires source_refs")
            decision_refs = bridge.get("decision_refs") if isinstance(bridge.get("decision_refs"), list) else []
            if not approval_decision_ref and not any(str(item or "").strip() for item in decision_refs):
                _add_reason(reasons, seen_codes, "review_not_approved", detail="evidence_closeout requires reviewer decision ref")

    # 5) Contamination + classification checks.
    contamination_guard = bridge.get("contamination_guard") if isinstance(bridge.get("contamination_guard"), Mapping) else None
    cross_lane_write_requested = False

    if contamination_guard is None:
        _add_reason(reasons, seen_codes, "schema_invalid", stage="schema", detail="contamination_guard must be an object")
    else:
        max_inline = contamination_guard.get("max_inline_context_bytes")
        if isinstance(max_inline, bool) or not isinstance(max_inline, int):
            _add_reason(reasons, seen_codes, "schema_invalid", stage="schema", detail="max_inline_context_bytes must be integer")
        elif max_inline > 512:
            _add_reason(reasons, seen_codes, "inline_context_over_limit", detail=f"max_inline_context_bytes={max_inline}")

        if bool(contamination_guard.get("contains_unverified_content") is True):
            _add_reason(reasons, seen_codes, "unverified_content_blocked")

        inline_excerpt = bridge.get("inline_excerpt")
        if isinstance(inline_excerpt, str) and inline_excerpt:
            excerpt_bytes = len(inline_excerpt.encode("utf-8"))
            allow_inline = bool(contamination_guard.get("allow_inline_excerpt") is True)
            if not allow_inline:
                _add_reason(reasons, seen_codes, "inline_context_over_limit", detail="inline_excerpt supplied while allow_inline_excerpt=false")
            if isinstance(max_inline, int) and excerpt_bytes > max_inline:
                _add_reason(
                    reasons,
                    seen_codes,
                    "inline_context_over_limit",
                    detail=f"inline excerpt bytes={excerpt_bytes} max_inline_context_bytes={max_inline}",
                )
            elif excerpt_bytes > 512:
                _add_reason(reasons, seen_codes, "inline_context_over_limit", detail=f"inline excerpt bytes={excerpt_bytes}")

        promotion_gate = str(contamination_guard.get("promotion_gate") or "").strip()
        if promotion_gate == "validator_required" and reviewer_role != "VALIDATOR":
            _add_reason(
                reasons,
                seen_codes,
                "promotion_gate_not_satisfied",
                detail=f"promotion_gate=validator_required reviewer_role={reviewer_role or 'unset'}",
            )
        if promotion_gate == "human_required" and reviewer_role not in {"SRE", "LIBRARIAN"}:
            _add_reason(
                reasons,
                seen_codes,
                "promotion_gate_not_satisfied",
                detail=f"promotion_gate=human_required reviewer_role={reviewer_role or 'unset'}",
            )

        cross_lane_write_requested = bool(contamination_guard.get("cross_lane_write_requested") is True)

        if isinstance(promotion, Mapping):
            classification = str(promotion.get("classification") or "").strip().lower()
            leakage_check = str(promotion.get("leakage_check") or "").strip().lower()
            promotion_redaction = bool(promotion.get("redaction_applied") is True)
            contamination_redaction = bool(contamination_guard.get("redaction_applied") is True)

            if classification == "secret":
                _add_reason(reasons, seen_codes, "classification_forbidden")
            if leakage_check == "fail":
                _add_reason(reasons, seen_codes, "leakage_risk", detail="promotion.leakage_check=fail")
            if classification == "restricted" and not (promotion_redaction and contamination_redaction):
                _add_reason(reasons, seen_codes, "redaction_required")

        if (
            "inline_context_over_limit" not in seen_codes
            and "unverified_content_blocked" not in seen_codes
            and "redaction_required" not in seen_codes
            and "classification_forbidden" not in seen_codes
            and "leakage_risk" not in seen_codes
        ):
            checks["contamination_guard_passed"] = True

    # 6) Expiry/time-window checks.
    expires_at = bridge.get("expires_at")
    if expires_at in {None, ""}:
        checks["expiry_valid"] = True
    else:
        expires_dt = _parse_dt(expires_at)
        if expires_dt is None:
            _add_reason(reasons, seen_codes, "schema_invalid", stage="schema", detail="expires_at must be RFC3339 date-time")
        elif expires_dt <= now_utc:
            _add_reason(reasons, seen_codes, "expired_bridge_object")
        else:
            checks["expiry_valid"] = True

    # 7) Consumer-local write-scope authorization.
    if contamination_guard is not None:
        if cross_lane_write_requested and not allow_cross_lane_write:
            _add_reason(reasons, seen_codes, "cross_lane_write_scope_violation")
        else:
            checks["write_scope_allowed"] = True

    # Class-gate summary check.
    class_gate_failures = {
        "promotion_missing",
        "promotion_state_invalid",
        "promotion_gate_not_satisfied",
        "review_not_approved",
        "leakage_risk",
        "redaction_required",
    }
    if not (set(seen_codes) & class_gate_failures):
        checks["class_gates_passed"] = True

    accepted = len(reasons) == 0

    decision = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at": _iso_utc(now_utc),
        "accepted": accepted,
        "gate_outcome": "ACCEPTED" if accepted else "REJECTED_INVALID",
        "checks": checks,
        "rejection_reasons": reasons,
        "rejection_codes": [item["code"] for item in reasons],
        "bridge": {
            "bridge_id": str(bridge.get("bridge_id") or "") or None,
            "object_class": object_class or None,
            "from_lane_id": from_lane_id or None,
            "from_lane_epoch": from_lane_epoch or None,
            "to_lane_id": to_lane_id or None,
            "to_lane_epoch": to_lane_epoch or None,
            "work_item_id": str(bridge.get("work_item_id") or "") or None,
            "object_path": str(((object_ref or {}).get("path") or "")) or None,
            "object_hash": str(((object_ref or {}).get("content_hash") or "")) or None,
        },
    }
    if resolved_object_path is not None and checks["object_ref_resolved"]:
        decision["bridge"]["resolved_object_path"] = str(resolved_object_path)

    return decision


def _append_decision_log(
    decision: Mapping[str, Any],
    *,
    workspace_root: Path,
    decision_log_path: Path | str,
) -> tuple[bool, str, str | None]:
    raw_path = Path(decision_log_path)
    candidate = raw_path if raw_path.is_absolute() else (workspace_root / raw_path)
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(workspace_root.resolve()):
            return False, "unsafe_log_path", None
    except Exception:
        return False, "unsafe_log_path", None

    row = {
        "ts": _iso_utc(_utc_now()),
        "schema_version": SCHEMA_VERSION,
        "accepted": bool(decision.get("accepted") is True),
        "gate_outcome": str(decision.get("gate_outcome") or ""),
        "bridge_id": str(((decision.get("bridge") or {}).get("bridge_id") or "")) or None,
        "object_class": str(((decision.get("bridge") or {}).get("object_class") or "")) or None,
        "rejection_codes": list(decision.get("rejection_codes") or []),
        "decision": dict(decision),
    }

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        return False, f"write_failed:{exc}", str(resolved)

    return True, "appended", str(resolved)


def ingest_cross_lane_bridge_object(
    packet: Any,
    *,
    receiver_lane_id: str,
    receiver_lane_epoch: str,
    sender_lane_epochs: Mapping[str, str] | None = None,
    allow_cross_lane_write: bool = False,
    now: Any = None,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    workspace_root: Path | str = ROOT,
    decision_log_path: Path | str = DEFAULT_DECISION_LOG,
) -> dict[str, Any]:
    """Validate cross-lane bridge object and append deterministic decision log."""

    workspace = Path(workspace_root).resolve()
    decision = evaluate_cross_lane_bridge_ingress(
        packet,
        receiver_lane_id=receiver_lane_id,
        receiver_lane_epoch=receiver_lane_epoch,
        sender_lane_epochs=sender_lane_epochs,
        allow_cross_lane_write=allow_cross_lane_write,
        now=now,
        schema_path=schema_path,
        workspace_root=workspace,
    )

    written, status, resolved_log_path = _append_decision_log(
        decision,
        workspace_root=workspace,
        decision_log_path=decision_log_path,
    )

    decision["logging"] = {
        "status": status,
        "written": written,
        "decision_log_path": resolved_log_path,
    }

    if not written:
        reasons = decision.get("rejection_reasons") if isinstance(decision.get("rejection_reasons"), list) else []
        seen = {str(item.get("code") or "") for item in reasons if isinstance(item, Mapping)}
        _add_reason(
            reasons,
            seen,
            "gate_unavailable",
            detail=f"decision_log:{status}",
        )
        decision["rejection_reasons"] = reasons
        decision["rejection_codes"] = [str(item.get("code") or "") for item in reasons if isinstance(item, Mapping)]
        decision["accepted"] = False
        decision["gate_outcome"] = "REJECTED_INVALID"

    return decision


def _load_packet(path_value: str) -> Any:
    if path_value == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path_value).read_text(encoding="utf-8"))


def _parse_allow_from(entries: list[str]) -> dict[str, str]:
    policy: dict[str, str] = {}
    for raw in entries:
        text = str(raw or "").strip()
        if not text:
            continue
        lane, sep, epoch = text.partition("=")
        lane = lane.strip()
        epoch = epoch.strip()
        if sep != "=" or not lane or not epoch:
            raise ValueError(f"invalid --allow-from entry (expected lane_id=epoch_id): {raw}")
        policy[lane] = epoch
    return policy


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic cross-lane bridge validator/ingest tool v1")
    parser.add_argument("--bridge", required=True, help="Bridge object JSON path or '-' for stdin")
    parser.add_argument("--to-lane-id", required=True, help="Expected receiver lane id")
    parser.add_argument("--to-lane-epoch", required=True, help="Expected receiver lane epoch")
    parser.add_argument(
        "--allow-from",
        action="append",
        default=[],
        metavar="LANE_ID=EPOCH_ID",
        help="Authorized sender lane + expected epoch (repeatable)",
    )
    parser.add_argument("--allow-cross-lane-write", action="store_true", help="Allow cross-lane write-requested bridge objects")
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Bridge schema path override")
    parser.add_argument("--workspace-root", default=str(ROOT), help="Workspace root for object/path resolution")
    parser.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Append-only decision JSONL path")
    parser.add_argument("--now", default=None, help="Override now (RFC3339)")
    args = parser.parse_args(argv)

    try:
        packet = _load_packet(args.bridge)
        sender_policy = _parse_allow_from(list(args.allow_from or []))
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "accepted": False,
            "gate_outcome": "REJECTED_INVALID",
            "rejection_reasons": [{"code": "schema_invalid", "detail": str(exc)}],
            "rejection_codes": ["schema_invalid"],
            "bridge": {},
            "logging": {
                "status": "not_attempted",
                "written": False,
                "decision_log_path": None,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2

    decision = ingest_cross_lane_bridge_object(
        packet,
        receiver_lane_id=args.to_lane_id,
        receiver_lane_epoch=args.to_lane_epoch,
        sender_lane_epochs=sender_policy,
        allow_cross_lane_write=bool(args.allow_cross_lane_write),
        now=args.now,
        schema_path=Path(args.schema_path),
        workspace_root=Path(args.workspace_root),
        decision_log_path=Path(args.decision_log),
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision.get("accepted") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
