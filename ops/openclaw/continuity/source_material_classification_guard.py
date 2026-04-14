#!/usr/bin/env python3
"""Deterministic source-material classification + routing firewall guard (v1)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

try:  # pragma: no cover - dependency wiring validated by tests
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "source_material_classification.schema.json"
SCHEMA_VERSION = "source.material.routing.decision.v1"

FIREWALL_LANE = "lane.quarantine.firewall"

_AUTO_REASON_ORDER = [
    "untrusted_tier",
    "unverified_content",
    "low_credibility_external_source",
    "model_output_unverified",
    "speculative_intent",
    "restricted_needs_redaction",
    "secret_sensitivity",
]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


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
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)
    parsed = _parse_dt(value)
    if parsed is not None:
        return parsed
    return _utc_now()


def _add_reason(reasons: list[dict[str, str]], seen: set[str], code: str, detail: str = "") -> None:
    if code in seen:
        return
    seen.add(code)
    row = {"code": code}
    if detail:
        row["detail"] = detail
    reasons.append(row)


def _schema_validate(packet: Mapping[str, Any], schema_path: Path) -> tuple[bool, Optional[str]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "schema_validator_unavailable"
    if not schema_path.exists():
        return False, f"schema_missing:{schema_path}"

    try:
        schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"schema_parse_failed:{exc}"

    if not isinstance(schema_doc, dict):
        return False, "schema_not_object"

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(packet),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None

    err = errors[0]
    return (
        False,
        "schema_validation_failed:"
        f"data_path={_json_ptr(err.absolute_path)}:"
        f"schema_path={_json_ptr(err.absolute_schema_path)}:"
        f"error={err.message}",
    )


def _ordered_subset(items: set[str], order: list[str]) -> list[str]:
    return [item for item in order if item in items]


def _resolve_egress_policy(*, decision: str, sensitivity: str, trust_tier: str) -> str:
    if decision in {"BLOCK", "QUARANTINE"}:
        return "deny"
    if sensitivity == "public" and trust_tier in {"t2_verified", "t3_canonical"}:
        return "allow"
    if sensitivity in {"internal", "restricted"} and trust_tier in {"t2_verified", "t3_canonical"}:
        return "review_required"
    return "deny"


def _derive_auto_quarantine(packet_obj: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    reasons: set[str] = set()
    release_requirements: set[str] = set()

    trust_tier = str(packet_obj.get("trust_tier") or "").strip()
    source_kind = str(packet_obj.get("source_kind") or "").strip()
    intent_class = str(packet_obj.get("intent_class") or "").strip()
    sensitivity = str(packet_obj.get("sensitivity") or "").strip()

    contamination_guard = packet_obj.get("contamination_guard")
    guard = contamination_guard if isinstance(contamination_guard, Mapping) else {}

    contains_unverified = bool(guard.get("contains_unverified_content") is True)
    redaction_applied = bool(guard.get("redaction_applied") is True)

    if trust_tier == "t0_untrusted":
        reasons.add("untrusted_tier")

    if contains_unverified:
        reasons.add("unverified_content")

    if source_kind in {"external_article", "social_post"} and trust_tier in {"t0_untrusted", "t1_provisional"}:
        reasons.add("low_credibility_external_source")
        release_requirements.add("corroboration_required")

    if source_kind == "model_output" and trust_tier != "t3_canonical":
        reasons.add("model_output_unverified")
        release_requirements.add("corroboration_required")

    if intent_class == "speculative_hypothesis":
        reasons.add("speculative_intent")

    if sensitivity == "secret":
        reasons.add("secret_sensitivity")
        release_requirements.add("human_approval")
        release_requirements.add("redaction_required")

    if sensitivity == "restricted" and not redaction_applied:
        reasons.add("restricted_needs_redaction")
        release_requirements.add("redaction_required")

    if reasons:
        release_requirements.add("validator_review")
        release_requirements.add("provenance_hash")

    return reasons, release_requirements


def evaluate_source_material_classification(
    packet: Any,
    *,
    now: Any = None,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Evaluate source-material classification packet with fail-closed route decision."""

    now_utc = _resolve_now(now)

    reasons: list[dict[str, str]] = []
    seen_codes: set[str] = set()

    checks = {
        "schema_valid": False,
        "quarantine_metadata_consistent": False,
        "contamination_policy_valid": False,
        "route_resolved": False,
    }

    if not isinstance(packet, Mapping):
        _add_reason(reasons, seen_codes, "schema_validation_failed", "packet_not_object")
        decision = "BLOCK"
        resolved_lane = None
        trust_tier = None
        sensitivity = None
        quarantine_required = False
    else:
        packet_obj = dict(packet)

        schema_ok, schema_issue = _schema_validate(packet_obj, Path(schema_path))
        if schema_ok:
            checks["schema_valid"] = True
        else:
            _add_reason(reasons, seen_codes, "schema_validation_failed", str(schema_issue or "unknown_schema_failure"))

        requested_lane = str(packet_obj.get("primary_lane_class") or "").strip() or None
        trust_tier = str(packet_obj.get("trust_tier") or "").strip() or None
        sensitivity = str(packet_obj.get("sensitivity") or "").strip() or None

        quarantine_obj = packet_obj.get("quarantine") if isinstance(packet_obj.get("quarantine"), Mapping) else {}
        contamination_guard = (
            packet_obj.get("contamination_guard") if isinstance(packet_obj.get("contamination_guard"), Mapping) else {}
        )

        status = str(quarantine_obj.get("status") or "").strip()
        declared_reasons = {
            str(item).strip()
            for item in (quarantine_obj.get("reason_codes") or [])
            if str(item).strip()
        }
        declared_release = {
            str(item).strip()
            for item in (quarantine_obj.get("release_requirements") or [])
            if str(item).strip()
        }

        auto_reasons, auto_release = _derive_auto_quarantine(packet_obj)
        quarantine_required = bool(auto_reasons)

        missing_reasons = auto_reasons - declared_reasons
        missing_release = auto_release - declared_release

        manual_hold = "manual_hold" in declared_reasons

        if quarantine_required:
            if status not in {"required", "active"}:
                _add_reason(
                    reasons,
                    seen_codes,
                    "quarantine_status_inconsistent",
                    f"status={status or 'unset'} expected=required|active",
                )
            if status == "released":
                _add_reason(reasons, seen_codes, "quarantine_release_forbidden")
            if missing_reasons:
                _add_reason(
                    reasons,
                    seen_codes,
                    "quarantine_reason_codes_incomplete",
                    f"missing={','.join(sorted(missing_reasons))}",
                )
            if missing_release:
                _add_reason(
                    reasons,
                    seen_codes,
                    "quarantine_release_requirements_incomplete",
                    f"missing={','.join(sorted(missing_release))}",
                )
        elif status in {"required", "active"} and not manual_hold:
            _add_reason(reasons, seen_codes, "manual_hold_reason_required")

        allow_inline_excerpt = bool(contamination_guard.get("allow_inline_excerpt") is True)
        max_inline_context_bytes = contamination_guard.get("max_inline_context_bytes")
        if isinstance(max_inline_context_bytes, bool) or not isinstance(max_inline_context_bytes, int):
            max_inline_context = 0
        else:
            max_inline_context = max_inline_context_bytes

        redaction_applied = bool(contamination_guard.get("redaction_applied") is True)
        cross_lane_write_requested = bool(contamination_guard.get("cross_lane_write_requested") is True)

        quarantine_active = quarantine_required or status in {"required", "active"}

        if sensitivity == "secret" and (allow_inline_excerpt or max_inline_context > 0):
            _add_reason(reasons, seen_codes, "secret_inline_context_forbidden")

        if sensitivity == "restricted" and allow_inline_excerpt and not redaction_applied:
            _add_reason(reasons, seen_codes, "restricted_inline_requires_redaction")

        if quarantine_active and cross_lane_write_requested:
            _add_reason(reasons, seen_codes, "cross_lane_write_forbidden_while_quarantined")

        checks["quarantine_metadata_consistent"] = not any(
            code in seen_codes
            for code in {
                "quarantine_reason_codes_incomplete",
                "quarantine_release_requirements_incomplete",
                "quarantine_status_inconsistent",
                "quarantine_release_forbidden",
                "manual_hold_reason_required",
            }
        )

        checks["contamination_policy_valid"] = not any(
            code in seen_codes
            for code in {
                "secret_inline_context_forbidden",
                "restricted_inline_requires_redaction",
                "cross_lane_write_forbidden_while_quarantined",
            }
        )

        accepted = len(reasons) == 0
        if accepted:
            decision = "QUARANTINE" if quarantine_active else "ROUTE"
        else:
            decision = "BLOCK"

        resolved_lane = FIREWALL_LANE if quarantine_active else requested_lane
        checks["route_resolved"] = bool(resolved_lane)

        packet_meta = {
            "material_id": str(packet_obj.get("material_id") or "") or None,
            "requested_lane_class": requested_lane,
            "resolved_lane_class": resolved_lane,
            "trust_tier": trust_tier,
            "source_kind": str(packet_obj.get("source_kind") or "") or None,
            "intent_class": str(packet_obj.get("intent_class") or "") or None,
            "sensitivity": sensitivity,
        }

        result = {
            "schema_version": SCHEMA_VERSION,
            "evaluated_at": _iso_utc(now_utc),
            "accepted": accepted,
            "decision": decision,
            "checks": checks,
            "rejection_reasons": reasons,
            "rejection_codes": [row["code"] for row in reasons],
            "classification": packet_meta,
            "quarantine": {
                "status": status or None,
                "required": quarantine_required,
                "active": quarantine_active,
                "auto_reason_codes": _ordered_subset(set(auto_reasons), _AUTO_REASON_ORDER),
                "declared_reason_codes": sorted(declared_reasons),
                "missing_reason_codes": sorted(missing_reasons),
                "auto_release_requirements": sorted(auto_release),
                "declared_release_requirements": sorted(declared_release),
                "missing_release_requirements": sorted(missing_release),
            },
            "routing": {
                "requested_lane_class": requested_lane,
                "resolved_lane_class": resolved_lane,
                "firewall_applied": resolved_lane == FIREWALL_LANE,
                "egress_policy": _resolve_egress_policy(
                    decision=decision,
                    sensitivity=str(sensitivity or ""),
                    trust_tier=str(trust_tier or ""),
                ),
            },
        }
        return result

    result = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at": _iso_utc(now_utc),
        "accepted": False,
        "decision": "BLOCK",
        "checks": checks,
        "rejection_reasons": reasons,
        "rejection_codes": [row["code"] for row in reasons],
        "classification": {
            "material_id": None,
            "requested_lane_class": None,
            "resolved_lane_class": resolved_lane,
            "trust_tier": trust_tier,
            "source_kind": None,
            "intent_class": None,
            "sensitivity": sensitivity,
        },
        "quarantine": {
            "status": None,
            "required": quarantine_required,
            "active": quarantine_required,
            "auto_reason_codes": [],
            "declared_reason_codes": [],
            "missing_reason_codes": [],
            "auto_release_requirements": [],
            "declared_release_requirements": [],
            "missing_release_requirements": [],
        },
        "routing": {
            "requested_lane_class": None,
            "resolved_lane_class": resolved_lane,
            "firewall_applied": False,
            "egress_policy": "deny",
        },
    }
    return result


def _load_packet(path_value: str) -> Any:
    if path_value == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path_value).read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic source-material classification/routing firewall guard")
    parser.add_argument("--classification", required=True, help="Classification packet JSON path or '-' for stdin")
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Schema path override")
    parser.add_argument("--now", default=None, help="Override now (RFC3339); defaults to current UTC")
    parser.add_argument("--json", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    try:
        packet = _load_packet(args.classification)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "accepted": False,
            "decision": "BLOCK",
            "rejection_reasons": [{"code": "gate_unavailable", "detail": f"classification_load_failed:{exc}"}],
            "rejection_codes": ["gate_unavailable"],
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2

    decision = evaluate_source_material_classification(packet, now=args.now, schema_path=Path(args.schema_path))

    if args.json:
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))

    return 0 if decision.get("decision") in {"ROUTE", "QUARANTINE"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
