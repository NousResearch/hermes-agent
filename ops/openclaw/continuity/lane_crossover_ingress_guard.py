#!/usr/bin/env python3
"""Fail-closed ingress guard for Lane Boundary Contract v1 crossover packets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional

try:  # pragma: no cover - dependency wiring validated by tests
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "lane_crossover_packet.schema.json"
DEFAULT_FAST_PATH_POLICY_PATH = ROOT / "state" / "continuity" / "latest" / "core_roadmap_dependency_unblock_policy_pack_v1.json"
DEFAULT_AUTHORITY_CONTRACT_PATH = ROOT / "docs" / "ops" / "templates" / "lane_topology_authority_contract.template.json"
SCHEMA_VERSION = "lane.crossover_ingress_guard.v1"
ALLOWED_PACKET_TYPES = {"signal", "ticket", "deep_review"}
_FAST_PATH_TAG_RE = re.compile(r"^([ABC])[0-9]+$")


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
        now = value
        if now.tzinfo is None:
            return now.replace(tzinfo=dt.timezone.utc)
        return now.astimezone(dt.timezone.utc)
    parsed = _parse_dt(value)
    if parsed is not None:
        return parsed
    return _utc_now()


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"parse_failed:{exc}"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def _lane_column_tag(lane_id: str) -> str | None:
    lane = str(lane_id or "").strip().lower()
    if lane.startswith("lane.column_a."):
        return "A"
    if lane.startswith("lane.column_b."):
        return "B"
    if lane.startswith("lane.column_c."):
        return "C"
    return None


def _tag_from_policy_lane(value: Any) -> str | None:
    token = str(value or "").strip().upper()
    if not token:
        return None
    m = _FAST_PATH_TAG_RE.fullmatch(token)
    if m is None:
        return None
    return m.group(1)


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


def _evaluate_cross_lane_ticket_policy(
    *,
    packet_obj: Mapping[str, Any],
    now_utc: dt.datetime,
    policy_path: Path,
    authority_contract_path: Path,
) -> tuple[bool, list[dict[str, str]]]:
    reasons: list[dict[str, str]] = []
    seen_codes: set[str] = set()

    contamination_guard = packet_obj.get("contamination_guard")
    if not isinstance(contamination_guard, Mapping):
        return False, [{"code": "contamination_guard_missing"}]

    cross_lane_write_requested = bool(contamination_guard.get("cross_lane_write_requested") is True)
    if not cross_lane_write_requested:
        return True, []

    packet_type = str(packet_obj.get("packet_type") or "").strip()
    if packet_type != "ticket":
        _add_reason(reasons, seen_codes, "cross_lane_ticket_required")
        return False, reasons

    payload = packet_obj.get("payload")
    if not isinstance(payload, Mapping):
        _add_reason(reasons, seen_codes, "cross_lane_ticket_payload_invalid")
        return False, reasons

    operation_id = str(payload.get("operation_id") or "").strip()
    lease_mode = str(payload.get("lease_mode") or "").strip()
    risk_tier = str(payload.get("risk_tier") or "").strip().lower()
    if not operation_id:
        _add_reason(reasons, seen_codes, "cross_lane_operation_missing")

    if lease_mode not in {"full_ticket", "fast_path"}:
        _add_reason(reasons, seen_codes, "cross_lane_lease_mode_invalid")

    if risk_tier not in {"low", "medium", "high", "critical"}:
        _add_reason(reasons, seen_codes, "cross_lane_risk_tier_invalid")

    attestation_refs = payload.get("attestation_refs") if isinstance(payload.get("attestation_refs"), list) else []
    if not any(str(item or "").strip() for item in attestation_refs):
        _add_reason(reasons, seen_codes, "cross_lane_attestation_missing")

    try:
        ticket_term = int(payload.get("fencing_term"))
    except Exception:
        ticket_term = None
        _add_reason(reasons, seen_codes, "cross_lane_fencing_term_invalid")

    authority_doc, authority_err = _load_json_object(authority_contract_path)
    if authority_doc is None:
        _add_reason(reasons, seen_codes, "lane_authority_contract_unavailable", detail=str(authority_err or "unknown"))
    elif ticket_term is not None:
        try:
            leases = authority_doc.get("authority_leases") or {}
            control_term = int((leases.get("control_lease") or {}).get("fencing_term"))
            workflow_term = int((leases.get("workflow_lease") or {}).get("fencing_term"))
            if ticket_term < max(control_term, workflow_term):
                _add_reason(reasons, seen_codes, "cross_lane_fencing_term_stale")
        except Exception:
            _add_reason(reasons, seen_codes, "lane_authority_contract_unavailable", detail="fencing_term_unreadable")

    policy_doc, policy_err = _load_json_object(policy_path)
    if policy_doc is None:
        _add_reason(reasons, seen_codes, "lane_topology_policy_unavailable", detail=str(policy_err or "unknown"))
        return len(reasons) == 0, reasons

    slice12 = ((policy_doc.get("slices") or {}).get("12")) if isinstance(policy_doc.get("slices"), Mapping) else None
    if not isinstance(slice12, Mapping):
        _add_reason(reasons, seen_codes, "lane_topology_policy_unavailable", detail="slice12_policy_missing")
        return len(reasons) == 0, reasons

    constraints = slice12.get("fast_path_constraints") if isinstance(slice12.get("fast_path_constraints"), Mapping) else {}
    fast_map = slice12.get("fast_path_map") if isinstance(slice12.get("fast_path_map"), list) else []

    deny_unknown = bool(constraints.get("deny_unknown_callsites") is True)
    allowed_risks = {
        str(item or "").strip().lower()
        for item in (constraints.get("allowed_risk_tiers") or [])
        if str(item or "").strip()
    }
    forbidden_risks = {
        str(item or "").strip().lower()
        for item in (constraints.get("forbidden_risk_tiers") or [])
        if str(item or "").strip()
    }

    from_tag = _lane_column_tag(str(packet_obj.get("from_lane_id") or ""))
    to_tag = _lane_column_tag(str(packet_obj.get("to_lane_id") or ""))

    matched_policy_row: Mapping[str, Any] | None = None
    for row in fast_map:
        if not isinstance(row, Mapping):
            continue
        op = str(row.get("operation") or "").strip()
        row_from_tag = _tag_from_policy_lane(row.get("from_lane"))
        row_to_tag = _tag_from_policy_lane(row.get("to_lane"))
        if op != operation_id:
            continue
        if row_from_tag and from_tag and row_from_tag != from_tag:
            continue
        if row_to_tag and to_tag and row_to_tag != to_tag:
            continue
        matched_policy_row = row
        break

    if deny_unknown and matched_policy_row is None:
        _add_reason(reasons, seen_codes, "cross_lane_unknown_callsite")

    if risk_tier and risk_tier in forbidden_risks:
        _add_reason(reasons, seen_codes, "cross_lane_risk_tier_forbidden")

    if lease_mode == "fast_path":
        if matched_policy_row is None:
            _add_reason(reasons, seen_codes, "cross_lane_fast_path_not_allowlisted")
        if allowed_risks and risk_tier and risk_tier not in allowed_risks:
            _add_reason(reasons, seen_codes, "cross_lane_fast_path_risk_tier_not_allowed")
        max_fast_ttl = int(constraints.get("max_fast_path_ttl_seconds") or 0)
        if max_fast_ttl > 0:
            created_at_dt = _parse_dt(packet_obj.get("created_at"))
            expires_at_dt = _parse_dt(packet_obj.get("expires_at"))
            if created_at_dt is None or expires_at_dt is None:
                _add_reason(reasons, seen_codes, "cross_lane_fast_path_ttl_invalid")
            else:
                ttl_seconds = int((expires_at_dt - created_at_dt).total_seconds())
                if ttl_seconds <= 0 or ttl_seconds > max_fast_ttl:
                    _add_reason(
                        reasons,
                        seen_codes,
                        "cross_lane_fast_path_ttl_exceeded",
                        detail=f"ttl_seconds={ttl_seconds} max={max_fast_ttl}",
                    )

    return len(reasons) == 0, reasons


def evaluate_lane_crossover_ingress(
    packet: Any,
    *,
    receiver_lane_id: str,
    receiver_lane_epoch: str,
    sender_lane_epochs: Optional[Mapping[str, str]] = None,
    now: Any = None,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    fast_path_policy_path: Path | str = DEFAULT_FAST_PATH_POLICY_PATH,
    authority_contract_path: Path | str = DEFAULT_AUTHORITY_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate lane crossover ingress packets with fail-closed rejection reasons."""

    now_utc = _resolve_now(now)
    reasons: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    checks = {
        "schema_valid": False,
        "contamination_guard_present": False,
        "receiver_lane_match": False,
        "receiver_epoch_match": False,
        "sender_lane_authorized": False,
        "sender_epoch_match": False,
        "expires_at_not_stale": False,
        "cross_lane_authority_policy_passed": False,
    }

    sender_policy: dict[str, str] = {}
    if isinstance(sender_lane_epochs, Mapping):
        for raw_lane, raw_epoch in sender_lane_epochs.items():
            lane = str(raw_lane or "").strip()
            epoch = str(raw_epoch or "").strip()
            if lane and epoch:
                sender_policy[lane] = epoch

    if not isinstance(packet, Mapping):
        _add_reason(reasons, seen_codes, "packet_not_object")
        return {
            "schema_version": SCHEMA_VERSION,
            "evaluated_at": _iso_utc(now_utc),
            "accepted": False,
            "gate_outcome": "REJECTED_INVALID",
            "checks": checks,
            "rejection_reasons": reasons,
            "rejection_codes": [row["code"] for row in reasons],
            "packet": {},
        }

    packet_obj = dict(packet)

    contamination_guard = packet_obj.get("contamination_guard")
    if isinstance(contamination_guard, Mapping):
        checks["contamination_guard_present"] = True
        inline_bytes = contamination_guard.get("max_inline_context_bytes")
        if isinstance(inline_bytes, bool) or not isinstance(inline_bytes, int):
            _add_reason(reasons, seen_codes, "contamination_guard_inline_context_invalid")
        elif inline_bytes < 0:
            _add_reason(reasons, seen_codes, "contamination_guard_inline_context_negative")
        elif inline_bytes > 2048:
            _add_reason(reasons, seen_codes, "contamination_guard_inline_context_exceeds_max")
    else:
        _add_reason(reasons, seen_codes, "contamination_guard_missing")

    to_lane_id = str(packet_obj.get("to_lane_id") or "").strip()
    to_lane_epoch = str(packet_obj.get("to_lane_epoch") or "").strip()
    from_lane_id = str(packet_obj.get("from_lane_id") or "").strip()
    from_lane_epoch = str(packet_obj.get("from_lane_epoch") or "").strip()

    expected_receiver_lane = str(receiver_lane_id or "").strip()
    expected_receiver_epoch = str(receiver_lane_epoch or "").strip()

    if to_lane_id == expected_receiver_lane and to_lane_id:
        checks["receiver_lane_match"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "receiver_lane_mismatch",
            detail=f"expected={expected_receiver_lane or 'unset'} actual={to_lane_id or 'unset'}",
        )

    if to_lane_epoch == expected_receiver_epoch and to_lane_epoch:
        checks["receiver_epoch_match"] = True
    else:
        _add_reason(
            reasons,
            seen_codes,
            "receiver_epoch_mismatch",
            detail=f"expected={expected_receiver_epoch or 'unset'} actual={to_lane_epoch or 'unset'}",
        )

    if from_lane_id and to_lane_id and from_lane_id == to_lane_id:
        _add_reason(reasons, seen_codes, "self_crossover_forbidden")

    if not sender_policy:
        _add_reason(reasons, seen_codes, "sender_lane_policy_missing")
    else:
        expected_sender_epoch = sender_policy.get(from_lane_id)
        if expected_sender_epoch is None:
            _add_reason(reasons, seen_codes, "sender_lane_not_authorized", detail=f"from_lane_id={from_lane_id or 'unset'}")
        else:
            checks["sender_lane_authorized"] = True
            if from_lane_epoch == expected_sender_epoch:
                checks["sender_epoch_match"] = True
            else:
                _add_reason(
                    reasons,
                    seen_codes,
                    "sender_epoch_mismatch",
                    detail=f"expected={expected_sender_epoch} actual={from_lane_epoch or 'unset'}",
                )

    packet_type = str(packet_obj.get("packet_type") or "").strip()
    if packet_type not in ALLOWED_PACKET_TYPES:
        _add_reason(reasons, seen_codes, "packet_type_unknown", detail=f"packet_type={packet_type or 'unset'}")
    elif packet_type in {"ticket", "deep_review"}:
        evidence_refs = packet_obj.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not any(str(item or "").strip() for item in evidence_refs):
            _add_reason(reasons, seen_codes, "required_evidence_refs_missing")

    expires_at = packet_obj.get("expires_at")
    if expires_at in {None, ""}:
        checks["expires_at_not_stale"] = True
    else:
        expires_at_dt = _parse_dt(expires_at)
        if expires_at_dt is None:
            _add_reason(reasons, seen_codes, "expires_at_invalid")
        elif expires_at_dt <= now_utc:
            _add_reason(reasons, seen_codes, "packet_expired")
        else:
            checks["expires_at_not_stale"] = True

    policy_ok, policy_reasons = _evaluate_cross_lane_ticket_policy(
        packet_obj=packet_obj,
        now_utc=now_utc,
        policy_path=Path(fast_path_policy_path),
        authority_contract_path=Path(authority_contract_path),
    )
    if policy_ok:
        checks["cross_lane_authority_policy_passed"] = True
    else:
        for row in policy_reasons:
            if not isinstance(row, Mapping):
                continue
            _add_reason(
                reasons,
                seen_codes,
                str(row.get("code") or "").strip() or "gate_policy_failed",
                detail=str(row.get("detail") or "").strip(),
            )

    schema_ok, schema_issue = _schema_validate(packet_obj, Path(schema_path))
    if schema_ok:
        checks["schema_valid"] = True
    else:
        _add_reason(reasons, seen_codes, "schema_validation_failed", detail=str(schema_issue or "unknown_schema_failure"))

    accepted = len(reasons) == 0
    decision = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at": _iso_utc(now_utc),
        "accepted": accepted,
        "gate_outcome": "ACCEPTED" if accepted else "REJECTED_INVALID",
        "checks": checks,
        "rejection_reasons": reasons,
        "rejection_codes": [row["code"] for row in reasons],
        "packet": {
            "packet_id": str(packet_obj.get("packet_id") or "") or None,
            "packet_type": packet_type or None,
            "from_lane_id": from_lane_id or None,
            "from_lane_epoch": from_lane_epoch or None,
            "to_lane_id": to_lane_id or None,
            "to_lane_epoch": to_lane_epoch or None,
        },
    }
    return decision


def _load_packet(path_value: str) -> Any:
    if path_value == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path_value).read_text(encoding="utf-8"))


def _parse_sender_policy(entries: list[str]) -> dict[str, str]:
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
    parser = argparse.ArgumentParser(description="Fail-closed ingress guard for Lane Boundary Contract v1 packets")
    parser.add_argument("--packet", required=True, help="Packet JSON path or '-' for stdin")
    parser.add_argument("--to-lane-id", required=True, help="Expected receiver lane_id")
    parser.add_argument("--to-lane-epoch", required=True, help="Expected receiver lane_epoch_id")
    parser.add_argument(
        "--allow-from",
        action="append",
        default=[],
        metavar="LANE_ID=EPOCH_ID",
        help="Authorized sender lane + expected epoch (repeatable)",
    )
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Schema path override")
    parser.add_argument(
        "--fast-path-policy",
        default=str(DEFAULT_FAST_PATH_POLICY_PATH),
        help="Cross-lane fast-path topology policy JSON path",
    )
    parser.add_argument(
        "--authority-contract",
        default=str(DEFAULT_AUTHORITY_CONTRACT_PATH),
        help="Lane authority contract JSON path",
    )
    parser.add_argument("--now", default=None, help="Override now (RFC3339); defaults to current UTC")
    args = parser.parse_args(argv)

    try:
        packet = _load_packet(args.packet)
        sender_policy = _parse_sender_policy(list(args.allow_from or []))
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "accepted": False,
            "gate_outcome": "REJECTED_INVALID",
            "rejection_reasons": [{"code": "packet_load_failed", "detail": str(exc)}],
            "rejection_codes": ["packet_load_failed"],
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2

    decision = evaluate_lane_crossover_ingress(
        packet,
        receiver_lane_id=args.to_lane_id,
        receiver_lane_epoch=args.to_lane_epoch,
        sender_lane_epochs=sender_policy,
        now=args.now,
        schema_path=Path(args.schema_path),
        fast_path_policy_path=Path(args.fast_path_policy),
        authority_contract_path=Path(args.authority_contract),
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision.get("accepted") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
