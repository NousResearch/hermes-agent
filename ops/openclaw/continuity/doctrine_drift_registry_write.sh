#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
ACTION_TOKEN=""
MUTATION_TICKET=""
ATTESTATIONS=()
ATTESTATION_OBJECTS=()
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
SHOW_HELP=0

i=1
while [[ $i -le $# ]]; do
  arg_i="${!i}"
  case "$arg_i" in
    --action-token|--truth-anchor)
      next_i=$((i + 1))
      ACTION_TOKEN="${!next_i:-}"
      i=$((i + 2)) ;;
    --mutation-ticket)
      next_i=$((i + 1))
      MUTATION_TICKET="${!next_i:-}"
      i=$((i + 2)) ;;
    --attestation)
      next_i=$((i + 1))
      ATTESTATIONS+=("${!next_i:-}")
      i=$((i + 2)) ;;
    --attestation-object)
      next_i=$((i + 1))
      ATTESTATION_OBJECTS+=("${!next_i:-}")
      i=$((i + 2)) ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1
      i=$((i + 1)) ;;
    -h|--help)
      SHOW_HELP=1
      i=$((i + 1)) ;;
    --registry|--incident-id|--reason|--severity|--status|--detail|--reported-at|--detected-at|--evidence)
      i=$((i + 2)) ;;
    --json)
      i=$((i + 1)) ;;
    *)
      i=$((i + 1)) ;;
  esac
done

if [[ "$SHOW_HELP" != "1" ]]; then
  guard_args=(
    --script "doctrine_drift_registry_write.sh"
    --risk-tier high
    --mutation-operation "doctrine_drift_registry_write:upsert"
  )
  if [[ -n "$ACTION_TOKEN" ]]; then
    guard_args+=(--action-token "$ACTION_TOKEN")
  fi
  if [[ -n "$MUTATION_TICKET" ]]; then
    guard_args+=(--mutation-ticket "$MUTATION_TICKET")
  fi
  att=""
  for att in "${ATTESTATIONS[@]}"; do
    if [[ -n "$att" ]]; then
      guard_args+=(--attestation "$att")
    fi
  done
  att_obj=""
  for att_obj in "${ATTESTATION_OBJECTS[@]}"; do
    if [[ -n "$att_obj" ]]; then
      guard_args+=(--attestation-object "$att_obj")
    fi
  done
  if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
    guard_args+=(--allow-legacy-anchor)
  fi
  "$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"
fi

python3 - "$ROOT" "$@" <<'PY'
import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
argv = sys.argv[2:]

DEFAULT_REGISTRY_REL = "state/continuity/latest/doctrine_drift_registry.json"
EXPECTED_SCHEMA = "clawd.continuity.doctrine_drift_registry.v1"

continuity_dir = (root / "ops" / "openclaw" / "continuity").resolve()
schema_path = (root / "ops" / "openclaw" / "architecture" / "schemas" / "doctrine_drift_registry.schema.json").resolve()
sys.path.insert(0, str(continuity_dir))

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(raw: Any) -> Optional[str]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    probe = txt[:-1] + "+00:00" if txt.endswith("Z") else txt
    try:
        parsed = dt.datetime.fromisoformat(probe)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = pathlib.Path(fh.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def unique_strings(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in (values or []):
        txt = str(raw or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def normalize_severity(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    if txt not in {"blocker", "warn", "info"}:
        return "warn"
    return txt


def normalize_status(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    if txt in {"", "open", "active", "new"}:
        return "open"
    if txt in {"resolved", "closed", "done", "dismissed"}:
        return txt
    return txt


def incident_id_from_reason(reason: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)[:64]
    if not slug:
        slug = "incident"
    return f"dd_{slug}"


def normalize_incident(raw: Dict[str, Any], *, fallback_reason: str, fallback_id: str) -> Dict[str, Any]:
    reason = str(raw.get("reason") or raw.get("code") or raw.get("incident_id") or fallback_reason).strip() or fallback_reason
    incident_id = str(raw.get("incident_id") or "").strip() or fallback_id or incident_id_from_reason(reason)

    out: Dict[str, Any] = {}

    # Preserve unknown fields by default to avoid destructive rewrites.
    for key, value in raw.items():
        if key in {
            "incident_id",
            "reason",
            "severity",
            "status",
            "detail",
            "reported_at",
            "detected_at",
            "evidence",
            "updated_at",
        }:
            continue
        out[key] = value

    out["incident_id"] = incident_id
    out["reason"] = reason
    out["severity"] = normalize_severity(raw.get("severity"))
    out["status"] = normalize_status(raw.get("status"))

    detail = str(raw.get("detail") or "").strip()
    if detail:
        out["detail"] = detail

    reported_at = parse_iso(raw.get("reported_at")) or parse_iso(raw.get("detected_at"))
    if reported_at:
        out["reported_at"] = reported_at

    detected_at = parse_iso(raw.get("detected_at"))
    if detected_at:
        out["detected_at"] = detected_at

    evidence = unique_strings(raw.get("evidence"))
    if evidence:
        out["evidence"] = evidence

    updated_at = parse_iso(raw.get("updated_at"))
    if updated_at:
        out["updated_at"] = updated_at

    return out


def minimal_validate(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("doctrine_drift_registry_contract_invalid:not_object")

    if payload.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError("doctrine_drift_registry_contract_invalid:schema")

    generated_at = parse_iso(payload.get("generated_at"))
    if generated_at is None:
        raise RuntimeError("doctrine_drift_registry_contract_invalid:generated_at")

    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        raise RuntimeError("doctrine_drift_registry_contract_invalid:incidents_not_array")

    for idx, row in enumerate(incidents):
        if not isinstance(row, dict):
            raise RuntimeError(f"doctrine_drift_registry_contract_invalid:incident_not_object:{idx}")
        if not str(row.get("incident_id") or "").strip():
            raise RuntimeError(f"doctrine_drift_registry_contract_invalid:incident_id_missing:{idx}")
        if not str(row.get("reason") or "").strip():
            raise RuntimeError(f"doctrine_drift_registry_contract_invalid:reason_missing:{idx}")
        severity = normalize_severity(row.get("severity"))
        if severity not in {"blocker", "warn", "info"}:
            raise RuntimeError(f"doctrine_drift_registry_contract_invalid:severity_invalid:{idx}")


def validate_contract(payload: Dict[str, Any]) -> None:
    if not schema_path.exists():
        raise RuntimeError(f"doctrine_drift_registry_schema_missing:{schema_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_doc, dict):
        raise RuntimeError("doctrine_drift_registry_schema_not_object")

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except Exception:
        minimal_validate(payload)
        return

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return

    err = errors[0]
    data_ptr = "$/" + "/".join(str(p) for p in err.absolute_path) if err.absolute_path else "$"
    schema_ptr = "$/" + "/".join(str(p) for p in err.absolute_schema_path) if err.absolute_schema_path else "$"
    raise RuntimeError(
        "doctrine_drift_registry_schema_validation_failed:"
        f"data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )


def sort_incidents(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    severity_rank = {"blocker": 0, "warn": 1, "info": 2}
    status_rank = {"open": 0, "active": 0, "new": 0, "resolved": 1, "closed": 1, "done": 1, "dismissed": 1}

    def key_fn(row: Dict[str, Any]) -> Tuple[int, int, str, str]:
        status = normalize_status(row.get("status"))
        severity = normalize_severity(row.get("severity"))
        reason = str(row.get("reason") or "").strip()
        incident_id = str(row.get("incident_id") or "").strip()
        return (
            status_rank.get(status, 0),
            severity_rank.get(severity, 1),
            reason,
            incident_id,
        )

    return sorted(rows, key=key_fn)


parser = argparse.ArgumentParser(
    prog="doctrine_drift_registry_write.sh",
    description="Deterministically upsert doctrine drift incidents into canonical registry.",
)
parser.add_argument(
    "--registry",
    default=(
        str(
            os.environ.get(
                "OPENCLAW_CONTINUITY_DOCTRINE_DRIFT_REGISTRY_REL",
                DEFAULT_REGISTRY_REL,
            )
        ).strip()
        or DEFAULT_REGISTRY_REL
    ),
    help="Registry JSON path (default: OPENCLAW_CONTINUITY_DOCTRINE_DRIFT_REGISTRY_REL or state/continuity/latest/doctrine_drift_registry.json)",
)
parser.add_argument("--incident-id", default=None, help="Explicit incident id (default: deterministic from reason)")
parser.add_argument("--reason", required=True, help="Doctrine drift reason key")
parser.add_argument("--severity", choices=["blocker", "warn", "info"], default=None, help="Incident severity")
parser.add_argument("--status", choices=["open", "resolved", "closed", "done", "dismissed"], default=None, help="Incident status")
parser.add_argument("--detail", default=None, help="Optional detail text")
parser.add_argument("--reported-at", default=None, help="Optional reported-at timestamp (ISO-8601)")
parser.add_argument("--detected-at", default=None, help="Optional detected-at timestamp (ISO-8601)")
parser.add_argument("--evidence", action="append", default=[], help="Evidence reference (repeatable)")
parser.add_argument("--json", action="store_true", help="Print resulting registry JSON")
parser.add_argument("--action-token", default=None, help="Canonical mutation token for direct mutating entrypoint use")
parser.add_argument("--truth-anchor", default=None, help="Legacy alias of --action-token")
parser.add_argument("--allow-legacy-anchor", action="store_true", help="Allow legacy anchor-only token mode for direct token validation")
parser.add_argument("--mutation-ticket", default=None, help="Authority ticket JSON string/path for high-risk token-path mutation")
parser.add_argument("--attestation", action="append", default=[], help="Authority attestation name (repeatable)")
parser.add_argument("--attestation-object", action="append", default=[], help="Structured authority attestation JSON string/path (repeatable)")
args = parser.parse_args(argv)

registry_path = pathlib.Path(str(args.registry or "").strip() or DEFAULT_REGISTRY_REL)
if not registry_path.is_absolute():
    registry_path = (root / registry_path).resolve()
else:
    registry_path = registry_path.resolve()

reason = str(args.reason or "").strip()
if not reason:
    raise SystemExit("--reason is required")

explicit_incident_id = str(args.incident_id or "").strip() or None
incident_id = explicit_incident_id or incident_id_from_reason(reason)

existing_payload: Dict[str, Any]
if registry_path.exists():
    try:
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"doctrine_drift_registry_read_failed:{exc}")
    if not isinstance(loaded, dict):
        raise SystemExit("doctrine_drift_registry_invalid:not_object")
    schema_name = str(loaded.get("schema") or "").strip()
    if schema_name and schema_name != EXPECTED_SCHEMA:
        raise SystemExit(f"doctrine_drift_registry_schema_mismatch:{schema_name}")
    existing_payload = loaded
else:
    existing_payload = {
        "schema": EXPECTED_SCHEMA,
        "generated_at": now_iso(),
        "incidents": [],
    }

existing_rows_raw = existing_payload.get("incidents") if isinstance(existing_payload.get("incidents"), list) else []
existing_rows: List[Dict[str, Any]] = []
for idx, raw in enumerate(existing_rows_raw):
    if not isinstance(raw, dict):
        continue
    fallback_reason = str(raw.get("reason") or raw.get("code") or raw.get("incident_id") or f"incident_{idx + 1}").strip() or f"incident_{idx + 1}"
    fallback_id = str(raw.get("incident_id") or "").strip() or incident_id_from_reason(fallback_reason)
    existing_rows.append(normalize_incident(raw, fallback_reason=fallback_reason, fallback_id=fallback_id))

match_idx: Optional[int] = None
for idx, row in enumerate(existing_rows):
    if str(row.get("incident_id") or "").strip() == incident_id:
        match_idx = idx
        break

if match_idx is None and explicit_incident_id is None:
    for idx, row in enumerate(existing_rows):
        if str(row.get("reason") or "").strip() == reason:
            match_idx = idx
            break

now = now_iso()
severity = normalize_severity(args.severity) if args.severity is not None else None
status = normalize_status(args.status) if args.status is not None else None
reported_at = parse_iso(args.reported_at) if args.reported_at is not None else None
detected_at = parse_iso(args.detected_at) if args.detected_at is not None else None
detail = str(args.detail or "").strip() if args.detail is not None else None
incoming_evidence = unique_strings(args.evidence)

if match_idx is not None:
    row = dict(existing_rows[match_idx])
    row["incident_id"] = incident_id
    row["reason"] = reason
    if severity is not None:
        row["severity"] = severity
    else:
        row["severity"] = normalize_severity(row.get("severity"))

    if status is not None:
        row["status"] = status
    else:
        row["status"] = normalize_status(row.get("status"))

    if detail is not None:
        if detail:
            row["detail"] = detail
        else:
            row.pop("detail", None)

    merged_evidence = unique_strings((row.get("evidence") or []) + incoming_evidence)
    if merged_evidence:
        row["evidence"] = merged_evidence
    else:
        row.pop("evidence", None)

    if reported_at is not None:
        row["reported_at"] = reported_at
    elif parse_iso(row.get("reported_at")) is None and parse_iso(row.get("detected_at")) is not None:
        row["reported_at"] = parse_iso(row.get("detected_at"))

    if detected_at is not None:
        row["detected_at"] = detected_at

    row["updated_at"] = now
    existing_rows[match_idx] = row
else:
    new_row: Dict[str, Any] = {
        "incident_id": incident_id,
        "reason": reason,
        "severity": severity or "warn",
        "status": status or "open",
        "updated_at": now,
    }
    if detail:
        new_row["detail"] = detail

    if reported_at is not None:
        new_row["reported_at"] = reported_at
    elif detected_at is not None:
        new_row["reported_at"] = detected_at
    else:
        new_row["reported_at"] = now

    if detected_at is not None:
        new_row["detected_at"] = detected_at

    if incoming_evidence:
        new_row["evidence"] = incoming_evidence

    existing_rows.append(new_row)

incidents_sorted = sort_incidents(existing_rows)
open_total = sum(1 for row in incidents_sorted if normalize_status(row.get("status")) not in {"resolved", "closed", "done", "dismissed"})

payload: Dict[str, Any] = {
    "schema": EXPECTED_SCHEMA,
    "generated_at": now,
    "incidents": incidents_sorted,
}

validate_contract(payload)
atomic_write(registry_path, payload)

if args.json:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(
        "DOCTRINE DRIFT REGISTRY: "
        f"path={to_rel(registry_path)} incident_id={incident_id} "
        f"status={status or ('updated' if match_idx is not None else 'open')} "
        f"incidents_total={len(incidents_sorted)} open_total={open_total}"
    )
PY
