#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
APPLY=0
LIMIT=5000
JSON_OUT=0
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: repair_provider_failure_summaries.sh [options]

Audit and optionally repair malformed autopilot.provider_failure_summary.v1 payloads
persisted in task_handoff_packets.gate_metadata_json.

Options:
  --db <path>           Continuity sqlite path (default: state/continuity/continuity_os.sqlite)
  --limit <n>           Max handoff rows to scan (default: 5000)
  --apply               Apply in-place repair (default: dry-run)
  --json                JSON output (default)
  --action-token <value>
                        Canonical mutation token for --apply mode.
  --truth-anchor <value>
                        Legacy alias of --action-token.
  --allow-legacy-anchor
                        Allow legacy anchor-only token mode for direct token validation.
  --mutation-ticket <value>
                        Authority ticket JSON string, @path, or path (high-risk token path).
  --attestation <name>
                        Satisfied authority attestation name (repeatable).
  --attestation-object <value>
                        Structured authority attestation JSON string, @path, or path (repeatable).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --limit)
      LIMIT="${2:-}"; shift 2 ;;
    --apply)
      APPLY=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "invalid --limit: $LIMIT (expected integer >= 1)" >&2
  exit 2
fi
if [[ "$LIMIT" -lt 1 ]]; then
  echo "invalid --limit: $LIMIT (expected integer >= 1)" >&2
  exit 2
fi

if [[ "$APPLY" == "1" ]]; then
  guard_args=(
    --script "repair_provider_failure_summaries.sh"
    --risk-tier "high"
    --mutation-operation "repair_provider_failure_summaries:apply"
  )
  if [[ -n "$ACTION_TOKEN" ]]; then
    guard_args+=(--action-token "$ACTION_TOKEN")
  fi
  if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
    guard_args+=(--allow-legacy-anchor)
  fi
  if [[ -n "$MUTATION_TICKET" ]]; then
    guard_args+=(--mutation-ticket "$MUTATION_TICKET")
  fi
  for att in "${MUTATION_ATTESTATIONS[@]}"; do
    if [[ -n "${att:-}" ]]; then
      guard_args+=(--attestation "$att")
    fi
  done
  for att_obj in "${MUTATION_ATTESTATION_OBJECTS[@]}"; do
    if [[ -n "${att_obj:-}" ]]; then
      guard_args+=(--attestation-object "$att_obj")
    fi
  done
  "$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"
fi

python3 - "$DB_PATH" "$LIMIT" "$APPLY" "$ROOT" <<'PY'
import datetime as dt
import json
import pathlib
import sqlite3
import sys
from typing import Any, Dict, List


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


db_path = pathlib.Path(sys.argv[1]).resolve()
limit = max(1, int(sys.argv[2] or 5000))
apply = bool(int(sys.argv[3] or 0))
root = pathlib.Path(sys.argv[4]).resolve()

if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))

try:
    from walletdb.provider_failure import (  # type: ignore
        PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION,
        validate_provider_failure_summary,
    )
except Exception as exc:  # pragma: no cover - defensive fallback
    PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION = "autopilot.provider_failure_summary.v1"
    _validator_import_error = str(exc)

    def validate_provider_failure_summary(summary: Any, *, strict: bool = True) -> Dict[str, Any]:
        return {
            "ok": False,
            "issues": [f"provider_failure_summary_validator_unavailable:{_validator_import_error}"],
            "schema_version": str((summary or {}).get("schema_version") if isinstance(summary, dict) else ""),
        }

if not db_path.exists():
    print(json.dumps({"ok": False, "error": "missing_db", "db_path": str(db_path)}, ensure_ascii=False))
    raise SystemExit(1)

con = sqlite3.connect(str(db_path))
con.row_factory = sqlite3.Row
cur = con.cursor()

rows_scanned = 0
provider_summary_rows_checked = 0
provider_summary_rows_invalid = 0
provider_summary_rows_repaired = 0
invalid_samples: List[Dict[str, Any]] = []

for row in cur.execute(
    """
SELECT packet_id, task_id, gate_metadata_json, failure_signature
FROM task_handoff_packets
WHERE gate_metadata_json IS NOT NULL
  AND TRIM(gate_metadata_json) <> ''
ORDER BY created_at DESC
LIMIT ?
""",
    (limit,),
).fetchall():
    rows_scanned += 1
    packet_id = str(row["packet_id"] or "")
    task_id = str(row["task_id"] or "")
    raw_meta = str(row["gate_metadata_json"] or "").strip()
    if not raw_meta:
        continue
    try:
        gate_meta = json.loads(raw_meta)
    except Exception:
        continue
    if not isinstance(gate_meta, dict):
        continue

    summary = gate_meta.get("gate_summary")
    if not isinstance(summary, dict):
        continue
    if str(summary.get("schema_version") or "") != PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION:
        continue

    provider_summary_rows_checked += 1
    verdict = validate_provider_failure_summary(summary, strict=True)
    if bool(verdict.get("ok") is True):
        continue

    provider_summary_rows_invalid += 1
    issues = verdict.get("issues") if isinstance(verdict.get("issues"), list) else []
    issue_text = [str(item) for item in issues][:8]
    sample = {
        "packet_id": packet_id,
        "task_id": task_id,
        "summary_signature": str(summary.get("summary_signature") or "")[:80] or None,
        "issues": issue_text,
    }
    if len(invalid_samples) < 10:
        invalid_samples.append(sample)

    if not apply:
        continue

    updated_gate_meta = dict(gate_meta)
    updated_gate_meta["gate_summary"] = None
    updated_gate_meta["gate_summary_repair"] = {
        "schema_version": "repair.provider_failure_summary.v1",
        "applied_at": now_iso(),
        "action": "drop_invalid_provider_failure_summary",
        "provider_failure_schema_version": PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION,
        "issues": issue_text,
        "original_summary_signature": str(summary.get("summary_signature") or "")[:120] or None,
    }

    current_failure_signature = str(row["failure_signature"] or "")
    summary_signature = str(summary.get("summary_signature") or "")
    next_failure_signature = current_failure_signature or None
    if summary_signature and current_failure_signature == summary_signature:
        fallback_signature = str(gate_meta.get("reason") or "").strip()[:240]
        if not fallback_signature:
            fallback_signature = f"repair_invalid_provider_summary:{packet_id}"[:240]
        next_failure_signature = fallback_signature

    cur.execute(
        """
UPDATE task_handoff_packets
SET gate_metadata_json = ?,
    failure_signature = ?
WHERE packet_id = ?
""",
        (json.dumps(updated_gate_meta, ensure_ascii=False, sort_keys=True), next_failure_signature, packet_id),
    )
    provider_summary_rows_repaired += int(cur.rowcount or 0)

if apply:
    con.commit()
else:
    con.rollback()
con.close()

out = {
    "ok": True,
    "apply": apply,
    "db_path": str(db_path),
    "scan_limit": limit,
    "rows_scanned": rows_scanned,
    "provider_summary_rows_checked": provider_summary_rows_checked,
    "provider_summary_rows_invalid": provider_summary_rows_invalid,
    "provider_summary_rows_repaired": provider_summary_rows_repaired,
    "invalid_samples": invalid_samples,
}
if not apply and provider_summary_rows_invalid > 0:
    out["recommended_apply"] = (
        "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/repair_provider_failure_summaries.sh "
        f"--db {db_path} --apply --json"
    )

print(json.dumps(out, ensure_ascii=False, indent=2))
PY
