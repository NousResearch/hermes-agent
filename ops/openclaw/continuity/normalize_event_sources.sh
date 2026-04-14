#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
JSON_OUT=0
ACTION_TOKEN=""
MUTATION_TICKET=""
ATTESTATIONS=()
ATTESTATION_OBJECTS=()
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"

usage() {
  cat <<'EOF'
Usage: normalize_event_sources.sh [options]

Normalize legacy/non-namespaced continuity_events.source values to canonical prefixes.

Options:
  --db <path>            Continuity sqlite path
  --json                 JSON output
  --action-token <value> Canonical mutation token for direct entrypoint use
  --truth-anchor <value> Legacy alias of --action-token
  --mutation-ticket <v>  Authority ticket JSON string, @path, or path (high-risk token path)
  --attestation <name>   Satisfied authority attestation (repeatable)
  --attestation-object <v>
                         Structured attestation JSON string, @path, or path (repeatable)
  --allow-legacy-anchor  Allow legacy anchor-only token mode for direct token validation
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --json)
      JSON_OUT=1; shift ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

guard_args=(
  --script "normalize_event_sources.sh"
  --risk-tier "high"
  --mutation-operation "normalize_event_sources:apply"
)
if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args+=(--action-token "$ACTION_TOKEN")
fi
if [[ -n "$MUTATION_TICKET" ]]; then
  guard_args+=(--mutation-ticket "$MUTATION_TICKET")
fi
for att in "${ATTESTATIONS[@]}"; do
  if [[ -n "${att:-}" ]]; then
    guard_args+=(--attestation "$att")
  fi
done
for att_obj in "${ATTESTATION_OBJECTS[@]}"; do
  if [[ -n "${att_obj:-}" ]]; then
    guard_args+=(--attestation-object "$att_obj")
  fi
done
if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  guard_args+=(--allow-legacy-anchor)
fi
"$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$ROOT" "$DB_PATH" "$JSON_OUT" "$ROOT/ops/openclaw/continuity/queue_arbitrator.sh" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from typing import Any, Dict, Optional

root = str(sys.argv[1])
db_path = str(sys.argv[2])
json_out = bool(int(sys.argv[3]))
queue_arb = str(sys.argv[4])

mapping = {
    "test.blocker.helper": "local.test.blocker_helper",
    "test.batch2": "local.test.batch2",
}

task_id = "continuity:normalize_event_sources"
agent = "continuity_event_normalizer"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def artifact_id(task_id: str, artifact_path: str, artifact_type: str) -> str:
    seed = f"{task_id}|{artifact_type}|{artifact_path}"
    return "tart_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def queue_cmd_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "normalize_event_sources.sh:queue_arbitrator"
    return env


def run_json_cmd(cmd: list[str], timeout: int = 30, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=env)
    payload = {}
    out = (cp.stdout or "").strip()
    if out:
        try:
            maybe = json.loads(out)
            if isinstance(maybe, dict):
                payload = maybe
        except Exception:
            payload = {}
    return {
        "ok": cp.returncode == 0,
        "returncode": cp.returncode,
        "payload": payload,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
    }


def is_ingress_denied(result: Dict[str, Any]) -> bool:
    if int(result.get("returncode") or 0) != 2:
        return False
    stderr = str(result.get("stderr") or "")
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    error = str(payload.get("error") or "")
    reason = str(payload.get("reason") or "")
    combined = " ".join([stderr, error, reason]).lower()
    return "mutator ingress denied" in combined or "requires --action-token" in combined


def queue_transition(
    to_status: str,
    reason: str,
    evidence_ref: str,
    actor_role: str = "sre_watchdog",
    allow_any_transition: bool = False,
) -> Dict[str, Any]:
    cmd = [
        queue_arb,
        "transition",
        "--task-id",
        task_id,
        "--to-status",
        to_status,
        "--actor-role",
        actor_role,
        "--reason",
        reason,
        "--evidence-ref",
        evidence_ref,
        "--release-locks",
        "--json",
    ]
    if allow_any_transition:
        cmd.append("--allow-any-transition")
    return run_json_cmd(cmd, extra_env=queue_cmd_env())


def upsert_queue_task() -> None:
    ts = now_iso()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, 'QUEUED', 'sre_watchdog', NULL, 0, 3, NULL, NULL, ?, ?)
ON CONFLICT(task_id) DO UPDATE SET
  source = excluded.source,
  title = excluded.title,
  acceptance_criteria = excluded.acceptance_criteria,
  status = COALESCE(NULLIF(TRIM(work_queue.status), ''), 'QUEUED'),
  role_required = CASE
    WHEN TRIM(COALESCE(work_queue.role_required, '')) <> '' THEN work_queue.role_required
    ELSE 'sre_watchdog'
  END,
  assigned_agent = CASE WHEN work_queue.status = 'RUNNING' THEN work_queue.assigned_agent ELSE NULL END,
  last_error_log = NULL,
  cooldown_until = NULL,
  updated_at = excluded.updated_at
""",
        (
            task_id,
            "continuity_ops",
            "Normalize continuity event source namespaces",
            "Normalize legacy continuity_events.source values to canonical namespaces without queue drift.",
            ts,
            ts,
        ),
    )

    db_rel = "state/continuity/continuity_os.sqlite"
    cur.execute(
        """
INSERT INTO task_file_targets (task_id, file_path, lock_mode, created_at)
VALUES (?, ?, 'exclusive', ?)
ON CONFLICT(task_id, file_path) DO UPDATE SET
  lock_mode = excluded.lock_mode,
  created_at = excluded.created_at
""",
        (task_id, db_rel, ts),
    )

    cur.execute(
        """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, 'sqlite', ?, NULL, ?, ?)
""",
        (
            artifact_id(task_id, db_rel, "sqlite"),
            task_id,
            db_rel,
            json.dumps({"producer": "normalize_event_sources"}, ensure_ascii=False),
            ts,
        ),
    )
    con.commit()
    con.close()


def current_queue_status() -> str:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    row = cur.execute("SELECT status FROM work_queue WHERE task_id = ?", (task_id,)).fetchone()
    con.close()
    return str(row[0] or "").strip().upper() if row else ""


def force_queue_role(role_required: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "UPDATE work_queue SET role_required = ?, updated_at = ? WHERE task_id = ? AND status = 'QUEUED'",
        (role_required, now_iso(), task_id),
    )
    changed = int(cur.rowcount or 0)
    con.commit()
    con.close()
    return changed


def reopen_queue_for_claim() -> Dict[str, Any]:
    prev_status = current_queue_status() or "QUEUED"
    if prev_status in {"QUEUED", "RUNNING"}:
        role_changed = force_queue_role("sre_watchdog") if prev_status == "QUEUED" else 0
        return {
            "changed": bool(role_changed),
            "from_status": prev_status,
            "to_status": prev_status,
            "transition": None,
            "role_adjusted": bool(role_changed),
            "ok": True,
        }

    transition = queue_transition(
        "QUEUED",
        "continuity_event_source_normalization_requeue",
        "state/continuity/continuity_os.sqlite",
        actor_role="sre_watchdog",
        allow_any_transition=True,
    )
    role_changed = force_queue_role("sre_watchdog") if transition.get("ok") else 0
    return {
        "changed": bool(transition.get("ok") or role_changed),
        "from_status": prev_status,
        "to_status": "QUEUED",
        "transition": {
            "returncode": transition.get("returncode"),
            "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
            "stderr": transition.get("stderr") or "",
        },
        "role_adjusted": bool(role_changed),
        "ok": bool(transition.get("ok")),
    }


result: Dict[str, Any] = {
    "ok": False,
    "task_id": task_id,
}
claimed = False
transition_done = False

try:
    upsert_queue_task()
    reopen = reopen_queue_for_claim()
    result["queue_reopen"] = reopen
    if reopen.get("ok") is False:
        result.update({"ok": False, "error": "queue_reopen_failed"})
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    claim = run_json_cmd(
        [
            queue_arb,
            "claim",
            "--agent",
            agent,
            "--actor-role",
            "sre_watchdog",
            "--task-id",
            task_id,
            "--lock-ttl-sec",
            "1800",
            "--json",
        ],
        extra_env=queue_cmd_env(),
    )

    result["queue_claim"] = {
        "returncode": claim.get("returncode"),
        "payload": claim.get("payload") if isinstance(claim.get("payload"), dict) else {},
        "stderr": claim.get("stderr") or "",
    }

    if not claim.get("ok"):
        if is_ingress_denied(claim):
            result.update({"ok": False, "error": "queue_claim_ingress_denied"})
            if json_out:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            raise SystemExit(1)

        payload = claim.get("payload") if isinstance(claim.get("payload"), dict) else {}
        skipped = payload.get("skipped") if isinstance(payload.get("skipped"), list) else []
        reason = "no_claimable_task"
        if isinstance(payload.get("error"), str) and payload.get("error"):
            reason = str(payload.get("error"))
        if skipped and isinstance(skipped[0], dict) and skipped[0].get("reason"):
            reason = str(skipped[0].get("reason"))
        result.update({"ok": True, "changed": False, "claim_deferred": True, "reason": reason})
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0)

    claimed = True

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    updated = {}
    for old, new in mapping.items():
        cur.execute("UPDATE continuity_events SET source = ? WHERE source = ?", (new, old))
        updated[f"{old}->{new}"] = int(cur.rowcount or 0)

    con.commit()
    remaining = [
        {"source": r[0], "count": int(r[1] or 0)}
        for r in cur.execute(
            """
SELECT source, COUNT(*)
FROM continuity_events
WHERE source NOT GLOB 'continuity.*'
  AND source NOT GLOB 'watchdog.*'
  AND source NOT GLOB 'runtime.*'
  AND source NOT GLOB 'local.*'
GROUP BY source
ORDER BY COUNT(*) DESC
"""
        ).fetchall()
    ]
    con.close()

    evidence_ref = "state/continuity/continuity_os.sqlite"
    transition = queue_transition(
        "DONE",
        "continuity_event_source_normalization_completed",
        evidence_ref,
    )
    if is_ingress_denied(transition):
        result["queue_transition"] = {
            "returncode": transition.get("returncode"),
            "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
            "stderr": transition.get("stderr") or "",
        }
        result.update({"ok": False, "error": "queue_transition_ingress_denied"})
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)
    transition_done = bool(transition.get("ok"))

    result.update(
        {
            "ok": bool(transition.get("ok")),
            "changed": bool(sum(updated.values())),
            "updated": updated,
            "remaining_noncanonical": remaining,
            "queue_transition": {
                "returncode": transition.get("returncode"),
                "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
                "stderr": transition.get("stderr") or "",
            },
        }
    )

    if not transition.get("ok"):
        result["error"] = "queue_transition_failed"
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    if json_out:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("EVENT SOURCE NORMALIZATION")
        print(f"- updated: {updated}")
        print(f"- remaining_noncanonical: {len(remaining)}")
    raise SystemExit(0)

except SystemExit:
    raise
except Exception as exc:
    result.update({"ok": False, "error": str(exc)})
    if claimed and not transition_done:
        try:
            transition = queue_transition(
                "BLOCKED",
                "continuity_event_source_normalization_failed",
                "state/continuity/continuity_os.sqlite",
            )
            result["queue_transition"] = {
                "returncode": transition.get("returncode"),
                "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
                "stderr": transition.get("stderr") or "",
            }
        except Exception as transition_exc:
            result["queue_transition_error"] = str(transition_exc)

    print(json.dumps(result, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)
PY
