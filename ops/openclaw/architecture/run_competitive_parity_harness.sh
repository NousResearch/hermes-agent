#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
INPUT_DIR="$ROOT/state/architecture/competitive_parity/inbox"
OUTPUT_ROOT="$ROOT/state/architecture/competitive_parity"
HARNESS_PATH="$ROOT/ops/openclaw/architecture/competitive_parity_harness.v1.yaml"
SCHEMA_PATH="$ROOT/ops/openclaw/architecture/schemas/competitive_scorecard.schema.json"
QUEUE_ARB_SCRIPT="$ROOT/ops/openclaw/continuity/queue_arbitrator.sh"
EVENT_ROUTER_SCRIPT="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
TASK_ID="parity:weekly_harness"
AGENT="competitive_parity_runner"
LOCK_TTL_SEC="7200"
EVENT_COOLDOWN_SEC="86400"
RUN_ID=""
JSON_OUT=0
EMIT_EVENTS=1

SCORECARDS=()

usage() {
  cat <<'EOF'
Usage: run_competitive_parity_harness.sh [options]

Low-noise competitive parity harness runner.
- Claims/locks through queue_arbitrator (non-autopilot producer path).
- Validates scorecards against competitive_scorecard schema.
- Writes run + dashboard artifacts under state/architecture/competitive_parity.
- Emits events only when blockers/regressions are detected.

Options:
  --scorecard <path>            Scorecard JSON path. Repeatable.
  --input-dir <path>            Auto-discovery root for scorecards when --scorecard omitted
                                (default: state/architecture/competitive_parity/inbox).
  --output-root <path>          Artifact output root
                                (default: state/architecture/competitive_parity).
  --task-id <id>                Queue task id (default: parity:weekly_harness)
  --agent <name>                Queue claim agent (default: competitive_parity_runner)
  --lock-ttl-sec <n>            Queue lock TTL seconds (default: 7200)
  --event-cooldown-sec <n>      Event dedupe cooldown seconds (default: 86400)
  --run-id <id>                 Explicit run id (default: parity_YYYYmmddTHHMMSSZ)
  --db <path>                   Continuity DB path override
  --harness <path>              Harness contract YAML path override
  --schema <path>               Scorecard JSON schema path override
  --queue-arb <path>            queue_arbitrator.sh path override
  --event-router <path>         event_router.sh path override
  --no-events                   Disable blocker/regression event emission
  --json                        JSON output
  -h, --help

Exit codes:
  0  success or claim-deferred (silent by default)
  1  runtime/infrastructure failure
  2  regression/blocker detected
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scorecard)
      SCORECARDS+=("${2:-}"); shift 2 ;;
    --input-dir)
      INPUT_DIR="${2:-}"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --task-id)
      TASK_ID="${2:-}"; shift 2 ;;
    --agent)
      AGENT="${2:-}"; shift 2 ;;
    --lock-ttl-sec)
      LOCK_TTL_SEC="${2:-}"; shift 2 ;;
    --event-cooldown-sec)
      EVENT_COOLDOWN_SEC="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --harness)
      HARNESS_PATH="${2:-}"; shift 2 ;;
    --schema)
      SCHEMA_PATH="${2:-}"; shift 2 ;;
    --queue-arb)
      QUEUE_ARB_SCRIPT="${2:-}"; shift 2 ;;
    --event-router)
      EVENT_ROUTER_SCRIPT="${2:-}"; shift 2 ;;
    --no-events)
      EMIT_EVENTS=0; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

scorecards_json="[]"
if [[ ${#SCORECARDS[@]} -gt 0 ]]; then
  scorecards_json="$(python3 - <<'PY' "${SCORECARDS[@]}"
import json
import sys
print(json.dumps(sys.argv[1:], ensure_ascii=False))
PY
)"
fi

python3 - "$ROOT" "$DB_PATH" "$INPUT_DIR" "$OUTPUT_ROOT" "$HARNESS_PATH" "$SCHEMA_PATH" "$QUEUE_ARB_SCRIPT" "$EVENT_ROUTER_SCRIPT" "$TASK_ID" "$AGENT" "$LOCK_TTL_SEC" "$EVENT_COOLDOWN_SEC" "$RUN_ID" "$JSON_OUT" "$EMIT_EVENTS" "$scorecards_json" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
db_path = pathlib.Path(sys.argv[2]).resolve()
input_dir = pathlib.Path(sys.argv[3]).resolve()
output_root = pathlib.Path(sys.argv[4]).resolve()
harness_path = pathlib.Path(sys.argv[5]).resolve()
schema_path = pathlib.Path(sys.argv[6]).resolve()
queue_arb_script = pathlib.Path(sys.argv[7]).resolve()
event_router_script = pathlib.Path(sys.argv[8]).resolve()
task_id = str(sys.argv[9]).strip()
agent = str(sys.argv[10]).strip()
try:
    lock_ttl_sec = max(60, int(sys.argv[11]))
except Exception:
    lock_ttl_sec = 7200
try:
    event_cooldown_sec = max(0, int(sys.argv[12]))
except Exception:
    event_cooldown_sec = 86400
run_id_arg = str(sys.argv[13]).strip()
json_out = bool(int(sys.argv[14]))
emit_events = bool(int(sys.argv[15]))
try:
    explicit_scorecards = [str(x) for x in json.loads(sys.argv[16])]
except Exception:
    explicit_scorecards = []

try:
    import jsonschema
except Exception as exc:
    out = {"ok": False, "error": f"jsonschema_missing:{exc}"}
    print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)

try:
    import yaml
except Exception as exc:
    out = {"ok": False, "error": f"pyyaml_missing:{exc}"}
    print(json.dumps(out, ensure_ascii=False, indent=2 if json_out else None))
    raise SystemExit(1)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(p: pathlib.Path) -> str:
    try:
        return p.resolve().relative_to(root).as_posix()
    except Exception:
        return str(p)


def atomic_write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def run_json_cmd(cmd: List[str], timeout: int = 45, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=env)
    except Exception as exc:
        return {"ok": False, "returncode": 127, "error": f"exec_failed:{exc}", "stdout": "", "stderr": str(exc), "payload": {}}

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
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
        "payload": payload,
    }


def queue_claim() -> Dict[str, Any]:
    queue_env = dict(os.environ)
    queue_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    queue_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "run_competitive_parity_harness.sh:queue_claim"
    return run_json_cmd(
        [
            str(queue_arb_script),
            "claim",
            "--agent",
            agent,
            "--actor-role",
            "validator",
            "--task-id",
            task_id,
            "--lock-ttl-sec",
            str(lock_ttl_sec),
            "--json",
        ],
        timeout=30,
        env=queue_env,
    )


def queue_transition(
    to_status: str,
    reason: str,
    evidence_ref: str = "",
    actor_role: str = "validator",
    release_locks: bool = True,
    allow_any_transition: bool = False,
) -> Dict[str, Any]:
    cmd = [
        str(queue_arb_script),
        "transition",
        "--task-id",
        task_id,
        "--to-status",
        to_status,
        "--actor-role",
        actor_role,
        "--reason",
        reason,
        "--json",
    ]
    if evidence_ref:
        cmd.extend(["--evidence-ref", evidence_ref])
    if release_locks:
        cmd.append("--release-locks")
    if allow_any_transition:
        cmd.append("--allow-any-transition")

    queue_env = dict(os.environ)
    queue_env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    queue_env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "run_competitive_parity_harness.sh:queue_transition"
    return run_json_cmd(cmd, timeout=30, env=queue_env)


def upsert_queue_task() -> None:
    ts = now_iso()
    acceptance = (
        "Publish schema-validated competitive parity dashboard artifacts. "
        "Block when score regression exceeds 10pp week-over-week or artifact bundle is missing."
    )
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, 'QUEUED', 'validator', NULL, 0, 3, NULL, NULL, ?, ?)
ON CONFLICT(task_id) DO UPDATE SET
  source = excluded.source,
  title = excluded.title,
  acceptance_criteria = excluded.acceptance_criteria,
  status = COALESCE(NULLIF(TRIM(work_queue.status), ''), 'QUEUED'),
  role_required = CASE
    WHEN TRIM(COALESCE(work_queue.role_required, '')) <> '' THEN work_queue.role_required
    ELSE 'validator'
  END,
  assigned_agent = CASE WHEN work_queue.status = 'RUNNING' THEN work_queue.assigned_agent ELSE NULL END,
  last_error_log = NULL,
  cooldown_until = NULL,
  updated_at = excluded.updated_at
""",
        (
            task_id,
            "competitive_parity",
            "Competitive parity harness weekly runner",
            acceptance,
            ts,
            ts,
        ),
    )

    lock_targets = [
        output_root,
        output_root / "dashboard" / "latest.json",
        output_root / "dashboard" / "history.jsonl",
        output_root / "runs",
    ]
    for p in lock_targets:
        cur.execute(
            """
INSERT INTO task_file_targets (task_id, file_path, lock_mode, created_at)
VALUES (?, ?, 'exclusive', ?)
ON CONFLICT(task_id, file_path) DO UPDATE SET
  lock_mode = excluded.lock_mode,
  created_at = excluded.created_at
""",
            (task_id, rel(pathlib.Path(p)), ts),
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
        role_changed = force_queue_role("validator") if prev_status == "QUEUED" else 0
        return {
            "changed": bool(role_changed),
            "from_status": prev_status,
            "to_status": prev_status,
            "transition": None,
            "role_adjusted": bool(role_changed),
            "ok": True,
        }

    transition = queue_transition(
        to_status="QUEUED",
        reason="competitive_parity_requeue_for_new_run",
        evidence_ref=rel(output_root),
        actor_role="validator",
        release_locks=False,
        allow_any_transition=True,
    )
    role_changed = force_queue_role("validator") if transition.get("ok") else 0
    return {
        "changed": bool(transition.get("ok") or role_changed),
        "from_status": prev_status,
        "to_status": "QUEUED",
        "transition": {
            "returncode": transition.get("returncode"),
            "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
        },
        "role_adjusted": bool(role_changed),
        "ok": bool(transition.get("ok")),
    }


def discover_scorecards() -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    if explicit_scorecards:
        for raw in explicit_scorecards:
            p = pathlib.Path(raw)
            if not p.is_absolute():
                p = (root / p).resolve()
            paths.append(p)
    else:
        if input_dir.exists():
            paths.extend(sorted(input_dir.rglob("scorecard.json")))
            paths.extend(sorted(input_dir.rglob("*.scorecard.json")))

    out: List[pathlib.Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p.resolve())
    return out


def clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


def normalize_density(value: float) -> float:
    if value <= 1.0:
        return clamp(value)
    return clamp(value / 100.0)


def latency_score(ms: float) -> float:
    return clamp(1.0 - min(max(ms, 0.0), 1000.0) / 1000.0)


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 64)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def artifact_id(task: str, artifact_path: str, artifact_type: str) -> str:
    seed = f"{task}|{artifact_type}|{artifact_path}"
    return "tart_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def insert_task_artifact(path: pathlib.Path, artifact_type: str, metadata: Dict[str, Any]) -> None:
    if not path.exists():
        return
    ts = now_iso()
    rel_path = rel(path)
    sha = hash_file(path)
    aid = artifact_id(task_id, rel_path, artifact_type)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
        (
            aid,
            task_id,
            artifact_type,
            rel_path,
            sha,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ts,
        ),
    )
    con.commit()
    con.close()


def emit_event(key: str, severity: str, summary: str, evidence_ref: str, fingerprint_input: str) -> Dict[str, Any]:
    if not emit_events:
        return {"enabled": False, "reason": "emit_events_disabled"}
    if not event_router_script.exists() or not event_router_script.is_file():
        return {"enabled": True, "ok": False, "error": "event_router_missing", "path": str(event_router_script)}

    cmd = [
        str(event_router_script),
        "--source",
        "architecture.competitive_parity",
        "--key",
        key,
        "--severity",
        severity,
        "--summary",
        summary,
        "--evidence-ref",
        evidence_ref,
        "--cooldown-sec",
        str(event_cooldown_sec),
        "--fingerprint-input",
        fingerprint_input,
    ]
    res = run_json_cmd(cmd, timeout=30)
    return {
        "enabled": True,
        "ok": res.get("returncode") in (0, 20),
        "emitted": res.get("returncode") == 0,
        "suppressed": res.get("returncode") == 20,
        "returncode": res.get("returncode"),
        "payload": res.get("payload") if isinstance(res.get("payload"), dict) else {},
        "stderr": str(res.get("stderr") or "")[:240],
    }


def load_previous_dashboard(latest_path: pathlib.Path) -> Dict[str, Any]:
    if not latest_path.exists():
        return {}
    try:
        data = json.loads(latest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def scorecard_for_competitors(paths: List[pathlib.Path], baseline_competitors: List[str]) -> Tuple[Dict[str, pathlib.Path], List[Dict[str, Any]]]:
    parse_errors: List[Dict[str, Any]] = []
    mapping: Dict[str, pathlib.Path] = {}

    for p in paths:
        if not p.exists():
            parse_errors.append({"type": "missing_scorecard_file", "path": rel(p)})
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append({"type": "invalid_json", "path": rel(p), "error": str(exc)})
            continue
        competitor = str(obj.get("competitor") or "").strip().lower()
        if competitor not in baseline_competitors:
            parse_errors.append(
                {
                    "type": "unknown_competitor",
                    "path": rel(p),
                    "competitor": competitor or None,
                    "allowed": baseline_competitors,
                }
            )
            continue
        mapping[competitor] = p

    return mapping, parse_errors


run_id = run_id_arg or ("parity_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
run_dir = output_root / "runs" / run_id
dashboard_dir = output_root / "dashboard"
dashboard_latest_path = dashboard_dir / "latest.json"
dashboard_history_jsonl = dashboard_dir / "history.jsonl"
dashboard_history_run = dashboard_dir / "history" / f"{run_id}.json"
run_summary_path = run_dir / "scorecard_summary.json"

claimed = False
transition_done = False
result: Dict[str, Any] = {
    "ok": False,
    "run_id": run_id,
    "task_id": task_id,
    "agent": agent,
    "input_dir": rel(input_dir),
    "output_root": rel(output_root),
    "queue_claim": None,
}

try:
    if not harness_path.exists():
        raise RuntimeError(f"missing_harness:{harness_path}")
    if not schema_path.exists():
        raise RuntimeError(f"missing_schema:{schema_path}")
    if not queue_arb_script.exists():
        raise RuntimeError(f"missing_queue_arbitrator:{queue_arb_script}")

    harness_obj = yaml.safe_load(harness_path.read_text(encoding="utf-8"))
    if not isinstance(harness_obj, dict):
        raise RuntimeError("invalid_harness_yaml")
    schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))

    baseline_competitors = [
        str(x).strip().lower()
        for x in (((harness_obj.get("competitors") or {}).get("baseline_set") or []))
        if str(x).strip()
    ]
    if not baseline_competitors:
        raise RuntimeError("missing_baseline_competitors")

    required_artifacts = [
        str(x).strip()
        for x in (((harness_obj.get("teardown_contract") or {}).get("required_artifacts") or []))
        if str(x).strip()
    ]
    if not required_artifacts:
        required_artifacts = ["scorecard.json", "teardown.md", "capture_manifest.json"]

    upsert_queue_task()

    reopen_res = reopen_queue_for_claim()
    result["queue_reopen"] = reopen_res
    if reopen_res.get("ok") is False:
        raise RuntimeError("queue_reopen_failed")

    claim_res = queue_claim()
    result["queue_claim"] = {
        "returncode": claim_res.get("returncode"),
        "payload": claim_res.get("payload") if isinstance(claim_res.get("payload"), dict) else {},
    }

    if not claim_res.get("ok"):
        payload = claim_res.get("payload") if isinstance(claim_res.get("payload"), dict) else {}
        skipped = payload.get("skipped") if isinstance(payload.get("skipped"), list) else []
        skip_reason = "no_claimable_task"
        if isinstance(payload.get("error"), str) and payload.get("error"):
            skip_reason = str(payload.get("error"))
        if skipped and isinstance(skipped[0], dict) and skipped[0].get("reason"):
            skip_reason = str(skipped[0].get("reason"))
        result.update(
            {
                "ok": True,
                "changed": False,
                "claim_deferred": True,
                "reason": skip_reason,
                "status": "SKIPPED",
            }
        )
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0)

    claimed = True

    ensure_dir(run_dir)
    ensure_dir(dashboard_dir)
    ensure_dir(dashboard_dir / "history")

    scorecard_paths = discover_scorecards()
    mapping, parse_errors = scorecard_for_competitors(scorecard_paths, baseline_competitors)

    blockers: List[Dict[str, Any]] = []
    blockers.extend(parse_errors)

    previous_dashboard = load_previous_dashboard(dashboard_latest_path)
    prev_comp = {}
    for row in (previous_dashboard.get("competitors") or []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("competitor") or "").strip().lower()
        if not key:
            continue
        prev_comp[key] = row

    competitor_rows = []
    regressions = []
    copied_scorecards = []

    for competitor in baseline_competitors:
        path = mapping.get(competitor)
        if path is None:
            blockers.append({"type": "missing_scorecard", "competitor": competitor})
            continue

        raw_obj = json.loads(path.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(instance=raw_obj, schema=schema_obj)
        except Exception as exc:
            blockers.append({"type": "schema_validation_failed", "competitor": competitor, "path": rel(path), "error": str(exc)[:300]})

        bundle_dir = path.parent
        bundle_paths: Dict[str, str] = {}
        missing_bundle: List[str] = []
        for artifact_name in required_artifacts:
            candidate = path if artifact_name == path.name else (bundle_dir / artifact_name)
            if candidate.exists():
                bundle_paths[artifact_name] = rel(candidate)
            else:
                missing_bundle.append(artifact_name)

        if missing_bundle:
            blockers.append(
                {
                    "type": "missing_artifact_bundle",
                    "competitor": competitor,
                    "missing": missing_bundle,
                    "scorecard": rel(path),
                }
            )

        comp_dir = run_dir / competitor
        ensure_dir(comp_dir)
        out_scorecard = comp_dir / "scorecard.json"
        shutil.copy2(path, out_scorecard)
        copied_scorecards.append(rel(out_scorecard))

        for artifact_name in ("teardown.md", "capture_manifest.json"):
            src = bundle_dir / artifact_name
            if src.exists():
                shutil.copy2(src, comp_dir / artifact_name)

        metrics = raw_obj.get("metrics") if isinstance(raw_obj.get("metrics"), dict) else {}
        info_density_raw = float(metrics.get("information_density") or 0.0)
        latency_ms = float(metrics.get("interaction_latency_ms") or 0.0)
        keyboard_raw = float(metrics.get("keyboard_operability") or 0.0)
        modularity_raw = float(metrics.get("component_modularity") or 0.0)
        trust_raw = float(metrics.get("trust_signaling") or 0.0)

        info_norm = normalize_density(info_density_raw)
        latency_norm = latency_score(latency_ms)
        keyboard_norm = clamp(keyboard_raw)
        modularity_norm = clamp(modularity_raw)
        trust_norm = clamp(trust_raw)

        parity_score_percent = round(((info_norm + latency_norm + keyboard_norm + modularity_norm + trust_norm) / 5.0) * 100.0, 2)
        density_score = round(info_norm * 100.0, 2)
        component_coverage_ratio = round(modularity_norm, 4)

        prev_row = prev_comp.get(competitor) if isinstance(prev_comp, dict) else None
        prev_parity = None
        prev_latency_ms = None
        if isinstance(prev_row, dict):
            try:
                prev_parity = float(prev_row.get("parity_score_percent"))
            except Exception:
                prev_parity = None
            try:
                prev_latency_ms = float((prev_row.get("raw_metrics") or {}).get("interaction_latency_ms"))
            except Exception:
                prev_latency_ms = None

        delta_pp = None
        if prev_parity is not None:
            delta_pp = round(parity_score_percent - prev_parity, 2)

        latency_delta_ms = None
        if prev_latency_ms is not None:
            latency_delta_ms = round(latency_ms - prev_latency_ms, 2)

        is_regression = bool(delta_pp is not None and delta_pp < -10.0)
        if is_regression:
            regressions.append(
                {
                    "competitor": competitor,
                    "delta_pp": delta_pp,
                    "current_parity_score_percent": parity_score_percent,
                    "previous_parity_score_percent": prev_parity,
                }
            )

        acceptance_rows = raw_obj.get("acceptance_tests") if isinstance(raw_obj.get("acceptance_tests"), list) else []
        test_counts = {"pass": 0, "fail": 0, "na": 0}
        for row in acceptance_rows:
            if not isinstance(row, dict):
                continue
            st = str(row.get("status") or "na").strip().lower()
            if st not in test_counts:
                st = "na"
            test_counts[st] += 1

        competitor_rows.append(
            {
                "competitor": competitor,
                "parity_score_percent": parity_score_percent,
                "density_score": density_score,
                "latency_delta_ms": latency_delta_ms,
                "component_coverage_ratio": component_coverage_ratio,
                "raw_metrics": {
                    "information_density": info_density_raw,
                    "interaction_latency_ms": latency_ms,
                    "keyboard_operability": keyboard_norm,
                    "component_modularity": modularity_norm,
                    "trust_signaling": trust_norm,
                },
                "acceptance_tests": {
                    "total": int(sum(test_counts.values())),
                    **test_counts,
                },
                "regression": {
                    "delta_pp": delta_pp,
                    "is_regression": is_regression,
                },
                "artifacts": {
                    "scorecard": rel(out_scorecard),
                    "teardown": rel(comp_dir / "teardown.md") if (comp_dir / "teardown.md").exists() else None,
                    "capture_manifest": rel(comp_dir / "capture_manifest.json") if (comp_dir / "capture_manifest.json").exists() else None,
                    "input_bundle": bundle_paths,
                },
            }
        )

    if regressions:
        blockers.extend({"type": "score_regression_gt_10pp", **r} for r in regressions)

    status = "BLOCKED" if blockers else "OK"

    dashboard = {
        "schema_version": "competitive.parity.dashboard.v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "task_id": task_id,
        "harness_id": harness_obj.get("id"),
        "harness_version": harness_obj.get("version"),
        "cadence": ((harness_obj.get("parity_dashboard") or {}).get("publish_cadence")),
        "required_metrics": (harness_obj.get("parity_dashboard") or {}).get("required_metrics") or [],
        "fail_conditions": (harness_obj.get("parity_dashboard") or {}).get("fail_conditions") or [],
        "status": status,
        "summary": {
            "competitor_count": len(competitor_rows),
            "expected_competitors": baseline_competitors,
            "regression_count": len(regressions),
            "blocker_count": len(blockers),
        },
        "competitors": competitor_rows,
        "regressions": regressions,
        "blockers": blockers,
        "input_refs": [rel(p) for p in scorecard_paths],
        "artifacts": {
            "run_dir": rel(run_dir),
            "scorecard_summary": rel(run_summary_path),
            "dashboard_latest": rel(dashboard_latest_path),
            "dashboard_history": rel(dashboard_history_run),
        },
    }

    atomic_write(run_summary_path, json.dumps(dashboard, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write(dashboard_latest_path, json.dumps(dashboard, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write(dashboard_history_run, json.dumps(dashboard, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    with dashboard_history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dashboard, ensure_ascii=False, sort_keys=True) + "\n")

    insert_task_artifact(run_summary_path, "parity_dashboard", {"run_id": run_id, "status": status})
    insert_task_artifact(dashboard_latest_path, "parity_dashboard_latest", {"run_id": run_id, "status": status})
    for rel_score in copied_scorecards:
        insert_task_artifact(root / rel_score, "parity_scorecard", {"run_id": run_id})

    evidence_ref = " | ".join([rel(run_summary_path), rel(dashboard_latest_path)])

    if blockers:
        transition = queue_transition(
            to_status="BLOCKED",
            reason="competitive_parity_blocked",
            evidence_ref=evidence_ref,
            actor_role="validator",
            release_locks=True,
        )
        transition_done = True
        event_info = emit_event(
            key="harness_blocked",
            severity="critical",
            summary=f"competitive parity blocked run_id={run_id} blockers={len(blockers)} regressions={len(regressions)}",
            evidence_ref=rel(dashboard_latest_path),
            fingerprint_input=(
                "run=competitive_parity|"
                + "|".join(sorted({json.dumps(b, ensure_ascii=False, sort_keys=True)[:200] for b in blockers}))
            ),
        )
        result.update(
            {
                "ok": False,
                "changed": True,
                "status": "BLOCKED",
                "dashboard": dashboard,
                "queue_transition": {
                    "returncode": transition.get("returncode"),
                    "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
                },
                "event": event_info,
            }
        )
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"BLOCKER: competitive parity harness blocked; run_id={run_id}; blockers={len(blockers)}; regressions={len(regressions)}")
        raise SystemExit(2)

    transition = queue_transition(
        to_status="DONE",
        reason="competitive_parity_run_completed",
        evidence_ref=evidence_ref,
        actor_role="validator",
        release_locks=True,
    )
    transition_done = True

    result.update(
        {
            "ok": True,
            "changed": True,
            "status": "DONE",
            "dashboard": dashboard,
            "queue_transition": {
                "returncode": transition.get("returncode"),
                "payload": transition.get("payload") if isinstance(transition.get("payload"), dict) else {},
            },
        }
    )

    if json_out:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0)

except SystemExit:
    raise
except Exception as exc:
    err_text = str(exc)
    result.update({"ok": False, "error": err_text, "status": "FAILED"})

    if claimed and not transition_done:
        try:
            evidence_ref = rel(run_summary_path) if run_summary_path.exists() else rel(output_root)
            fail_transition = queue_transition(
                to_status="BLOCKED",
                reason="competitive_parity_runner_exception",
                evidence_ref=evidence_ref,
                actor_role="sre_watchdog",
                release_locks=True,
            )
            result["queue_transition"] = {
                "returncode": fail_transition.get("returncode"),
                "payload": fail_transition.get("payload") if isinstance(fail_transition.get("payload"), dict) else {},
            }
        except Exception as transition_exc:
            result["queue_transition_error"] = str(transition_exc)

    if json_out:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"BLOCKER: competitive parity harness failed; error={err_text}")
    raise SystemExit(1)
PY
