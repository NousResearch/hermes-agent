#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
WRAPPER_SCRIPT="$ROOT/ops/openclaw/run_web_capture_macro.sh"
MACROS_DIR="$ROOT/ops/web_capture/macros"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
STATE_PATH="$ROOT/state/continuity/latest/web_capture_scheduler_state.json"
LOCK_PATH="$ROOT/state/continuity/locks/web_capture_scheduler.lock"
SCHEDULER_SCHEMA_PATH="$ROOT/ops/openclaw/architecture/schemas/web_capture_scheduler_state.schema.json"

MODE="auto"
DRY_RUN=0
JSON_OUT=0
FORCE=0
KEEP_TAB=0
NO_QUEUE=0
MIN_INTERVAL_SEC=""
declare -a MACRO_PATHS=()

usage() {
  cat <<'EOF'
Usage: run_web_capture_scheduler.sh [options]

Deterministic multi-domain fairness scheduler for web-capture macros.
Selects one eligible macro per invocation using domain+macro round-robin,
then executes ops/openclaw/run_web_capture_macro.sh for the selected macro.

Options:
  --macro <path>            Explicit macro path (repeatable). If omitted, auto-discovers *.yaml in macros dir.
  --macros-dir <path>       Macro discovery directory (default: ops/web_capture/macros)
  --mode <auto|fetch|browser>
  --dry-run                 Evaluate fairness + eligibility only (do not execute selected macro)
  --json                    Print machine JSON output
  --force                   Pass --force to wrapper (bypass cadence/domain precheck)
  --keep-tab                Pass --keep-tab to wrapper
  --min-interval-sec <n>    Pass cadence override to wrapper dry-run/execute
  --db <path>               Continuity sqlite path passed to wrapper
  --no-queue                Pass --no-queue to wrapper
  --state <path>            Scheduler state output path
  --lock <path>             Scheduler lock path
  --schema <path>           Scheduler state contract schema path
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --macro)
      MACRO_PATHS+=("${2:-}"); shift 2 ;;
    --macros-dir)
      MACROS_DIR="${2:-}"; shift 2 ;;
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    --force)
      FORCE=1; shift ;;
    --keep-tab)
      KEEP_TAB=1; shift ;;
    --min-interval-sec)
      MIN_INTERVAL_SEC="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --no-queue)
      NO_QUEUE=1; shift ;;
    --state)
      STATE_PATH="${2:-}"; shift 2 ;;
    --lock)
      LOCK_PATH="${2:-}"; shift 2 ;;
    --schema)
      SCHEDULER_SCHEMA_PATH="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$MODE" in
  auto|fetch|browser) ;;
  *)
    echo "invalid --mode: $MODE (expected auto|fetch|browser)" >&2
    exit 2 ;;
esac

if [[ -n "$MIN_INTERVAL_SEC" ]] && ! [[ "$MIN_INTERVAL_SEC" =~ ^[0-9]+$ ]]; then
  echo "invalid --min-interval-sec: $MIN_INTERVAL_SEC (expected integer >= 0)" >&2
  exit 2
fi

if [[ ! -x "$WRAPPER_SCRIPT" ]]; then
  echo "missing executable wrapper script: $WRAPPER_SCRIPT" >&2
  exit 1
fi

if [[ "${#MACRO_PATHS[@]}" -eq 0 ]]; then
  if [[ -d "$MACROS_DIR" ]]; then
    while IFS= read -r -d '' p; do
      MACRO_PATHS+=("$p")
    done < <(find "$MACROS_DIR" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z)
  fi
fi

if [[ "${#MACRO_PATHS[@]}" -eq 0 ]]; then
  if [[ "$JSON_OUT" -eq 1 ]]; then
    printf '{"ok":false,"error":"no_macros_found","macros_dir":"%s"}\n' "$MACROS_DIR"
  else
    printf 'WEB_CAPTURE_SCHEDULER: no macros found (%s)\n' "$MACROS_DIR"
  fi
  exit 1
fi

macro_blob="$(printf '%s\n' "${MACRO_PATHS[@]}")"

python3 - "$ROOT" "$WRAPPER_SCRIPT" "$MODE" "$DRY_RUN" "$JSON_OUT" "$FORCE" "$KEEP_TAB" "$MIN_INTERVAL_SEC" "$DB_PATH" "$NO_QUEUE" "$STATE_PATH" "$LOCK_PATH" "$SCHEDULER_SCHEMA_PATH" "$macro_blob" <<'PY'
import collections
import datetime as dt
import fcntl
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
wrapper_script = pathlib.Path(sys.argv[2]).resolve()
mode = str(sys.argv[3])
dry_run = bool(int(sys.argv[4]))
json_out = bool(int(sys.argv[5]))
force = bool(int(sys.argv[6]))
keep_tab = bool(int(sys.argv[7]))
min_interval_sec = str(sys.argv[8] or "").strip()
db_path = str(sys.argv[9] or "").strip()
no_queue = bool(int(sys.argv[10]))
state_path = pathlib.Path(sys.argv[11]).resolve()
lock_path = pathlib.Path(sys.argv[12]).resolve()
scheduler_schema_path = pathlib.Path(sys.argv[13]).resolve()
macro_blob = sys.argv[14] if len(sys.argv) > 14 else ""

try:
    import yaml
except Exception as exc:
    out = {"ok": False, "error": f"pyyaml_missing:{exc}"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(1)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def now_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


SCHEDULER_SCHEMA_VERSION = "openclaw.web_capture.scheduler_state.v1"


def safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "web_capture"


def macro_slug_from_path(path: pathlib.Path) -> str:
    name = path.name
    if name.endswith(".yaml"):
        name = name[:-5]
    elif name.endswith(".yml"):
        name = name[:-4]
    return safe_slug(name)


def maybe_rel(path_like: Any) -> str:
    p = pathlib.Path(str(path_like)).resolve()
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    txt = str(raw or "").strip().lower()
    if txt in {"1", "true", "yes", "y"}:
        return True
    if txt in {"0", "false", "no", "n"}:
        return False
    return default


def parse_iso(value: Any):
    txt = str(value or "").strip()
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


def validate_scheduler_state_obj(obj: Dict[str, Any], *, require_contract: bool = False) -> List[str]:
    errors: List[str] = []

    if not isinstance(obj, dict):
        return ["state_not_object"]

    if str(obj.get("schema_version") or "") != SCHEDULER_SCHEMA_VERSION:
        errors.append("schema_version_invalid")

    if parse_iso(obj.get("updated_at")) is None:
        errors.append("updated_at_invalid")

    status = str(obj.get("selection_status") or "")
    if status not in {"idle_no_eligible_macro", "selected_dry_run", "executed"}:
        errors.append("selection_status_invalid")

    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else None
    if not summary:
        errors.append("summary_missing")
    else:
        for key in ("total_macros", "valid_macros", "eligible_macros", "blocked_or_skipped_macros"):
            raw = summary.get(key)
            if not isinstance(raw, int) or raw < 0:
                errors.append(f"summary_{key}_invalid")

        total_macros = summary.get("total_macros") if isinstance(summary.get("total_macros"), int) else None
        valid_macros = summary.get("valid_macros") if isinstance(summary.get("valid_macros"), int) else None
        eligible_macros = summary.get("eligible_macros") if isinstance(summary.get("eligible_macros"), int) else None
        blocked_macros = summary.get("blocked_or_skipped_macros") if isinstance(summary.get("blocked_or_skipped_macros"), int) else None

        if total_macros is not None and valid_macros is not None and valid_macros > total_macros:
            errors.append("summary_valid_gt_total")
        if valid_macros is not None and eligible_macros is not None and eligible_macros > valid_macros:
            errors.append("summary_eligible_gt_valid")
        if total_macros is not None and eligible_macros is not None and blocked_macros is not None:
            if blocked_macros != max(0, total_macros - eligible_macros):
                errors.append("summary_blocked_count_mismatch")

    macros = obj.get("macros") if isinstance(obj.get("macros"), list) else None
    if macros is None:
        errors.append("macros_missing")
        macro_count = None
    else:
        macro_count = len(macros)
        if isinstance(summary, dict) and isinstance(summary.get("total_macros"), int) and macro_count != int(summary.get("total_macros")):
            errors.append("macros_total_mismatch")

    domain_cursor = obj.get("domain_cursor") if isinstance(obj.get("domain_cursor"), dict) else None
    if domain_cursor is None:
        errors.append("domain_cursor_missing")

    contract = obj.get("contract") if isinstance(obj.get("contract"), dict) else None
    if require_contract:
        if contract is None:
            errors.append("contract_missing")
        else:
            if str(contract.get("schema_version") or "") != SCHEDULER_SCHEMA_VERSION:
                errors.append("contract_schema_version_invalid")
            if parse_iso(contract.get("validated_at")) is None:
                errors.append("contract_validated_at_invalid")
            if not isinstance(contract.get("state_valid"), bool):
                errors.append("contract_state_valid_invalid")
            if not isinstance(contract.get("validation_errors"), list):
                errors.append("contract_validation_errors_invalid")

    return sorted(set(errors))


def run_wrapper_dry(macro_path: pathlib.Path) -> Dict[str, Any]:
    cmd = ["bash", str(wrapper_script), "--macro", str(macro_path), "--mode", mode, "--dry-run", "--json"]
    if force:
        cmd.append("--force")
    if min_interval_sec:
        cmd.extend(["--min-interval-sec", min_interval_sec])
    if db_path:
        cmd.extend(["--db", db_path])
    if no_queue:
        cmd.append("--no-queue")

    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    stdout = (cp.stdout or "").strip()
    payload: Dict[str, Any] = {}
    parse_error = None
    if stdout:
        try:
            payload = json.loads(stdout)
            if not isinstance(payload, dict):
                payload = {}
        except Exception as exc:
            parse_error = str(exc)
    reason = "unknown"
    run_allowed = False
    if payload:
        run_allowed = bool(payload.get("run_allowed"))
        reason = str(payload.get("reason") or ("ready" if run_allowed else "unknown")).strip() or "unknown"
    elif parse_error:
        reason = "dry_run_output_unparseable"
    elif cp.returncode != 0:
        reason = f"dry_run_rc_{cp.returncode}"

    domain_guard = payload.get("domain_guard") if isinstance(payload.get("domain_guard"), dict) else {}

    return {
        "ok": cp.returncode == 0,
        "returncode": cp.returncode,
        "run_allowed": run_allowed,
        "reason": reason,
        "payload": payload,
        "domain_guard": {
            "reason": domain_guard.get("reason"),
            "cooldown_remaining_sec": domain_guard.get("cooldown_remaining_sec"),
            "operator_action_required": parse_bool(domain_guard.get("operator_action_required"), False),
            "operator_contract_json": domain_guard.get("operator_contract_json"),
        },
        "stdout_tail": stdout[-1000:] if stdout else "",
        "stderr_tail": (cp.stderr or "")[-1000:],
    }


def choose_domain(all_domains: List[str], eligible_domains: List[str], last_domain: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ordered_domains": list(all_domains),
        "last_domain": last_domain,
        "start_index": 0,
    }
    if not eligible_domains:
        meta["eligible_domains"] = []
        return None, meta

    eligible_set = set(eligible_domains)
    if not all_domains:
        all_domains = sorted(eligible_set)
        meta["ordered_domains"] = list(all_domains)

    start = 0
    if last_domain and last_domain in all_domains:
        start = (all_domains.index(last_domain) + 1) % len(all_domains)
    meta["start_index"] = start
    meta["eligible_domains"] = sorted(eligible_set)

    for off in range(len(all_domains)):
        dom = all_domains[(start + off) % len(all_domains)]
        if dom in eligible_set:
            meta["selected_index"] = (start + off) % len(all_domains)
            return dom, meta

    dom = sorted(eligible_set)[0]
    meta["selected_index"] = None
    return dom, meta


def choose_macro_for_domain(
    domain: str,
    domain_all_slugs: List[str],
    eligible_rows: List[Dict[str, Any]],
    last_macro_slug: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "domain": domain,
        "ordered_macros": list(domain_all_slugs),
        "last_macro_slug": last_macro_slug,
        "start_index": 0,
    }

    by_slug = {str(r.get("macro_slug")): r for r in eligible_rows if str(r.get("macro_slug"))}
    eligible_slugs = sorted(by_slug.keys())
    meta["eligible_macros"] = eligible_slugs
    if not eligible_slugs:
        return None, meta

    ordered = list(domain_all_slugs) if domain_all_slugs else list(eligible_slugs)
    if not ordered:
        ordered = list(eligible_slugs)

    start = 0
    if last_macro_slug and last_macro_slug in ordered:
        start = (ordered.index(last_macro_slug) + 1) % len(ordered)
    meta["start_index"] = start

    for off in range(len(ordered)):
        slug = ordered[(start + off) % len(ordered)]
        if slug in by_slug:
            meta["selected_index"] = (start + off) % len(ordered)
            return by_slug[slug], meta

    return by_slug[eligible_slugs[0]], meta


macro_paths: List[pathlib.Path] = []
for line in str(macro_blob or "").splitlines():
    p = pathlib.Path(line.strip())
    if not str(p):
        continue
    macro_paths.append(p.resolve())

if not macro_paths:
    out = {"ok": False, "error": "no_macros_found"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(1)

lock_path.parent.mkdir(parents=True, exist_ok=True)
state_path.parent.mkdir(parents=True, exist_ok=True)

if not scheduler_schema_path.exists():
    out = {
        "ok": False,
        "error": "scheduler_schema_missing",
        "schema_path": maybe_rel(scheduler_schema_path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) if json_out else f"WEB_CAPTURE_SCHEDULER: missing schema {maybe_rel(scheduler_schema_path)}")
    raise SystemExit(1)

with lock_path.open("a+") as lock_fh:
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

    previous_state_raw = load_json(state_path)
    previous_state_contract_errors = validate_scheduler_state_obj(previous_state_raw, require_contract=False)
    previous_state_valid = len(previous_state_contract_errors) == 0
    previous_state = previous_state_raw if previous_state_valid else {}
    now = now_iso()

    macro_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []

    for macro_path in macro_paths:
        slug = macro_slug_from_path(macro_path)
        row: Dict[str, Any] = {
            "macro_slug": slug,
            "macro_path": maybe_rel(macro_path),
            "exists": macro_path.exists(),
            "domain": slug,
            "target_url": "",
            "valid": False,
        }

        if not macro_path.exists():
            row["reason"] = "macro_missing"
            macro_rows.append(row)
            continue

        try:
            obj = yaml.safe_load(macro_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("macro_top_level_not_object")
            domain = str(obj.get("domain") or slug).strip().lower() or slug
            row["domain"] = domain
            row["target_url"] = str(obj.get("target_url") or "").strip()
            row["valid"] = True
        except Exception as exc:
            row["reason"] = f"macro_parse_failed:{exc}"
            macro_rows.append(row)
            continue

        eval_res = run_wrapper_dry(macro_path)
        row.update(
            {
                "dry_run_ok": eval_res.get("ok"),
                "dry_run_rc": eval_res.get("returncode"),
                "run_allowed": bool(eval_res.get("run_allowed")),
                "reason": str(eval_res.get("reason") or "unknown"),
                "cooldown_remaining_sec": eval_res.get("domain_guard", {}).get("cooldown_remaining_sec"),
                "operator_action_required": bool(eval_res.get("domain_guard", {}).get("operator_action_required")),
                "operator_contract_json": eval_res.get("domain_guard", {}).get("operator_contract_json"),
            }
        )
        macro_rows.append(row)
        valid_rows.append(row)

    # Deduplicate by macro_slug deterministically (first discovered wins).
    dedup_rows: List[Dict[str, Any]] = []
    seen_slugs = set()
    for row in macro_rows:
        slug = str(row.get("macro_slug") or "")
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        dedup_rows.append(row)
    macro_rows = dedup_rows
    valid_rows = [r for r in macro_rows if bool(r.get("valid"))]

    eligible_rows = [r for r in valid_rows if bool(r.get("run_allowed"))]

    all_domains = sorted({str(r.get("domain") or "") for r in valid_rows if str(r.get("domain") or "")})
    eligible_domains = sorted({str(r.get("domain") or "") for r in eligible_rows if str(r.get("domain") or "")})

    last_domain = str(previous_state.get("last_selected_domain") or "").strip() or None
    domain_cursor_obj = previous_state.get("domain_cursor") if isinstance(previous_state.get("domain_cursor"), dict) else {}

    selected_domain, domain_rr_meta = choose_domain(all_domains, eligible_domains, last_domain)
    selected_row: Optional[Dict[str, Any]] = None
    macro_rr_meta: Dict[str, Any] = {}

    if selected_domain:
        domain_all_slugs = sorted(
            [str(r.get("macro_slug")) for r in valid_rows if str(r.get("domain") or "") == selected_domain and str(r.get("macro_slug") or "")]
        )
        domain_eligible_rows = [r for r in eligible_rows if str(r.get("domain") or "") == selected_domain]
        last_macro = None
        if isinstance(domain_cursor_obj.get(selected_domain), dict):
            last_macro = str((domain_cursor_obj.get(selected_domain) or {}).get("last_macro_slug") or "").strip() or None
        selected_row, macro_rr_meta = choose_macro_for_domain(selected_domain, domain_all_slugs, domain_eligible_rows, last_macro)

    execution: Dict[str, Any] = {
        "executed": False,
        "wrapper_exit_code": None,
        "wrapper_status": None,
        "wrapper_reason": None,
        "wrapper_command": None,
    }

    if selected_row and not dry_run:
        cmd = ["bash", str(wrapper_script), "--macro", str(root / str(selected_row.get("macro_path"))), "--mode", mode]
        if force:
            cmd.append("--force")
        if keep_tab:
            cmd.append("--keep-tab")
        if min_interval_sec:
            cmd.extend(["--min-interval-sec", min_interval_sec])
        if db_path:
            cmd.extend(["--db", db_path])
        if no_queue:
            cmd.append("--no-queue")

        cp = subprocess.run(cmd, text=True, capture_output=True, check=False)

        macro_slug = str(selected_row.get("macro_slug") or "")
        schedule_path = root / "state" / "cron_watchdog" / f"web_capture_{macro_slug}_schedule_state.json"
        schedule_obj = load_json(schedule_path)

        execution.update(
            {
                "executed": True,
                "wrapper_exit_code": int(cp.returncode),
                "wrapper_status": schedule_obj.get("last_status"),
                "wrapper_reason": schedule_obj.get("last_exit_code"),
                "wrapper_command": " ".join(cmd),
                "stdout_tail": (cp.stdout or "")[-2000:],
                "stderr_tail": (cp.stderr or "")[-2000:],
                "schedule_state_path": maybe_rel(schedule_path),
                "schedule_state": schedule_obj,
            }
        )

    reason_counts = collections.Counter()
    for row in macro_rows:
        reason_counts[str(row.get("reason") or "unknown")] += 1

    domain_rollup: List[Dict[str, Any]] = []
    for dom in all_domains:
        dom_rows = [r for r in macro_rows if str(r.get("domain") or "") == dom]
        dom_eligible = [r for r in dom_rows if bool(r.get("run_allowed"))]
        dom_operator = [r for r in dom_rows if bool(r.get("operator_action_required"))]
        cooldown_active = 0
        for r in dom_rows:
            try:
                rem = int(r.get("cooldown_remaining_sec") or 0)
            except Exception:
                rem = 0
            if rem > 0:
                cooldown_active += 1
        domain_rollup.append(
            {
                "domain": dom,
                "macro_count": len(dom_rows),
                "eligible_macro_count": len(dom_eligible),
                "operator_action_required_count": len(dom_operator),
                "cooldown_active_macro_count": cooldown_active,
                "macros": sorted([str(r.get("macro_slug")) for r in dom_rows if str(r.get("macro_slug") or "")]),
            }
        )

    selection_payload = None
    if selected_row:
        selection_payload = {
            "domain": selected_row.get("domain"),
            "macro_slug": selected_row.get("macro_slug"),
            "macro_path": selected_row.get("macro_path"),
            "reason": selected_row.get("reason"),
            "fairness": {
                "domain_rr": domain_rr_meta,
                "macro_rr": macro_rr_meta,
            },
        }

    status = "idle_no_eligible_macro"
    if selection_payload and dry_run:
        status = "selected_dry_run"
    elif selection_payload and execution.get("executed"):
        status = "executed"

    published_at = now_iso()

    new_state: Dict[str, Any] = {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "updated_at": published_at,
        "mode": mode,
        "dry_run": dry_run,
        "force": force,
        "last_selected_domain": selection_payload.get("domain") if selection_payload else previous_state.get("last_selected_domain"),
        "last_selected_macro_slug": selection_payload.get("macro_slug") if selection_payload else previous_state.get("last_selected_macro_slug"),
        "last_selected_at": now if selection_payload else previous_state.get("last_selected_at"),
        "selection_status": status,
        "selection": selection_payload,
        "execution": execution,
        "summary": {
            "total_macros": len(macro_rows),
            "valid_macros": len(valid_rows),
            "eligible_macros": len(eligible_rows),
            "blocked_or_skipped_macros": len(macro_rows) - len(eligible_rows),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "domains": domain_rollup,
        "macros": sorted(macro_rows, key=lambda x: (str(x.get("domain") or ""), str(x.get("macro_slug") or ""))),
        "state_path": maybe_rel(state_path),
    }

    domain_cursor = domain_cursor_obj if isinstance(domain_cursor_obj, dict) else {}
    if selection_payload:
        sel_domain = str(selection_payload.get("domain") or "").strip()
        sel_macro = str(selection_payload.get("macro_slug") or "").strip()
        if sel_domain and sel_macro:
            domain_cursor = dict(domain_cursor)
            domain_cursor[sel_domain] = {
                "last_macro_slug": sel_macro,
                "updated_at": now,
            }
    new_state["domain_cursor"] = domain_cursor
    new_state["contract"] = {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "schema_path": maybe_rel(scheduler_schema_path),
        "validated_at": published_at,
        "state_valid": True,
        "validation_errors": [],
        "previous_state_valid": previous_state_valid,
        "previous_state_errors": previous_state_contract_errors[:8],
    }

    state_contract_errors = validate_scheduler_state_obj(new_state, require_contract=True)
    new_state["contract"]["state_valid"] = len(state_contract_errors) == 0
    new_state["contract"]["validation_errors"] = state_contract_errors

    atomic_write(state_path, new_state)

    out = {
        "ok": len(state_contract_errors) == 0,
        "status": status,
        "selection": selection_payload,
        "execution": execution,
        "summary": new_state.get("summary"),
        "state_path": maybe_rel(state_path),
        "contract": new_state.get("contract"),
    }

    if state_contract_errors:
        out["error"] = "scheduler_state_contract_invalid"
        if json_out:
            print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "WEB_CAPTURE_SCHEDULER: contract_invalid "
                f"errors={','.join(state_contract_errors)} state={maybe_rel(state_path)}"
            )
        raise SystemExit(1)

    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if status == "idle_no_eligible_macro":
            print(
                "WEB_CAPTURE_SCHEDULER: idle "
                f"eligible=0/{len(macro_rows)} "
                f"reasons={dict(sorted(reason_counts.items()))}"
            )
        elif status == "selected_dry_run":
            print(
                "WEB_CAPTURE_SCHEDULER: ready "
                f"domain={selection_payload.get('domain')} macro={selection_payload.get('macro_slug')}"
            )
        else:
            print(
                "WEB_CAPTURE_SCHEDULER: executed "
                f"domain={selection_payload.get('domain')} macro={selection_payload.get('macro_slug')} "
                f"wrapper_status={execution.get('wrapper_status') or 'unknown'}"
            )
PY
