#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"

usage() {
  cat <<'EOF'
Usage: librarian_curator.sh <command> [options]

Librarian/Curator runtime scaffold for curation, promotion, and retrieval hygiene.

Commands:
  ingest       Build deterministic curation queue from reports/docs/memory surfaces
  lint         Lint queue/promotions integrity + supersession hygiene
  promote      Promote one candidate/path into canonical ledger
  build-index  Rebuild canonical index from promotions ledger
  hygiene      Run Obsidian governance checks (schema + secrets + retrieval smoke)

Common options:
  --json
  --action-token <value>  Canonical mutation token for direct mutating entrypoint use
  --truth-anchor <value>  Legacy alias of --action-token
  --mutation-ticket <v>   Authority ticket JSON string, @path, or path (high-risk token path)
  --attestation <name>    Satisfied authority attestation (repeatable)
  --attestation-object <v>
                         Structured attestation JSON string, @path, or path (repeatable)
  --allow-legacy-anchor   Allow legacy anchor-only token mode for direct token validation

ingest options:
  --since-hours <n>   Include files modified within n hours (default: 168)
  --limit <n>         Max candidates (default: 300)

lint options:
  --strict            Exit non-zero on warnings/failures

promote options:
  --candidate-id <id>
  --source-path <path>
  --reason <text>     Required
  --operator <name>   Required

hygiene options:
  --strict            Exit non-zero when hard checks fail
  --skip-retrieval-eval
                      Skip retrieval_eval.py (faster health pass)

EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

CMD="$1"
shift || true

JSON_OUT=0
SINCE_HOURS=168
LIMIT=300
STRICT=0
SKIP_RETRIEVAL_EVAL=0
CANDIDATE_ID=""
SOURCE_PATH=""
REASON=""
OPERATOR=""
ACTION_TOKEN=""
MUTATION_TICKET=""
ATTESTATIONS=()
ATTESTATION_OBJECTS=()
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    --since-hours)
      SINCE_HOURS="${2:-}"; shift 2 ;;
    --limit)
      LIMIT="${2:-}"; shift 2 ;;
    --strict)
      STRICT=1; shift ;;
    --skip-retrieval-eval)
      SKIP_RETRIEVAL_EVAL=1; shift ;;
    --candidate-id)
      CANDIDATE_ID="${2:-}"; shift 2 ;;
    --source-path)
      SOURCE_PATH="${2:-}"; shift 2 ;;
    --reason)
      REASON="${2:-}"; shift 2 ;;
    --operator)
      OPERATOR="${2:-}"; shift 2 ;;
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

is_mutating=0
case "$CMD" in
  ingest|promote|build-index)
    is_mutating=1 ;;
  *)
    is_mutating=0 ;;
esac

if [[ "$is_mutating" == "1" ]]; then
  guard_args=(
    --script "librarian_curator.sh"
    --risk-tier "high"
    --mutation-operation "librarian_curator:${CMD}"
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
fi

python3 - "$ROOT" "$CMD" "$JSON_OUT" "$SINCE_HOURS" "$LIMIT" "$STRICT" "$SKIP_RETRIEVAL_EVAL" "$CANDIDATE_ID" "$SOURCE_PATH" "$REASON" "$OPERATOR" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

root = pathlib.Path(sys.argv[1]).resolve()
cmd = str(sys.argv[2] or "").strip()
json_out = bool(int(sys.argv[3]))
since_hours = max(1, int(sys.argv[4] or 168))
limit = max(1, min(2000, int(sys.argv[5] or 300)))
strict = bool(int(sys.argv[6]))
skip_retrieval_eval = bool(int(sys.argv[7]))
candidate_id_arg = str(sys.argv[8] or "").strip()
source_path_arg = str(sys.argv[9] or "").strip()
reason_arg = str(sys.argv[10] or "").strip()
operator_arg = str(sys.argv[11] or "").strip()

state_dir = root / "state" / "continuity" / "librarian"
queue_path = state_dir / "curation_queue.json"
promotions_path = state_dir / "promotions.jsonl"
index_path = state_dir / "canonical_index.json"
index_md_path = state_dir / "canonical_index.md"
hygiene_path = state_dir / "retrieval_hygiene.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: pathlib.Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: pathlib.Path) -> str:
    p = rel(path)
    if p.startswith("reports/"):
        return "report"
    if p.startswith("docs/ops/"):
        return "ops_doc"
    if p.startswith("memory/"):
        return "memory_note"
    return "misc"


def logical_key(path: str) -> str:
    base = pathlib.Path(path).name.lower()
    base = re.sub(r"\.[a-z0-9]+$", "", base)
    base = re.sub(r"_20\d{2}-\d{2}-\d{2}.*$", "", base)
    base = re.sub(r"_v\d+$", "", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "unknown"


def load_queue() -> Dict[str, Any]:
    if not queue_path.exists():
        return {"candidates": []}
    try:
        return json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return {"candidates": []}


def load_promotions() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not promotions_path.exists():
        return rows
    with promotions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    rows.sort(key=lambda r: (str(r.get("promoted_at") or ""), str(r.get("promotion_id") or "")))
    return rows


def build_index_payload(promotions: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for row in promotions:
        key = str(row.get("logical_key") or logical_key(str(row.get("source_path") or "")))
        by_key.setdefault(key, []).append(row)

    canonical_rows: List[Dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        history = sorted(by_key[key], key=lambda r: (str(r.get("promoted_at") or ""), str(r.get("promotion_id") or "")))
        latest = history[-1]
        canonical_rows.append(
            {
                "logical_key": key,
                "source_path": latest.get("source_path"),
                "sha256": latest.get("sha256"),
                "promoted_at": latest.get("promoted_at"),
                "operator": latest.get("operator"),
                "reason": latest.get("reason"),
                "promotion_id": latest.get("promotion_id"),
                "history_count": len(history),
                "superseded_promotions": [r.get("promotion_id") for r in history[:-1]],
            }
        )

    payload = {
        "schema": "clawd.librarian.canonical_index.v1",
        "generated_at": now_iso(),
        "workspace_id": "clawd-architect",
        "summary": {
            "promotion_events": len(promotions),
            "canonical_entries": len(canonical_rows),
        },
        "entries": canonical_rows,
    }
    return payload


def write_index_markdown(payload: Dict[str, Any]) -> None:
    lines = [
        "# Librarian Canonical Index",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- canonical_entries: {(payload.get('summary') or {}).get('canonical_entries')}",
        "",
        "## Entries",
        "",
    ]
    for row in payload.get("entries") or []:
        lines.extend(
            [
                f"- `{row.get('logical_key')}` -> `{row.get('source_path')}`",
                f"  - promoted_at: {row.get('promoted_at')} | operator: {row.get('operator')} | history_count: {row.get('history_count')}",
            ]
        )
    index_md_path.parent.mkdir(parents=True, exist_ok=True)
    index_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit(payload: Dict[str, Any], code: int = 0) -> None:
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = payload.get("status") or ("ok" if payload.get("ok", True) else "fail")
        print(f"LIBRARIAN {cmd}: {status}")
        summary = payload.get("summary")
        if isinstance(summary, dict):
            for k in sorted(summary.keys()):
                print(f"- {k}: {summary[k]}")
    raise SystemExit(code)


if cmd == "ingest":
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=since_hours)
    roots = [root / "reports", root / "docs" / "ops", root / "memory"]
    candidates: List[Dict[str, Any]] = []

    for base in roots:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            rel_path = rel(path)
            if rel_path.startswith("memory/pdf_"):
                continue
            if "/pdf_" in rel_path:
                continue
            st = path.stat()
            updated = dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
            if updated < cutoff:
                continue
            cand_seed = f"{rel_path}|{int(st.st_mtime)}|{st.st_size}"
            cand_id = "cur_" + hashlib.sha256(cand_seed.encode("utf-8")).hexdigest()[:18]
            candidates.append(
                {
                    "candidate_id": cand_id,
                    "source_path": rel_path,
                    "source_class": classify(path),
                    "logical_key": logical_key(rel_path),
                    "updated_at": updated.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "bytes": int(st.st_size),
                    "sha256": sha_file(path),
                }
            )

    candidates.sort(key=lambda r: (r["updated_at"], r["source_path"]), reverse=True)
    candidates = candidates[:limit]

    payload = {
        "schema": "clawd.librarian.curation_queue.v1",
        "generated_at": now_iso(),
        "workspace_id": "clawd-architect",
        "source_window_hours": since_hours,
        "summary": {
            "candidate_count": len(candidates),
            "limit": limit,
        },
        "candidates": candidates,
    }
    atomic_write(queue_path, payload)
    emit(payload)

if cmd == "build-index":
    promotions = load_promotions()
    payload = build_index_payload(promotions)
    atomic_write(index_path, payload)
    write_index_markdown(payload)
    emit(payload)

if cmd == "promote":
    if not reason_arg:
        emit({"ok": False, "status": "fail", "error": "missing_reason"}, 2)
    if not operator_arg:
        emit({"ok": False, "status": "fail", "error": "missing_operator"}, 2)
    if not candidate_id_arg and not source_path_arg:
        emit({"ok": False, "status": "fail", "error": "missing_candidate_or_source"}, 2)

    queue = load_queue()
    candidates = queue.get("candidates") or []

    target: Optional[Dict[str, Any]] = None
    if candidate_id_arg:
        for row in candidates:
            if str(row.get("candidate_id") or "") == candidate_id_arg:
                target = dict(row)
                break
    if target is None and source_path_arg:
        sp = str(source_path_arg).strip()
        p = (root / sp).resolve() if not pathlib.Path(sp).is_absolute() else pathlib.Path(sp).resolve()
        if not p.exists():
            emit({"ok": False, "status": "fail", "error": "source_path_not_found", "source_path": sp}, 2)
        rel_path = rel(p)
        target = {
            "candidate_id": None,
            "source_path": rel_path,
            "source_class": classify(p),
            "logical_key": logical_key(rel_path),
            "updated_at": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "bytes": int(p.stat().st_size),
            "sha256": sha_file(p),
        }

    if target is None:
        emit({"ok": False, "status": "fail", "error": "candidate_not_found", "candidate_id": candidate_id_arg}, 2)

    promotions = load_promotions()
    key = str(target.get("logical_key") or logical_key(str(target.get("source_path") or "")))
    same_key = [r for r in promotions if str(r.get("logical_key") or "") == key]
    supersedes = same_key[-1].get("promotion_id") if same_key else None

    promoted_at = now_iso()
    seed = f"{target.get('source_path')}|{target.get('sha256')}|{promoted_at}|{operator_arg}|{reason_arg}"
    promotion_id = "prm_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:18]

    row = {
        "promotion_id": promotion_id,
        "promoted_at": promoted_at,
        "operator": operator_arg,
        "reason": reason_arg,
        "candidate_id": target.get("candidate_id"),
        "source_path": target.get("source_path"),
        "source_class": target.get("source_class"),
        "logical_key": key,
        "sha256": target.get("sha256"),
        "supersedes_promotion_id": supersedes,
    }
    append_jsonl(promotions_path, row)

    all_promotions = load_promotions()
    index_payload = build_index_payload(all_promotions)
    atomic_write(index_path, index_payload)
    write_index_markdown(index_payload)

    emit(
        {
            "schema": "clawd.librarian.promotion.v1",
            "generated_at": promoted_at,
            "ok": True,
            "status": "ok",
            "promotion": row,
            "summary": {
                "promotion_events": (index_payload.get("summary") or {}).get("promotion_events"),
                "canonical_entries": (index_payload.get("summary") or {}).get("canonical_entries"),
            },
        }
    )

if cmd == "lint":
    queue = load_queue()
    promotions = load_promotions()
    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    candidate_ids = set()
    for row in queue.get("candidates") or []:
        cid = str(row.get("candidate_id") or "")
        if not cid:
            violations.append({"kind": "candidate_missing_id", "source_path": row.get("source_path")})
            continue
        if cid in candidate_ids:
            violations.append({"kind": "duplicate_candidate_id", "candidate_id": cid})
        candidate_ids.add(cid)
        sp = str(row.get("source_path") or "")
        if not sp or not (root / sp).exists():
            warnings.append({"kind": "candidate_source_missing", "candidate_id": cid, "source_path": sp})

    key_to_promos: Dict[str, List[Dict[str, Any]]] = {}
    for row in promotions:
        key = str(row.get("logical_key") or "")
        key_to_promos.setdefault(key, []).append(row)

    for key, rows in key_to_promos.items():
        rows = sorted(rows, key=lambda r: (str(r.get("promoted_at") or ""), str(r.get("promotion_id") or "")))
        seen = {str(r.get("promotion_id") or "") for r in rows}
        for r in rows[1:]:
            sup = str(r.get("supersedes_promotion_id") or "")
            if not sup:
                warnings.append({"kind": "missing_supersedes_link", "logical_key": key, "promotion_id": r.get("promotion_id")})
            elif sup not in seen:
                violations.append({"kind": "broken_supersedes_link", "logical_key": key, "promotion_id": r.get("promotion_id"), "supersedes": sup})

    status = "pass"
    if violations:
        status = "fail"
    elif warnings:
        status = "warn"

    payload = {
        "schema": "clawd.librarian.lint.v1",
        "generated_at": now_iso(),
        "workspace_id": "clawd-architect",
        "status": status,
        "ok": status != "fail",
        "summary": {
            "candidate_count": len(queue.get("candidates") or []),
            "promotion_events": len(promotions),
            "violations": len(violations),
            "warnings": len(warnings),
        },
        "violations": violations,
        "warnings": warnings,
    }

    if strict and status != "pass":
        emit(payload, 1)
    emit(payload)

if cmd == "hygiene":
    checks: List[Dict[str, Any]] = []

    def run_cmd(name: str, argv: List[str]) -> Dict[str, Any]:
        cp = subprocess.run(argv, text=True, capture_output=True, check=False)
        out = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()
        obj: Any = None
        if out:
            try:
                obj = json.loads(out)
            except Exception:
                obj = None
        return {
            "name": name,
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout_json": obj,
            "stdout": out[:600] if obj is None else None,
            "stderr": err[:400] or None,
        }

    checks.append(
        run_cmd(
            "vault_validate",
            ["python3", str(root / "ops" / "obsidian" / "vault_validate.py"), "--strict"],
        )
    )
    checks.append(
        run_cmd(
            "secret_scan",
            ["python3", str(root / "ops" / "obsidian" / "secret_scan.py"), "--json", "--warn-only"],
        )
    )
    if skip_retrieval_eval:
        checks.append(
            {
                "name": "retrieval_eval",
                "ok": True,
                "returncode": 0,
                "stdout_json": {"skipped": True, "reason": "skip_retrieval_eval"},
                "stdout": None,
                "stderr": None,
            }
        )
    else:
        checks.append(
            run_cmd(
                "retrieval_eval",
                ["python3", str(root / "ops" / "obsidian" / "retrieval_eval.py"), "--no-fail"],
            )
        )

    hard_fail = False
    warn_count = 0
    for c in checks:
        if c["name"] == "vault_validate" and not c["ok"]:
            hard_fail = True
        if c["name"] == "secret_scan":
            obj = c.get("stdout_json")
            if isinstance(obj, dict) and int(obj.get("count") or 0) > 0:
                warn_count += int(obj.get("count") or 0)

    status = "pass"
    if hard_fail:
        status = "fail"
    elif warn_count > 0:
        status = "warn"

    payload = {
        "schema": "clawd.librarian.hygiene.v1",
        "generated_at": now_iso(),
        "workspace_id": "clawd-architect",
        "status": status,
        "ok": status != "fail",
        "summary": {
            "hard_fail": hard_fail,
            "secret_findings": warn_count,
            "checks": len(checks),
        },
        "checks": checks,
    }
    atomic_write(hygiene_path, payload)

    if strict and status == "fail":
        emit(payload, 1)
    emit(payload)

emit({"ok": False, "status": "fail", "error": f"unknown_command:{cmd}"}, 2)
PY
