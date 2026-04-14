#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"

ACTION_TOKEN=""
MUTATION_TICKET=""
ATTESTATIONS=()
ATTESTATION_OBJECTS=()
ALLOW_LEGACY_ANCHOR=0
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"
      shift 2 ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"
      shift 2 ;;
    --attestation)
      ATTESTATIONS+=("${2:-}")
      shift 2 ;;
    --attestation-object)
      ATTESTATION_OBJECTS+=("${2:-}")
      shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1
      shift ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift ;;
  esac
done

ACTION="show"
for arg in "${FORWARDED_ARGS[@]}"; do
  if [[ "$arg" == "show" || "$arg" == "claim" || "$arg" == "commit" ]]; then
    ACTION="$arg"
    break
  fi
done

if [[ "$ACTION" != "show" ]]; then
  guard_script="$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh"
  if [[ -x "$guard_script" ]]; then
    risk_tier="medium"
    if [[ "$ACTION" == "commit" ]]; then
      risk_tier="high"
    fi

    guard_args=(
      --script "core_roadmap_queue_layer_txn.sh"
      --risk-tier "$risk_tier"
      --mutation-operation "core_roadmap_queue_txn:${ACTION}"
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
    "$guard_script" "${guard_args[@]}" >/dev/null
  fi
fi

python3 - "$ROOT" "${FORWARDED_ARGS[@]}" <<'PY'
import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import secrets
import tempfile
import sys
from typing import Any, Dict, List, Optional, Tuple


root = pathlib.Path(sys.argv[1]).resolve()
argv = sys.argv[2:]

queue_layer_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_CORE_ROADMAP_QUEUE_LAYER_PATH",
        str(root / "state" / "continuity" / "latest" / "core_roadmap_queue_layer.json"),
    )
).resolve()
runtime_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_CORE_ROADMAP_QUEUE_TXN_RUNTIME_PATH",
        str(root / "state" / "continuity" / "latest" / "core_roadmap_queue_transaction_runtime.json"),
    )
).resolve()
lock_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_CORE_ROADMAP_QUEUE_TXN_LOCK_PATH",
        str(root / "state" / "continuity" / "locks" / "core_roadmap_queue_transaction_runtime.lock"),
    )
).resolve()
stale_claim_retry_cooldown_sec = max(
    0,
    int(os.environ.get("OPENCLAW_CORE_ROADMAP_QUEUE_STALE_CLAIM_RETRY_COOLDOWN_SEC", "120") or 120),
)


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def now_dt() -> dt.datetime:
    override = parse_iso(os.environ.get("OPENCLAW_QUEUE_TXN_NOW"))
    if override is not None:
        return override
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return now_dt().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: pathlib.Path) -> str:
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
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def load_json_object(path: pathlib.Path) -> Dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("json_not_object")
    return loaded


def default_runtime() -> Dict[str, Any]:
    return {
        "schema": "clawd.core_roadmap_queue_transaction_runtime.v1",
        "generated_at": now_iso(),
        "fencing_epoch": 0,
        "active_claim": None,
        "task_runtime": {},
        "transition_history": [],
        "recovery": {
            "counters": {
                "stale_claim_expired": 0,
                "total": 0,
            },
            "last_recovery": None,
        },
    }


def normalize_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = default_runtime()
    out["generated_at"] = str(payload.get("generated_at") or out["generated_at"])

    try:
        out["fencing_epoch"] = max(0, int(payload.get("fencing_epoch") or 0))
    except Exception:
        out["fencing_epoch"] = 0

    active_claim = payload.get("active_claim") if isinstance(payload.get("active_claim"), dict) else None
    if active_claim:
        claim_epoch: Optional[int]
        try:
            claim_epoch = int(active_claim.get("claim_epoch"))
        except Exception:
            claim_epoch = None
        claim_token = str(active_claim.get("claim_token") or "").strip()
        task_id = str(active_claim.get("task_id") or "").strip()
        state = str(active_claim.get("state") or "claimed").strip().lower() or "claimed"
        if claim_epoch is not None and claim_epoch >= 0 and claim_token and task_id and state in {"claimed", "running"}:
            out["active_claim"] = {
                "task_id": task_id,
                "worker_id": str(active_claim.get("worker_id") or "").strip() or None,
                "claim_epoch": claim_epoch,
                "claim_token": claim_token,
                "state": state,
                "claimed_at": str(active_claim.get("claimed_at") or "").strip() or None,
                "running_at": str(active_claim.get("running_at") or "").strip() or None,
                "lease_sec": max(1, int(active_claim.get("lease_sec") or 900)),
                "lease_expires_at": str(active_claim.get("lease_expires_at") or "").strip() or None,
            }

    task_runtime: Dict[str, Dict[str, Any]] = {}
    raw_tasks = payload.get("task_runtime") if isinstance(payload.get("task_runtime"), dict) else {}
    for key, raw in raw_tasks.items():
        task_id = str(key or "").strip()
        if not task_id or not isinstance(raw, dict):
            continue
        state = str(raw.get("state") or "").strip().lower()
        if state not in {"claimed", "running", "done", "blocked", "retry"}:
            continue
        retry_count = 0
        try:
            retry_count = max(0, int(raw.get("retry_count") or 0))
        except Exception:
            retry_count = 0
        claim_epoch = raw.get("claim_epoch")
        try:
            claim_epoch = int(claim_epoch) if claim_epoch is not None else None
        except Exception:
            claim_epoch = None
        task_runtime[task_id] = {
            "task_id": task_id,
            "state": state,
            "updated_at": str(raw.get("updated_at") or "").strip() or None,
            "last_transition": str(raw.get("last_transition") or state).strip().lower() or state,
            "last_transition_at": str(raw.get("last_transition_at") or raw.get("updated_at") or "").strip() or None,
            "claim_epoch": claim_epoch,
            "claim_token": str(raw.get("claim_token") or "").strip() or None,
            "worker_id": str(raw.get("worker_id") or "").strip() or None,
            "retry_count": retry_count,
            "cooldown_until": str(raw.get("cooldown_until") or "").strip() or None,
            "reason": str(raw.get("reason") or "").strip() or None,
        }

    out["task_runtime"] = task_runtime

    recovery = payload.get("recovery") if isinstance(payload.get("recovery"), dict) else {}
    counters = recovery.get("counters") if isinstance(recovery.get("counters"), dict) else {}
    recovery_counters: Dict[str, int] = {}
    for key, value in counters.items():
        if not isinstance(key, str):
            continue
        try:
            recovery_counters[str(key).strip()] = max(0, int(value))
        except Exception:
            continue
    out["recovery"] = {
        "counters": {
            "stale_claim_expired": max(0, recovery_counters.get("stale_claim_expired", 0)),
            "total": max(0, recovery_counters.get("total", recovery_counters.get("stale_claim_expired", 0))),
        },
        "last_recovery": recovery.get("last_recovery") if isinstance(recovery.get("last_recovery"), dict) else None,
    }

    history = payload.get("transition_history") if isinstance(payload.get("transition_history"), list) else []
    normalized_history: List[Dict[str, Any]] = []
    for raw in history[-200:]:
        if not isinstance(raw, dict):
            continue
        normalized_history.append(raw)
    out["transition_history"] = normalized_history

    return out


def append_history(runtime: Dict[str, Any], event: Dict[str, Any]) -> None:
    history = runtime.get("transition_history") if isinstance(runtime.get("transition_history"), list) else []
    history.append(event)
    runtime["transition_history"] = history[-200:]


def acquire_lock() -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except Exception:
        os.close(fd)
        raise
    return fd


def release_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def load_queue_layer() -> Dict[str, Any]:
    if not queue_layer_path.exists():
        raise RuntimeError("queue_layer_missing")
    payload = load_json_object(queue_layer_path)
    status = str(payload.get("contract_status") or "missing").strip().lower()
    authoritative = bool(payload.get("authoritative") is True)
    if status != "ok" or not authoritative:
        raise RuntimeError("queue_layer_not_authoritative")
    return payload


def load_runtime() -> Dict[str, Any]:
    if not runtime_path.exists():
        return default_runtime()
    try:
        payload = load_json_object(runtime_path)
    except Exception:
        return default_runtime()
    return normalize_runtime(payload)


def cooldown_remaining_sec(cooldown_until: Optional[str], now: dt.datetime) -> int:
    deadline = parse_iso(cooldown_until)
    if deadline is None:
        return 0
    return max(0, int((deadline - now).total_seconds()))


def ensure_task_meta(runtime: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task_runtime = runtime.get("task_runtime") if isinstance(runtime.get("task_runtime"), dict) else {}
    runtime["task_runtime"] = task_runtime
    row = task_runtime.get(task_id)
    if not isinstance(row, dict):
        row = {
            "task_id": task_id,
            "state": "retry",
            "updated_at": None,
            "last_transition": None,
            "last_transition_at": None,
            "claim_epoch": None,
            "claim_token": None,
            "worker_id": None,
            "retry_count": 0,
            "cooldown_until": None,
            "reason": None,
        }
        task_runtime[task_id] = row
    return row


def expire_stale_claim_if_needed(runtime: Dict[str, Any], now: dt.datetime) -> Optional[Dict[str, Any]]:
    active = runtime.get("active_claim") if isinstance(runtime.get("active_claim"), dict) else None
    if not active:
        return None

    lease_expires_at = parse_iso(active.get("lease_expires_at"))
    if lease_expires_at is None or lease_expires_at > now:
        return None

    task_id = str(active.get("task_id") or "").strip()
    claim_epoch = active.get("claim_epoch")
    claim_token = str(active.get("claim_token") or "").strip() or None

    meta = ensure_task_meta(runtime, task_id)
    try:
        retry_count = max(0, int(meta.get("retry_count") or 0)) + 1
    except Exception:
        retry_count = 1

    cooldown_until = (now + dt.timedelta(seconds=stale_claim_retry_cooldown_sec)).replace(microsecond=0)
    cooldown_iso = cooldown_until.isoformat().replace("+00:00", "Z")

    now_iso = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    meta.update(
        {
            "state": "retry",
            "updated_at": now_iso,
            "last_transition": "retry",
            "last_transition_at": now_iso,
            "claim_epoch": claim_epoch if isinstance(claim_epoch, int) else None,
            "claim_token": None,
            "worker_id": str(active.get("worker_id") or "").strip() or None,
            "retry_count": retry_count,
            "cooldown_until": cooldown_iso,
            "reason": "stale_claim_expired",
        }
    )

    recovery = runtime.get("recovery") if isinstance(runtime.get("recovery"), dict) else {}
    counters = recovery.get("counters") if isinstance(recovery.get("counters"), dict) else {}
    counters = {str(k): max(0, int(v)) for k, v in counters.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
    counters["stale_claim_expired"] = counters.get("stale_claim_expired", 0) + 1
    counters["total"] = counters.get("total", counters.get("stale_claim_expired", 0) - 1) + 1

    runtime["recovery"] = {
        "counters": {
            "stale_claim_expired": counters["stale_claim_expired"],
            "total": counters["total"],
        },
        "last_recovery": {
            "transition": "stale_claim_expired",
            "task_id": task_id,
            "claim_epoch": claim_epoch if isinstance(claim_epoch, int) else None,
            "claim_token": claim_token,
            "retry_count": retry_count,
            "cooldown_until": cooldown_iso,
            "recovered_at": now_iso,
        },
    }

    runtime["active_claim"] = None
    event = {
        "ts": now_iso,
        "transition": "stale_claim_expired",
        "task_id": task_id,
        "claim_epoch": claim_epoch if isinstance(claim_epoch, int) else None,
        "claim_token": claim_token,
        "retry_count": retry_count,
        "cooldown_until": cooldown_iso,
    }
    append_history(runtime, event)
    return event


def parse_ready_candidates(queue_layer: Dict[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ready_rows = queue_layer.get("ready_candidates") if isinstance(queue_layer.get("ready_candidates"), list) else []
    order: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for row in ready_rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        order.append(task_id)
        meta[task_id] = row
    return order, meta


def build_runtime_projection(runtime: Dict[str, Any], queue_layer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = now_dt()
    task_runtime = runtime.get("task_runtime") if isinstance(runtime.get("task_runtime"), dict) else {}

    state_counts = {"claimed": 0, "running": 0, "done": 0, "blocked": 0, "retry": 0}
    cooldown_active: List[Dict[str, Any]] = []
    retry_ready: List[str] = []

    for task_id, raw in sorted(task_runtime.items()):
        if not isinstance(raw, dict):
            continue
        state = str(raw.get("state") or "").strip().lower()
        if state in state_counts:
            state_counts[state] += 1

        if state == "retry":
            cooldown_until = str(raw.get("cooldown_until") or "").strip() or None
            remaining = cooldown_remaining_sec(cooldown_until, now)
            if remaining > 0:
                cooldown_active.append(
                    {
                        "task_id": task_id,
                        "cooldown_until": cooldown_until,
                        "cooldown_remaining_sec": remaining,
                    }
                )
            else:
                retry_ready.append(task_id)

    ready_candidates = []
    if isinstance(queue_layer, dict):
        ready_candidates, _ = parse_ready_candidates(queue_layer)

    eligible_ready: List[str] = []
    for task_id in ready_candidates:
        row = task_runtime.get(task_id) if isinstance(task_runtime.get(task_id), dict) else {}
        state = str(row.get("state") or "").strip().lower()
        if state in {"done", "blocked", "claimed", "running"}:
            continue
        if state == "retry":
            if cooldown_remaining_sec(str(row.get("cooldown_until") or ""), now) > 0:
                continue
        eligible_ready.append(task_id)

    recovery = runtime.get("recovery") if isinstance(runtime.get("recovery"), dict) else {}

    return {
        "schema": "clawd.core_roadmap_queue_transaction_runtime_projection.v1",
        "runtime_path": rel(runtime_path),
        "lock_path": rel(lock_path),
        "fencing_epoch": int(runtime.get("fencing_epoch") or 0),
        "active_claim": runtime.get("active_claim") if isinstance(runtime.get("active_claim"), dict) else None,
        "task_state_counts": state_counts,
        "cooldown_active": cooldown_active,
        "retry_ready": retry_ready,
        "eligible_ready": eligible_ready,
        "history_size": len(runtime.get("transition_history") if isinstance(runtime.get("transition_history"), list) else []),
        "recovery": {
            "counters": recovery.get("counters") if isinstance(recovery.get("counters"), dict) else {"stale_claim_expired": 0, "total": 0},
            "last_recovery": recovery.get("last_recovery") if isinstance(recovery.get("last_recovery"), dict) else None,
        },
    }


def emit(payload: Dict[str, Any], rc: int = 0) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        ok = bool(payload.get("ok") is True)
        status = "ok" if ok else "error"
        action = str(payload.get("action") or args.action)
        detail = str(payload.get("error") or payload.get("decision") or "")
        print(f"CORE ROADMAP QUEUE TXN: action={action} status={status} detail={detail}")
    raise SystemExit(rc)


parser = argparse.ArgumentParser(
    prog="core_roadmap_queue_layer_txn.sh",
    description="Transactional claim/commit runtime for core-roadmap queue layer",
)
parser.add_argument("action", choices=["show", "claim", "commit"], nargs="?", default="show")
parser.add_argument("--task-id", dest="task_id")
parser.add_argument("--worker", dest="worker_id")
parser.add_argument("--lease-sec", dest="lease_sec", type=int, default=900)
parser.add_argument("--claim-token", dest="claim_token")
parser.add_argument("--claim-epoch", dest="claim_epoch", type=int)
parser.add_argument("--to-state", dest="to_state", choices=["running", "done", "blocked", "retry"])
parser.add_argument("--reason", dest="reason", default="")
parser.add_argument("--cooldown-sec", dest="cooldown_sec", type=int, default=300)
parser.add_argument("--json", action="store_true")
args = parser.parse_args(argv)

lock_fd: Optional[int] = None
try:
    lock_fd = acquire_lock()
    queue_layer: Optional[Dict[str, Any]] = None
    queue_error: Optional[str] = None
    try:
        queue_layer = load_queue_layer()
    except RuntimeError as exc:
        queue_error = str(exc)

    runtime = load_runtime()
    stale_event = expire_stale_claim_if_needed(runtime, now_dt())

    if args.action == "show":
        projection = build_runtime_projection(runtime, queue_layer)
        payload = {
            "ok": True,
            "action": "show",
            "schema": "clawd.core_roadmap_queue_transaction_runtime_surface.v1",
            "generated_at": now_iso(),
            "queue_layer_path": rel(queue_layer_path),
            "queue_layer_authoritative": bool(queue_error is None and queue_layer is not None),
            "queue_layer_error": queue_error,
            "runtime": runtime,
            "projection": projection,
        }
        if stale_event:
            runtime["generated_at"] = now_iso()
            atomic_write(runtime_path, runtime)
            payload["stale_claim_recovered"] = stale_event
        emit(payload, rc=0)

    if queue_error is not None or queue_layer is None:
        emit(
            {
                "ok": False,
                "action": args.action,
                "error": queue_error or "queue_layer_unavailable",
                "decision": "BLOCK",
                "queue_layer_path": rel(queue_layer_path),
            },
            rc=3,
        )

    now = now_dt()
    ready_order, ready_meta = parse_ready_candidates(queue_layer)
    task_runtime = runtime.get("task_runtime") if isinstance(runtime.get("task_runtime"), dict) else {}

    if args.action == "claim":
        worker_id = str(args.worker_id or "").strip()
        if not worker_id:
            emit({"ok": False, "action": "claim", "error": "worker_required"}, rc=2)

        active = runtime.get("active_claim") if isinstance(runtime.get("active_claim"), dict) else None
        if active is not None:
            emit(
                {
                    "ok": False,
                    "action": "claim",
                    "error": "active_claim_inflight",
                    "decision": "BLOCK",
                    "active_claim": active,
                    "projection": build_runtime_projection(runtime, queue_layer),
                },
                rc=3,
            )

        requested_task = str(args.task_id or "").strip() or None

        candidate_blockers: Dict[str, str] = {}
        eligible: List[str] = []
        cooldown_evidence: List[Dict[str, Any]] = []
        for task_id in ready_order:
            meta = task_runtime.get(task_id) if isinstance(task_runtime.get(task_id), dict) else {}
            state = str(meta.get("state") or "").strip().lower()
            if state in {"done", "blocked", "claimed", "running"}:
                candidate_blockers[task_id] = f"terminal_state:{state}"
                continue
            if state == "retry":
                remaining = cooldown_remaining_sec(str(meta.get("cooldown_until") or ""), now)
                if remaining > 0:
                    candidate_blockers[task_id] = "cooldown_active"
                    cooldown_evidence.append(
                        {
                            "task_id": task_id,
                            "cooldown_until": str(meta.get("cooldown_until") or "").strip() or None,
                            "cooldown_remaining_sec": remaining,
                        }
                    )
                    continue
            eligible.append(task_id)

        if requested_task:
            if requested_task not in ready_order:
                emit(
                    {
                        "ok": False,
                        "action": "claim",
                        "error": "requested_task_not_ready_candidate",
                        "decision": "BLOCK",
                        "task_id": requested_task,
                    },
                    rc=3,
                )
            if requested_task not in eligible:
                blocker = candidate_blockers.get(requested_task) or "requested_task_not_eligible"
                emit(
                    {
                        "ok": False,
                        "action": "claim",
                        "error": blocker,
                        "decision": "BLOCK",
                        "task_id": requested_task,
                        "cooldown": cooldown_evidence,
                    },
                    rc=3,
                )
            selected_task = requested_task
        else:
            if not eligible:
                error_code = "no_ready_candidate"
                if cooldown_evidence:
                    error_code = "cooldown_active"
                emit(
                    {
                        "ok": False,
                        "action": "claim",
                        "error": error_code,
                        "decision": "BLOCK",
                        "ready_candidates": ready_order,
                        "cooldown": cooldown_evidence,
                    },
                    rc=3,
                )
            selected_task = eligible[0]

        claim_epoch = int(runtime.get("fencing_epoch") or 0) + 1
        lease_sec = max(1, int(args.lease_sec or 900))
        claim_issued_at = now.replace(microsecond=0)
        lease_expires_at = (claim_issued_at + dt.timedelta(seconds=lease_sec)).replace(microsecond=0)
        claim_token = hashlib.sha256(
            f"{selected_task}|{claim_epoch}|{worker_id}|{claim_issued_at.isoformat()}|{os.getpid()}|{secrets.token_hex(8)}".encode(
                "utf-8"
            )
        ).hexdigest()

        claim_obj = {
            "task_id": selected_task,
            "worker_id": worker_id,
            "claim_epoch": claim_epoch,
            "claim_token": claim_token,
            "state": "claimed",
            "claimed_at": claim_issued_at.isoformat().replace("+00:00", "Z"),
            "running_at": None,
            "lease_sec": lease_sec,
            "lease_expires_at": lease_expires_at.isoformat().replace("+00:00", "Z"),
        }

        runtime["fencing_epoch"] = claim_epoch
        runtime["active_claim"] = claim_obj

        row = ensure_task_meta(runtime, selected_task)
        row.update(
            {
                "state": "claimed",
                "updated_at": claim_issued_at.isoformat().replace("+00:00", "Z"),
                "last_transition": "claimed",
                "last_transition_at": claim_issued_at.isoformat().replace("+00:00", "Z"),
                "claim_epoch": claim_epoch,
                "claim_token": claim_token,
                "worker_id": worker_id,
                "cooldown_until": None,
                "reason": str(args.reason or "").strip() or None,
            }
        )

        append_history(
            runtime,
            {
                "ts": claim_issued_at.isoformat().replace("+00:00", "Z"),
                "transition": "claim",
                "task_id": selected_task,
                "claim_epoch": claim_epoch,
                "claim_token": claim_token,
                "worker_id": worker_id,
            },
        )
        runtime["generated_at"] = now_iso()
        atomic_write(runtime_path, runtime)

        emit(
            {
                "ok": True,
                "action": "claim",
                "decision": "APPLY",
                "task_id": selected_task,
                "claim": claim_obj,
                "task": ready_meta.get(selected_task) if isinstance(ready_meta.get(selected_task), dict) else {"task_id": selected_task},
                "projection": build_runtime_projection(runtime, queue_layer),
                "runtime_path": rel(runtime_path),
                "queue_layer_path": rel(queue_layer_path),
            },
            rc=0,
        )

    if args.action == "commit":
        task_id = str(args.task_id or "").strip()
        claim_token = str(args.claim_token or "").strip()
        claim_epoch = args.claim_epoch
        to_state = str(args.to_state or "").strip().lower()
        if not task_id or not claim_token or claim_epoch is None or not to_state:
            emit({"ok": False, "action": "commit", "error": "task_id_claim_token_claim_epoch_to_state_required"}, rc=2)

        active = runtime.get("active_claim") if isinstance(runtime.get("active_claim"), dict) else None
        if not active:
            emit(
                {
                    "ok": False,
                    "action": "commit",
                    "error": "stale_claim_rejected",
                    "decision": "BLOCK",
                    "reason": "active_claim_missing",
                },
                rc=3,
            )

        active_task = str(active.get("task_id") or "").strip()
        active_token = str(active.get("claim_token") or "").strip()
        try:
            active_epoch = int(active.get("claim_epoch"))
        except Exception:
            active_epoch = -1

        if task_id != active_task or claim_token != active_token or int(claim_epoch) != active_epoch:
            emit(
                {
                    "ok": False,
                    "action": "commit",
                    "error": "stale_claim_rejected",
                    "decision": "BLOCK",
                    "provided": {
                        "task_id": task_id,
                        "claim_epoch": int(claim_epoch),
                        "claim_token": claim_token,
                    },
                    "active_claim": active,
                },
                rc=3,
            )

        active_state = str(active.get("state") or "claimed").strip().lower() or "claimed"
        expected_from = "claimed" if to_state == "running" else "running"
        if active_state != expected_from:
            emit(
                {
                    "ok": False,
                    "action": "commit",
                    "error": "invalid_transition",
                    "decision": "BLOCK",
                    "from_state": active_state,
                    "expected_from": expected_from,
                    "to_state": to_state,
                    "active_claim": active,
                },
                rc=3,
            )

        row = ensure_task_meta(runtime, task_id)
        ts = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        reason = str(args.reason or "").strip() or None

        if to_state == "running":
            active["state"] = "running"
            active["running_at"] = ts
            runtime["active_claim"] = active

            row.update(
                {
                    "state": "running",
                    "updated_at": ts,
                    "last_transition": "running",
                    "last_transition_at": ts,
                    "claim_epoch": int(claim_epoch),
                    "claim_token": claim_token,
                    "worker_id": str(active.get("worker_id") or "").strip() or None,
                    "reason": reason,
                }
            )
        else:
            runtime["active_claim"] = None
            base_retry = max(0, int(row.get("retry_count") or 0))
            cooldown_until = None
            retry_count = base_retry
            if to_state == "retry":
                retry_count += 1
                cooldown_sec = max(0, int(args.cooldown_sec or 0))
                if cooldown_sec > 0:
                    cooldown_until = (now + dt.timedelta(seconds=cooldown_sec)).replace(microsecond=0).isoformat().replace(
                        "+00:00", "Z"
                    )
            row.update(
                {
                    "state": to_state,
                    "updated_at": ts,
                    "last_transition": to_state,
                    "last_transition_at": ts,
                    "claim_epoch": int(claim_epoch),
                    "claim_token": None,
                    "worker_id": str(active.get("worker_id") or "").strip() or None,
                    "retry_count": retry_count,
                    "cooldown_until": cooldown_until,
                    "reason": reason,
                }
            )

        append_history(
            runtime,
            {
                "ts": ts,
                "transition": to_state,
                "task_id": task_id,
                "claim_epoch": int(claim_epoch),
                "claim_token": claim_token,
                "reason": reason,
                "cooldown_sec": max(0, int(args.cooldown_sec or 0)) if to_state == "retry" else 0,
            },
        )
        runtime["generated_at"] = now_iso()
        atomic_write(runtime_path, runtime)

        payload: Dict[str, Any] = {
            "ok": True,
            "action": "commit",
            "decision": "APPLY",
            "task_id": task_id,
            "to_state": to_state,
            "claim_epoch": int(claim_epoch),
            "runtime_path": rel(runtime_path),
            "projection": build_runtime_projection(runtime, queue_layer),
        }
        if to_state == "retry":
            payload["retry_contract"] = {
                "schema": "clawd.core_roadmap_queue_retry_contract.v1",
                "policy": "fixed_cooldown",
                "retry_count": int((runtime.get("task_runtime") or {}).get(task_id, {}).get("retry_count") or 0),
                "cooldown_sec": max(0, int(args.cooldown_sec or 0)),
                "cooldown_until": (runtime.get("task_runtime") or {}).get(task_id, {}).get("cooldown_until"),
            }
        emit(payload, rc=0)

    emit({"ok": False, "action": args.action, "error": "unknown_action"}, rc=2)
except SystemExit:
    raise
except Exception as exc:
    emit(
        {
            "ok": False,
            "action": getattr(args, "action", "unknown"),
            "error": "core_roadmap_queue_txn_runtime_error",
            "detail": str(exc),
        },
        rc=2,
    )
finally:
    release_lock(lock_fd)
PY
