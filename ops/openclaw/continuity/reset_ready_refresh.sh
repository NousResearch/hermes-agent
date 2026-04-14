#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: reset_ready_refresh.sh [options]

Refresh continuity/current, mint a generation-aligned successor-safe proof,
rebuild handover/latest surfaces, and fail closed if the published handover does
not agree with the fresh proof artifacts.

Options:
  --json        Print JSON summary
  -h, --help    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

python3 - "$ROOT" "$JSON_OUT" <<'PY'
import datetime as dt
import importlib.util
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
from typing import Any, List, Mapping, Optional

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_positive_int_env(name: str, *, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(int(default)))))
    except Exception:
        return int(default)


current_timeout_sec = read_positive_int_env(
    "OPENCLAW_RESET_READY_REFRESH_CURRENT_TIMEOUT_SEC",
    default=120,
)
handover_timeout_sec = read_positive_int_env(
    "OPENCLAW_RESET_READY_REFRESH_HANDOVER_TIMEOUT_SEC",
    default=120,
)
kill_grace_sec = read_positive_int_env(
    "OPENCLAW_RESET_READY_REFRESH_KILL_GRACE_SEC",
    default=3,
)


artifacts = {
    "current": "state/continuity/current.json",
    "proof": "state/continuity/latest/successor_safe_handover_proof.json",
    "proof_status": "state/continuity/latest/successor_safe_handover_proof_status.json",
    "successor_packet": "state/continuity/successor_packet/latest.json",
    "successor_hydrated_state": "state/continuity/successor_packet/hydrated_state.json",
    "successor_handover_ack": "state/continuity/successor_packet/handover_ack.jsonl",
    "handover_json": "state/handover/latest.json",
    "handover_md": "state/handover/latest.md",
    "result": "state/continuity/latest/reset_ready_refresh_latest.json",
}
result_artifact_path = root / artifacts["result"]

# Avoid self-poisoning the new refresh with the previous wrapper outcome.
# continuity_current/continuity_now project reset_ready_refresh_latest.json into
# readiness, so a stale degraded artifact can block the very run that is trying
# to replace it. This wrapper always writes a fresh result artifact on exit.
try:
    if result_artifact_path.exists():
        result_artifact_path.unlink()
except Exception:
    pass


def truncate(raw: Any, *, limit: int = 400) -> str:
    txt = str(raw or "")
    if len(txt) <= limit:
        return txt
    return txt[:limit]


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_bounded_command(cmd: list[str], *, timeout_sec: int) -> dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return {
            "ok": proc.returncode == 0,
            "timed_out": False,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_sec": timeout_sec,
            "command": shell_join(cmd),
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            stdout, stderr = proc.communicate(timeout=kill_grace_sec)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            stdout, stderr = proc.communicate()

        return {
            "ok": False,
            "timed_out": True,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_sec": timeout_sec,
            "command": shell_join(cmd),
        }


def classify_command_failure(*, phase: str, run: Mapping[str, Any]) -> str:
    if run.get("timed_out") is True:
        return f"{phase}_timeout"

    haystack = (
        f"{run.get('stderr') or ''}\n{run.get('stdout') or ''}"
    ).lower()
    if "publish lock timeout" in haystack or (
        "current_publish.lock" in haystack and "timeout" in haystack
    ):
        return "publish_lock_timeout"
    return f"{phase}_failed"


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    txt = str(raw or "").strip()
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
    return out.astimezone(dt.timezone.utc)


def read_text_env(name: str, *, default: str) -> str:
    raw = os.environ.get(name)
    txt = str(raw).strip().lower() if raw is not None else ""
    return txt or str(default).strip().lower()


def load_json_object(path: pathlib.Path) -> Optional[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def load_ack_evidence_for_packet(path: pathlib.Path, *, packet_id: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or not packet_id:
        return {
            "latest_row": None,
        }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {
            "latest_row": None,
        }
    latest_row: Optional[dict[str, Any]] = None
    for raw_line in reversed(lines):
        line = str(raw_line or "").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, Mapping):
            continue
        if str(row.get("packet_id") or "").strip() != packet_id:
            continue
        row_obj = dict(row)
        if latest_row is None:
            latest_row = row_obj
            break
    return {
        "latest_row": latest_row,
    }


def evaluate_successor_inheritance_contract(*, now_dt: dt.datetime) -> dict[str, Any]:
    gate_mode = read_text_env("OPENCLAW_RESET_READY_SUCCESSOR_INHERITANCE_GATE", default="auto")
    packet_path = root / artifacts["successor_packet"]
    hydrated_state_path = root / artifacts["successor_hydrated_state"]
    ack_path = root / artifacts["successor_handover_ack"]

    packet_obj = load_json_object(packet_path)
    packet_present = isinstance(packet_obj, Mapping)

    if gate_mode in {"off", "false", "0", "disabled"}:
        gate_enforced = False
    elif gate_mode in {"required", "true", "1", "on"}:
        gate_enforced = True
    else:
        gate_enforced = packet_present

    packet_id = str((packet_obj or {}).get("packet_id") or "").strip() or None
    packet_generated_at = str((packet_obj or {}).get("generated_at") or "").strip() or None
    packet_generated_dt = parse_iso(packet_generated_at)
    packet_continuity_hash = str((packet_obj or {}).get("continuity_hash") or "").strip() or None

    hydrated_state_obj = load_json_object(hydrated_state_path)
    hydrated_state_present = isinstance(hydrated_state_obj, Mapping)
    hydrated_state_packet_id = str((hydrated_state_obj or {}).get("packet_id") or "").strip() or None
    hydrated_state_continuity_hash = str((hydrated_state_obj or {}).get("continuity_hash") or "").strip() or None

    ack_evidence = load_ack_evidence_for_packet(ack_path, packet_id=str(packet_id or ""))
    ack_latest_obj = ack_evidence.get("latest_row") if isinstance(ack_evidence.get("latest_row"), Mapping) else None
    ack_latest_present = isinstance(ack_latest_obj, Mapping)
    ack_latest_result = str((ack_latest_obj or {}).get("hydration_result") or "").strip() or None
    ack_obj = ack_latest_obj if ack_latest_result == "success" else None
    ack_present = isinstance(ack_obj, Mapping)
    ack_latest_error_code = str((ack_latest_obj or {}).get("error_code") or "").strip() or None
    ack_successor_session_id = str((ack_obj or {}).get("successor_session_id") or "").strip() or None
    ack_handover_at = str((ack_obj or {}).get("handover_at") or "").strip() or None
    ack_handover_dt = parse_iso(ack_handover_at)

    mismatches: List[str] = []

    if gate_enforced:
        if not packet_present:
            mismatches.append("successor_packet_missing")
        if not packet_id:
            mismatches.append("successor_packet_id_missing")
        if not packet_continuity_hash:
            mismatches.append("successor_packet_continuity_hash_missing")

        if not hydrated_state_present:
            mismatches.append("hydrated_state_missing")
        if packet_id and hydrated_state_packet_id != packet_id:
            mismatches.append("hydrated_state_packet_id_mismatch")
        if packet_continuity_hash and hydrated_state_continuity_hash != packet_continuity_hash:
            mismatches.append("hydrated_state_continuity_hash_mismatch")

        if not ack_latest_present:
            mismatches.append("handover_ack_missing_for_packet")
        elif ack_latest_result != "success":
            mismatches.append("handover_ack_latest_not_success")
        else:
            if not ack_successor_session_id:
                mismatches.append("handover_ack_successor_session_missing")
            if packet_generated_dt is not None and ack_handover_dt is not None and ack_handover_dt < packet_generated_dt:
                mismatches.append("handover_ack_before_packet_generated")
            if packet_generated_dt is not None and ack_handover_dt is None:
                mismatches.append("handover_ack_timestamp_invalid")
            if packet_generated_dt is not None and ack_handover_dt is not None and ack_handover_dt > now_dt:
                mismatches.append("handover_ack_timestamp_in_future")

    return {
        "gate_mode": gate_mode,
        "gate_enforced": bool(gate_enforced),
        "packet_present": packet_present,
        "packet_id": packet_id,
        "packet_generated_at": packet_generated_at,
        "packet_continuity_hash": packet_continuity_hash,
        "hydrated_state_present": hydrated_state_present,
        "hydrated_state_packet_id": hydrated_state_packet_id,
        "hydrated_state_continuity_hash": hydrated_state_continuity_hash,
        "ack_present": ack_present,
        "ack_latest_present": ack_latest_present,
        "ack_latest_result": ack_latest_result,
        "ack_latest_error_code": ack_latest_error_code,
        "ack_successor_session_id": ack_successor_session_id,
        "ack_handover_at": ack_handover_at,
        "mismatches": mismatches,
    }


def normalize_result_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    ok = out.get("ok") is True

    phase = str(out.get("phase") or "").strip()
    if not phase:
        out["phase"] = "complete" if ok else "failed"

    result = str(out.get("result") or "").strip()
    if not result:
        out["result"] = "success" if ok else "failure"

    return out


def emit(payload: dict[str, Any], *, exit_code: int) -> None:
    payload = normalize_result_contract(payload)

    try:
        result_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        result_artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    if payload.get("ok"):
        if json_out:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "RESET READY REFRESH: "
                f"ok={payload['ok']} "
                f"readiness={payload.get('current', {}).get('readiness')} "
                f"proof_state={payload.get('proof_status', {}).get('proof_state')} "
                f"reset_allowed={payload.get('proof_status', {}).get('reset_allowed')} "
                f"resume_allowed={payload.get('proof_status', {}).get('resume_allowed')}"
            )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


base_payload = {
    "ok": False,
    "schema": "clawd.reset_ready_refresh.result.v1",
    "generated_at": now_iso(),
    "runtime_bounds": {
        "current_timeout_sec": current_timeout_sec,
        "handover_timeout_sec": handover_timeout_sec,
        "kill_grace_sec": kill_grace_sec,
    },
    "partial_refresh": {
        "current_refreshed": False,
        "proof_refreshed": False,
        "handover_refreshed": False,
    },
    "artifacts": dict(artifacts),
}

current_cmd = ["bash", str(root / "ops" / "openclaw" / "continuity.sh"), "current", "--refresh", "--json"]
current_run = run_bounded_command(current_cmd, timeout_sec=current_timeout_sec)
current_payload: Optional[Mapping[str, Any]] = None

if current_run.get("timed_out") is True:
    error_code = classify_command_failure(phase="current_refresh", run=current_run)
    payload = dict(base_payload)
    payload["phase"] = "current_refresh"
    payload["error"] = {
        "code": error_code,
        "message": truncate(current_run.get("stderr") or current_run.get("stdout") or error_code),
        "command": current_run.get("command"),
        "timeout_sec": current_run.get("timeout_sec"),
        "timed_out": True,
        "returncode": current_run.get("returncode"),
    }
    emit(payload, exit_code=1)

if not current_run.get("ok"):
    raw_stdout = str(current_run.get("stdout") or "")
    if raw_stdout.strip():
        try:
            nonzero_payload = json.loads(raw_stdout)
        except Exception:
            nonzero_payload = None
        if isinstance(nonzero_payload, Mapping):
            # continuity_current fail-close contract may intentionally return a
            # structured degraded payload with nonzero exit status.
            current_payload = nonzero_payload

    if current_payload is None:
        error_code = classify_command_failure(phase="current_refresh", run=current_run)
        payload = dict(base_payload)
        payload["phase"] = "current_refresh"
        payload["error"] = {
            "code": error_code,
            "message": truncate(current_run.get("stderr") or current_run.get("stdout") or error_code),
            "command": current_run.get("command"),
            "timeout_sec": current_run.get("timeout_sec"),
            "timed_out": False,
            "returncode": current_run.get("returncode"),
        }
        emit(payload, exit_code=1)

if current_payload is None:
    try:
        current_payload = json.loads(current_run.get("stdout") or "{}")
    except Exception as exc:
        payload = dict(base_payload)
        payload["phase"] = "current_refresh"
        payload["partial_refresh"] = {
            "current_refreshed": False,
            "proof_refreshed": False,
            "handover_refreshed": False,
        }
        payload["error"] = {
            "code": "current_refresh_invalid_json",
            "message": truncate(exc),
            "command": current_run.get("command"),
            "timed_out": False,
            "returncode": current_run.get("returncode"),
        }
        emit(payload, exit_code=1)

if not isinstance(current_payload, Mapping):
    payload = dict(base_payload)
    payload["phase"] = "current_refresh"
    payload["error"] = {
        "code": "current_refresh_invalid_payload",
        "message": "continuity_current_payload_not_object",
        "command": current_run.get("command"),
        "timed_out": False,
        "returncode": current_run.get("returncode"),
    }
    emit(payload, exit_code=1)

proof_status: Optional[dict[str, Any]] = None
try:
    proof_module_path = root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py"
    spec = importlib.util.spec_from_file_location("successor_safe_handover_proof", proof_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("successor_safe_handover_proof_module_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    proof = module.build_successor_safe_handover_proof(root=root, trigger="reset_ready_refresh")
    proof_eval = module.evaluate_proof_consumability(proof, mode="resume")
    projected_status = module.project_proof_status(proof=proof, evaluation=proof_eval)
    module.write_proof_artifact(root=root, proof=proof)

    proof_status_path = root / artifacts["proof_status"]
    proof_status_path.parent.mkdir(parents=True, exist_ok=True)
    proof_status_path.write_text(
        json.dumps(projected_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    proof_status = dict(projected_status)
except Exception as exc:
    payload = dict(base_payload)
    payload["phase"] = "proof_refresh"
    payload["partial_refresh"] = {
        "current_refreshed": True,
        "proof_refreshed": False,
        "handover_refreshed": False,
    }
    payload["current"] = {
        "generated_at": current_payload.get("generated_at"),
        "readiness": current_payload.get("readiness"),
        "action_token": current_payload.get("action_token"),
        "mutation_gate": (current_payload.get("mutation_gate") or {}).get("status") if isinstance(current_payload.get("mutation_gate"), Mapping) else None,
    }
    payload["error"] = {
        "code": "proof_refresh_failed",
        "message": truncate(exc),
    }
    emit(payload, exit_code=1)

# Rebuild handover surfaces against the already-refreshed current/proof artifacts.
# Avoid --refresh here: handover_latest may refresh current again, which can rotate the
# coherence generation after the proof is minted and create a false proof-generation mismatch.
handover_cmd = ["bash", str(root / "ops" / "openclaw" / "continuity" / "handover_latest.sh")]
handover_run = run_bounded_command(handover_cmd, timeout_sec=handover_timeout_sec)
if not handover_run.get("ok"):
    error_code = classify_command_failure(phase="handover_refresh", run=handover_run)
    payload = dict(base_payload)
    payload["phase"] = "handover_refresh"
    payload["partial_refresh"] = {
        "current_refreshed": True,
        "proof_refreshed": True,
        "handover_refreshed": False,
    }
    payload["current"] = {
        "generated_at": current_payload.get("generated_at"),
        "readiness": current_payload.get("readiness"),
        "action_token": current_payload.get("action_token"),
        "mutation_gate": (current_payload.get("mutation_gate") or {}).get("status") if isinstance(current_payload.get("mutation_gate"), Mapping) else None,
    }
    payload["proof_status"] = dict(proof_status or {})
    payload["error"] = {
        "code": error_code,
        "message": truncate(handover_run.get("stderr") or handover_run.get("stdout") or error_code),
        "command": handover_run.get("command"),
        "timeout_sec": handover_run.get("timeout_sec"),
        "timed_out": bool(handover_run.get("timed_out") is True),
        "returncode": handover_run.get("returncode"),
    }
    emit(payload, exit_code=1)

handover_path = root / artifacts["handover_json"]
try:
    handover = json.loads(handover_path.read_text(encoding="utf-8"))
except Exception as exc:
    payload = dict(base_payload)
    payload["phase"] = "handover_refresh"
    payload["partial_refresh"] = {
        "current_refreshed": True,
        "proof_refreshed": True,
        "handover_refreshed": False,
    }
    payload["current"] = {
        "generated_at": current_payload.get("generated_at"),
        "readiness": current_payload.get("readiness"),
        "action_token": current_payload.get("action_token"),
        "mutation_gate": (current_payload.get("mutation_gate") or {}).get("status") if isinstance(current_payload.get("mutation_gate"), Mapping) else None,
    }
    payload["proof_status"] = dict(proof_status or {})
    payload["error"] = {
        "code": "handover_payload_unreadable",
        "message": truncate(exc),
        "command": handover_run.get("command"),
    }
    emit(payload, exit_code=1)

if not isinstance(handover, Mapping):
    payload = dict(base_payload)
    payload["phase"] = "handover_refresh"
    payload["partial_refresh"] = {
        "current_refreshed": True,
        "proof_refreshed": True,
        "handover_refreshed": False,
    }
    payload["current"] = {
        "generated_at": current_payload.get("generated_at"),
        "readiness": current_payload.get("readiness"),
        "action_token": current_payload.get("action_token"),
        "mutation_gate": (current_payload.get("mutation_gate") or {}).get("status") if isinstance(current_payload.get("mutation_gate"), Mapping) else None,
    }
    payload["proof_status"] = dict(proof_status or {})
    payload["error"] = {
        "code": "handover_payload_not_object",
        "message": "handover_payload_not_object",
        "command": handover_run.get("command"),
    }
    emit(payload, exit_code=1)

final_proof_status = dict(proof_status or {})
proof_status_path = root / artifacts["proof_status"]
try:
    proof_status_disk = json.loads(proof_status_path.read_text(encoding="utf-8"))
    if isinstance(proof_status_disk, Mapping):
        final_proof_status = dict(proof_status_disk)
except Exception:
    pass

handover_proof = handover.get("proof_status") if isinstance(handover.get("proof_status"), Mapping) else {}
handover_safe = handover.get("safe_signals") if isinstance(handover.get("safe_signals"), Mapping) else {}
proof_mismatches: list[str] = []

for field in ("proof_id", "proof_state", "reset_allowed", "resume_allowed", "top_blocker"):
    if handover_proof.get(field) != final_proof_status.get(field):
        proof_mismatches.append(f"proof_status.{field}")

if handover_safe.get("proof_id") != final_proof_status.get("proof_id"):
    proof_mismatches.append("safe_signals.proof_id")
if handover_safe.get("proof_state") != final_proof_status.get("proof_state"):
    proof_mismatches.append("safe_signals.proof_state")
if handover_safe.get("proof_reset_allowed") != final_proof_status.get("reset_allowed"):
    proof_mismatches.append("safe_signals.proof_reset_allowed")
if handover_safe.get("proof_resume_allowed") != final_proof_status.get("resume_allowed"):
    proof_mismatches.append("safe_signals.proof_resume_allowed")

successor_inheritance = evaluate_successor_inheritance_contract(now_dt=dt.datetime.now(dt.timezone.utc))
successor_inheritance_mismatches = list(successor_inheritance.get("mismatches") or [])

ok = len(proof_mismatches) == 0 and len(successor_inheritance_mismatches) == 0
payload = {
    "ok": ok,
    "schema": "clawd.reset_ready_refresh.result.v1",
    "generated_at": now_iso(),
    "runtime_bounds": {
        "current_timeout_sec": current_timeout_sec,
        "handover_timeout_sec": handover_timeout_sec,
        "kill_grace_sec": kill_grace_sec,
    },
    "partial_refresh": {
        "current_refreshed": True,
        "proof_refreshed": True,
        "handover_refreshed": True,
    },
    "current": {
        "generated_at": current_payload.get("generated_at"),
        "readiness": current_payload.get("readiness"),
        "action_token": current_payload.get("action_token"),
        "mutation_gate": (current_payload.get("mutation_gate") or {}).get("status") if isinstance(current_payload.get("mutation_gate"), Mapping) else None,
    },
    "proof_status": dict(final_proof_status),
    "handover_proof_status": dict(handover_proof),
    "handover_safe_signals": {
        "proof_id": handover_safe.get("proof_id"),
        "proof_state": handover_safe.get("proof_state"),
        "proof_reset_allowed": handover_safe.get("proof_reset_allowed"),
        "proof_resume_allowed": handover_safe.get("proof_resume_allowed"),
        "safe_to_reset": handover_safe.get("safe_to_reset"),
        "safe_to_resume": handover_safe.get("safe_to_resume"),
    },
    "proof_mismatches": proof_mismatches,
    "successor_inheritance": successor_inheritance,
    "successor_inheritance_mismatches": successor_inheritance_mismatches,
    "artifacts": dict(artifacts),
}

if not ok:
    payload["phase"] = "alignment_check"
    if proof_mismatches:
        payload["error"] = {
            "code": "proof_alignment_mismatch",
            "message": "handover proof fields diverged from freshly projected successor-safe proof status",
        }
    else:
        payload["error"] = {
            "code": "successor_inheritance_mismatch",
            "message": "successor packet takeover inheritance is missing or stale for reset-ready handoff",
        }
    emit(payload, exit_code=1)

emit(payload, exit_code=0)
PY
