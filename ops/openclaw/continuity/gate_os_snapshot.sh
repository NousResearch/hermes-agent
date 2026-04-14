#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
REFRESH=0
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: gate_os_snapshot.sh [options]

Emit unified GateOS snapshot from continuity/queue/parity/web-capture surfaces.

Options:
  --refresh     Refresh continuity current before snapshot
  --json        Print JSON
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1; shift ;;
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

python3 - "$ROOT" "$REFRESH" "$JSON_OUT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
refresh = bool(int(sys.argv[2]))
json_out = bool(int(sys.argv[3]))

out_path = root / "state" / "continuity" / "latest" / "gate_os_latest.json"

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from continuity_policy import (
        DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC as _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
        continuity_now_contract_declared as _continuity_now_contract_declared,
        continuity_now_contract_expected_fields as _continuity_now_contract_expected_fields,
        continuity_now_contract_failclose_reasons as _continuity_now_contract_failclose_reasons,
        generation_pointer_core_failclose_reasons as _generation_pointer_core_failclose_reasons,
        is_severe_verify_gate_preflight_blocker as _is_severe_verify_gate_preflight_blocker,
        read_nonnegative_int_env as _read_nonnegative_int_env,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC = 21600

    def _read_nonnegative_int_env(name: str, *, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, str(int(default)))))
        except Exception:
            return int(default)

    def _continuity_now_contract_expected_fields(
        *,
        contract_obj: Any,
        source_refs: Any,
    ) -> tuple[str, str, str]:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}
        expected_sha = str(contract_map.get("sha256") or source_map.get("continuity_now_sha256") or "").strip()
        expected_generated_at = str(contract_map.get("generated_at") or "").strip()
        expected_generation = str(contract_map.get("coherence_build_generation_id") or "").strip()
        return expected_sha, expected_generated_at, expected_generation

    def _continuity_now_contract_declared(
        *,
        contract_obj: Any,
        source_refs: Any,
        require_sha_pin: bool,
    ) -> bool:
        if not require_sha_pin:
            return isinstance(contract_obj, dict)
        expected_sha, _, _ = _continuity_now_contract_expected_fields(
            contract_obj=contract_obj,
            source_refs=source_refs,
        )
        return bool(expected_sha)

    def _continuity_now_contract_failclose_reasons(
        *,
        contract_declared: Any,
        contract_path: pathlib.Path,
        expected_sha256: Any,
        expected_generated_at: Any,
        expected_coherence_build_generation_id: Any,
    ) -> tuple[List[str], Optional[str], Optional[Dict[str, Any]]]:
        declared = bool(contract_declared)
        if not declared:
            return [], None, None

        expected_sha = str(expected_sha256 or "").strip()
        expected_generated = str(expected_generated_at or "").strip()
        expected_generation = str(expected_coherence_build_generation_id or "").strip()

        if not contract_path.exists():
            return ["continuity_now_contract_missing"], None, None

        try:
            raw = contract_path.read_text(encoding="utf-8")
            actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("continuity_now_contract_not_object")

            reasons: List[str] = []
            if expected_sha and actual_sha != expected_sha:
                reasons.append("continuity_now_contract_sha_mismatch")

            actual_generated = str(payload.get("generated_at") or "").strip()
            if expected_generated and actual_generated != expected_generated:
                reasons.append("continuity_now_contract_generated_at_mismatch")

            actual_generation = str((((payload.get("coherence") or {}).get("build_generation_id") or "")).strip())
            if expected_generation and actual_generation != expected_generation:
                reasons.append("continuity_now_contract_generation_mismatch")

            return unique_preserve(reasons), actual_sha, payload
        except Exception:
            return ["continuity_now_contract_unreadable"], None, None

    def _generation_pointer_core_failclose_reasons(
        *,
        pointer_current_sha256: Any,
        current_sha256: Any,
        pointer_current_generated_at: Any,
        current_generated_at: Any,
        pointer_generation_id: Any,
        current_generation_id: Any,
    ) -> List[str]:
        reasons: List[str] = []

        pointer_current_sha = str(pointer_current_sha256 or "").strip()
        current_sha = str(current_sha256 or "").strip()
        pointer_current_ts = str(pointer_current_generated_at or "").strip()
        current_ts = str(current_generated_at or "").strip()
        pointer_generation = str(pointer_generation_id or "").strip()
        current_generation = str(current_generation_id or "").strip()

        if not pointer_current_sha:
            reasons.append("generation_pointer_missing_current_sha256")
        elif current_sha and pointer_current_sha != current_sha:
            reasons.append("generation_pointer_current_sha_mismatch")

        if not pointer_current_ts:
            reasons.append("generation_pointer_missing_current_generated_at")
        elif current_ts and pointer_current_ts != current_ts:
            reasons.append("generation_pointer_current_generated_at_mismatch")

        current_dt = parse_iso(current_ts)
        pointer_dt = parse_iso(pointer_current_ts)
        if current_dt is not None and pointer_dt is not None and pointer_dt < current_dt:
            reasons.append("generation_pointer_stale")

        if current_generation and not pointer_generation:
            reasons.append("generation_pointer_missing_generation_id")
        elif current_generation and pointer_generation and current_generation != pointer_generation:
            reasons.append("generation_pointer_generation_mismatch")

        return unique_preserve(reasons)

    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS = {"strict_autonomy_required_override_denied"}
    _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES = (
        "layered_health_gate:",
        "execution_supervisor_launch_readiness_severity_gate:",
        "execution_supervisor_probe_execution_gate:",
        "execution_supervisor_worker_health_canary_gate:",
    )

    def _is_severe_verify_gate_preflight_blocker(reason: Any) -> bool:
        blocker = str(reason or "").strip()
        if not blocker:
            return False
        if blocker in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKERS:
            return True
        return any(blocker.startswith(prefix) for prefix in _SEVERE_VERIFY_GATE_PREFLIGHT_BLOCKER_PREFIXES)

try:
    from continuity_now_paths import (
        CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON as _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON,
        DEFAULT_CONTINUITY_NOW_LATEST_REL as _CONTINUITY_NOW_LATEST_REL,
        continuity_now_contract_path_conflict_reason as _continuity_now_contract_path_conflict_reason,
        resolve_continuity_now_contract_path as _resolve_continuity_now_contract_path,
        resolve_continuity_now_evidence_path as _resolve_continuity_now_evidence_path,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _CONTINUITY_NOW_LATEST_REL = "state/continuity/latest/continuity_now_latest.json"
    _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON = "continuity_now_contract_path_conflict"

    def _resolve_continuity_now_contract_path(root_path: pathlib.Path, *, contract_obj: Any = None, source_refs: Any = None) -> pathlib.Path:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}
        path_txt = str(
            contract_map.get("path")
            or source_map.get("continuity_now")
            or _CONTINUITY_NOW_LATEST_REL
        ).strip() or _CONTINUITY_NOW_LATEST_REL
        path = pathlib.Path(path_txt)
        if not path.is_absolute():
            path = (pathlib.Path(root_path).resolve() / path).resolve()
        else:
            path = path.resolve()
        return path

    def _continuity_now_contract_path_conflict_reason(root_path: pathlib.Path, *, contract_obj: Any = None, source_refs: Any = None) -> str | None:
        contract_map = contract_obj if isinstance(contract_obj, dict) else {}
        source_map = source_refs if isinstance(source_refs, dict) else {}

        contract_raw = str(contract_map.get("path") or "").strip()
        source_raw = str(source_map.get("continuity_now") or "").strip()
        if not contract_raw or not source_raw:
            return None

        contract_path = pathlib.Path(contract_raw)
        if not contract_path.is_absolute():
            contract_path = (pathlib.Path(root_path).resolve() / contract_path).resolve()
        else:
            contract_path = contract_path.resolve()

        source_path = pathlib.Path(source_raw)
        if not source_path.is_absolute():
            source_path = (pathlib.Path(root_path).resolve() / source_path).resolve()
        else:
            source_path = source_path.resolve()

        if contract_path == source_path:
            return None
        return _CONTINUITY_NOW_CONTRACT_PATH_CONFLICT_REASON

    def _resolve_continuity_now_evidence_path(
        root_path: pathlib.Path,
        *,
        contract_obj: Any = None,
        source_refs: Any = None,
        raw_path: Any = None,
        fallback_rel: str = _CONTINUITY_NOW_LATEST_REL,
    ) -> str:
        raw_txt = str(raw_path or "").strip()
        if raw_txt:
            raw_path_obj = pathlib.Path(raw_txt)
            if raw_path_obj.is_absolute():
                resolved_root = pathlib.Path(root_path).resolve()
                try:
                    return str(raw_path_obj.resolve().relative_to(resolved_root))
                except Exception:
                    return str(raw_path_obj.resolve())
            return str(raw_path_obj)
        contract_path = _resolve_continuity_now_contract_path(
            pathlib.Path(root_path),
            contract_obj=contract_obj,
            source_refs=source_refs,
        )
        resolved_root = pathlib.Path(root_path).resolve()
        try:
            return str(contract_path.relative_to(resolved_root))
        except Exception:
            return str(contract_path)


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def clock_now_dt() -> dt.datetime:
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc)


def clock_now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_iso() -> str:
    return clock_now_iso()


def parse_iso(raw: Any):
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
    return out


def age_sec(raw: Any):
    ts = parse_iso(raw)
    if ts is None:
        return None
    return max(0, int((clock_now_dt() - ts).total_seconds()))


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_preserve(rows: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        txt = str(row or "").strip()
        if not txt or txt in seen:
            continue
        out.append(txt)
        seen.add(txt)
    return out


def read_reason_rows(raw: Any) -> List[str]:
    out: List[str] = []
    for item in raw or []:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def evaluate_generation_pointer_contract(
    *,
    current_obj: Dict[str, Any],
    now_obj: Dict[str, Any],
    current_sha: str,
) -> Dict[str, Any]:
    pointer_path = (root / "state" / "continuity" / "latest" / "continuity_read_pointer.json").resolve()
    source_refs = current_obj.get("source_refs") if isinstance(current_obj.get("source_refs"), dict) else {}
    raw_continuity_now_contract = current_obj.get("continuity_now_contract")
    continuity_now_contract = raw_continuity_now_contract if isinstance(raw_continuity_now_contract, dict) else {}

    contract_path = _resolve_continuity_now_contract_path(
        root,
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )

    expected_contract_sha, expected_contract_generated_at, expected_contract_generation = _continuity_now_contract_expected_fields(
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )

    contract_declared = _continuity_now_contract_declared(
        contract_obj=raw_continuity_now_contract,
        source_refs=source_refs,
        require_sha_pin=False,
    )
    path_conflict_reason = _continuity_now_contract_path_conflict_reason(
        root,
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )
    contract_failclose_reasons, contract_actual_sha, _ = _continuity_now_contract_failclose_reasons(
        contract_declared=contract_declared,
        contract_path=contract_path,
        expected_sha256=expected_contract_sha,
        expected_generated_at=expected_contract_generated_at,
        expected_coherence_build_generation_id=expected_contract_generation,
    )
    contract_failclose_reasons = unique_preserve(
        ([path_conflict_reason] if path_conflict_reason else [])
        + contract_failclose_reasons
    )
    continuity_now_contract_info: Dict[str, Any] = {
        "declared": contract_declared,
        "path": to_rel(contract_path),
        "expected_sha256": expected_contract_sha or None,
        "expected_generated_at": expected_contract_generated_at or None,
        "expected_coherence_build_generation_id": expected_contract_generation or None,
        "actual_sha256": contract_actual_sha,
    }

    out: Dict[str, Any] = {
        "path": to_rel(pointer_path),
        "present": pointer_path.exists(),
        "schema": None,
        "pointer_generation_id": None,
        "current_generation_id": None,
        "continuity_now_generation_id": None,
        "pointer_current_sha256": None,
        "current_sha256": current_sha,
        "pointer_current_generated_at": None,
        "current_generated_at": str(current_obj.get("generated_at") or "").strip() or None,
        "continuity_now_generated_at": str(now_obj.get("generated_at") or "").strip() or None,
        "failclose_reasons": [],
        "continuity_now_contract": continuity_now_contract_info,
    }

    if not pointer_path.exists():
        out["failclose_reasons"] = unique_preserve(contract_failclose_reasons)
        return out

    try:
        pointer_obj = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["failclose_reasons"] = unique_preserve(contract_failclose_reasons + ["generation_pointer_unreadable"])
        out["error"] = str(exc)
        return out

    if not isinstance(pointer_obj, dict):
        out["failclose_reasons"] = unique_preserve(contract_failclose_reasons + ["generation_pointer_not_object"])
        return out

    out["schema"] = pointer_obj.get("schema")
    if str(pointer_obj.get("schema") or "").strip() != "clawd.continuity.pointer.v1":
        out["failclose_reasons"].append("generation_pointer_schema_invalid")

    contract = pointer_obj.get("continuity_read_contract") if isinstance(pointer_obj.get("continuity_read_contract"), dict) else {}
    source_current = pointer_obj.get("source_current") if isinstance(pointer_obj.get("source_current"), dict) else {}

    pointer_generation = str(
        contract.get("coherence_build_generation_id")
        or pointer_obj.get("coherence_build_generation_id")
        or ""
    ).strip()
    current_generation = str((((current_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())
    now_generation = str((((now_obj.get("coherence") or {}).get("build_generation_id") or "")).strip())

    out["pointer_generation_id"] = pointer_generation or None
    out["current_generation_id"] = current_generation or None
    out["continuity_now_generation_id"] = now_generation or None

    pointer_current_sha = str(
        contract.get("continuity_current_sha256")
        or source_current.get("sha256")
        or ""
    ).strip()
    out["pointer_current_sha256"] = pointer_current_sha or None

    pointer_current_generated_at = str(
        contract.get("continuity_current_generated_at")
        or source_current.get("generated_at")
        or ""
    ).strip()
    out["pointer_current_generated_at"] = pointer_current_generated_at or None

    current_generated_at = str(current_obj.get("generated_at") or "").strip()

    out["failclose_reasons"] = unique_preserve(
        read_reason_rows(out.get("failclose_reasons"))
        + _generation_pointer_core_failclose_reasons(
            pointer_current_sha256=pointer_current_sha,
            current_sha256=current_sha,
            pointer_current_generated_at=pointer_current_generated_at,
            current_generated_at=current_generated_at,
            pointer_generation_id=pointer_generation,
            current_generation_id=current_generation,
        )
        + contract_failclose_reasons
    )
    return out


def run_json(cmd: List[str], *, phase: str) -> Dict[str, Any]:
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    raw_stdout = cp.stdout or ""
    parsed: Any = {}
    parse_exc: Optional[Exception] = None

    if raw_stdout.strip():
        try:
            parsed = json.loads(raw_stdout)
        except Exception as exc:
            parse_exc = exc

    if cp.returncode != 0:
        if parse_exc is None and isinstance(parsed, dict):
            return parsed
        if parse_exc is not None:
            raise SystemExit(
                f"gate_os_snapshot_{phase}_invalid_json_nonzero:{parse_exc}"
            )
        err = (cp.stderr or raw_stdout or "command_failed").strip()
        raise SystemExit(f"gate_os_snapshot_{phase}_failed:{err[:240]}")

    if parse_exc is not None:
        raise SystemExit(f"gate_os_snapshot_{phase}_invalid_json:{parse_exc}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"gate_os_snapshot_{phase}_invalid_json:not_object")
    return parsed


def gate_result(gate_id: str, status: str, severity: str, blocked: bool, code: str, summary: str, evidence: List[str]) -> Dict[str, Any]:
    ts = now_iso()
    return {
        "schema": "clawd.gate_result.v1",
        "gate_id": gate_id,
        "gate_version": "1.0.0",
        "run": {
            "run_id": f"gr_{gate_id.replace('.', '_')}_{ts.replace(':', '').replace('-', '')}",
            "timestamp": ts,
            "host_id": os.environ.get("HOSTNAME"),
            "toolchain_fingerprint": "openclaw.localfirst.gateos.v1",
        },
        "status": status,
        "severity": severity,
        "mutation": {
            "blocked": blocked,
            "block_scope": "global" if blocked else None,
        },
        "reason": {
            "code": code,
            "summary": summary,
            "details": None,
        },
        "evidence_refs": [{"ref": ref} for ref in evidence],
    }

if refresh:
    run_json(
        [str(root / "ops" / "openclaw" / "continuity" / "continuity_current.sh"), "--refresh", "--json"],
        phase="continuity_current_refresh",
    )

current = run_json(
    [str(root / "ops" / "openclaw" / "continuity" / "continuity_current.sh"), "--json"],
    phase="continuity_current",
)
now_obj = run_json(
    [str(root / "ops" / "openclaw" / "continuity" / "continuity_now.sh"), "--json"],
    phase="continuity_now",
)

results: List[Dict[str, Any]] = []
current_path = (root / "state" / "continuity" / "current.json").resolve()
if current_path.exists():
    try:
        current_source_sha = sha256_file(current_path)
    except Exception:
        current_source_sha = hashlib.sha256(json.dumps(current, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
else:
    current_source_sha = hashlib.sha256(json.dumps(current, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

generation_pointer = evaluate_generation_pointer_contract(
    current_obj=current,
    now_obj=now_obj,
    current_sha=current_source_sha,
)
generation_failclose_reasons = read_reason_rows(generation_pointer.get("failclose_reasons"))
continuity_now_contract_evidence_path = _resolve_continuity_now_evidence_path(
    root,
    raw_path=(((generation_pointer.get("continuity_now_contract") or {}).get("path")) if isinstance(generation_pointer.get("continuity_now_contract"), dict) else None),
    fallback_rel=_CONTINUITY_NOW_LATEST_REL,
)

read_contract_evidence = unique_preserve(
    [
        to_rel(current_path),
        str(generation_pointer.get("path") or "state/continuity/latest/continuity_read_pointer.json"),
        str((((generation_pointer.get("continuity_now_contract") or {}).get("path")) or "")).strip(),
    ]
)
if generation_failclose_reasons:
    results.append(
        gate_result(
            "continuity.read_contract",
            "fail",
            "hard_fail",
            True,
            "READ_CONTRACT_INTEGRITY_FAIL",
            f"generation/read contract mismatch reasons={generation_failclose_reasons}",
            read_contract_evidence,
        )
    )
else:
    results.append(
        gate_result(
            "continuity.read_contract",
            "pass",
            "info",
            False,
            "READ_CONTRACT_INTEGRITY_OK",
            "generation/read contract coherent",
            read_contract_evidence,
        )
    )

readiness = str(current.get("readiness") or "UNKNOWN")
effective_readiness = readiness
if generation_failclose_reasons and readiness in {"READY", "READY_WITH_DEBT"}:
    effective_readiness = "RECONCILE_REQUIRED"

mutation_allowed = ((current.get("mutation_gate") or {}).get("status") == "allowed") and not generation_failclose_reasons

if effective_readiness == "READY":
    results.append(gate_result("continuity.readiness", "pass", "info", False, "READY", "Continuity readiness is READY", ["state/continuity/current.json"]))
elif effective_readiness == "READY_WITH_DEBT":
    results.append(gate_result("continuity.readiness", "warn", "soft_fail", True, "READY_WITH_DEBT", "Read-only resume allowed; mutation blocked by debt", ["state/continuity/current.json"]))
else:
    results.append(gate_result("continuity.readiness", "fail", "hard_fail", True, "NOT_READY", f"Continuity readiness={effective_readiness}", ["state/continuity/current.json"]))

verify = now_obj.get("verify") or {}
verify_status = str(verify.get("status") or "UNKNOWN").upper()
verify_gate_preflight = verify.get("gate_preflight") if isinstance(verify.get("gate_preflight"), dict) else {}
verify_gate_predicted = verify_gate_preflight.get("predicted_gate") if isinstance(verify_gate_preflight.get("predicted_gate"), dict) else {}
verify_gate_predicted_blocker = str(verify_gate_predicted.get("predicted_blocker_reason") or "").strip()
verify_gate_ready_to_run = verify_gate_predicted.get("ready_to_run") if isinstance(verify_gate_predicted.get("ready_to_run"), bool) else None
verify_gate_preflight_available = bool(verify_gate_preflight.get("available") is True)
verify_gate_status_evidence = (
    verify_gate_preflight.get("status_evidence_gate")
    if isinstance(verify_gate_preflight.get("status_evidence_gate"), dict)
    else {}
)
verify_status_evidence_failure_reason = str(verify_gate_status_evidence.get("failure_reason") or "").strip()
verify_status_evidence_untrusted = bool(verify_status == "READY" and verify_status_evidence_failure_reason)

if verify_status == "READY" and not verify_status_evidence_untrusted:
    verify_required_status = "pass"
    verify_required_severity = "info"
    verify_required_blocked = False
    verify_required_code = "READY"
    verify_required_summary = "verify status=READY"
elif verify_status_evidence_untrusted:
    verify_required_status = "fail"
    verify_required_severity = "hard_fail"
    verify_required_blocked = True
    verify_required_code = "VERIFY_STATUS_EVIDENCE_UNTRUSTED"
    verify_required_summary = f"verify status READY but status evidence gate failed reason={verify_status_evidence_failure_reason}"
else:
    verify_required_status = "fail"
    verify_required_severity = "hard_fail"
    verify_required_blocked = True
    verify_required_code = verify_status or "UNKNOWN"
    verify_required_summary = f"verify status={verify_status}"

results.append(
    gate_result(
        "continuity.verify.required",
        verify_required_status,
        verify_required_severity,
        verify_required_blocked,
        verify_required_code,
        verify_required_summary,
        ["state/continuity/latest/verify_last.json", continuity_now_contract_evidence_path],
    )
)

def is_severe_verify_gate_preflight_blocker(reason: Optional[str]) -> bool:
    return bool(_is_severe_verify_gate_preflight_blocker(reason))

if not verify_gate_preflight_available:
    results.append(
        gate_result(
            "continuity.verify.preflight",
            "warn",
            "soft_fail",
            False,
            "VERIFY_PREFLIGHT_UNAVAILABLE",
            "verify-gate preflight unavailable in continuity_now",
            [continuity_now_contract_evidence_path],
        )
    )
elif verify_gate_predicted_blocker:
    severe_preflight_blocker = is_severe_verify_gate_preflight_blocker(verify_gate_predicted_blocker)
    results.append(
        gate_result(
            "continuity.verify.preflight",
            "fail" if severe_preflight_blocker else "warn",
            "hard_fail" if severe_preflight_blocker else "soft_fail",
            False,
            "VERIFY_PREFLIGHT_BLOCKER_PREDICTED_SEVERE" if severe_preflight_blocker else "VERIFY_PREFLIGHT_BLOCKER_PREDICTED",
            f"verify-gate preflight predicted blocker={verify_gate_predicted_blocker}",
            [continuity_now_contract_evidence_path],
        )
    )
elif verify_gate_ready_to_run is False:
    results.append(
        gate_result(
            "continuity.verify.preflight",
            "warn",
            "soft_fail",
            False,
            "VERIFY_PREFLIGHT_NOT_READY",
            "verify-gate preflight indicates not ready_to_run",
            [continuity_now_contract_evidence_path],
        )
    )
else:
    results.append(
        gate_result(
            "continuity.verify.preflight",
            "pass",
            "info",
            False,
            "VERIFY_PREFLIGHT_READY",
            "verify-gate preflight ready_to_run",
            [continuity_now_contract_evidence_path],
        )
    )

parity = now_obj.get("parity") or {}
parity_due = bool(parity.get("due")) or not bool(parity.get("fresh", True))
results.append(
    gate_result(
        "parity.freshness",
        "warn" if parity_due else "pass",
        "soft_fail" if parity_due else "info",
        parity_due,
        "PARITY_STALE" if parity_due else "PARITY_FRESH",
        "parity stale beyond ttl" if parity_due else "parity fresh",
        ["state/architecture/competitive_parity/dashboard/latest.json"],
    )
)

queue = now_obj.get("queue") or {}
stale_locks = int(queue.get("stale_active_file_lock_count") or 0)
running = int((queue.get("status_counts") or {}).get("RUNNING") or 0)
results.append(
    gate_result(
        "queue.lock_lease_sanity",
        "fail" if stale_locks > 0 else "pass",
        "hard_fail" if stale_locks > 0 else "info",
        stale_locks > 0,
        "STALE_LOCKS_PRESENT" if stale_locks > 0 else "LOCKS_HEALTHY",
        f"stale_active_locks={stale_locks}; running_tasks={running}",
        ["state/continuity/continuity_os.sqlite"],
    )
)

latest_run_path = root / "ops" / "web_capture" / "artifacts" / "latest_run.json"
if latest_run_path.exists():
    try:
        meta = json.loads(latest_run_path.read_text(encoding="utf-8"))
        run_id = str(meta.get("run_id") or "")
        idx_path = root / "ops" / "web_capture" / "artifacts" / run_id / "index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            s = str(idx.get("status") or "unknown")
            results.append(
                gate_result(
                    "web.artifact.latest",
                    "pass" if s == "ok" else ("warn" if s == "blocked" else "fail"),
                    "info" if s == "ok" else ("soft_fail" if s == "blocked" else "hard_fail"),
                    s != "ok",
                    f"WEB_CAPTURE_{s.upper()}",
                    f"latest web capture status={s}",
                    [f"ops/web_capture/artifacts/{run_id}/index.json"],
                )
            )
    except Exception:
        pass

web_capture = now_obj.get("web_capture") or {}
web_operator_domains = int(web_capture.get("operator_action_required_domains") or 0)
web_cooldown_domains = int(web_capture.get("cooldown_active_domains") or 0)
web_guard_refs = []
for row in list(web_capture.get("domains") or [])[:6]:
    if isinstance(row, dict):
        if row.get("state_path"):
            web_guard_refs.append(str(row.get("state_path")))
        if row.get("operator_action_required") and row.get("operator_contract_json"):
            web_guard_refs.append(str(row.get("operator_contract_json")))

results.append(
    gate_result(
        "web.domain_guard.operator_contract",
        "warn" if web_operator_domains > 0 else "pass",
        "soft_fail" if web_operator_domains > 0 else "info",
        False,
        "WEB_OPERATOR_LOGIN_REQUIRED" if web_operator_domains > 0 else "WEB_OPERATOR_LOGIN_CLEAR",
        f"domains_requiring_operator_login={web_operator_domains}",
        web_guard_refs or ["state/continuity/latest"],
    )
)

results.append(
    gate_result(
        "web.domain_guard.backoff_window",
        "warn" if web_cooldown_domains > 0 else "pass",
        "soft_fail" if web_cooldown_domains > 0 else "info",
        False,
        "WEB_DOMAIN_BACKOFF_ACTIVE" if web_cooldown_domains > 0 else "WEB_DOMAIN_BACKOFF_CLEAR",
        f"domains_in_backoff={web_cooldown_domains}",
        web_guard_refs or ["state/continuity/latest"],
    )
)

scheduler = web_capture.get("scheduler") if isinstance(web_capture.get("scheduler"), dict) else {}
scheduler_freshness_limit_sec = _read_nonnegative_int_env(
    "OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC",
    default=_DEFAULT_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC,
)

scheduler_updated_at = scheduler.get("updated_at")
scheduler_age_sec = age_sec(scheduler_updated_at)
scheduler_exists = scheduler.get("state_exists") is True
scheduler_contract_valid = scheduler.get("contract_state_valid")
if scheduler_contract_valid is None:
    scheduler_contract_valid = True if scheduler_exists else None
scheduler_fresh = scheduler.get("fresh")
if scheduler_fresh is None and scheduler_age_sec is not None:
    scheduler_fresh = scheduler_age_sec <= scheduler_freshness_limit_sec

scheduler_evidence = [
    str(scheduler.get("state_path") or "state/continuity/latest/web_capture_scheduler_state.json"),
]
if scheduler.get("contract_schema_path"):
    scheduler_evidence.append(str(scheduler.get("contract_schema_path")))

if not scheduler_exists:
    results.append(
        gate_result(
            "web.scheduler.contract",
            "fail",
            "hard_fail",
            True,
            "WEB_SCHEDULER_STATE_MISSING",
            "web scheduler state artifact missing",
            scheduler_evidence,
        )
    )
elif scheduler_contract_valid is False:
    results.append(
        gate_result(
            "web.scheduler.contract",
            "fail",
            "hard_fail",
            True,
            "WEB_SCHEDULER_CONTRACT_INVALID",
            f"web scheduler contract invalid errors={scheduler.get('contract_validation_errors') or []}",
            scheduler_evidence,
        )
    )
elif scheduler_fresh is False:
    results.append(
        gate_result(
            "web.scheduler.contract",
            "warn",
            "soft_fail",
            False,
            "WEB_SCHEDULER_STALE",
            f"web scheduler stale age_sec={scheduler_age_sec} limit_sec={scheduler_freshness_limit_sec}",
            scheduler_evidence,
        )
    )
else:
    results.append(
        gate_result(
            "web.scheduler.contract",
            "pass",
            "info",
            False,
            "WEB_SCHEDULER_GOVERNED",
            f"web scheduler governed status={scheduler.get('selection_status')} age_sec={scheduler_age_sec}",
            scheduler_evidence,
        )
    )

summary = {
    "pass": sum(1 for r in results if r.get("status") == "pass"),
    "warn": sum(1 for r in results if r.get("status") == "warn"),
    "fail": sum(1 for r in results if r.get("status") == "fail"),
    "mutation_allowed": mutation_allowed,
}

payload = {
    "schema": "clawd.gate_os.snapshot.v1",
    "generated_at": now_iso(),
    "workspace_id": "clawd-architect",
    "summary": summary,
    "results": results,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
tmp = out_path.with_name(out_path.name + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, out_path)

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(f"GATE OS SNAPSHOT: pass={summary['pass']} warn={summary['warn']} fail={summary['fail']} mutation_allowed={summary['mutation_allowed']}")
PY
