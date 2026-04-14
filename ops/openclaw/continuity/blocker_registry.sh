#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CURRENT_PATH="$ROOT/state/continuity/current.json"
OUT_PATH="$ROOT/state/continuity/latest/blocker_registry.json"
JSON_OUT=0
MAX_AGE_SEC_RAW="${OPENCLAW_BLOCKER_REGISTRY_MAX_CURRENT_AGE_SEC:-${OPENCLAW_CONTINUITY_CURRENT_CACHE_TTL_SEC:-300}}"

usage() {
  cat <<'EOF'
Usage: blocker_registry.sh [options]

Generate canonical blocker registry as a freshness-bounded derivative of continuity/current truth.

Options:
  --current <path>      Source continuity current JSON.
                        Default: state/continuity/current.json
  --out <path>          Output blocker registry JSON.
                        Default: state/continuity/latest/blocker_registry.json
  --max-age-sec <sec>   Max allowed age for source continuity/current before fail-close BLOCKER.
                        Default: OPENCLAW_BLOCKER_REGISTRY_MAX_CURRENT_AGE_SEC
                                 or OPENCLAW_CONTINUITY_CURRENT_CACHE_TTL_SEC
                                 or 300
  --json                Print machine JSON output.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --current)
      CURRENT_PATH="${2:-}"; shift 2 ;;
    --out)
      OUT_PATH="${2:-}"; shift 2 ;;
    --max-age-sec)
      MAX_AGE_SEC_RAW="${2:-}"; shift 2 ;;
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

python3 - "$ROOT" "$CURRENT_PATH" "$OUT_PATH" "$MAX_AGE_SEC_RAW" "$JSON_OUT" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
current_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
max_age_raw = sys.argv[4]
json_out = bool(int(sys.argv[5]))

if not current_path.is_absolute():
    current_path = (root / current_path).resolve()
if not out_path.is_absolute():
    out_path = (root / out_path).resolve()

continuity_dir = (root / "ops" / "openclaw" / "continuity").resolve()
schema_path = (root / "ops" / "openclaw" / "architecture" / "schemas" / "blocker_registry.schema.json").resolve()
_RESET_READY_REFRESH_LATEST_REL = "state/continuity/latest/reset_ready_refresh_latest.json"
_DOCTRINE_DRIFT_REGISTRY_REL = "state/continuity/latest/doctrine_drift_registry.json"
_CURRENT_PUBLISH_LOCK_OWNER_REL = "state/continuity/latest/current_publish.lock.owner.json"
sys.path.insert(0, str(continuity_dir))

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None

try:
    from continuity_policy import (
        CURRENT_PUBLISH_LOCK_OWNER_REL as _CURRENT_PUBLISH_LOCK_OWNER_REL,
        PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON as _PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON,
        PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON as _PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON,
        PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON as _PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON,
        continuity_now_contract_declared as _continuity_now_contract_declared,
        continuity_now_contract_expected_fields as _continuity_now_contract_expected_fields,
        continuity_now_contract_failclose_reasons as _continuity_now_contract_failclose_reasons,
        generation_pointer_core_failclose_reasons as _generation_pointer_core_failclose_reasons,
        project_reset_ready_refresh_blocker_warning_metadata as _project_reset_ready_refresh_blocker_warning_metadata,
        project_reset_ready_refresh_posture as _project_reset_ready_refresh_posture,
    )
except Exception:  # pragma: no cover - sidecar fixtures may omit helper module
    _CURRENT_PUBLISH_LOCK_OWNER_REL = "state/continuity/latest/current_publish.lock.owner.json"
    _PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON = "continuity_current_publish_lock_wait_budget_exceeded"
    _PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON = "continuity_current_publish_lock_hold_budget_exceeded"
    _PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON = "continuity_current_publish_lock_owner_not_alive"
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

    def _project_reset_ready_refresh_posture(
        *,
        surface: Any = None,
        latest_payload: Any = None,
        path: Any = None,
        sha256: Any = None,
        present: Any = None,
        now_ts: Any = None,
        freshness_max_age_sec: Any = None,
    ) -> Dict[str, Any]:
        surface_map = surface if isinstance(surface, dict) else {}
        latest_map = latest_payload if isinstance(latest_payload, dict) else {}

        path_text = str(path or surface_map.get("path") or "").strip()
        sha_text = str(sha256 or surface_map.get("sha256") or "").strip() or None

        present_value: bool
        if isinstance(present, bool):
            present_value = present
        else:
            present_value = bool(surface_map.get("present") is True or bool(latest_map))

        ok = surface_map.get("ok") if isinstance(surface_map.get("ok"), bool) else None
        if ok is None and isinstance(latest_map.get("ok"), bool):
            ok = latest_map.get("ok")

        phase = str(surface_map.get("phase") or latest_map.get("phase") or "").strip() or None
        if phase is None and ok is True:
            phase = "complete"

        partial_refresh = surface_map.get("partial_refresh") if isinstance(surface_map.get("partial_refresh"), dict) else {}
        if not partial_refresh and isinstance(latest_map.get("partial_refresh"), dict):
            partial_refresh = latest_map.get("partial_refresh")

        def _partial_flag(name: str) -> bool | None:
            raw_value = partial_refresh.get(name)
            return raw_value if isinstance(raw_value, bool) else None

        partial_current = _partial_flag("current_refreshed")
        partial_proof = _partial_flag("proof_refreshed")
        partial_handover = _partial_flag("handover_refreshed")

        explicit_partial_failure = surface_map.get("partial_failure")
        if isinstance(explicit_partial_failure, bool):
            partial_failure = explicit_partial_failure
        else:
            partial_failure = bool(
                present_value
                and any(value is False for value in [partial_current, partial_proof, partial_handover])
            )

        error_code = str(
            surface_map.get("error_code")
            or (((latest_map.get("error") or {}).get("code")) if isinstance(latest_map.get("error"), dict) else "")
            or ""
        ).strip() or None

        explicit_degraded = surface_map.get("degraded")
        if isinstance(explicit_degraded, bool):
            degraded = explicit_degraded
        else:
            degraded = bool(present_value and (ok is False or partial_failure))

        generated_at = str(surface_map.get("generated_at") or latest_map.get("generated_at") or "").strip() or None

        def _coerce_nonnegative_int(raw: Any) -> Optional[int]:
            if isinstance(raw, bool):
                return None
            try:
                return max(0, int(raw))
            except Exception:
                return None

        freshness_limit_sec = _coerce_nonnegative_int(surface_map.get("freshness_limit_sec"))
        if freshness_limit_sec is None:
            freshness_limit_sec = _coerce_nonnegative_int(freshness_max_age_sec)
        if freshness_limit_sec is None:
            freshness_limit_sec = 21600

        age_sec = _coerce_nonnegative_int(surface_map.get("age_sec"))
        fresh = surface_map.get("fresh") if isinstance(surface_map.get("fresh"), bool) else None
        stale = surface_map.get("stale") if isinstance(surface_map.get("stale"), bool) else None

        if fresh is None and isinstance(stale, bool):
            fresh = not stale
        if stale is None and isinstance(fresh, bool):
            stale = not fresh

        if stale is None:
            stale = fresh is False

        status = "missing"
        if present_value:
            if degraded:
                status = "degraded"
            elif ok is True:
                status = "ok"
            else:
                status = "present"

        recommended_action = None
        if degraded or stale:
            recommended_action = "rerun_reset_ready_refresh"
        elif present_value:
            recommended_action = "inspect_reset_ready_refresh_result"

        return {
            "path": path_text,
            "sha256": sha_text,
            "generated_at": generated_at,
            "present": present_value,
            "status": status,
            "ok": ok,
            "phase": phase,
            "error_code": error_code,
            "freshness_limit_sec": freshness_limit_sec,
            "age_sec": age_sec,
            "fresh": fresh,
            "stale": stale,
            "partial_refresh": {
                "current_refreshed": partial_current,
                "proof_refreshed": partial_proof,
                "handover_refreshed": partial_handover,
            },
            "degraded": degraded,
            "partial_failure": partial_failure,
            "action_required": bool(degraded or stale),
            "recommended_action": recommended_action,
        }

    def _project_reset_ready_refresh_blocker_warning_metadata(
        *,
        posture: Any = None,
    ) -> Optional[Dict[str, Any]]:
        posture_map = posture if isinstance(posture, dict) else {}
        if posture_map.get("present") is not True:
            return None

        return {
            "recommended_action": str(posture_map.get("recommended_action") or "").strip() or None,
            "action_required": posture_map.get("action_required") is True,
            "context": {
                "status": posture_map.get("status"),
                "ok": posture_map.get("ok"),
                "phase": posture_map.get("phase"),
                "error_code": posture_map.get("error_code"),
                "partial_failure": posture_map.get("partial_failure"),
                "generated_at": posture_map.get("generated_at"),
                "freshness_limit_sec": posture_map.get("freshness_limit_sec"),
                "age_sec": posture_map.get("age_sec"),
                "fresh": posture_map.get("fresh"),
                "stale": posture_map.get("stale"),
            },
        }

_CURRENT_PUBLISH_LOCK_OWNER_REL = str(_CURRENT_PUBLISH_LOCK_OWNER_REL or "state/continuity/latest/current_publish.lock.owner.json").strip() or "state/continuity/latest/current_publish.lock.owner.json"

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


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return clock_now_dt().replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    return out


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_int(raw: Any) -> Optional[int]:
    try:
        return int(raw)
    except Exception:
        return None


def _coerce_float(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _pid_alive(pid: Optional[int]) -> Optional[bool]:
    if pid is None or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_reason_rows(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return unique_preserve([str(x).strip() for x in value if str(x).strip()])


def evaluate_generation_pointer_contract(
    *,
    current_obj: Dict[str, Any],
    current_sha: str,
    current_generated_at: Any,
) -> Dict[str, Any]:
    pointer_path = (root / "state" / "continuity" / "latest" / "continuity_read_pointer.json").resolve()
    out: Dict[str, Any] = {
        "path": to_rel(pointer_path),
        "present": pointer_path.exists(),
        "source": "state/continuity/latest/continuity_read_pointer.json",
        "schema": None,
        "pointer_generation_id": None,
        "current_generation_id": None,
        "pointer_current_sha256": None,
        "current_sha256": current_sha,
        "pointer_current_generated_at": None,
        "current_generated_at": str(current_generated_at or "").strip() or None,
        "continuity_now_contract": None,
        "failclose_reasons": [],
    }

    if not pointer_path.exists():
        return out

    try:
        pointer_obj = load_json(pointer_path)
    except Exception as exc:
        out["failclose_reasons"] = ["generation_pointer_unreadable"]
        out["error"] = str(exc)
        return out

    if not isinstance(pointer_obj, dict):
        out["failclose_reasons"] = ["generation_pointer_not_object"]
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
    out["pointer_generation_id"] = pointer_generation or None
    out["current_generation_id"] = current_generation or None

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

    current_generated_txt = str(current_generated_at or "").strip()
    out["failclose_reasons"] = unique_preserve(
        (out.get("failclose_reasons") or [])
        + _generation_pointer_core_failclose_reasons(
            pointer_current_sha256=pointer_current_sha,
            current_sha256=current_sha,
            pointer_current_generated_at=pointer_current_generated_at,
            current_generated_at=current_generated_txt,
            pointer_generation_id=pointer_generation,
            current_generation_id=current_generation,
        )
    )

    continuity_now_contract = current_obj.get("continuity_now_contract") if isinstance(current_obj.get("continuity_now_contract"), dict) else {}
    source_refs = current_obj.get("source_refs") if isinstance(current_obj.get("source_refs"), dict) else {}

    expected_now_sha, expected_now_generated_at, expected_now_generation = _continuity_now_contract_expected_fields(
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )
    path_conflict_reason = _continuity_now_contract_path_conflict_reason(
        root,
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )

    # Treat continuity_now contract as fail-close only when an explicit
    # content pin (sha256) is present. Lightweight current payloads may
    # carry a continuity_now path hint without a pinned artifact.
    contract_declared = _continuity_now_contract_declared(
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
        require_sha_pin=True,
    )
    contract_path = _resolve_continuity_now_contract_path(
        root,
        contract_obj=continuity_now_contract,
        source_refs=source_refs,
    )

    contract_info: Dict[str, Any] = {
        "declared": contract_declared,
        "path": to_rel(contract_path),
        "expected_sha256": expected_now_sha or None,
        "expected_generated_at": expected_now_generated_at or None,
        "expected_coherence_build_generation_id": expected_now_generation or None,
        "actual_sha256": None,
    }

    contract_reasons, contract_actual_sha, _ = _continuity_now_contract_failclose_reasons(
        contract_declared=contract_declared,
        contract_path=contract_path,
        expected_sha256=expected_now_sha,
        expected_generated_at=expected_now_generated_at,
        expected_coherence_build_generation_id=expected_now_generation,
    )
    contract_reasons = unique_preserve(
        ([path_conflict_reason] if path_conflict_reason else [])
        + contract_reasons
    )
    contract_info["actual_sha256"] = contract_actual_sha
    out["failclose_reasons"].extend(contract_reasons)

    out["continuity_now_contract"] = contract_info
    out["failclose_reasons"] = unique_preserve(out.get("failclose_reasons") or [])
    return out


def evaluate_publish_lock_posture(
    *,
    source_refs: Dict[str, Any],
) -> Dict[str, Any]:
    raw_path = str(source_refs.get("current_publish_lock_owner") or _CURRENT_PUBLISH_LOCK_OWNER_REL).strip() or _CURRENT_PUBLISH_LOCK_OWNER_REL
    owner_path = pathlib.Path(raw_path)
    if not owner_path.is_absolute():
        owner_path = (root / owner_path).resolve()
    else:
        owner_path = owner_path.resolve()

    out: Dict[str, Any] = {
        "path": to_rel(owner_path),
        "present": owner_path.exists(),
        "schema": None,
        "status": "missing",
        "owner_pid": None,
        "owner_alive": None,
        "owner_token": None,
        "owner_started_at": None,
        "owner_command": None,
        "owner_host": None,
        "owner_age_sec": None,
        "lock_wait_sec": None,
        "lock_hold_warn_sec": None,
        "owner_exceeds_wait_budget": None,
        "owner_exceeds_lock_hold_warn": None,
        "recommended_action": None,
        "action_required": False,
        "inspect_command": f"cat {to_rel(owner_path)}",
    }

    if not owner_path.exists():
        return out

    try:
        payload = load_json(owner_path)
    except Exception as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)
        out["recommended_action"] = "inspect_current_publish_lock_owner"
        out["action_required"] = True
        return out

    if not isinstance(payload, dict):
        out["status"] = "invalid"
        out["recommended_action"] = "inspect_current_publish_lock_owner"
        out["action_required"] = True
        return out

    out["schema"] = payload.get("schema")
    out["owner_token"] = str(payload.get("owner_token") or "").strip() or None
    out["owner_started_at"] = str(payload.get("owner_started_at") or "").strip() or None
    out["owner_command"] = str(payload.get("owner_command") or "").strip()[:240] or None
    out["owner_host"] = str(payload.get("owner_host") or "").strip() or None

    owner_pid = _coerce_int(payload.get("owner_pid"))
    out["owner_pid"] = owner_pid
    out["owner_alive"] = _pid_alive(owner_pid)

    age_dt = parse_iso(payload.get("owner_started_at")) or parse_iso(payload.get("updated_at"))
    if age_dt is not None:
        out["owner_age_sec"] = max(0, int((clock_now_dt() - age_dt).total_seconds()))

    lock_wait_sec = _coerce_float(payload.get("lock_wait_sec"))
    if lock_wait_sec is not None and lock_wait_sec >= 0:
        out["lock_wait_sec"] = lock_wait_sec

    lock_hold_warn_sec = _coerce_float(payload.get("lock_hold_warn_sec"))
    if lock_hold_warn_sec is not None and lock_hold_warn_sec >= 0:
        out["lock_hold_warn_sec"] = lock_hold_warn_sec

    owner_age_sec = out.get("owner_age_sec")
    if isinstance(owner_age_sec, int) and out.get("lock_wait_sec") is not None:
        out["owner_exceeds_wait_budget"] = bool(float(owner_age_sec) >= float(out["lock_wait_sec"]))
    if isinstance(owner_age_sec, int) and out.get("lock_hold_warn_sec") is not None:
        out["owner_exceeds_lock_hold_warn"] = bool(float(owner_age_sec) >= float(out["lock_hold_warn_sec"]))

    owner_not_alive = out.get("owner_alive") is False

    out["recommended_action"] = "inspect_current_publish_lock_owner"
    out["action_required"] = bool(
        owner_not_alive
        or out.get("owner_exceeds_wait_budget") is True
        or out.get("owner_exceeds_lock_hold_warn") is True
    )

    # Fail-closed truthfulness: dead owner must dominate publish-lock posture
    # even when age budgets are also exceeded. Otherwise stale owner sidecars
    # can be misclassified as merely long-held active locks under pressure.
    if owner_not_alive:
        out["status"] = "owner_not_alive"
    elif out.get("owner_exceeds_lock_hold_warn") is True:
        out["status"] = "hold_budget_exceeded"
    elif out.get("owner_exceeds_wait_budget") is True:
        out["status"] = "wait_budget_exceeded"
    else:
        out["status"] = "active"

    return out


def reset_ready_refresh_evidence_route_contract(*, severity: str) -> Dict[str, bool]:
    """Local-only evidence routing contract for reset-ready-refresh reasons.

    The blocker_registry intentionally owns blocker-vs-warn reset evidence
    assembly. Keep this contract explicit here (instead of lifting to shared
    policy) because only blocker_registry knows row-severity semantics.
    """

    severity_value = str(severity or "").strip().lower()
    if severity_value == "blocker":
        return {
            "include_mutation_gate": True,
            "include_supporting_drifts_when_present": False,
        }
    if severity_value == "warn":
        return {
            "include_mutation_gate": False,
            "include_supporting_drifts_when_present": True,
        }
    raise ValueError(f"unsupported reset_ready_refresh evidence severity: {severity}")


def reset_ready_refresh_evidence_refs(
    *,
    reset_ready_refresh_path: str,
    reset_ready_refresh_current_present: bool,
    include_mutation_gate: bool,
    include_supporting_drifts_when_present: bool,
) -> List[str]:
    """Canonical reset-ready-refresh evidence anchors for blocker/warn rows.

    Keep routing local to blocker_registry because blocker-vs-warn row assembly
    semantics intentionally diverge even when they share base evidence anchors.
    """

    evidence = [
        "state/continuity/current.json#reset_ready_refresh"
        if reset_ready_refresh_current_present
        else "state/continuity/current.json#drifts",
        str(reset_ready_refresh_path or "").strip() or _RESET_READY_REFRESH_LATEST_REL,
    ]
    if include_supporting_drifts_when_present and reset_ready_refresh_current_present:
        evidence.append("state/continuity/current.json#drifts")
    if include_mutation_gate:
        evidence.append("state/continuity/current.json#mutation_gate")
    return unique_preserve(evidence)


def blocker_evidence_for_reason(
    reason: str,
    *,
    continuity_now_contract_path: str,
    reset_ready_refresh_path: str,
    reset_ready_refresh_current_present: bool,
    publish_lock_owner_path: str,
) -> List[str]:
    if reason == "continuity_current_stale":
        return ["state/continuity/current.json#generated_at"]
    if reason.startswith("readiness:"):
        return ["state/continuity/current.json#readiness", "state/continuity/current.json#mutation_gate"]
    if reason.startswith("reset_ready_refresh_"):
        route_contract = reset_ready_refresh_evidence_route_contract(severity="blocker")
        return reset_ready_refresh_evidence_refs(
            reset_ready_refresh_path=reset_ready_refresh_path,
            reset_ready_refresh_current_present=reset_ready_refresh_current_present,
            include_mutation_gate=route_contract["include_mutation_gate"],
            include_supporting_drifts_when_present=route_contract["include_supporting_drifts_when_present"],
        )
    if reason.startswith("continuity_current_publish_lock_"):
        return unique_preserve([
            "state/continuity/current.json#generated_at",
            str(publish_lock_owner_path or "").strip() or _CURRENT_PUBLISH_LOCK_OWNER_REL,
        ])
    if reason.startswith("generation_pointer_"):
        return [
            "state/continuity/current.json#coherence.build_generation_id",
            "state/continuity/latest/continuity_read_pointer.json",
        ]
    if reason.startswith("continuity_now_contract_") or reason == "continuity_now_contract_missing":
        continuity_now_evidence = str(continuity_now_contract_path or "").strip() or _CONTINUITY_NOW_LATEST_REL
        return [
            "state/continuity/current.json#continuity_now_contract",
            continuity_now_evidence,
        ]
    if "pointer" in reason or "truth_anchor" in reason:
        return ["state/continuity/current.json#truth_anchor", "state/continuity/current.json#mutation_gate"]
    if reason.startswith("coherence") or reason.startswith("policy"):
        return ["state/continuity/current.json#coherence", "state/continuity/current.json#mutation_gate"]
    return ["state/continuity/current.json#mutation_gate", "state/continuity/current.json#drifts"]


def warning_evidence_for_reason(
    reason: str,
    *,
    reset_ready_refresh_path: str,
    reset_ready_refresh_current_present: bool,
    publish_lock_owner_path: str,
    doctrine_drift_registry_path: str,
) -> List[str]:
    if reason.startswith("reset_ready_refresh_"):
        route_contract = reset_ready_refresh_evidence_route_contract(severity="warn")
        return reset_ready_refresh_evidence_refs(
            reset_ready_refresh_path=reset_ready_refresh_path,
            reset_ready_refresh_current_present=reset_ready_refresh_current_present,
            include_mutation_gate=route_contract["include_mutation_gate"],
            include_supporting_drifts_when_present=route_contract["include_supporting_drifts_when_present"],
        )
    if reason.startswith("continuity_current_publish_lock_"):
        return unique_preserve([
            "state/continuity/current.json#generated_at",
            "state/continuity/current.json#drifts",
            str(publish_lock_owner_path or "").strip() or _CURRENT_PUBLISH_LOCK_OWNER_REL,
        ])
    if reason.startswith("doctrine_drift:"):
        return unique_preserve([
            "state/continuity/current.json#doctrine_drift",
            "state/continuity/current.json#drifts",
            str(doctrine_drift_registry_path or "").strip() or _DOCTRINE_DRIFT_REGISTRY_REL,
        ])
    return ["state/continuity/current.json#drifts"]


def _raise_contract_schema_validation_error(*, data_ptr: str, schema_ptr: str, message: str) -> None:
    raise RuntimeError(
        "blocker_registry_contract_schema_validation_failed:"
        f"data_path={data_ptr}:schema_path={schema_ptr}:error={message}"
    )


def _validate_contract_minimal(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        _raise_contract_schema_validation_error(
            data_ptr="$",
            schema_ptr="$/type",
            message="payload must be object",
        )

    required_top = [
        "schema",
        "generated_at",
        "status",
        "readiness",
        "freshness",
        "source_current",
        "truth_anchor",
        "mutation_gate",
        "blockers",
    ]
    missing = [key for key in required_top if key not in payload]
    if missing:
        _raise_contract_schema_validation_error(
            data_ptr="$",
            schema_ptr="$/required",
            message=f"missing required properties: {', '.join(missing)}",
        )

    if payload.get("schema") != "clawd.continuity.blocker_registry.v1":
        _raise_contract_schema_validation_error(
            data_ptr="$/schema",
            schema_ptr="$/properties/schema/const",
            message="schema must equal clawd.continuity.blocker_registry.v1",
        )

    status = str(payload.get("status") or "").strip()
    if status not in {"READY", "BLOCKER"}:
        _raise_contract_schema_validation_error(
            data_ptr="$/status",
            schema_ptr="$/properties/status/enum",
            message="status must be READY or BLOCKER",
        )

    readiness = str(payload.get("readiness") or "").strip()
    if readiness not in {"READY", "READY_WITH_DEBT", "RECONCILE_REQUIRED", "NOT_READY", "UNKNOWN"}:
        _raise_contract_schema_validation_error(
            data_ptr="$/readiness",
            schema_ptr="$/properties/readiness/enum",
            message="readiness must be one of READY/READY_WITH_DEBT/RECONCILE_REQUIRED/NOT_READY/UNKNOWN",
        )

    mutation_gate = payload.get("mutation_gate") if isinstance(payload.get("mutation_gate"), dict) else None
    if mutation_gate is None:
        _raise_contract_schema_validation_error(
            data_ptr="$/mutation_gate",
            schema_ptr="$/properties/mutation_gate/type",
            message="mutation_gate must be object",
        )
    gate_status = str(mutation_gate.get("status") or "").strip()
    if gate_status not in {"allowed", "forbidden"}:
        _raise_contract_schema_validation_error(
            data_ptr="$/mutation_gate/status",
            schema_ptr="$/properties/mutation_gate/properties/status/enum",
            message="mutation_gate.status must be allowed or forbidden",
        )

    blockers = payload.get("blockers")
    if not isinstance(blockers, list):
        _raise_contract_schema_validation_error(
            data_ptr="$/blockers",
            schema_ptr="$/properties/blockers/type",
            message="blockers must be array",
        )


def validate_contract(payload: Dict[str, Any]) -> None:
    if not schema_path.exists():
        raise RuntimeError(f"blocker_registry_contract_schema_missing:{schema_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_doc, dict):
        raise RuntimeError("blocker_registry_contract_schema_not_object")

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except Exception:
        _validate_contract_minimal(payload)
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
    _raise_contract_schema_validation_error(
        data_ptr=data_ptr,
        schema_ptr=schema_ptr,
        message=str(err.message),
    )


try:
    max_age_sec = int(max_age_raw)
except Exception:
    raise SystemExit(f"invalid --max-age-sec: {max_age_raw}")

if max_age_sec < 0:
    raise SystemExit(f"invalid --max-age-sec: {max_age_sec}")

if not current_path.exists():
    raise SystemExit(f"continuity current missing: {current_path}")

try:
    current = load_json(current_path)
except Exception as exc:
    raise SystemExit(f"continuity current parse failed: {exc}")

if not isinstance(current, dict):
    raise SystemExit("continuity current must be JSON object")

source_sha = sha256_file(current_path)
current_generated_raw = current.get("generated_at")
current_generated_dt = parse_iso(current_generated_raw)
source_age_sec = None
fresh = True
freshness_reasons: List[str] = []
if current_generated_dt is None:
    fresh = False
    freshness_reasons.append("continuity_current_generated_at_invalid")
else:
    source_age_sec = max(0, int((clock_now_dt() - current_generated_dt).total_seconds()))
    if max_age_sec > 0 and source_age_sec > max_age_sec:
        fresh = False
        freshness_reasons.append("continuity_current_stale")

readiness = str(current.get("readiness") or "UNKNOWN").strip() or "UNKNOWN"
ready_like = readiness in {"READY", "READY_WITH_DEBT"}

mutation_gate = current.get("mutation_gate") if isinstance(current.get("mutation_gate"), dict) else {}
gate_status = str(mutation_gate.get("status") or "forbidden").strip().lower()
gate_reasons = read_reason_rows(mutation_gate.get("reason"))
blocking_reasons = read_reason_rows(mutation_gate.get("blocking_reasons"))
concurrency_reasons = read_reason_rows(mutation_gate.get("concurrency_reasons"))

if not blocking_reasons:
    blocking_reasons = [
        reason
        for reason in gate_reasons
        if reason not in {"all_resume_gates_green"}
    ]

if not ready_like:
    blocking_reasons.append(f"readiness:{readiness}")

generation_pointer = evaluate_generation_pointer_contract(
    current_obj=current,
    current_sha=source_sha,
    current_generated_at=current_generated_raw,
)
generation_failclose_reasons = read_reason_rows(generation_pointer.get("failclose_reasons"))
continuity_now_contract_path = _resolve_continuity_now_evidence_path(
    root,
    raw_path=((generation_pointer.get("continuity_now_contract") or {}).get("path") if isinstance(generation_pointer.get("continuity_now_contract"), dict) else None),
    fallback_rel=_CONTINUITY_NOW_LATEST_REL,
)
source_refs = current.get("source_refs") if isinstance(current.get("source_refs"), dict) else {}
doctrine_drift_registry_path = str(source_refs.get("doctrine_drift_registry") or _DOCTRINE_DRIFT_REGISTRY_REL).strip() or _DOCTRINE_DRIFT_REGISTRY_REL
reset_ready_refresh_raw_path = str(source_refs.get("reset_ready_refresh") or _RESET_READY_REFRESH_LATEST_REL).strip() or _RESET_READY_REFRESH_LATEST_REL
reset_ready_refresh_path_obj = pathlib.Path(reset_ready_refresh_raw_path)
if not reset_ready_refresh_path_obj.is_absolute():
    reset_ready_refresh_path_obj = (root / reset_ready_refresh_path_obj).resolve()
else:
    reset_ready_refresh_path_obj = reset_ready_refresh_path_obj.resolve()
reset_ready_refresh_evidence_path = to_rel(reset_ready_refresh_path_obj)
reset_ready_refresh_current = current.get("reset_ready_refresh") if isinstance(current.get("reset_ready_refresh"), dict) else {}
reset_ready_refresh_posture = _project_reset_ready_refresh_posture(
    surface=reset_ready_refresh_current,
    path=reset_ready_refresh_evidence_path,
    present=(
        reset_ready_refresh_current.get("present")
        if isinstance(reset_ready_refresh_current.get("present"), bool)
        else bool(reset_ready_refresh_current)
    ),
)
reset_ready_refresh_current_present = bool(reset_ready_refresh_posture.get("present") is True)
reset_ready_refresh_evidence_path = str(
    reset_ready_refresh_posture.get("path") or reset_ready_refresh_evidence_path
).strip() or _RESET_READY_REFRESH_LATEST_REL
reset_ready_refresh_warning_metadata = _project_reset_ready_refresh_blocker_warning_metadata(
    posture=reset_ready_refresh_posture,
)
publish_lock = evaluate_publish_lock_posture(source_refs=source_refs)
publish_lock_owner_path = str(publish_lock.get("path") or "").strip() or _CURRENT_PUBLISH_LOCK_OWNER_REL
publish_lock_warning_metadata = {
    "recommended_action": publish_lock.get("recommended_action"),
    "action_required": publish_lock.get("action_required") is True,
    "context": {
        "path": publish_lock_owner_path,
        "status": publish_lock.get("status"),
        "owner_pid": publish_lock.get("owner_pid"),
        "owner_alive": publish_lock.get("owner_alive"),
        "owner_age_sec": publish_lock.get("owner_age_sec"),
        "lock_wait_sec": publish_lock.get("lock_wait_sec"),
        "lock_hold_warn_sec": publish_lock.get("lock_hold_warn_sec"),
        "owner_exceeds_wait_budget": publish_lock.get("owner_exceeds_wait_budget"),
        "owner_exceeds_lock_hold_warn": publish_lock.get("owner_exceeds_lock_hold_warn"),
        "owner_host": publish_lock.get("owner_host"),
        "owner_command": publish_lock.get("owner_command"),
        "inspect_command": publish_lock.get("inspect_command"),
    },
}

# Bootstrap-friendly parity: if no canonical continuity read pointer exists yet,
# do not fail-close solely on source age. Once a read pointer is present,
# continuity_current_stale remains a hard blocker.
if "continuity_current_stale" in freshness_reasons and not bool(generation_pointer.get("present")):
    freshness_reasons = [reason for reason in freshness_reasons if reason != "continuity_current_stale"]
    fresh = len(freshness_reasons) == 0

blocking_reasons.extend(freshness_reasons)
blocking_reasons.extend(generation_failclose_reasons)

if gate_status != "allowed" and not blocking_reasons:
    blocking_reasons = gate_reasons or concurrency_reasons or ["mutation_gate_forbidden"]

blocking_reasons = unique_preserve(blocking_reasons)

warning_reasons: List[str] = []
for drift in (current.get("drifts") or []):
    if not isinstance(drift, dict):
        continue
    code = str(drift.get("code") or "").strip()
    detail = str(drift.get("detail") or "").strip()
    if not detail:
        continue
    if code == "CONTINUITY_WARNING":
        warning_reasons.append(detail)
if publish_lock.get("owner_exceeds_wait_budget") is True:
    warning_reasons.append(_PUBLISH_LOCK_WAIT_BUDGET_WARNING_REASON)
if publish_lock.get("owner_exceeds_lock_hold_warn") is True:
    warning_reasons.append(_PUBLISH_LOCK_HOLD_BUDGET_WARNING_REASON)
if publish_lock.get("owner_alive") is False:
    warning_reasons.append(_PUBLISH_LOCK_OWNER_NOT_ALIVE_WARNING_REASON)
warning_reasons = unique_preserve(warning_reasons)

status = "READY"
if gate_status != "allowed" or blocking_reasons:
    status = "BLOCKER"

if status == "READY":
    mutation_gate_out: Dict[str, Any] = {
        "status": "allowed",
        "posture": str(mutation_gate.get("posture") or "open"),
        "reason": gate_reasons or ["all_resume_gates_green"],
        "blocking_reasons": [],
        "concurrency_reasons": [],
        "expected_in_flight_guard": False,
    }
else:
    mutation_reason_rows = unique_preserve(blocking_reasons + concurrency_reasons)
    mutation_gate_out = {
        "status": "forbidden",
        "posture": "blocker",
        "reason": mutation_reason_rows or ["blocker_registry_failclose"],
        "blocking_reasons": blocking_reasons,
        "concurrency_reasons": concurrency_reasons,
        "expected_in_flight_guard": False,
    }

blockers: List[Dict[str, Any]] = []
for reason in blocking_reasons:
    blockers.append(
        {
            "reason": reason,
            "severity": "blocker",
            "owner": "sre_watchdog",
            "status": "open",
            "evidence": blocker_evidence_for_reason(
                reason,
                continuity_now_contract_path=continuity_now_contract_path,
                reset_ready_refresh_path=reset_ready_refresh_evidence_path,
                reset_ready_refresh_current_present=reset_ready_refresh_current_present,
                publish_lock_owner_path=publish_lock_owner_path,
            ),
        }
    )

for reason in warning_reasons:
    warning_row = {
        "reason": reason,
        "severity": "warn",
        "owner": "sre_watchdog",
        "status": "open",
        "evidence": warning_evidence_for_reason(
            reason,
            reset_ready_refresh_path=reset_ready_refresh_evidence_path,
            reset_ready_refresh_current_present=reset_ready_refresh_current_present,
            publish_lock_owner_path=publish_lock_owner_path,
            doctrine_drift_registry_path=doctrine_drift_registry_path,
        ),
    }
    if reason.startswith("reset_ready_refresh_") and isinstance(reset_ready_refresh_warning_metadata, dict):
        warning_row["recommended_action"] = reset_ready_refresh_warning_metadata.get("recommended_action")
        warning_row["action_required"] = reset_ready_refresh_warning_metadata.get("action_required") is True
        warning_row["context"] = dict(
            reset_ready_refresh_warning_metadata.get("context")
            if isinstance(reset_ready_refresh_warning_metadata.get("context"), dict)
            else {}
        )
    if reason.startswith("continuity_current_publish_lock_"):
        warning_row["recommended_action"] = publish_lock_warning_metadata.get("recommended_action")
        warning_row["action_required"] = publish_lock_warning_metadata.get("action_required") is True
        warning_row["context"] = dict(
            publish_lock_warning_metadata.get("context")
            if isinstance(publish_lock_warning_metadata.get("context"), dict)
            else {}
        )
    blockers.append(warning_row)

truth_anchor = current.get("truth_anchor") if isinstance(current.get("truth_anchor"), dict) else {}
truth_anchor_out = {
    "snapshot_id": truth_anchor.get("snapshot_id"),
    "journal_offset": truth_anchor.get("journal_offset"),
    "pointer_hash": truth_anchor.get("pointer_hash"),
    "pointer_source": truth_anchor.get("pointer_source"),
}

current_rel = to_rel(current_path)
out_rel = to_rel(out_path)

payload = {
    "schema": "clawd.continuity.blocker_registry.v1",
    "generated_at": now_iso(),
    "status": status,
    "readiness": readiness,
    "freshness": {
        "source": current_rel,
        "source_generated_at": current_generated_raw,
        "source_age_sec": source_age_sec,
        "max_age_sec": max_age_sec,
        "fresh": fresh,
        "failclose_reasons": freshness_reasons,
    },
    "source_current": {
        "path": current_rel,
        "sha256": source_sha,
        "generated_at": current_generated_raw,
    },
    "source_precedence": [
        out_rel,
        "state/continuity/current.json",
        publish_lock_owner_path,
        "state/continuity/latest/continuity_read_pointer.json",
        "state/handover/latest.json",
        "reports/handover_context_latest.md",
    ],
    "truth_anchor": truth_anchor_out,
    "generation_pointer": generation_pointer,
    "publish_lock": publish_lock,
    "mutation_gate": mutation_gate_out,
    "blockers": blockers,
    "notes": [
        "Generated from state/continuity/current.json; do not edit manually.",
        "Fail-close on stale continuity/current source age breach.",
        "Fail-close on continuity generation-pointer mismatch/staleness when canonical pointer is present.",
        "Use continuity.sh blocker-registry to refresh this artifact deterministically.",
    ],
}

validate_contract(payload)
atomic_write(out_path, payload)

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(
        f"BLOCKER REGISTRY: status={status} readiness={readiness} "
        f"fresh={fresh} blockers={len([x for x in blockers if (x.get('severity') == 'blocker')])}"
    )
PY
