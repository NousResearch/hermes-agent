#!/usr/bin/env python3
"""Deterministic model rollout health snapshot producer (v1).

Builds `state/continuity/model_rollout_health/latest.json` from canonical
continuity surfaces so rollout-controller dwell/health gates do not depend on
manual snapshot staging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_CONTINUITY_NOW = Path("state/continuity/latest/continuity_now_latest.json")
DEFAULT_VERIFY_LAST = Path("state/continuity/latest/verify_last.json")
DEFAULT_GATE_OS = Path("state/continuity/latest/gate_os_latest.json")
DEFAULT_OUTPUT = Path("state/continuity/model_rollout_health/latest.json")

SCHEMA_VERSION = "clawd.model_rollout_health.v1"
RING_STATES = ("CANARY", "RING_1", "RING_2")
VERIFY_MAX_AGE_SEC = 1800


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: Path | str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def safe_repo_path(repo_root: Path, raw_path: Path | str) -> Tuple[bool, Path, Optional[str]]:
    try:
        resolved = resolve_repo_path(repo_root, raw_path)
    except Exception as exc:
        return False, repo_root, f"path_resolve_failed:{exc}"
    if not is_within(repo_root, resolved):
        return False, resolved, "path_outside_repo"
    return True, resolved, None


def _bool_check(ok: bool, reason: Optional[str], details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "reason": reason,
        "details": details or {},
    }


def evaluate_health(
    *,
    continuity_now: Dict[str, Any],
    verify_last: Dict[str, Any],
    gate_os: Dict[str, Any],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}

    continuity_ready = continuity_now.get("ready") is True
    checks["continuity_ready"] = _bool_check(continuity_ready, None if continuity_ready else "continuity_not_ready")

    not_ready_reasons = continuity_now.get("not_ready_reasons")
    if not isinstance(not_ready_reasons, list):
        not_ready_reasons = []

    mutation = continuity_now.get("mutation_gate_projection") if isinstance(continuity_now.get("mutation_gate_projection"), dict) else {}
    mutation_status = mutation.get("status")
    mutation_allowed = mutation_status == "allowed"
    checks["mutation_gate_allowed"] = _bool_check(
        mutation_allowed,
        None if mutation_allowed else "mutation_gate_not_allowed",
        {"status": mutation_status},
    )

    verify_status = verify_last.get("status")
    verify_ready = verify_status == "READY"
    checks["verify_status_ready"] = _bool_check(
        verify_ready,
        None if verify_ready else "verify_not_ready",
        {"status": verify_status},
    )

    verify_age_sec = None
    verify_surface = continuity_now.get("verify") if isinstance(continuity_now.get("verify"), dict) else {}
    raw_verify_age = verify_surface.get("age_sec")
    if isinstance(raw_verify_age, int) and raw_verify_age >= 0:
        verify_age_sec = raw_verify_age
    verify_fresh = isinstance(verify_age_sec, int) and verify_age_sec <= VERIFY_MAX_AGE_SEC
    checks["verify_surface_fresh"] = _bool_check(
        verify_fresh,
        None if verify_fresh else "verify_surface_stale_or_missing",
        {
            "age_sec": verify_age_sec,
            "max_age_sec": VERIFY_MAX_AGE_SEC,
        },
    )

    gate_summary = gate_os.get("summary") if isinstance(gate_os.get("summary"), dict) else {}
    gate_fail = gate_summary.get("fail")
    gate_fail_zero = isinstance(gate_fail, int) and gate_fail == 0
    checks["gate_os_fail_zero"] = _bool_check(
        gate_fail_zero,
        None if gate_fail_zero else "gate_os_fail_nonzero_or_missing",
        {"fail": gate_fail},
    )

    gate_mutation_allowed = gate_summary.get("mutation_allowed") is True
    checks["gate_os_mutation_allowed"] = _bool_check(
        gate_mutation_allowed,
        None if gate_mutation_allowed else "gate_os_mutation_not_allowed",
        {"mutation_allowed": gate_summary.get("mutation_allowed")},
    )

    fail_reasons: List[str] = []
    for cid, row in checks.items():
        if row.get("ok") is True:
            continue
        fail_reasons.append(str(row.get("reason") or f"{cid}_failed"))

    for reason in not_ready_reasons:
        if isinstance(reason, str) and reason:
            fail_reasons.append(f"continuity_not_ready_reason:{reason}")

    fail_reasons = sorted(set(fail_reasons))
    overall_ok = not fail_reasons

    return overall_ok, fail_reasons, {
        "checks": checks,
        "continuity_not_ready_reasons": not_ready_reasons,
    }


def write_output(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists() and not path.is_file():
            return {"written": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return {"written": True, "path": str(path)}
    except Exception as exc:
        return {"written": False, "reason": "write_failed", "error": str(exc), "path": str(path)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic model rollout health snapshot producer (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--continuity-now", default=str(DEFAULT_CONTINUITY_NOW), help="continuity_now latest JSON path")
    ap.add_argument("--verify-last", default=str(DEFAULT_VERIFY_LAST), help="verify_last JSON path")
    ap.add_argument("--gate-os", default=str(DEFAULT_GATE_OS), help="gate_os latest JSON path")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT), help="output snapshot path")
    ap.add_argument("--fail-on-unhealthy", action="store_true", help="Return non-zero when computed status is unhealthy")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()

    ok, continuity_now_path, reason = safe_repo_path(repo_root, args.continuity_now)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(continuity_now_path)}))
        return 2

    ok, verify_last_path, reason = safe_repo_path(repo_root, args.verify_last)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(verify_last_path)}))
        return 2

    ok, gate_os_path, reason = safe_repo_path(repo_root, args.gate_os)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(gate_os_path)}))
        return 2

    ok, out_path, reason = safe_repo_path(repo_root, args.out)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(out_path)}))
        return 2

    source_docs: Dict[str, Dict[str, Any]] = {}
    loaded: Dict[str, Dict[str, Any]] = {}

    for source_key, source_path in (
        ("continuity_now", continuity_now_path),
        ("verify_last", verify_last_path),
        ("gate_os", gate_os_path),
    ):
        if not source_path.exists():
            result = {"ok": False, "error": f"{source_key}_missing", "path": str(source_path)}
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2
        if not source_path.is_file():
            result = {"ok": False, "error": f"{source_key}_not_file", "path": str(source_path)}
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2
        try:
            doc = load_json_file(source_path)
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"{source_key}_unreadable",
                "path": str(source_path),
                "detail": str(exc),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2
        if not isinstance(doc, dict):
            result = {
                "ok": False,
                "error": f"{source_key}_not_object",
                "path": str(source_path),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2

        loaded[source_key] = doc
        source_docs[source_key] = {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "generated_at": doc.get("generated_at") or doc.get("timestamp"),
        }

    overall_ok, fail_reasons, details = evaluate_health(
        continuity_now=loaded["continuity_now"],
        verify_last=loaded["verify_last"],
        gate_os=loaded["gate_os"],
    )

    ring_payload: Dict[str, Dict[str, Any]] = {}
    for ring_state in RING_STATES:
        ring_payload[ring_state] = {
            "slo_ok": overall_ok,
            "status": "healthy" if overall_ok else "unhealthy",
            "reasons": [] if overall_ok else fail_reasons,
        }

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "overall_status": "healthy" if overall_ok else "unhealthy",
        "rings": ring_payload,
        "checks": details.get("checks"),
        "sources": source_docs,
    }

    write_result = write_output(out_path, snapshot)
    result = {
        "ok": write_result.get("written") is True,
        "status": snapshot.get("overall_status"),
        "snapshot": snapshot,
        "write": write_result,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    if write_result.get("written") is not True:
        return 2
    if args.fail_on_unhealthy and not overall_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
