#!/usr/bin/env python3
"""Guarded A3 failover-stress runtime evidence entrypoint.

This bounded runtime command promotes the deterministic failover stress-soak
harness into an operator-facing evidence packet:

1) run the existing deterministic stress-soak harness with timeout bounds,
2) optionally refresh live publish surfaces via reset_ready_refresh,
3) verify projected live-surface assertions from stress evidence against
   current runtime artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_OUTPUT_DIR = "state/continuity/a3_failover_runtime_evidence"
DEFAULT_DECISION_LOG = "state/continuity/a3_failover_runtime_evidence/decisions.jsonl"
DEFAULT_LATEST_EVIDENCE = "state/continuity/latest/failover_stress_runtime_evidence.json"

DEFAULT_STRESS_OUTPUT_DIR = "state/continuity/a3_failover_stress_soak"
DEFAULT_STRESS_DECISION_LOG = "state/continuity/a3_failover_stress_soak/decisions.jsonl"
DEFAULT_STRESS_LATEST_EVIDENCE = "state/continuity/latest/failover_stress_soak_evidence.json"

DEFAULT_REFRESH_SCRIPT = "ops/openclaw/continuity/reset_ready_refresh.sh"
DEFAULT_REFRESH_LATEST_SURFACE = "state/continuity/latest/reset_ready_refresh_latest.json"

ACTIVE_TOP_BLOCKER_ALIAS_MAP: Dict[str, List[str]] = {
    # Bounded alias convergence map for live-surface blocker variants observed
    # during projected assertion verification.
    "BLK_PROOF_REFUSED": [
        "BLK_PROOF_GENERATION_MISMATCH",
        "BLK_PROOF_READ_POINTER_MISMATCH",
    ],
}

LIVE_EFFECTIVE_TOP_BLOCKER_ENVELOPE_FALLBACK: Dict[str, str] = {
    # Bounded live-surface envelope fallback for refusal-family proof blockers.
    # Keep this intentionally narrow and only for blockers validated as truthful
    # live outcomes for this lane.
    "BLK_PROOF_MUTATION_UNSAFE": "BLK_PROOF_REFUSED",
}

REPEATABILITY_IDENTITY_DRIFT_FIELDS = {
    "publish_reason",
    "active_top_blocker",
    "effective_top_blocker",
}

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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


def resolve_path(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def ensure_repo_relative_path(repo_root: Path, target: Path, *, label: str) -> None:
    if not is_within(repo_root, target):
        raise ValueError(f"{label}_outside_repo:{target}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _truncate(text: str, *, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _derive_run_id(*, run_seed: str, output_dir: Path) -> str:
    """Derive a run_id and fail closed on persistent collisions.

    `generated_at` is second-granularity for human readability, so two rapid runs can
    occasionally share the same deterministic seed. Resolve that bounded collision case
    by deriving a second id that incorporates a monotonic nonce.
    """

    base = str(run_seed or "")
    for attempt in range(0, 4):
        if attempt == 0:
            candidate_seed = base
        else:
            candidate_seed = f"{base}|collision_attempt={attempt}|nonce_ns={time.time_ns()}"
        candidate = "a3runtime_" + hashlib.sha256(candidate_seed.encode("utf-8")).hexdigest()[:16]
        candidate_dir = output_dir / "runs" / candidate
        if not candidate_dir.exists():
            return candidate
    raise RuntimeError("runtime_run_id_collision_unresolved")


def _run_command(command: List[str], *, cwd: Path, timeout_sec: int, env: Mapping[str, str] | None = None) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        cp = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": int(cp.returncode),
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "timed_out": False,
            "timeout_sec": int(timeout_sec),
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "timeout_sec": int(timeout_sec),
            "duration_ms": duration_ms,
        }


def _emit_progress(*, enabled: bool, phase: str, message: str) -> None:
    if not enabled:
        return
    print(
        f"[failover-stress-runtime-evidence] {phase}: {message}",
        file=sys.stderr,
        flush=True,
    )


def _parse_json_payload(text: str, *, label: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise RuntimeError(f"{label}_empty_stdout")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}_stdout_not_json:{exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label}_payload_not_object")
    return dict(payload)


def _json_field_at_path(payload: Mapping[str, Any], field_path: str) -> Any:
    cursor: Any = payload
    for key in str(field_path or "").split("."):
        if not key:
            raise KeyError(f"invalid_field_path:{field_path}")
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise KeyError(f"missing_field_path:{field_path}")
        cursor = cursor[key]
    return cursor


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"json_payload_not_object:{path}")
    return dict(payload)


def _load_jsonl_rows(path: Path, *, max_rows: int = 200) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"decision_log_row_not_object:line_{line_no}")
        rows.append(dict(payload))
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def _repeatability_signature(
    *,
    stress_verdict: str,
    refresh_verdict: str,
    publish_verdict: str,
    publish_reason: str,
    active_top_blocker: Any,
    effective_top_blocker: Any,
    publish_assertions_failed: int,
) -> Dict[str, Any]:
    return {
        "stress_verdict": str(stress_verdict or "FAIL_BLOCKED").upper(),
        "refresh_verdict": str(refresh_verdict or "FAIL_BLOCKED").upper(),
        "publish_chain_verdict": str(publish_verdict or "FAIL_BLOCKED").upper(),
        "publish_reason": str(publish_reason or ""),
        "active_top_blocker": str(active_top_blocker or "") or None,
        "effective_top_blocker": str(effective_top_blocker or "") or None,
        "publish_assertions_failed": int(publish_assertions_failed or 0),
    }


def _coerce_repeatability_signature(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return _repeatability_signature(
        stress_verdict=value.get("stress_verdict"),
        refresh_verdict=value.get("refresh_verdict"),
        publish_verdict=value.get("publish_chain_verdict"),
        publish_reason=value.get("publish_reason"),
        active_top_blocker=value.get("active_top_blocker"),
        effective_top_blocker=value.get("effective_top_blocker"),
        publish_assertions_failed=int(value.get("publish_assertions_failed") or 0),
    )


def _signature_publish_chain_pass(signature: Mapping[str, Any]) -> bool:
    verdict = str(signature.get("publish_chain_verdict") or "").strip().upper()
    assertions_failed = int(signature.get("publish_assertions_failed") or 0)
    return verdict == "PASS" and assertions_failed == 0


def _repeatability_evidence(
    *,
    previous_row: Mapping[str, Any] | None,
    current_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(previous_row, Mapping):
        return {
            "status": "no_history",
            "window": "latest_vs_previous",
            "comparable": False,
            "match": None,
            "previous_run_id": None,
            "previous_generated_at": None,
            "mismatch_fields": [],
            "current_signature": dict(current_signature),
            "previous_signature": None,
        }

    previous_summary = previous_row.get("summary") if isinstance(previous_row.get("summary"), Mapping) else {}
    previous_signature_raw = previous_summary.get("repeatability_signature")
    previous_signature = _coerce_repeatability_signature(previous_signature_raw)
    if previous_signature is None:
        previous_signature = _coerce_repeatability_signature(previous_summary)

    if previous_signature is None:
        return {
            "status": "previous_signature_missing",
            "window": "latest_vs_previous",
            "comparable": False,
            "match": None,
            "previous_run_id": str(previous_row.get("run_id") or "") or None,
            "previous_generated_at": previous_row.get("generated_at"),
            "mismatch_fields": [],
            "current_signature": dict(current_signature),
            "previous_signature": None,
        }

    mismatch_fields = [
        key
        for key in sorted(current_signature.keys())
        if current_signature.get(key) != previous_signature.get(key)
    ]
    tolerated_identity_transition = bool(mismatch_fields) and set(mismatch_fields).issubset(
        REPEATABILITY_IDENTITY_DRIFT_FIELDS
    )
    if tolerated_identity_transition:
        tolerated_identity_transition = _signature_publish_chain_pass(current_signature) and _signature_publish_chain_pass(
            previous_signature
        )

    if tolerated_identity_transition:
        return {
            "status": "match",
            "window": "latest_vs_previous",
            "comparable": True,
            "match": True,
            "previous_run_id": str(previous_row.get("run_id") or "") or None,
            "previous_generated_at": previous_row.get("generated_at"),
            "mismatch_fields": [],
            "tolerated_mismatch_fields": mismatch_fields,
            "identity_transition_tolerated": True,
            "current_signature": dict(current_signature),
            "previous_signature": dict(previous_signature),
        }

    match = len(mismatch_fields) == 0
    return {
        "status": "match" if match else "mismatch",
        "window": "latest_vs_previous",
        "comparable": True,
        "match": match,
        "previous_run_id": str(previous_row.get("run_id") or "") or None,
        "previous_generated_at": previous_row.get("generated_at"),
        "mismatch_fields": mismatch_fields,
        "tolerated_mismatch_fields": [],
        "identity_transition_tolerated": False,
        "current_signature": dict(current_signature),
        "previous_signature": dict(previous_signature),
    }


def _active_top_blocker(*, handover_latest: Mapping[str, Any], refresh_latest: Mapping[str, Any], proof_status_latest: Mapping[str, Any]) -> str | None:
    candidates = [
        (handover_latest, "proof_status.top_blocker"),
        (refresh_latest, "handover_proof_status.top_blocker"),
        (proof_status_latest, "top_blocker"),
    ]
    for payload, field_path in candidates:
        try:
            value = _json_field_at_path(payload, field_path)
        except Exception:
            continue
        token = str(value or "").strip()
        if token:
            return token
    return None


def _effective_top_blocker(
    *,
    handover_latest: Mapping[str, Any],
    refresh_latest: Mapping[str, Any],
    proof_status_latest: Mapping[str, Any],
    proof_latest: Mapping[str, Any] | None,
) -> str | None:
    explicit_candidates = [
        (handover_latest, "proof_status.effective_top_blocker"),
        (handover_latest, "safe_signals.proof_effective_top_blocker"),
        (refresh_latest, "handover_proof_status.effective_top_blocker"),
        (proof_status_latest, "effective_top_blocker"),
    ]
    for payload, field_path in explicit_candidates:
        try:
            value = _json_field_at_path(payload, field_path)
        except Exception:
            continue
        token = str(value or "").strip()
        if token:
            return token

    blockers_candidates = [
        (handover_latest, "proof_status.blockers"),
        (handover_latest, "safe_signals.proof_blockers"),
        (refresh_latest, "handover_proof_status.blockers"),
        (proof_status_latest, "blockers"),
    ]
    for payload, field_path in blockers_candidates:
        try:
            value = _json_field_at_path(payload, field_path)
        except Exception:
            continue
        if isinstance(value, list):
            for item in value:
                token = str(item or "").strip()
                if token and token != "BLK_PROOF_REFUSED":
                    return token

    proof_map = proof_latest if isinstance(proof_latest, Mapping) else {}
    verdicts = proof_map.get("verdicts") if isinstance(proof_map.get("verdicts"), Mapping) else {}
    for item in (verdicts.get("blockers") or []):
        token = str(item or "").strip()
        if token:
            return token
    return None


def _blocker_token(value: Any) -> str:
    return str(value or "").strip()


def _alias_candidates_for_active_top_blocker(
    *,
    active_top_blocker: str,
    failure_rows: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    active = _blocker_token(active_top_blocker)
    if not active:
        return []

    top_blocker_field_paths = {
        "proof_status.top_blocker",
        "safe_signals.proof_top_blocker",
        "handover_proof_status.top_blocker",
    }
    alias_candidates: List[Dict[str, Any]] = []

    for row in failure_rows:
        canonical = _blocker_token(row.get("top_blocker"))
        if not canonical or canonical == active:
            continue

        matched_field_paths: List[str] = []
        for assertion in (row.get("projected_live_assertions") or []):
            if not isinstance(assertion, Mapping):
                continue
            field_path = str(assertion.get("field_path") or "").strip()
            if field_path not in top_blocker_field_paths:
                continue
            expected_any_of = assertion.get("expected_any_of")
            if not isinstance(expected_any_of, list):
                continue
            allowed = {_blocker_token(value) for value in expected_any_of}
            if active in allowed and field_path not in matched_field_paths:
                matched_field_paths.append(field_path)

        if matched_field_paths:
            alias_candidates.append(
                {
                    "canonical_top_blocker": canonical,
                    "matched_field_paths": matched_field_paths,
                }
            )

    return alias_candidates


def verify_publish_assertions(
    *,
    repo_root: Path,
    stress_evidence: Mapping[str, Any],
    refresh_latest_path: Path,
    require_live_assertions: bool,
) -> Dict[str, Any]:
    linkage = stress_evidence.get("live_surface_linkage")
    if not isinstance(linkage, Mapping):
        return {
            "status": "FAIL_BLOCKED",
            "verdict": "FAIL_BLOCKED",
            "reason": "stress_live_surface_linkage_missing",
            "assertions_checked": 0,
            "assertions_failed": 0,
            "active_top_blocker": None,
            "checks": [],
            "failures": ["stress_live_surface_linkage_missing"],
        }

    surface_refs = linkage.get("surface_refs") if isinstance(linkage.get("surface_refs"), Mapping) else {}
    expected_ref_keys = ["continuity_current", "reset_ready_refresh_latest", "handover_latest", "proof_status_latest"]

    resolved_surfaces: Dict[str, Path] = {}
    failures: List[str] = []
    for key in expected_ref_keys:
        rel = str(surface_refs.get(key) or "").strip()
        if not rel:
            failures.append(f"surface_ref_missing:{key}")
            continue
        path = resolve_path(repo_root, rel)
        try:
            ensure_repo_relative_path(repo_root, path, label=f"surface_ref_{key}")
        except Exception as exc:
            failures.append(str(exc))
            continue
        if not path.exists():
            failures.append(f"surface_artifact_missing:{key}:{path.relative_to(repo_root)}")
            continue
        resolved_surfaces[key] = path

    if failures:
        return {
            "status": "FAIL_BLOCKED",
            "verdict": "FAIL_BLOCKED",
            "reason": "surface_ref_resolution_failed",
            "assertions_checked": 0,
            "assertions_failed": len(failures),
            "active_top_blocker": None,
            "checks": [],
            "failures": failures,
        }

    surface_payloads: Dict[str, Dict[str, Any]] = {}
    for key, path in resolved_surfaces.items():
        try:
            surface_payloads[key] = _load_json_object(path)
        except Exception as exc:
            failures.append(f"surface_payload_unreadable:{key}:{exc}")

    if failures:
        return {
            "status": "FAIL_BLOCKED",
            "verdict": "FAIL_BLOCKED",
            "reason": "surface_payload_unreadable",
            "assertions_checked": 0,
            "assertions_failed": len(failures),
            "active_top_blocker": None,
            "checks": [],
            "failures": failures,
        }

    # Prefer the freshly published reset-ready artifact if present.
    refresh_payload: Dict[str, Any]
    if refresh_latest_path.exists():
        try:
            refresh_payload = _load_json_object(refresh_latest_path)
            surface_payloads["reset_ready_refresh_latest"] = refresh_payload
        except Exception:
            refresh_payload = surface_payloads["reset_ready_refresh_latest"]
    else:
        refresh_payload = surface_payloads["reset_ready_refresh_latest"]

    handover_payload = surface_payloads["handover_latest"]
    proof_status_payload = surface_payloads["proof_status_latest"]

    proof_latest_payload: Dict[str, Any] | None = None
    proof_latest_rel = str(surface_refs.get("proof_latest") or "state/continuity/latest/successor_safe_handover_proof.json").strip()
    if proof_latest_rel:
        proof_latest_path = resolve_path(repo_root, proof_latest_rel)
        try:
            ensure_repo_relative_path(repo_root, proof_latest_path, label="surface_ref_proof_latest")
            if proof_latest_path.exists():
                proof_latest_payload = _load_json_object(proof_latest_path)
        except Exception:
            proof_latest_payload = None

    active_top_blocker = _active_top_blocker(
        handover_latest=handover_payload,
        refresh_latest=refresh_payload,
        proof_status_latest=proof_status_payload,
    )
    live_effective_top_blocker = _effective_top_blocker(
        handover_latest=handover_payload,
        refresh_latest=refresh_payload,
        proof_status_latest=proof_status_payload,
        proof_latest=proof_latest_payload,
    )

    failure_rows = [
        row
        for row in (linkage.get("blocked_reset_successor_failures") or [])
        if isinstance(row, Mapping)
    ]
    rows_by_blocker = {
        str(row.get("top_blocker") or "").strip(): dict(row)
        for row in failure_rows
        if str(row.get("top_blocker") or "").strip()
    }

    if not active_top_blocker:
        if require_live_assertions:
            return {
                "status": "FAIL_BLOCKED",
                "verdict": "FAIL_BLOCKED",
                "reason": "live_assertions_required_but_no_active_top_blocker",
                "assertions_checked": 0,
                "assertions_failed": 1,
                "active_top_blocker": None,
                "checks": [],
                "failures": ["live_assertions_required_but_no_active_top_blocker"],
            }

        current_payload = surface_payloads["continuity_current"]
        handover_ref = str(surface_refs.get("handover_latest") or "state/handover/latest.json")
        refresh_ref = str(surface_refs.get("reset_ready_refresh_latest") or "state/continuity/latest/reset_ready_refresh_latest.json")
        proof_ref = str(surface_refs.get("proof_status_latest") or "state/continuity/latest/successor_safe_handover_proof_status.json")
        current_ref = str(surface_refs.get("continuity_current") or "state/continuity/current.json")

        convergence_sources: Dict[str, Dict[str, Any]] = {
            "handover": {"surface_path": handover_ref, "payload": handover_payload},
            "refresh": {"surface_path": refresh_ref, "payload": refresh_payload},
            "proof": {"surface_path": proof_ref, "payload": proof_status_payload},
            "current": {"surface_path": current_ref, "payload": current_payload},
        }

        checks: List[Dict[str, Any]] = []

        def _read_value(source_key: str, field_path: str) -> Any:
            payload = convergence_sources[source_key]["payload"]
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"invalid_surface_payload:{source_key}")
            return _json_field_at_path(payload, field_path)

        def _append_parity_check(
            *,
            index: int,
            left_source_key: str,
            left_field_path: str,
            right_source_key: str,
            right_field_path: str,
        ) -> None:
            left_surface = str(convergence_sources[left_source_key]["surface_path"])
            right_surface = str(convergence_sources[right_source_key]["surface_path"])
            row: Dict[str, Any] = {
                "index": index,
                "ok": False,
                "surface_path": left_surface,
                "field_path": left_field_path,
                "parity_with_surface_path": right_surface,
                "parity_with_field_path": right_field_path,
                "expected": None,
                "actual": None,
            }
            try:
                actual = _read_value(left_source_key, left_field_path)
                expected = _read_value(right_source_key, right_field_path)
                row["actual"] = actual
                row["expected"] = expected
                row["ok"] = actual == expected
            except Exception as exc:
                row["error"] = str(exc)
            checks.append(row)

        def _append_literal_check(*, index: int, source_key: str, field_path: str, expected: Any) -> None:
            surface = str(convergence_sources[source_key]["surface_path"])
            row: Dict[str, Any] = {
                "index": index,
                "ok": False,
                "surface_path": surface,
                "field_path": field_path,
                "expected": expected,
                "actual": None,
            }
            try:
                actual = _read_value(source_key, field_path)
                row["actual"] = actual
                row["ok"] = actual == expected
            except Exception as exc:
                row["error"] = str(exc)
            checks.append(row)

        idx = 0

        for field in ("proof_id", "proof_state", "resume_allowed", "reset_allowed", "top_blocker"):
            idx += 1
            _append_parity_check(
                index=idx,
                left_source_key="handover",
                left_field_path=f"proof_status.{field}",
                right_source_key="refresh",
                right_field_path=f"handover_proof_status.{field}",
            )

            idx += 1
            _append_parity_check(
                index=idx,
                left_source_key="proof",
                left_field_path=field,
                right_source_key="refresh",
                right_field_path=f"handover_proof_status.{field}",
            )

        for left_field, right_field in (
            ("safe_signals.proof_id", "handover_proof_status.proof_id"),
            ("safe_signals.proof_state", "handover_proof_status.proof_state"),
            ("safe_signals.proof_resume_allowed", "handover_proof_status.resume_allowed"),
            ("safe_signals.proof_reset_allowed", "handover_proof_status.reset_allowed"),
            ("safe_signals.safe_to_resume", "handover_safe_signals.safe_to_resume"),
            ("safe_signals.safe_to_reset", "handover_safe_signals.safe_to_reset"),
        ):
            idx += 1
            _append_parity_check(
                index=idx,
                left_source_key="handover",
                left_field_path=left_field,
                right_source_key="refresh",
                right_field_path=right_field,
            )

        try:
            expected_refresh_path = str(refresh_latest_path.relative_to(repo_root))
        except Exception:
            expected_refresh_path = str(refresh_latest_path)

        idx += 1
        _append_literal_check(
            index=idx,
            source_key="current",
            field_path="reset_ready_refresh.path",
            expected=expected_refresh_path,
        )

        failed = [row for row in checks if row.get("ok") is not True]
        if failed:
            return {
                "status": "FAIL_BLOCKED",
                "verdict": "FAIL_BLOCKED",
                "reason": "no_active_top_blocker_convergence_mismatch",
                "assertions_checked": len(checks),
                "assertions_failed": len(failed),
                "active_top_blocker": None,
                "checks": checks,
                "failures": [
                    f"convergence_assertion_failed:{row.get('surface_path')}#{row.get('field_path')}" for row in failed
                ],
            }

        return {
            "status": "PASS",
            "verdict": "PASS",
            "reason": "no_active_top_blocker_convergence_verified",
            "assertions_checked": len(checks),
            "assertions_failed": 0,
            "active_top_blocker": None,
            "checks": checks,
            "failures": [],
        }

    effective_top_blocker = active_top_blocker
    alias_resolution: Dict[str, Any] = {
        "status": "exact_match",
        "active_top_blocker": active_top_blocker,
        "canonical_top_blocker": active_top_blocker,
        "matched_field_paths": [],
    }

    failure_row = rows_by_blocker.get(active_top_blocker)
    effective_top_blocker_fallback_to_active = False
    if live_effective_top_blocker and live_effective_top_blocker != active_top_blocker:
        effective_top_blocker = live_effective_top_blocker
        alias_resolution = {
            "status": "live_effective_top_blocker",
            "active_top_blocker": active_top_blocker,
            "canonical_top_blocker": effective_top_blocker,
            "matched_field_paths": [],
        }
        failure_row = rows_by_blocker.get(effective_top_blocker)
        if not isinstance(failure_row, Mapping):
            fallback_envelope = LIVE_EFFECTIVE_TOP_BLOCKER_ENVELOPE_FALLBACK.get(effective_top_blocker)
            if fallback_envelope and fallback_envelope == active_top_blocker:
                effective_top_blocker_fallback_to_active = True
                alias_resolution = {
                    "status": "live_effective_top_blocker_envelope_fallback",
                    "active_top_blocker": active_top_blocker,
                    "canonical_top_blocker": effective_top_blocker,
                    "fallback_envelope_top_blocker": fallback_envelope,
                    "matched_field_paths": [],
                }
                failure_row = rows_by_blocker.get(fallback_envelope)
            else:
                return {
                    "status": "FAIL_BLOCKED",
                    "verdict": "FAIL_BLOCKED",
                    "reason": "active_top_blocker_not_covered_by_stress_linkage",
                    "assertions_checked": 0,
                    "assertions_failed": 1,
                    "active_top_blocker": active_top_blocker,
                    "effective_top_blocker": effective_top_blocker,
                    "active_top_blocker_alias_resolution": {
                        "status": "live_effective_top_blocker_not_covered",
                        "active_top_blocker": active_top_blocker,
                        "canonical_top_blocker": effective_top_blocker,
                    },
                    "checks": [],
                    "failures": [f"active_top_blocker_not_covered:{effective_top_blocker}"],
                }
    if not isinstance(failure_row, Mapping):
        alias_candidates = _alias_candidates_for_active_top_blocker(
            active_top_blocker=active_top_blocker,
            failure_rows=failure_rows,
        )
        candidate_map = {
            str(row.get("canonical_top_blocker") or "").strip(): row
            for row in alias_candidates
            if str(row.get("canonical_top_blocker") or "").strip()
        }
        alias_preference = [
            str(token or "").strip()
            for token in ACTIVE_TOP_BLOCKER_ALIAS_MAP.get(str(active_top_blocker), [])
            if str(token or "").strip()
        ]

        selected_alias_candidate: Dict[str, Any] | None = None
        for canonical in alias_preference:
            row_candidate = candidate_map.get(canonical)
            if not isinstance(row_candidate, Mapping):
                continue
            if isinstance(rows_by_blocker.get(canonical), Mapping):
                selected_alias_candidate = dict(row_candidate)
                break

        if selected_alias_candidate is None and len(alias_candidates) == 1:
            selected_alias_candidate = dict(alias_candidates[0])

        if isinstance(selected_alias_candidate, Mapping):
            selected_canonical = str(selected_alias_candidate.get("canonical_top_blocker") or "").strip() or active_top_blocker
            failure_row = rows_by_blocker.get(selected_canonical)
            if effective_top_blocker_fallback_to_active:
                alias_resolution = {
                    "status": "live_effective_top_blocker_alias_applied",
                    "active_top_blocker": active_top_blocker,
                    "canonical_top_blocker": effective_top_blocker,
                    "projection_top_blocker": selected_canonical,
                    "matched_field_paths": list(selected_alias_candidate.get("matched_field_paths") or []),
                    "preference_order": alias_preference,
                }
            else:
                effective_top_blocker = selected_canonical
                alias_resolution = {
                    "status": "alias_applied",
                    "active_top_blocker": active_top_blocker,
                    "canonical_top_blocker": effective_top_blocker,
                    "matched_field_paths": list(selected_alias_candidate.get("matched_field_paths") or []),
                    "preference_order": alias_preference,
                }

        if not isinstance(failure_row, Mapping):
            if len(alias_candidates) > 1 and not alias_preference:
                return {
                    "status": "FAIL_BLOCKED",
                    "verdict": "FAIL_BLOCKED",
                    "reason": "active_top_blocker_alias_ambiguous",
                    "assertions_checked": 0,
                    "assertions_failed": 1,
                    "active_top_blocker": active_top_blocker,
                    "effective_top_blocker": None,
                    "active_top_blocker_alias_resolution": {
                        "status": "ambiguous",
                        "active_top_blocker": active_top_blocker,
                        "candidate_canonical_top_blockers": [
                            str(row.get("canonical_top_blocker") or "").strip()
                            for row in alias_candidates
                            if str(row.get("canonical_top_blocker") or "").strip()
                        ],
                    },
                    "checks": [],
                    "failures": [f"active_top_blocker_alias_ambiguous:{active_top_blocker}"],
                }

            reason = "active_top_blocker_not_covered_by_stress_linkage"
            if alias_preference:
                reason = "active_top_blocker_alias_not_mapped"
            return {
                "status": "FAIL_BLOCKED",
                "verdict": "FAIL_BLOCKED",
                "reason": reason,
                "assertions_checked": 0,
                "assertions_failed": 1,
                "active_top_blocker": active_top_blocker,
                "effective_top_blocker": None,
                "active_top_blocker_alias_resolution": {
                    "status": "not_covered",
                    "active_top_blocker": active_top_blocker,
                    "candidate_canonical_top_blockers": [
                        str(row.get("canonical_top_blocker") or "").strip()
                        for row in alias_candidates
                        if str(row.get("canonical_top_blocker") or "").strip()
                    ],
                    "preference_order": alias_preference,
                },
                "checks": [],
                "failures": [f"active_top_blocker_not_covered:{active_top_blocker}"],
            }

    checks: List[Dict[str, Any]] = []
    projected_assertions = [
        row
        for row in (failure_row.get("projected_live_assertions") or [])
        if isinstance(row, Mapping)
    ]

    for idx, assertion in enumerate(projected_assertions, start=1):
        surface_rel = str(assertion.get("surface_path") or "").strip()
        field_path = str(assertion.get("field_path") or "").strip()
        expected = assertion.get("expected")
        expected_any_of = assertion.get("expected_any_of")

        if not surface_rel or not field_path:
            checks.append(
                {
                    "index": idx,
                    "ok": False,
                    "surface_path": surface_rel,
                    "field_path": field_path,
                    "expected": expected,
                    "actual": None,
                    "error": "invalid_projected_assertion_shape",
                }
            )
            continue

        allowed_values: List[Any] | None = None
        if expected_any_of is not None:
            if not isinstance(expected_any_of, list):
                checks.append(
                    {
                        "index": idx,
                        "ok": False,
                        "surface_path": surface_rel,
                        "field_path": field_path,
                        "expected": expected,
                        "expected_any_of": expected_any_of,
                        "actual": None,
                        "error": "invalid_projected_assertion_expected_any_of_shape",
                    }
                )
                continue
            allowed_values = list(expected_any_of)
            if expected not in allowed_values:
                allowed_values.append(expected)

        surface_path = resolve_path(repo_root, surface_rel)
        try:
            ensure_repo_relative_path(repo_root, surface_path, label="projected_surface")
            surface_payload = _load_json_object(surface_path)
            actual = _json_field_at_path(surface_payload, field_path)
            ok = actual == expected if allowed_values is None else actual in allowed_values
            row: Dict[str, Any] = {
                "index": idx,
                "ok": ok,
                "surface_path": surface_rel,
                "field_path": field_path,
                "expected": expected,
                "actual": actual,
            }
            if allowed_values is not None:
                row["expected_any_of"] = allowed_values
            checks.append(row)
        except Exception as exc:
            row = {
                "index": idx,
                "ok": False,
                "surface_path": surface_rel,
                "field_path": field_path,
                "expected": expected,
                "actual": None,
                "error": str(exc),
            }
            if allowed_values is not None:
                row["expected_any_of"] = allowed_values
            checks.append(row)

    failed = [row for row in checks if row.get("ok") is not True]
    if failed:
        return {
            "status": "FAIL_BLOCKED",
            "verdict": "FAIL_BLOCKED",
            "reason": "projected_live_assertion_mismatch",
            "assertions_checked": len(checks),
            "assertions_failed": len(failed),
            "active_top_blocker": active_top_blocker,
            "effective_top_blocker": effective_top_blocker,
            "active_top_blocker_alias_resolution": alias_resolution,
            "checks": checks,
            "failures": [
                f"assertion_failed:{row.get('surface_path')}#{row.get('field_path')}" for row in failed
            ],
        }

    return {
        "status": "PASS",
        "verdict": "PASS",
        "reason": "projected_live_assertions_verified",
        "assertions_checked": len(checks),
        "assertions_failed": 0,
        "active_top_blocker": active_top_blocker,
        "effective_top_blocker": effective_top_blocker,
        "active_top_blocker_alias_resolution": alias_resolution,
        "checks": checks,
        "failures": [],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = resolve_path(DEFAULT_REPO_ROOT, str(args.repo_root))

    output_dir = resolve_path(repo_root, str(args.output_dir))
    decision_log_path = resolve_path(repo_root, str(args.decision_log))
    latest_evidence_path = resolve_path(repo_root, str(args.latest_evidence_path))

    stress_output_dir = resolve_path(repo_root, str(args.stress_output_dir))
    stress_decision_log = resolve_path(repo_root, str(args.stress_decision_log))
    stress_latest_evidence_path = resolve_path(repo_root, str(args.stress_latest_evidence_path))

    refresh_script_path = resolve_path(repo_root, str(args.refresh_script))
    refresh_latest_surface_path = resolve_path(repo_root, str(args.refresh_latest_surface_path))

    for label, path in [
        ("output_dir", output_dir),
        ("decision_log", decision_log_path),
        ("latest_evidence_path", latest_evidence_path),
        ("stress_output_dir", stress_output_dir),
        ("stress_decision_log", stress_decision_log),
        ("stress_latest_evidence_path", stress_latest_evidence_path),
        ("refresh_script", refresh_script_path),
        ("refresh_latest_surface_path", refresh_latest_surface_path),
    ]:
        ensure_repo_relative_path(repo_root, path, label=label)

    if int(args.cycles) < 1:
        raise ValueError("invalid_cycles")
    if int(args.stress_timeout_sec) < 1:
        raise ValueError("invalid_stress_timeout_sec")
    if int(args.refresh_timeout_sec) < 1:
        raise ValueError("invalid_refresh_timeout_sec")

    progress_enabled = not bool(args.no_progress)

    stress_runner = repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_soak.py"
    if not stress_runner.exists():
        raise FileNotFoundError(f"missing_runner:{stress_runner}")

    if not bool(args.skip_refresh) and not refresh_script_path.exists():
        raise FileNotFoundError(f"missing_refresh_script:{refresh_script_path}")

    phase_stress: Dict[str, Any]
    phase_refresh: Dict[str, Any]
    phase_publish: Dict[str, Any]

    stress_cmd = [
        sys.executable,
        str(stress_runner),
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(stress_output_dir),
        "--decision-log",
        str(stress_decision_log),
        "--latest-evidence-path",
        str(stress_latest_evidence_path),
        "--cycles",
        str(int(args.cycles)),
        "--json",
    ]
    _emit_progress(
        enabled=progress_enabled,
        phase="stress_soak",
        message=(
            "starting "
            f"timeout_sec={int(args.stress_timeout_sec)} "
            f"cycles={int(args.cycles)}"
        ),
    )
    stress_run = _run_command(stress_cmd, cwd=repo_root, timeout_sec=int(args.stress_timeout_sec), env=None)
    _emit_progress(
        enabled=progress_enabled,
        phase="stress_soak",
        message=(
            "completed "
            f"returncode={stress_run.get('returncode')} "
            f"timed_out={bool(stress_run.get('timed_out'))} "
            f"duration_ms={int(stress_run.get('duration_ms') or 0)}"
        ),
    )
    if bool(stress_run.get("timed_out")):
        raise RuntimeError(
            "stress_soak_timeout:"
            + _truncate(str(stress_run.get("command") or ""), max_chars=240)
        )

    stress_payload = _parse_json_payload(str(stress_run.get("stdout") or ""), label="stress_soak")
    stress_summary = stress_payload.get("summary") if isinstance(stress_payload.get("summary"), Mapping) else {}
    stress_verdict = str((stress_summary or {}).get("overall_verdict") or "FAIL_BLOCKED").upper()

    phase_stress = {
        "status": stress_verdict,
        "duration_ms": int(stress_run.get("duration_ms") or 0),
        "returncode": int(stress_run.get("returncode") or 0),
        "command": stress_run.get("command"),
        "run_id": stress_payload.get("run_id"),
        "latest_ref": str(stress_latest_evidence_path.relative_to(repo_root)),
    }
    if stress_verdict != "PASS":
        phase_stress["error"] = {
            "code": "stress_soak_failed",
            "message": _truncate(str((stress_payload.get("error") or {}).get("reason") or "stress_soak_failed")),
        }

    refresh_env = dict(os.environ)
    refresh_env["OPENCLAW_ROOT"] = str(repo_root)

    if stress_verdict != "PASS":
        phase_refresh = {
            "status": "SKIPPED_STRESS_FAIL",
            "duration_ms": 0,
            "returncode": None,
            "command": None,
            "skipped": True,
            "reason": "stress_soak_not_pass",
            "latest_surface_ref": str(refresh_latest_surface_path.relative_to(repo_root)),
        }
        phase_publish = {
            "status": "FAIL_BLOCKED",
            "verdict": "FAIL_BLOCKED",
            "reason": "publish_assertion_verification_skipped_stress_fail",
            "assertions_checked": 0,
            "assertions_failed": 1,
            "active_top_blocker": None,
            "checks": [],
            "failures": ["stress_soak_not_pass"],
        }
    else:
        if bool(args.skip_refresh):
            _emit_progress(
                enabled=progress_enabled,
                phase="refresh_publish_chain",
                message="skipped (skip_refresh_flag)",
            )
            phase_refresh = {
                "status": "SKIPPED",
                "duration_ms": 0,
                "returncode": None,
                "command": None,
                "skipped": True,
                "reason": "skip_refresh_flag",
                "latest_surface_ref": str(refresh_latest_surface_path.relative_to(repo_root)),
            }
        else:
            refresh_cmd = ["bash", str(refresh_script_path), "--json"]
            _emit_progress(
                enabled=progress_enabled,
                phase="refresh_publish_chain",
                message=f"starting timeout_sec={int(args.refresh_timeout_sec)}",
            )
            refresh_run = _run_command(
                refresh_cmd,
                cwd=repo_root,
                timeout_sec=int(args.refresh_timeout_sec),
                env=refresh_env,
            )
            _emit_progress(
                enabled=progress_enabled,
                phase="refresh_publish_chain",
                message=(
                    "completed "
                    f"returncode={refresh_run.get('returncode')} "
                    f"timed_out={bool(refresh_run.get('timed_out'))} "
                    f"duration_ms={int(refresh_run.get('duration_ms') or 0)}"
                ),
            )
            if bool(refresh_run.get("timed_out")):
                raise RuntimeError(
                    "refresh_publish_timeout:"
                    + _truncate(str(refresh_run.get("command") or ""), max_chars=240)
                )

            refresh_status = "PASS" if int(refresh_run.get("returncode") or 0) == 0 else "FAIL_BLOCKED"
            phase_refresh = {
                "status": refresh_status,
                "duration_ms": int(refresh_run.get("duration_ms") or 0),
                "returncode": int(refresh_run.get("returncode") or 0),
                "command": refresh_run.get("command"),
                "skipped": False,
                "latest_surface_ref": str(refresh_latest_surface_path.relative_to(repo_root)),
            }
            if refresh_status != "PASS":
                refresh_inner_error_code = None
                refresh_inner_phase = None
                try:
                    refresh_payload = _parse_json_payload(
                        str(refresh_run.get("stdout") or ""),
                        label="refresh_publish",
                    )
                except Exception:
                    refresh_payload = {}
                if isinstance(refresh_payload, Mapping):
                    refresh_inner_phase = str(refresh_payload.get("phase") or "").strip() or None
                    refresh_error = refresh_payload.get("error") if isinstance(refresh_payload.get("error"), Mapping) else {}
                    refresh_inner_error_code = str(refresh_error.get("code") or "").strip() or None
                phase_refresh["error"] = {
                    "code": "refresh_publish_failed",
                    "inner_phase": refresh_inner_phase,
                    "inner_error_code": refresh_inner_error_code,
                    "stdout_tail": _truncate(str(refresh_run.get("stdout") or ""), max_chars=300),
                    "stderr_tail": _truncate(str(refresh_run.get("stderr") or ""), max_chars=300),
                }

        if phase_refresh.get("status") not in {"PASS", "SKIPPED"}:
            phase_publish = {
                "status": "FAIL_BLOCKED",
                "verdict": "FAIL_BLOCKED",
                "reason": "publish_assertion_verification_skipped_refresh_fail",
                "assertions_checked": 0,
                "assertions_failed": 1,
                "active_top_blocker": None,
                "checks": [],
                "failures": ["refresh_publish_failed"],
            }
        else:
            _emit_progress(
                enabled=progress_enabled,
                phase="publish_assertion_verification",
                message="starting",
            )
            phase_publish = verify_publish_assertions(
                repo_root=repo_root,
                stress_evidence=stress_payload,
                refresh_latest_path=refresh_latest_surface_path,
                require_live_assertions=bool(args.require_live_assertions),
            )
            _emit_progress(
                enabled=progress_enabled,
                phase="publish_assertion_verification",
                message=(
                    "completed "
                    f"verdict={phase_publish.get('verdict')} "
                    f"reason={phase_publish.get('reason')}"
                ),
            )

    refresh_verdict = "PASS" if phase_refresh.get("status") in {"PASS", "SKIPPED"} else "FAIL_BLOCKED"
    publish_verdict = str(phase_publish.get("verdict") or "FAIL_BLOCKED").upper()
    publish_reason = str(phase_publish.get("reason") or "")

    blocked_reasons: List[str] = []
    if stress_verdict != "PASS":
        blocked_reasons.append("stress_soak_not_pass")
    if refresh_verdict != "PASS":
        blocked_reasons.append("refresh_publish_not_pass")
    if publish_verdict != "PASS":
        blocked_reasons.extend(str(row) for row in (phase_publish.get("failures") or []) if str(row).strip())

    repeatability_signature = _repeatability_signature(
        stress_verdict=stress_verdict,
        refresh_verdict=refresh_verdict,
        publish_verdict=publish_verdict,
        publish_reason=publish_reason,
        active_top_blocker=phase_publish.get("active_top_blocker"),
        effective_top_blocker=phase_publish.get("effective_top_blocker"),
        publish_assertions_failed=int(phase_publish.get("assertions_failed") or 0),
    )

    previous_decision_rows = _load_jsonl_rows(decision_log_path, max_rows=20)
    previous_row = previous_decision_rows[-1] if previous_decision_rows else None
    repeatability = _repeatability_evidence(previous_row=previous_row, current_signature=repeatability_signature)

    overall_verdict = "PASS" if not blocked_reasons else "FAIL_BLOCKED"

    generated_at = now_iso()
    run_seed = json_dumps(
        {
            "generated_at": generated_at,
            "stress_run_id": stress_payload.get("run_id"),
            "stress_verdict": stress_verdict,
            "publish_verdict": publish_verdict,
            "publish_reason": publish_reason,
            "active_top_blocker": phase_publish.get("active_top_blocker"),
            "effective_top_blocker": phase_publish.get("effective_top_blocker"),
            "active_top_blocker_alias_resolution": phase_publish.get("active_top_blocker_alias_resolution"),
            "repeatability_status": repeatability.get("status"),
            "repeatability_mismatch_fields": repeatability.get("mismatch_fields"),
            "blocked_reasons": blocked_reasons,
        }
    )
    run_id = _derive_run_id(run_seed=run_seed, output_dir=output_dir)
    run_dir = output_dir / "runs" / run_id

    evidence = {
        "object_type": "clawd.a3_failover_stress_runtime_evidence.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "source_lane": "A3",
        "harness": {
            "component": "continuity.failover_stress_runtime_evidence",
            "version": "v1",
            "stress_runner": "continuity.failover_stress_soak",
        },
        "config": {
            "cycles_per_profile": int(args.cycles),
            "stress_timeout_sec": int(args.stress_timeout_sec),
            "refresh_timeout_sec": int(args.refresh_timeout_sec),
            "refresh_enabled": not bool(args.skip_refresh),
            "require_live_assertions": bool(args.require_live_assertions),
        },
        "summary": {
            "overall_verdict": overall_verdict,
            "stress_verdict": stress_verdict,
            "refresh_verdict": refresh_verdict,
            "publish_chain_verdict": publish_verdict,
            "active_top_blocker": phase_publish.get("active_top_blocker"),
            "effective_top_blocker": phase_publish.get("effective_top_blocker"),
            "active_top_blocker_alias_resolution": (
                phase_publish.get("active_top_blocker_alias_resolution")
                if isinstance(phase_publish.get("active_top_blocker_alias_resolution"), Mapping)
                else None
            ),
            "publish_reason": publish_reason,
            "publish_assertions_checked": int(phase_publish.get("assertions_checked") or 0),
            "publish_assertions_failed": int(phase_publish.get("assertions_failed") or 0),
            "repeatability_signature": repeatability_signature,
            "repeatability": repeatability,
            "blocked_reasons": blocked_reasons,
        },
        "phases": {
            "stress_soak": phase_stress,
            "refresh_publish_chain": phase_refresh,
            "publish_assertion_verification": phase_publish,
        },
        "stress_evidence": {
            "run_id": stress_payload.get("run_id"),
            "summary": dict(stress_summary or {}),
            "live_surface_linkage": dict(stress_payload.get("live_surface_linkage") or {}),
        },
        "repeatability": repeatability,
        "artifacts": {
            "run_dir": str(run_dir.relative_to(repo_root)),
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
            "decision_log_ref": str(decision_log_path.relative_to(repo_root)),
            "latest_ref": str(latest_evidence_path.relative_to(repo_root)),
            "stress_latest_ref": str(stress_latest_evidence_path.relative_to(repo_root)),
            "refresh_latest_ref": str(refresh_latest_surface_path.relative_to(repo_root)),
        },
        "source_refs": [
            {
                "path": str((repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_soak.py").relative_to(repo_root)),
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_soak.py"),
            },
            {
                "path": str((repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_runtime_evidence.py").relative_to(repo_root)),
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_runtime_evidence.py"),
            },
            {
                "path": str(refresh_script_path.relative_to(repo_root)),
                "sha256": file_sha256(refresh_script_path) if refresh_script_path.exists() else None,
            },
        ],
    }

    write_json(run_dir / "evidence.json", evidence)
    write_json(latest_evidence_path, evidence)
    append_jsonl(
        decision_log_path,
        {
            "run_id": run_id,
            "generated_at": generated_at,
            "verdict": overall_verdict,
            "summary": dict(evidence.get("summary") or {}),
            "publish_reason": publish_reason,
            "repeatability_status": repeatability.get("status"),
            "repeatability_match": repeatability.get("match"),
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
        },
    )

    _emit_progress(
        enabled=progress_enabled,
        phase="runtime_evidence",
        message=(
            "completed "
            f"run_id={run_id} "
            f"overall_verdict={overall_verdict}"
        ),
    )

    return evidence


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Guarded failover-stress runtime evidence command")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root (default: auto-detected)")

    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Runtime evidence output directory")
    ap.add_argument("--decision-log", default=DEFAULT_DECISION_LOG, help="Append-only runtime evidence decision log JSONL")
    ap.add_argument("--latest-evidence-path", default=DEFAULT_LATEST_EVIDENCE, help="Latest runtime evidence artifact path")

    ap.add_argument("--cycles", type=int, default=2, help="Stress-soak cycles per profile")
    ap.add_argument("--stress-timeout-sec", type=int, default=120, help="Timeout bound for stress-soak phase")
    ap.add_argument("--refresh-timeout-sec", type=int, default=120, help="Timeout bound for refresh/publish phase")

    ap.add_argument("--stress-output-dir", default=DEFAULT_STRESS_OUTPUT_DIR, help="Stress-soak output directory")
    ap.add_argument("--stress-decision-log", default=DEFAULT_STRESS_DECISION_LOG, help="Stress-soak decision log path")
    ap.add_argument(
        "--stress-latest-evidence-path",
        default=DEFAULT_STRESS_LATEST_EVIDENCE,
        help="Stress-soak latest evidence path",
    )

    ap.add_argument("--refresh-script", default=DEFAULT_REFRESH_SCRIPT, help="Refresh/publish script path")
    ap.add_argument(
        "--refresh-latest-surface-path",
        default=DEFAULT_REFRESH_LATEST_SURFACE,
        help="Reset-ready refresh latest surface artifact path",
    )
    ap.add_argument("--skip-refresh", action="store_true", help="Skip refresh/publish phase and verify current surfaces as-is")
    ap.add_argument(
        "--require-live-assertions",
        action="store_true",
        help="Fail closed when no active top blocker is present for assertion verification",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress phase progress logs on stderr",
    )
    ap.add_argument("--json", action="store_true", help="Print runtime evidence JSON")
    return ap


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = run(args)
    except Exception as exc:
        error_payload = {
            "object_type": "clawd.a3_failover_stress_runtime_evidence.v1",
            "run_id": None,
            "generated_at": now_iso(),
            "summary": {
                "overall_verdict": "FAIL_BLOCKED",
                "stress_verdict": "FAIL_BLOCKED",
                "refresh_verdict": "FAIL_BLOCKED",
                "publish_chain_verdict": "FAIL_BLOCKED",
                "active_top_blocker": None,
                "publish_assertions_checked": 0,
                "publish_assertions_failed": 1,
                "blocked_reasons": ["runtime_error"],
            },
            "error": {
                "reason": str(exc),
            },
        }
        if bool(args.json):
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print(f"BLOCKER: failover stress runtime evidence command failed: {exc}", file=sys.stderr)
        return 2

    verdict = str((((payload or {}).get("summary") or {}).get("overall_verdict") or "FAIL_BLOCKED")).upper()
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{verdict}: failover_runtime_evidence_run={payload.get('run_id')}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
