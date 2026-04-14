#!/usr/bin/env python3
"""Deterministic A3 replay fixture taxonomy + evidence packet emitter (Wave 3).

This runner executes a bounded set of critical failover/succession replay scenarios,
then emits contract-shaped replay artifacts that can be consumed by rollout/soak gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_OUTPUT_DIR = "state/continuity/wave3_a3_replay"
DEFAULT_DECISION_LOG = "state/continuity/wave3_a3_replay/decisions.jsonl"
DEFAULT_LATEST_INDEX = "state/continuity/latest/wave2_replay_evidence_index.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(payload: Mapping[str, Any]) -> str:
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


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_load_spec_failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proof_base() -> Dict[str, Any]:
    return {
        "object_type": "clawd.successor_safe_handover_proof.v1",
        "proof_id": "proof_fixture_base",
        "proof_generation_id": "cohgen_1",
        "produced_at": "2026-03-20T10:00:00Z",
        "expires_at": "2026-03-20T10:30:00Z",
        "status": "ACTIVE",
        "source_refs": {
            "continuity_read_pointer": {
                "path": "state/continuity/latest/continuity_read_pointer.json",
                "build_generation_id": "cohgen_1",
                "sha256": "sha_cur",
            }
        },
        "coherence_tuple_ref": {},
        "safety_inputs": {
            "verify_gate": {"status": "PASS"},
            "mutation_safety": {"unsafe_in_flight_mutation": False},
            "queue_authority": {"status": "CLEAR"},
            "connector_freshness": {"status": "FRESH"},
        },
        "verdicts": {"reset_safety": "PASS", "resume_safety": "PASS", "blockers": []},
        "invalidation": {"invalidated_at": None, "reason_codes": [], "superseded_by_proof_id": None},
    }


def scenario_trigger_conflict_precedence(*, fsm: Any, proof_mod: Any) -> Dict[str, Any]:
    del proof_mod
    reduced = fsm.reduce_failover_state(
        fsm.build_state_snapshot(
            state="FAILOVER_PREP",
            reset_required=True,
            entered_at="2026-03-20T11:00:00Z",
        ),
        triggers=[
            "TR_WARN_THRESHOLD_CLEARED_STABLE",
            "TR_RESET_READINESS_PASS",
            "TR_RESET_READINESS_FAIL",
        ],
        now="2026-03-20T11:00:10Z",
    )

    selected = str((reduced.get("evaluation") or {}).get("selected_trigger") or "")
    suppressed = set((reduced.get("evaluation") or {}).get("suppressed_triggers") or [])
    observed = "FAIL_BLOCKED" if reduced.get("state") == "BLOCKED_RESET" else "PASS"

    checks = {
        "state_blocked_reset": reduced.get("state") == "BLOCKED_RESET",
        "selected_trigger_fail_closed": selected == "TR_RESET_READINESS_FAIL",
        "suppressed_trigger_contains_reset_readiness_pass": "TR_RESET_READINESS_PASS" in suppressed,
    }

    return {
        "scenario_id": "W2H_A3_F11_TRIGGER_CONFLICT",
        "fixture_family": "F11",
        "taxonomy": ["trigger_precedence", "fail_closed_conflict"],
        "description": "Conflicting reset readiness triggers must choose FAIL and stay blocked-reset.",
        "expected_verdict": "FAIL_BLOCKED",
        "observed_verdict": observed,
        "pass": all(checks.values()) and observed == "FAIL_BLOCKED",
        "blockers": ["BLK_PROOF_STALE_OR_INVALID"] if observed == "FAIL_BLOCKED" else [],
        "details": {
            "selected_trigger": selected,
            "state": reduced.get("state"),
            "suppressed_triggers": sorted(suppressed),
            "checks": checks,
        },
    }


def scenario_proof_expired_blocks_reset(*, fsm: Any, proof_mod: Any) -> Dict[str, Any]:
    proof = _proof_base()
    proof["proof_id"] = "proof_expired"
    proof["expires_at"] = "2026-03-20T10:04:59Z"

    report = proof_mod.evaluate_reset_readiness_with_proof(
        proof=proof,
        evaluated_at="2026-03-20T10:05:00Z",
        expected_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
    )
    trigger = proof_mod.reset_readiness_report_to_trigger(report)
    reduced = fsm.reduce_failover_state(
        fsm.build_state_snapshot(state="FAILOVER_PREP", reset_required=True, entered_at="2026-03-20T10:04:00Z"),
        triggers=[trigger],
        now="2026-03-20T10:05:01Z",
    )

    verdict = str(report.get("verdict") or "")
    blockers = [str(row) for row in (report.get("blockers") or [])]
    observed = "FAIL_BLOCKED" if verdict == "FAIL_BLOCKED" and reduced.get("state") == "BLOCKED_RESET" else "PASS"

    checks = {
        "verdict_fail_blocked": verdict == "FAIL_BLOCKED",
        "blocker_contains_proof_expired": "BLK_PROOF_EXPIRED" in blockers,
        "fsm_state_blocked_reset": reduced.get("state") == "BLOCKED_RESET",
        "trigger_is_reset_readiness_fail": trigger == "TR_RESET_READINESS_FAIL",
    }

    return {
        "scenario_id": "W2H_A3_F10_PROOF_EXPIRED",
        "fixture_family": "F10",
        "taxonomy": ["proof_lifecycle", "reset_readiness"],
        "description": "Expired proof must fail reset readiness and keep FSM blocked-reset.",
        "expected_verdict": "FAIL_BLOCKED",
        "observed_verdict": observed,
        "pass": all(checks.values()) and observed == "FAIL_BLOCKED",
        "blockers": blockers,
        "details": {
            "trigger": trigger,
            "report_verdict": verdict,
            "report_blockers": blockers,
            "state": reduced.get("state"),
            "checks": checks,
        },
    }


def scenario_proof_invalidated_blocks_successor_validation(*, fsm: Any, proof_mod: Any) -> Dict[str, Any]:
    proof = _proof_base()
    proof["proof_id"] = "proof_invalidated"
    proof["status"] = "INVALIDATED"
    proof["invalidation"] = {
        "invalidated_at": "2026-03-20T10:06:00Z",
        "reason_codes": ["INV_POST_PROOF_MUTATION"],
        "superseded_by_proof_id": None,
    }

    report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=proof,
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        post_proof_invalidation_absent=False,
        evaluated_at="2026-03-20T10:06:30Z",
    )
    trigger = proof_mod.successor_validation_report_to_trigger(report)
    reduced = fsm.reduce_failover_state(
        fsm.build_state_snapshot(state="SUCCESSOR_RESUME_VALIDATION", reset_required=True, entered_at="2026-03-20T10:06:15Z"),
        triggers=[trigger],
        now="2026-03-20T10:06:31Z",
    )

    verdict = str(report.get("verdict") or "")
    blockers = [str(row) for row in (report.get("blockers") or [])]
    observed = "FAIL_BLOCKED" if verdict == "FAIL_BLOCKED" and reduced.get("state") == "BLOCKED_RESET" else "PASS"

    checks = {
        "verdict_fail_blocked": verdict == "FAIL_BLOCKED",
        "blocker_contains_proof_invalidated": "BLK_PROOF_INVALIDATED" in blockers,
        "fsm_state_blocked_reset": reduced.get("state") == "BLOCKED_RESET",
        "trigger_is_successor_validation_fail": trigger == "TR_SUCCESSOR_VALIDATION_FAIL",
    }

    return {
        "scenario_id": "W2H_A3_F10_PROOF_INVALIDATED",
        "fixture_family": "F10",
        "taxonomy": ["proof_lifecycle", "successor_resume_validation"],
        "description": "Invalidated proof must fail successor validation and return FSM to blocked-reset.",
        "expected_verdict": "FAIL_BLOCKED",
        "observed_verdict": observed,
        "pass": all(checks.values()) and observed == "FAIL_BLOCKED",
        "blockers": blockers,
        "details": {
            "trigger": trigger,
            "report_verdict": verdict,
            "report_blockers": blockers,
            "state": reduced.get("state"),
            "checks": checks,
        },
    }


def scenario_successor_validation_conflict_fail_precedence(*, fsm: Any, proof_mod: Any) -> Dict[str, Any]:
    del proof_mod
    reduced = fsm.reduce_failover_state(
        fsm.build_state_snapshot(state="SUCCESSOR_RESUME_VALIDATION", reset_required=True, entered_at="2026-03-20T11:01:00Z"),
        triggers=["TR_SUCCESSOR_VALIDATION_PASS", "TR_SUCCESSOR_VALIDATION_FAIL"],
        now="2026-03-20T11:01:10Z",
    )

    selected = str((reduced.get("evaluation") or {}).get("selected_trigger") or "")
    suppressed = set((reduced.get("evaluation") or {}).get("suppressed_triggers") or [])
    observed = "FAIL_BLOCKED" if reduced.get("state") == "BLOCKED_RESET" else "PASS"

    checks = {
        "state_blocked_reset": reduced.get("state") == "BLOCKED_RESET",
        "selected_trigger_successor_fail": selected == "TR_SUCCESSOR_VALIDATION_FAIL",
        "suppressed_contains_successor_pass": "TR_SUCCESSOR_VALIDATION_PASS" in suppressed,
    }

    return {
        "scenario_id": "W2H_A3_F11_SUCCESSOR_VALIDATION_CONFLICT",
        "fixture_family": "F11",
        "taxonomy": ["successor_validation", "trigger_precedence"],
        "description": "Conflicting successor pass/fail triggers must choose fail-closed transition.",
        "expected_verdict": "FAIL_BLOCKED",
        "observed_verdict": observed,
        "pass": all(checks.values()) and observed == "FAIL_BLOCKED",
        "blockers": ["BLK_PROOF_INVALIDATED"] if observed == "FAIL_BLOCKED" else [],
        "details": {
            "selected_trigger": selected,
            "state": reduced.get("state"),
            "suppressed_triggers": sorted(suppressed),
            "checks": checks,
        },
    }


SCENARIO_RUNNERS: List[Callable[..., Dict[str, Any]]] = [
    scenario_trigger_conflict_precedence,
    scenario_proof_expired_blocks_reset,
    scenario_proof_invalidated_blocks_successor_validation,
    scenario_successor_validation_conflict_fail_precedence,
]


def _scenario_signature(scenario_results: List[Dict[str, Any]]) -> str:
    rows = []
    for row in scenario_results:
        details = row.get("details") if isinstance(row.get("details"), Mapping) else {}
        rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "pass": bool(row.get("pass")),
                "observed_verdict": row.get("observed_verdict"),
                "state": details.get("state"),
                "selected_trigger": details.get("selected_trigger"),
                "blockers": list(row.get("blockers") or []),
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute_soak_runs(*, fsm: Any, proof_mod: Any, soak_runs: int) -> Dict[str, Any]:
    run_rows: List[Dict[str, Any]] = []
    signatures: List[str] = []

    for idx in range(soak_runs):
        scenario_results = [runner(fsm=fsm, proof_mod=proof_mod) for runner in SCENARIO_RUNNERS]
        run_pass = all(bool(row.get("pass")) for row in scenario_results)
        signature = _scenario_signature(scenario_results)
        signatures.append(signature)
        run_rows.append(
            {
                "run_index": idx + 1,
                "pass": run_pass,
                "signature": signature,
                "scenario_results": scenario_results,
            }
        )

    unique_signatures = sorted(set(signatures))
    runs_fail = sum(1 for row in run_rows if not bool(row.get("pass")))

    return {
        "runs": run_rows,
        "summary": {
            "runs_total": soak_runs,
            "runs_pass": soak_runs - runs_fail,
            "runs_fail": runs_fail,
            "deterministic_signatures": unique_signatures,
            "deterministic_signature_count": len(unique_signatures),
            "drift_detected": len(unique_signatures) > 1,
        },
    }


def build_artifacts(
    *,
    repo_root: Path,
    generated_at: str,
    run_id: str,
    scenario_results: List[Dict[str, Any]],
    soak_summary: Mapping[str, Any],
    full_trace: bool,
) -> Dict[str, Dict[str, Any]]:
    scenario_ids = [str(row.get("scenario_id") or "") for row in scenario_results]
    scenario_count = len(scenario_results)
    pass_count = sum(1 for row in scenario_results if bool(row.get("pass")))
    fail_count = scenario_count - pass_count

    soak_runs_total = int(soak_summary.get("runs_total") or 1)
    soak_runs_fail = int(soak_summary.get("runs_fail") or 0)
    soak_drift_detected = bool(soak_summary.get("drift_detected"))

    overall_verdict = "PASS" if fail_count == 0 and soak_runs_fail == 0 and not soak_drift_detected else "FAIL_BLOCKED"

    first_failure = next((row for row in scenario_results if not bool(row.get("pass"))), None)
    first_blocking_gate = str(first_failure.get("scenario_id") or "") if isinstance(first_failure, Mapping) else None
    if first_blocking_gate is None and soak_runs_fail > 0:
        first_blocking_gate = "SOAK_RUN_FAILURE"
    if first_blocking_gate is None and soak_drift_detected:
        first_blocking_gate = "SOAK_DRIFT_DETECTED"

    fixture_manifest = {
        "object_type": "clawd.wave2_replay.fixture_manifest.v1",
        "generated_at": generated_at,
        "fixture_family": "A3_CRITICAL",
        "source_lane": "A3",
        "scenarios": [
            {
                "scenario_id": row.get("scenario_id"),
                "fixture_family": row.get("fixture_family"),
                "taxonomy": list(row.get("taxonomy") or []),
                "description": row.get("description"),
                "expected_verdict": row.get("expected_verdict"),
            }
            for row in scenario_results
        ],
    }

    run_request = {
        "object_type": "clawd.wave2_replay.run_request.v1",
        "run_id": run_id,
        "requested_at": generated_at,
        "mode": "full_trace" if full_trace else "first_fail",
        "scenario_ids": scenario_ids,
        "runner": {
            "component": "continuity.failover_replay_evidence",
            "version": "v1",
        },
    }

    gate_trace = {
        "object_type": "clawd.wave2_replay.gate_trace.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "first_blocking_gate": first_blocking_gate,
        "gates": [
            {
                "gate_id": row.get("scenario_id"),
                "fixture_family": row.get("fixture_family"),
                "status": "pass" if bool(row.get("pass")) else "fail",
                "expected_verdict": row.get("expected_verdict"),
                "observed_verdict": row.get("observed_verdict"),
                "blockers": list(row.get("blockers") or []),
                "details": row.get("details") if full_trace else {
                    "selected_trigger": (row.get("details") or {}).get("selected_trigger"),
                    "state": (row.get("details") or {}).get("state"),
                },
            }
            for row in scenario_results
        ],
    }

    decision = {
        "object_type": "clawd.wave2_replay.decision.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "verdict": overall_verdict,
        "first_blocking_gate": first_blocking_gate,
        "summary": {
            "scenario_count": scenario_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "soak_runs_total": soak_runs_total,
            "soak_runs_fail": soak_runs_fail,
            "soak_drift_detected": soak_drift_detected,
        },
        "scenario_results": [
            {
                "scenario_id": row.get("scenario_id"),
                "expected_verdict": row.get("expected_verdict"),
                "observed_verdict": row.get("observed_verdict"),
                "pass": bool(row.get("pass")),
                "fixture_family": row.get("fixture_family"),
                "blockers": list(row.get("blockers") or []),
            }
            for row in scenario_results
        ],
    }

    fsm_path = repo_root / "ops" / "openclaw" / "continuity" / "failover_fsm.py"
    proof_path = repo_root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py"

    evidence_index = {
        "object_type": "clawd.wave2_replay.evidence_index.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "source_lane": "A3",
        "fixture_family": "A3_CRITICAL",
        "runner": {
            "component": "continuity.failover_replay_evidence",
            "version": "v1",
        },
        "summary": {
            "scenario_count": scenario_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "overall_verdict": overall_verdict,
            "first_blocking_gate": first_blocking_gate,
            "soak_runs_total": soak_runs_total,
            "soak_runs_fail": soak_runs_fail,
            "soak_drift_detected": soak_drift_detected,
            "soak_deterministic_signature_count": int(soak_summary.get("deterministic_signature_count") or 0),
        },
        "artifacts": {},
        "source_refs": [
            {
                "path": str(fsm_path.relative_to(repo_root)),
                "sha256": file_sha256(fsm_path) if fsm_path.exists() else None,
            },
            {
                "path": str(proof_path.relative_to(repo_root)),
                "sha256": file_sha256(proof_path) if proof_path.exists() else None,
            },
        ],
        "soak": dict(soak_summary),
    }

    return {
        "fixture_manifest": fixture_manifest,
        "run_request": run_request,
        "gate_trace": gate_trace,
        "decision": decision,
        "evidence_index": evidence_index,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = resolve_path(DEFAULT_REPO_ROOT, str(args.repo_root))

    output_dir = resolve_path(repo_root, str(args.output_dir))
    decision_log = resolve_path(repo_root, str(args.decision_log))
    latest_index_path = resolve_path(repo_root, str(args.latest_index_path))

    ensure_repo_relative_path(repo_root, output_dir, label="output_dir")
    ensure_repo_relative_path(repo_root, decision_log, label="decision_log")
    ensure_repo_relative_path(repo_root, latest_index_path, label="latest_index_path")

    fsm_path = repo_root / "ops" / "openclaw" / "continuity" / "failover_fsm.py"
    proof_path = repo_root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py"

    if not fsm_path.exists():
        raise FileNotFoundError(f"missing_module:{fsm_path}")
    if not proof_path.exists():
        raise FileNotFoundError(f"missing_module:{proof_path}")

    fsm = load_module(fsm_path, "failover_fsm_replay_evidence")
    proof_mod = load_module(proof_path, "successor_proof_replay_evidence")

    soak_runs = int(args.soak_runs)
    if soak_runs < 1:
        raise ValueError("invalid_soak_runs")

    soak = execute_soak_runs(fsm=fsm, proof_mod=proof_mod, soak_runs=soak_runs)
    runs = list(soak.get("runs") or [])
    if not runs:
        raise RuntimeError("soak_execution_failed_no_runs")

    scenario_results = list((runs[0] or {}).get("scenario_results") or [])
    soak_summary = (soak.get("summary") or {}) if isinstance(soak.get("summary"), Mapping) else {}

    generated_at = now_iso()
    run_seed = json_dumps({
        "generated_at": generated_at,
        "scenario_results": scenario_results,
        "soak_summary": soak_summary,
    })
    run_id = "w2replay_" + hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:16]

    artifacts = build_artifacts(
        repo_root=repo_root,
        generated_at=generated_at,
        run_id=run_id,
        scenario_results=scenario_results,
        soak_summary=soak_summary,
        full_trace=bool(args.full_trace),
    )

    run_dir = output_dir / "runs" / run_id
    fixture_manifest_path = run_dir / "fixture_manifest.json"
    run_request_path = run_dir / "run_request.json"
    gate_trace_path = run_dir / "gate_trace.json"
    decision_path = run_dir / "decision.json"
    evidence_index_path = run_dir / "evidence_index.json"

    for key, path in [
        ("fixture_manifest", fixture_manifest_path),
        ("run_request", run_request_path),
        ("gate_trace", gate_trace_path),
        ("decision", decision_path),
        ("evidence_index", evidence_index_path),
    ]:
        write_json(path, artifacts[key])

    evidence_index = dict(artifacts["evidence_index"])
    evidence_index["artifacts"] = {
        "run_dir": str(run_dir.relative_to(repo_root)),
        "fixture_manifest_ref": str(fixture_manifest_path.relative_to(repo_root)),
        "run_request_ref": str(run_request_path.relative_to(repo_root)),
        "gate_trace_ref": str(gate_trace_path.relative_to(repo_root)),
        "decision_ref": str(decision_path.relative_to(repo_root)),
    }

    write_json(evidence_index_path, evidence_index)
    write_json(latest_index_path, evidence_index)

    append_jsonl(
        decision_log,
        {
            "run_id": run_id,
            "generated_at": generated_at,
            "verdict": ((artifacts["decision"].get("verdict") or "FAIL_BLOCKED")),
            "summary": dict((artifacts["decision"].get("summary") or {})),
            "evidence_index_ref": str(evidence_index_path.relative_to(repo_root)),
        },
    )

    return evidence_index


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministic failover replay evidence emitter (Wave 3 A3 critical scenarios)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root (default: auto-detected)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Replay artifact output directory")
    ap.add_argument("--decision-log", default=DEFAULT_DECISION_LOG, help="Append-only decision log JSONL path")
    ap.add_argument(
        "--latest-index-path",
        default=DEFAULT_LATEST_INDEX,
        help="Latest evidence index output path",
    )
    ap.add_argument("--full-trace", action="store_true", help="Emit detailed gate-trace details")
    ap.add_argument("--soak-runs", type=int, default=1, help="Replay soak run count (determinism + confidence)")
    ap.add_argument("--json", action="store_true", help="Print evidence index JSON")
    return ap


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = run(args)
    except Exception as exc:
        error_payload = {
            "object_type": "clawd.wave2_replay.evidence_index.v1",
            "run_id": None,
            "generated_at": now_iso(),
            "summary": {
                "scenario_count": 0,
                "pass_count": 0,
                "fail_count": 1,
                "overall_verdict": "FAIL_BLOCKED",
                "first_blocking_gate": "runtime_error",
            },
            "error": {
                "reason": str(exc),
            },
        }
        if bool(args.json):
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print(f"BLOCKER: failover replay evidence emitter failed: {exc}", file=sys.stderr)
        return 2

    verdict = str((((payload or {}).get("summary") or {}).get("overall_verdict") or "FAIL_BLOCKED")).upper()
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{verdict}: replay_evidence_run={payload.get('run_id')}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
