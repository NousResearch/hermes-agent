#!/usr/bin/env python3
"""Deterministic A3 failover/succession stress-soak harness.

This MVP harness exercises bounded synthetic load profiles against the canonical
failover FSM and successor-safe proof helpers, then emits contract-shaped stress
artifacts for operator surfaces and validation.
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
DEFAULT_OUTPUT_DIR = "state/continuity/a3_failover_stress_soak"
DEFAULT_DECISION_LOG = "state/continuity/a3_failover_stress_soak/decisions.jsonl"
DEFAULT_LATEST_EVIDENCE = "state/continuity/latest/failover_stress_soak_evidence.json"
FRESHNESS_TTL_SEC = 6 * 60 * 60
_BASE_TS = dt.datetime(2026, 3, 26, 0, 0, 0, tzinfo=dt.timezone.utc)


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


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_load_spec_failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_iso(*, profile_index: int, cycle_index: int, tick_index: int) -> str:
    ts = _BASE_TS + dt.timedelta(seconds=(profile_index * 10_000) + (cycle_index * 100) + tick_index)
    return ts.isoformat().replace("+00:00", "Z")


def _proof_base(*, proof_id: str, expires_at: str = "2026-03-27T00:30:00Z") -> Dict[str, Any]:
    return {
        "object_type": "clawd.successor_safe_handover_proof.v1",
        "proof_id": proof_id,
        "proof_generation_id": "cohgen_1",
        "produced_at": "2026-03-26T00:00:00Z",
        "expires_at": expires_at,
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


def _cycle_signature(payload: Mapping[str, Any]) -> str:
    stable = {
        "profile_id": payload.get("profile_id"),
        "expected_terminal_state": payload.get("expected_terminal_state"),
        "terminal_state": payload.get("terminal_state"),
        "pass": bool(payload.get("pass")),
        "convergence_tick_index": payload.get("convergence_tick_index"),
        "blocked_ticks": payload.get("blocked_ticks"),
        "reset_ready_ticks": payload.get("reset_ready_ticks"),
        "synthetic_load_units_total": payload.get("synthetic_load_units_total"),
        "transition_path": list(payload.get("transition_path") or []),
        "selected_trigger_path": list(payload.get("selected_trigger_path") or []),
        "step_outcomes": [
            {
                "label": row.get("label"),
                "state": row.get("state"),
                "selected_trigger": row.get("selected_trigger"),
                "proof_state": row.get("proof_state"),
                "verdict": row.get("verdict"),
                "blockers": list(row.get("blockers") or []),
                "synthetic_load_units": row.get("synthetic_load_units"),
            }
            for row in (payload.get("step_outcomes") or [])
        ],
    }
    return hashlib.sha256(json_dumps(stable).encode("utf-8")).hexdigest()


def _workload_signature(profile_rows: List[Mapping[str, Any]]) -> str:
    stable = [
        {
            "profile_id": row.get("profile_id"),
            "pass_count": row.get("pass_count"),
            "fail_count": row.get("fail_count"),
            "blocked_cycle_count": row.get("blocked_cycle_count"),
            "max_transition_tick_latency": row.get("max_transition_tick_latency"),
            "deterministic_signature_count": row.get("deterministic_signature_count"),
            "representative_signature": row.get("representative_signature"),
            "terminal_state_counts": row.get("terminal_state_counts"),
        }
        for row in profile_rows
    ]
    return hashlib.sha256(json_dumps(stable).encode("utf-8")).hexdigest()


def _classify_successor_failure_class(blockers: List[str]) -> str:
    blocker_set = {str(row).strip() for row in blockers if str(row).strip()}
    if "BLK_PROOF_GENERATION_MISMATCH" in blocker_set:
        return "generation_mismatch"
    if "BLK_PROOF_READ_POINTER_MISMATCH" in blocker_set:
        return "read_pointer_mismatch"
    if "BLK_PROOF_VERIFY_GATE_NOT_PASS" in blocker_set:
        return "verify_gate_not_pass"
    if "BLK_PROOF_INVALIDATED" in blocker_set:
        return "proof_invalidated"
    return "other_successor_validation_fail"


def _projected_top_blocker_expectations(*, top_blocker: str) -> List[Any]:
    """Return bounded accepted top-blocker values for projected live assertions.

    Stress-cycle classification keeps the synthetic top blocker at failure-source granularity.
    During full publish refresh, successor-proof status can collapse into an envelope
    blocker (`BLK_PROOF_REFUSED`) or temporarily clear while safe-signals remain blocked.
    Encode only those known bounded variants so assertion checks stay deterministic.
    """

    expected: List[Any] = [top_blocker]
    if top_blocker in {"BLK_PROOF_GENERATION_MISMATCH", "BLK_PROOF_READ_POINTER_MISMATCH"}:
        expected.append("BLK_PROOF_REFUSED")
    elif top_blocker in {"BLK_PROOF_INVALIDATED", "BLK_PROOF_VERIFY_GATE_NOT_PASS"}:
        expected.append(None)

    seen = set()
    deduped: List[Any] = []
    for value in expected:
        marker = json_dumps(value)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


def _projected_live_assertions(*, top_blocker: str) -> List[Dict[str, Any]]:
    top_blocker_expected_any_of = _projected_top_blocker_expectations(top_blocker=top_blocker)
    return [
        {
            "surface_path": "state/handover/latest.json",
            "field_path": "proof_status.top_blocker",
            "expected": top_blocker,
            "expected_any_of": top_blocker_expected_any_of,
        },
        {
            "surface_path": "state/handover/latest.json",
            "field_path": "safe_signals.proof_top_blocker",
            "expected": top_blocker,
            "expected_any_of": top_blocker_expected_any_of,
        },
        {
            "surface_path": "state/handover/latest.json",
            "field_path": "safe_signals.safe_to_resume",
            "expected": False,
        },
        {
            "surface_path": "state/handover/latest.json",
            "field_path": "safe_signals.safe_to_reset",
            "expected": False,
        },
        {
            "surface_path": "state/continuity/latest/reset_ready_refresh_latest.json",
            "field_path": "handover_proof_status.top_blocker",
            "expected": top_blocker,
            "expected_any_of": top_blocker_expected_any_of,
        },
        {
            "surface_path": "state/continuity/latest/reset_ready_refresh_latest.json",
            "field_path": "handover_safe_signals.safe_to_resume",
            "expected": False,
        },
        {
            "surface_path": "state/continuity/latest/reset_ready_refresh_latest.json",
            "field_path": "handover_safe_signals.safe_to_reset",
            "expected": False,
        },
        {
            "surface_path": "state/continuity/latest/successor_safe_handover_proof_status.json",
            "field_path": "top_blocker",
            "expected": top_blocker,
            "expected_any_of": top_blocker_expected_any_of,
        },
        {
            "surface_path": "state/continuity/current.json",
            "field_path": "reset_ready_refresh.path",
            "expected": "state/continuity/latest/reset_ready_refresh_latest.json",
        },
    ]


def _build_live_surface_linkage(cycle_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}

    for cycle in cycle_rows:
        profile_id = str(cycle.get("profile_id") or "")
        cycle_index = int(cycle.get("cycle_index") or 0)
        for step in (cycle.get("step_outcomes") or []):
            if str(step.get("selected_trigger") or "") != "TR_SUCCESSOR_VALIDATION_FAIL":
                continue
            if str(step.get("state") or "") != "BLOCKED_RESET":
                continue

            blockers = [str(row).strip() for row in (step.get("blockers") or []) if str(row).strip()]
            failure_class = _classify_successor_failure_class(blockers)
            top_blocker = blockers[0] if blockers else "BLK_PROOF_STALE_OR_INVALID"
            key = (failure_class, top_blocker)

            row = grouped.get(key)
            if row is None:
                row = {
                    "failure_class": failure_class,
                    "trigger": "TR_SUCCESSOR_VALIDATION_FAIL",
                    "top_blocker": top_blocker,
                    "count": 0,
                    "profiles": set(),
                    "step_labels": set(),
                    "proof_states": set(),
                    "sample_refs": [],
                }
                grouped[key] = row

            row["count"] = int(row.get("count") or 0) + 1
            row["profiles"].add(profile_id)
            row["step_labels"].add(str(step.get("label") or ""))
            proof_state = str(step.get("proof_state") or "").strip()
            if proof_state:
                row["proof_states"].add(proof_state)
            if len(row["sample_refs"]) < 3:
                row["sample_refs"].append(
                    {
                        "profile_id": profile_id,
                        "cycle_index": cycle_index,
                        "step_label": str(step.get("label") or ""),
                    }
                )

    failure_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        row = grouped[key]
        failure_rows.append(
            {
                "failure_class": row["failure_class"],
                "trigger": row["trigger"],
                "top_blocker": row["top_blocker"],
                "count": int(row["count"]),
                "profiles": sorted(x for x in row["profiles"] if x),
                "step_labels": sorted(x for x in row["step_labels"] if x),
                "proof_states": sorted(x for x in row["proof_states"] if x),
                "sample_refs": list(row["sample_refs"]),
                "projected_live_assertions": _projected_live_assertions(top_blocker=row["top_blocker"]),
            }
        )

    return {
        "schema": "clawd.a3_failover_stress_soak.live_surface_linkage.v1",
        "focus": "blocked_reset_successor_failures",
        "surface_refs": {
            "continuity_current": "state/continuity/current.json",
            "reset_ready_refresh_latest": "state/continuity/latest/reset_ready_refresh_latest.json",
            "handover_latest": "state/handover/latest.json",
            "proof_status_latest": "state/continuity/latest/successor_safe_handover_proof_status.json",
        },
        "blocked_reset_successor_failures": failure_rows,
    }


def _apply_step(
    *,
    fsm: Any,
    snapshot: Mapping[str, Any],
    now: str,
    label: str,
    synthetic_load_units: int,
    triggers: List[str],
    blockers: List[str] | None = None,
    proof_state: str | None = None,
    verdict: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    next_snapshot = fsm.reduce_failover_state(
        snapshot,
        triggers=triggers,
        now=now,
        blockers=blockers,
    )
    evaluation = next_snapshot.get("evaluation") if isinstance(next_snapshot.get("evaluation"), Mapping) else {}
    step_row = {
        "label": label,
        "at": now,
        "state": next_snapshot.get("state"),
        "selected_trigger": evaluation.get("selected_trigger"),
        "suppressed_triggers": list(evaluation.get("suppressed_triggers") or []),
        "blockers": list(next_snapshot.get("active_blockers") or []),
        "reset_allowed": bool(next_snapshot.get("reset_allowed")),
        "proof_state": proof_state,
        "verdict": verdict,
        "synthetic_load_units": int(synthetic_load_units),
    }
    return dict(next_snapshot), step_row


def _finalize_cycle(
    *,
    profile_id: str,
    expected_terminal_state: str,
    step_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    transition_path = [str(row.get("state") or "") for row in step_outcomes]
    selected_trigger_path = [str(row.get("selected_trigger") or "") for row in step_outcomes]
    terminal_state = transition_path[-1] if transition_path else "UNKNOWN"
    convergence_tick_index = next(
        (idx + 1 for idx, state in enumerate(transition_path) if state == expected_terminal_state),
        None,
    )
    blocked_ticks = sum(1 for row in step_outcomes if row.get("state") == "BLOCKED_RESET")
    reset_ready_ticks = sum(1 for row in step_outcomes if row.get("state") == "RESET_READY")
    synthetic_load_units_total = sum(int(row.get("synthetic_load_units") or 0) for row in step_outcomes)

    fail_reasons: List[str] = []
    if terminal_state != expected_terminal_state:
        fail_reasons.append(f"terminal_state_mismatch:{terminal_state}")
    if convergence_tick_index is None:
        fail_reasons.append("convergence_not_reached")

    cycle = {
        "profile_id": profile_id,
        "expected_terminal_state": expected_terminal_state,
        "terminal_state": terminal_state,
        "convergence_tick_index": convergence_tick_index,
        "blocked_ticks": blocked_ticks,
        "reset_ready_ticks": reset_ready_ticks,
        "synthetic_load_units_total": synthetic_load_units_total,
        "transition_path": transition_path,
        "selected_trigger_path": selected_trigger_path,
        "step_outcomes": step_outcomes,
        "fail_reasons": fail_reasons,
    }
    cycle["pass"] = not fail_reasons
    cycle["cycle_signature"] = _cycle_signature(cycle)
    return cycle


def _step_with_label(step_outcomes: List[Dict[str, Any]], label: str) -> Dict[str, Any] | None:
    for row in step_outcomes:
        if str(row.get("label") or "") == label:
            return row
    return None


def _append_profile_checks(
    cycle: Dict[str, Any],
    *,
    checks: Mapping[str, bool],
) -> Dict[str, Any]:
    fail_reasons = [str(row) for row in (cycle.get("fail_reasons") or []) if str(row).strip()]
    for check_name, passed in checks.items():
        if bool(passed):
            continue
        fail_reasons.append(f"profile_check_failed:{check_name}")

    cycle["fail_reasons"] = fail_reasons
    cycle["pass"] = not fail_reasons
    cycle["cycle_signature"] = _cycle_signature(cycle)
    return cycle


def _profile_recovery_under_sustained_load(*, fsm: Any, proof_mod: Any, profile_index: int, cycle_index: int) -> Dict[str, Any]:
    tick = 0
    snapshot = fsm.build_state_snapshot(
        state="HEALTHY",
        entered_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    steps: List[Dict[str, Any]] = []

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="warning_burst",
        synthetic_load_units=18,
        triggers=["TR_WARN_THRESHOLD_REACHED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="critical_burst_requires_reset",
        synthetic_load_units=35,
        triggers=["TR_CRITICAL_THRESHOLD_REACHED", "TR_RESET_REQUIRED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="failover_prep",
        synthetic_load_units=28,
        triggers=["TR_FAILOVER_PREP_STARTED"],
    )
    steps.append(step)

    tick += 1
    reset_report = proof_mod.evaluate_reset_readiness_with_proof(
        proof=_proof_base(proof_id=f"proof_recovery_{cycle_index}"),
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        expected_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_readiness_pass",
        synthetic_load_units=12,
        triggers=[proof_mod.reset_readiness_report_to_trigger(reset_report)],
        blockers=list(reset_report.get("blockers") or []),
        proof_state=str(((reset_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(reset_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed",
        synthetic_load_units=10,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=_proof_base(proof_id=f"proof_resume_{cycle_index}"),
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_pass",
        synthetic_load_units=14,
        triggers=[proof_mod.successor_validation_report_to_trigger(successor_report)],
        blockers=list(successor_report.get("blockers") or []),
        proof_state=str(((successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(successor_report.get("verdict") or ""),
    )
    steps.append(step)

    return _finalize_cycle(profile_id="steady_recovery_under_load", expected_terminal_state="HEALTHY", step_outcomes=steps)


def _profile_expired_proof_fail_closed(*, fsm: Any, proof_mod: Any, profile_index: int, cycle_index: int) -> Dict[str, Any]:
    tick = 0
    snapshot = fsm.build_state_snapshot(
        state="HEALTHY",
        entered_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    steps: List[Dict[str, Any]] = []

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="critical_burst_requires_reset",
        synthetic_load_units=42,
        triggers=["TR_CRITICAL_THRESHOLD_REACHED", "TR_RESET_REQUIRED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="failover_prep",
        synthetic_load_units=16,
        triggers=["TR_FAILOVER_PREP_STARTED"],
    )
    steps.append(step)

    for label, load_units in [("expired_proof_reset_check_1", 31), ("expired_proof_reset_check_2", 29), ("expired_proof_reset_check_3", 26)]:
        tick += 1
        report = proof_mod.evaluate_reset_readiness_with_proof(
            proof=_proof_base(proof_id=f"proof_expired_{cycle_index}_{tick}", expires_at="2026-03-25T23:59:59Z"),
            evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
            expected_generation_id="cohgen_1",
            expected_pointer_sha256="sha_cur",
        )
        snapshot, step = _apply_step(
            fsm=fsm,
            snapshot=snapshot,
            now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
            label=label,
            synthetic_load_units=load_units,
            triggers=[proof_mod.reset_readiness_report_to_trigger(report)],
            blockers=list(report.get("blockers") or []),
            proof_state=str(((report.get("proof") or {}).get("proof_state") or "")),
            verdict=str(report.get("verdict") or ""),
        )
        steps.append(step)

    return _finalize_cycle(profile_id="expired_proof_fail_closed_under_load", expected_terminal_state="BLOCKED_RESET", step_outcomes=steps)


def _profile_blocker_clear_then_resume(*, fsm: Any, proof_mod: Any, profile_index: int, cycle_index: int) -> Dict[str, Any]:
    tick = 0
    snapshot = fsm.build_state_snapshot(
        state="HEALTHY",
        entered_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    steps: List[Dict[str, Any]] = []

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="critical_burst_requires_reset",
        synthetic_load_units=40,
        triggers=["TR_CRITICAL_THRESHOLD_REACHED", "TR_RESET_REQUIRED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="failover_prep",
        synthetic_load_units=15,
        triggers=["TR_FAILOVER_PREP_STARTED"],
    )
    steps.append(step)

    tick += 1
    first_reset_report = proof_mod.evaluate_reset_readiness_with_proof(
        proof=_proof_base(proof_id=f"proof_first_expired_{cycle_index}", expires_at="2026-03-25T23:59:59Z"),
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        expected_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_readiness_fail",
        synthetic_load_units=24,
        triggers=[proof_mod.reset_readiness_report_to_trigger(first_reset_report)],
        blockers=list(first_reset_report.get("blockers") or []),
        proof_state=str(((first_reset_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(first_reset_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="blockers_cleared",
        synthetic_load_units=11,
        triggers=["TR_BLOCKERS_CLEARED"],
        blockers=[],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_first",
        synthetic_load_units=9,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    verify_gate_blocked_proof = _proof_base(proof_id=f"proof_verify_gate_not_pass_{cycle_index}")
    verify_gate = verify_gate_blocked_proof.get("safety_inputs", {}).get("verify_gate")
    if isinstance(verify_gate, dict):
        verify_gate["status"] = "FAIL"
    first_successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=verify_gate_blocked_proof,
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_verify_gate_not_pass",
        synthetic_load_units=21,
        triggers=[proof_mod.successor_validation_report_to_trigger(first_successor_report), "TR_BLOCKERS_CLEARED"],
        blockers=list(first_successor_report.get("blockers") or []),
        proof_state=str(((first_successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(first_successor_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="blockers_cleared_after_verify_gate",
        synthetic_load_units=10,
        triggers=["TR_BLOCKERS_CLEARED"],
        blockers=[],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_second",
        synthetic_load_units=8,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    invalidated_proof = _proof_base(proof_id=f"proof_invalidated_{cycle_index}")
    invalidated_proof["status"] = "INVALIDATED"
    invalidated_proof["invalidation"] = {
        "invalidated_at": deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        "reason_codes": ["INV_POST_PROOF_MUTATION"],
        "superseded_by_proof_id": None,
    }
    second_successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=invalidated_proof,
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        post_proof_invalidation_absent=False,
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_fail",
        synthetic_load_units=22,
        triggers=[proof_mod.successor_validation_report_to_trigger(second_successor_report), "TR_BLOCKERS_CLEARED"],
        blockers=list(second_successor_report.get("blockers") or []),
        proof_state=str(((second_successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(second_successor_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="blockers_cleared_again",
        synthetic_load_units=10,
        triggers=["TR_BLOCKERS_CLEARED"],
        blockers=[],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_third",
        synthetic_load_units=8,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    final_successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=_proof_base(proof_id=f"proof_resume_final_{cycle_index}"),
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_pass",
        synthetic_load_units=15,
        triggers=[proof_mod.successor_validation_report_to_trigger(final_successor_report)],
        blockers=list(final_successor_report.get("blockers") or []),
        proof_state=str(((final_successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(final_successor_report.get("verdict") or ""),
    )
    steps.append(step)

    cycle = _finalize_cycle(profile_id="blocker_clear_recovery_under_load", expected_terminal_state="HEALTHY", step_outcomes=steps)

    verify_gate_step = _step_with_label(steps, "successor_validation_verify_gate_not_pass")
    successor_fail_step = _step_with_label(steps, "successor_validation_fail")
    checks = {
        "verify_gate_step_present": verify_gate_step is not None,
        "verify_gate_blocks_reset": bool(verify_gate_step) and verify_gate_step.get("state") == "BLOCKED_RESET",
        "verify_gate_reports_expected_blocker": bool(verify_gate_step)
        and "BLK_PROOF_VERIFY_GATE_NOT_PASS" in (verify_gate_step.get("blockers") or []),
        "verify_gate_selected_trigger": bool(verify_gate_step)
        and verify_gate_step.get("selected_trigger") == "TR_SUCCESSOR_VALIDATION_FAIL",
        "verify_gate_suppresses_blockers_cleared_noise": bool(verify_gate_step)
        and "TR_BLOCKERS_CLEARED" in (verify_gate_step.get("suppressed_triggers") or []),
        "successor_fail_step_present": successor_fail_step is not None,
        "successor_fail_blocks_reset": bool(successor_fail_step) and successor_fail_step.get("state") == "BLOCKED_RESET",
        "successor_fail_selected_trigger": bool(successor_fail_step)
        and successor_fail_step.get("selected_trigger") == "TR_SUCCESSOR_VALIDATION_FAIL",
        "successor_fail_suppresses_blockers_cleared_noise": bool(successor_fail_step)
        and "TR_BLOCKERS_CLEARED" in (successor_fail_step.get("suppressed_triggers") or []),
    }
    return _append_profile_checks(cycle, checks=checks)


def _profile_successor_generation_race_recovery(*, fsm: Any, proof_mod: Any, profile_index: int, cycle_index: int) -> Dict[str, Any]:
    tick = 0
    snapshot = fsm.build_state_snapshot(
        state="HEALTHY",
        entered_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    steps: List[Dict[str, Any]] = []

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="critical_burst_requires_reset",
        synthetic_load_units=38,
        triggers=["TR_CRITICAL_THRESHOLD_REACHED", "TR_RESET_REQUIRED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="failover_prep",
        synthetic_load_units=17,
        triggers=["TR_FAILOVER_PREP_STARTED"],
    )
    steps.append(step)

    tick += 1
    reset_report = proof_mod.evaluate_reset_readiness_with_proof(
        proof=_proof_base(proof_id=f"proof_reset_generation_race_{cycle_index}"),
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        expected_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_readiness_pass",
        synthetic_load_units=13,
        triggers=[proof_mod.reset_readiness_report_to_trigger(reset_report)],
        blockers=list(reset_report.get("blockers") or []),
        proof_state=str(((reset_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(reset_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_first",
        synthetic_load_units=10,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    mismatched_generation_proof = _proof_base(proof_id=f"proof_generation_mismatch_{cycle_index}")
    mismatched_generation_proof["proof_generation_id"] = "cohgen_2"
    pointer_ref = mismatched_generation_proof.get("source_refs", {}).get("continuity_read_pointer")
    if isinstance(pointer_ref, dict):
        pointer_ref["build_generation_id"] = "cohgen_2"
        pointer_ref["coherence_build_generation_id"] = "cohgen_2"
    mismatch_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=mismatched_generation_proof,
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_generation_mismatch",
        synthetic_load_units=23,
        triggers=[proof_mod.successor_validation_report_to_trigger(mismatch_report), "TR_BLOCKERS_CLEARED"],
        blockers=list(mismatch_report.get("blockers") or []),
        proof_state=str(((mismatch_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(mismatch_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="generation_alignment_repaired",
        synthetic_load_units=12,
        triggers=["TR_BLOCKERS_CLEARED"],
        blockers=[],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_second",
        synthetic_load_units=9,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    aligned_successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=_proof_base(proof_id=f"proof_generation_aligned_{cycle_index}"),
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_pass",
        synthetic_load_units=16,
        triggers=[proof_mod.successor_validation_report_to_trigger(aligned_successor_report)],
        blockers=list(aligned_successor_report.get("blockers") or []),
        proof_state=str(((aligned_successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(aligned_successor_report.get("verdict") or ""),
    )
    steps.append(step)

    cycle = _finalize_cycle(
        profile_id="successor_generation_race_recovery_under_load",
        expected_terminal_state="HEALTHY",
        step_outcomes=steps,
    )

    mismatch_step = _step_with_label(steps, "successor_validation_generation_mismatch")
    checks = {
        "mismatch_step_present": mismatch_step is not None,
        "mismatch_blocks_reset": bool(mismatch_step) and mismatch_step.get("state") == "BLOCKED_RESET",
        "mismatch_reports_generation_blocker": bool(mismatch_step)
        and "BLK_PROOF_GENERATION_MISMATCH" in (mismatch_step.get("blockers") or []),
        "mismatch_selected_trigger": bool(mismatch_step)
        and mismatch_step.get("selected_trigger") == "TR_SUCCESSOR_VALIDATION_FAIL",
        "mismatch_suppresses_blockers_cleared_noise": bool(mismatch_step)
        and "TR_BLOCKERS_CLEARED" in (mismatch_step.get("suppressed_triggers") or []),
    }
    return _append_profile_checks(cycle, checks=checks)


def _profile_successor_pointer_drift_recovery(*, fsm: Any, proof_mod: Any, profile_index: int, cycle_index: int) -> Dict[str, Any]:
    tick = 0
    snapshot = fsm.build_state_snapshot(
        state="HEALTHY",
        entered_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    steps: List[Dict[str, Any]] = []

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="critical_burst_requires_reset",
        synthetic_load_units=39,
        triggers=["TR_CRITICAL_THRESHOLD_REACHED", "TR_RESET_REQUIRED"],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="failover_prep",
        synthetic_load_units=18,
        triggers=["TR_FAILOVER_PREP_STARTED"],
    )
    steps.append(step)

    tick += 1
    reset_report = proof_mod.evaluate_reset_readiness_with_proof(
        proof=_proof_base(proof_id=f"proof_reset_pointer_drift_{cycle_index}"),
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        expected_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_readiness_pass",
        synthetic_load_units=13,
        triggers=[proof_mod.reset_readiness_report_to_trigger(reset_report)],
        blockers=list(reset_report.get("blockers") or []),
        proof_state=str(((reset_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(reset_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_first",
        synthetic_load_units=11,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    pointer_drift_proof = _proof_base(proof_id=f"proof_pointer_drift_{cycle_index}")
    pointer_ref = pointer_drift_proof.get("source_refs", {}).get("continuity_read_pointer")
    if isinstance(pointer_ref, dict):
        pointer_ref["sha256"] = "sha_stale"
        pointer_ref["current_sha256"] = "sha_stale"
        pointer_ref["continuity_current_sha256"] = "sha_stale"
    mismatch_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=pointer_drift_proof,
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_pointer_drift_mismatch",
        synthetic_load_units=24,
        triggers=[proof_mod.successor_validation_report_to_trigger(mismatch_report), "TR_BLOCKERS_CLEARED"],
        blockers=list(mismatch_report.get("blockers") or []),
        proof_state=str(((mismatch_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(mismatch_report.get("verdict") or ""),
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="pointer_alignment_repaired",
        synthetic_load_units=12,
        triggers=["TR_BLOCKERS_CLEARED"],
        blockers=[],
    )
    steps.append(step)

    tick += 1
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="reset_executed_second",
        synthetic_load_units=9,
        triggers=["TR_RESET_EXECUTED"],
    )
    steps.append(step)

    tick += 1
    aligned_successor_report = proof_mod.evaluate_successor_resume_validation_with_proof(
        proof=_proof_base(proof_id=f"proof_pointer_aligned_{cycle_index}"),
        successor_generation_id="cohgen_1",
        expected_pointer_sha256="sha_cur",
        evaluated_at=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
    )
    snapshot, step = _apply_step(
        fsm=fsm,
        snapshot=snapshot,
        now=deterministic_iso(profile_index=profile_index, cycle_index=cycle_index, tick_index=tick),
        label="successor_validation_pass",
        synthetic_load_units=16,
        triggers=[proof_mod.successor_validation_report_to_trigger(aligned_successor_report)],
        blockers=list(aligned_successor_report.get("blockers") or []),
        proof_state=str(((aligned_successor_report.get("proof") or {}).get("proof_state") or "")),
        verdict=str(aligned_successor_report.get("verdict") or ""),
    )
    steps.append(step)

    cycle = _finalize_cycle(
        profile_id="successor_pointer_drift_recovery_under_load",
        expected_terminal_state="HEALTHY",
        step_outcomes=steps,
    )

    mismatch_step = _step_with_label(steps, "successor_validation_pointer_drift_mismatch")
    checks = {
        "mismatch_step_present": mismatch_step is not None,
        "mismatch_blocks_reset": bool(mismatch_step) and mismatch_step.get("state") == "BLOCKED_RESET",
        "mismatch_reports_pointer_blocker": bool(mismatch_step)
        and "BLK_PROOF_READ_POINTER_MISMATCH" in (mismatch_step.get("blockers") or []),
        "mismatch_pointer_drift_not_generation": bool(mismatch_step)
        and "BLK_PROOF_GENERATION_MISMATCH" not in (mismatch_step.get("blockers") or []),
        "mismatch_selected_trigger": bool(mismatch_step)
        and mismatch_step.get("selected_trigger") == "TR_SUCCESSOR_VALIDATION_FAIL",
        "mismatch_suppresses_blockers_cleared_noise": bool(mismatch_step)
        and "TR_BLOCKERS_CLEARED" in (mismatch_step.get("suppressed_triggers") or []),
    }
    return _append_profile_checks(cycle, checks=checks)


PROFILE_RUNNERS: List[Dict[str, Any]] = [
    {
        "profile_id": "steady_recovery_under_load",
        "description": "Critical synthetic load converges through reset-ready and successor validation back to HEALTHY.",
        "runner": _profile_recovery_under_sustained_load,
    },
    {
        "profile_id": "expired_proof_fail_closed_under_load",
        "description": "Sustained load with expired proof must converge to BLOCKED_RESET and stay fail-closed.",
        "runner": _profile_expired_proof_fail_closed,
    },
    {
        "profile_id": "blocker_clear_recovery_under_load",
        "description": "Blocked reset under load recovers deterministically once blockers clear and successor validation passes.",
        "runner": _profile_blocker_clear_then_resume,
    },
    {
        "profile_id": "successor_generation_race_recovery_under_load",
        "description": "Successor generation mismatch under load must fail closed, then converge once the proof is reminted against the active generation.",
        "runner": _profile_successor_generation_race_recovery,
    },
    {
        "profile_id": "successor_pointer_drift_recovery_under_load",
        "description": "Successor pointer drift under load must fail closed, then converge once pointer parity is restored and validation reruns.",
        "runner": _profile_successor_pointer_drift_recovery,
    },
]


def execute_profiles(*, fsm: Any, proof_mod: Any, cycles: int) -> Dict[str, Any]:
    profile_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []

    for profile_index, profile in enumerate(PROFILE_RUNNERS):
        profile_cycles: List[Dict[str, Any]] = []
        for cycle_index in range(cycles):
            cycle = profile["runner"](fsm=fsm, proof_mod=proof_mod, profile_index=profile_index, cycle_index=cycle_index)
            cycle["cycle_index"] = cycle_index + 1
            profile_cycles.append(cycle)
            cycle_rows.append(cycle)

        signatures = sorted({str(row.get("cycle_signature") or "") for row in profile_cycles})
        terminal_state_counts: Dict[str, int] = {}
        for row in profile_cycles:
            terminal_state = str(row.get("terminal_state") or "UNKNOWN")
            terminal_state_counts[terminal_state] = terminal_state_counts.get(terminal_state, 0) + 1

        profile_rows.append(
            {
                "profile_id": profile["profile_id"],
                "description": profile["description"],
                "cycles": cycles,
                "pass_count": sum(1 for row in profile_cycles if bool(row.get("pass"))),
                "fail_count": sum(1 for row in profile_cycles if not bool(row.get("pass"))),
                "blocked_cycle_count": sum(1 for row in profile_cycles if row.get("terminal_state") == "BLOCKED_RESET"),
                "expected_terminal_state": profile_cycles[0].get("expected_terminal_state") if profile_cycles else None,
                "terminal_state_counts": terminal_state_counts,
                "max_transition_tick_latency": max(int(row.get("convergence_tick_index") or 0) for row in profile_cycles),
                "synthetic_load_units_total": sum(int(row.get("synthetic_load_units_total") or 0) for row in profile_cycles),
                "representative_signature": signatures[0] if signatures else None,
                "deterministic_signature_count": len(signatures),
                "drift_detected": len(signatures) > 1,
            }
        )

    total_cycles = len(cycle_rows)
    total_ticks = sum(len(row.get("step_outcomes") or []) for row in cycle_rows)
    convergence_fail_count = sum(1 for row in cycle_rows if not bool(row.get("pass")))
    blocked_cycle_count = sum(1 for row in cycle_rows if row.get("terminal_state") == "BLOCKED_RESET")
    stress_drift_detected = any(bool(row.get("drift_detected")) for row in profile_rows)

    return {
        "profiles": profile_rows,
        "cycles": cycle_rows,
        "summary": {
            "profile_count": len(profile_rows),
            "total_cycles": total_cycles,
            "total_ticks": total_ticks,
            "convergence_pass_count": total_cycles - convergence_fail_count,
            "convergence_fail_count": convergence_fail_count,
            "blocked_cycle_count": blocked_cycle_count,
            "synthetic_load_units_total": sum(int(row.get("synthetic_load_units_total") or 0) for row in cycle_rows),
            "max_transition_tick_latency": max(int(row.get("convergence_tick_index") or 0) for row in cycle_rows),
            "stress_drift_detected": stress_drift_detected,
        },
    }


def build_evidence(
    *,
    repo_root: Path,
    run_id: str,
    generated_at: str,
    cycles: int,
    execution: Mapping[str, Any],
    latest_evidence_path: Path,
    run_dir: Path,
    decision_log_path: Path,
) -> Dict[str, Any]:
    profiles = list(execution.get("profiles") or [])
    cycle_rows = list(execution.get("cycles") or [])
    summary = dict(execution.get("summary") or {})
    workload_signature = _workload_signature(profiles)
    live_surface_linkage = _build_live_surface_linkage(cycle_rows)

    determinism = {
        "workload_signature": workload_signature,
        "drift_detected": bool(summary.get("stress_drift_detected")),
        "profile_signatures": [
            {
                "profile_id": row.get("profile_id"),
                "signature_count": row.get("deterministic_signature_count"),
                "representative_signature": row.get("representative_signature"),
            }
            for row in profiles
        ],
    }

    overall_verdict = "PASS"
    if int(summary.get("convergence_fail_count") or 0) > 0 or bool(determinism.get("drift_detected")):
        overall_verdict = "FAIL_BLOCKED"

    evidence = {
        "object_type": "clawd.a3_failover_stress_soak_evidence.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "source_lane": "A3",
        "harness": {
            "component": "continuity.failover_stress_soak",
            "version": "v1",
            "synthetic_load_model": "deterministic_mvp",
        },
        "config": {
            "cycles_per_profile": cycles,
            "profile_count": len(profiles),
            "freshness_ttl_sec": FRESHNESS_TTL_SEC,
        },
        "summary": {
            **summary,
            "overall_verdict": overall_verdict,
        },
        "profiles": profiles,
        "determinism": determinism,
        "live_surface_linkage": live_surface_linkage,
        "artifacts": {
            "run_dir": str(run_dir.relative_to(repo_root)),
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
            "cycle_log_ref": str((run_dir / "cycle_log.json").relative_to(repo_root)),
            "decision_log_ref": str(decision_log_path.relative_to(repo_root)),
            "latest_ref": str(latest_evidence_path.relative_to(repo_root)),
        },
        "source_refs": [
            {
                "path": "ops/openclaw/continuity/failover_fsm.py",
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "failover_fsm.py"),
            },
            {
                "path": "ops/openclaw/continuity/successor_safe_handover_proof.py",
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py"),
            },
            {
                "path": "ops/openclaw/continuity/failover_stress_soak.py",
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "failover_stress_soak.py"),
            },
        ],
    }
    return evidence


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = resolve_path(DEFAULT_REPO_ROOT, str(args.repo_root))
    output_dir = resolve_path(repo_root, str(args.output_dir))
    decision_log_path = resolve_path(repo_root, str(args.decision_log))
    latest_evidence_path = resolve_path(repo_root, str(args.latest_evidence_path))

    ensure_repo_relative_path(repo_root, output_dir, label="output_dir")
    ensure_repo_relative_path(repo_root, decision_log_path, label="decision_log")
    ensure_repo_relative_path(repo_root, latest_evidence_path, label="latest_evidence_path")

    fsm_path = repo_root / "ops" / "openclaw" / "continuity" / "failover_fsm.py"
    proof_path = repo_root / "ops" / "openclaw" / "continuity" / "successor_safe_handover_proof.py"
    if not fsm_path.exists():
        raise FileNotFoundError(f"missing_module:{fsm_path}")
    if not proof_path.exists():
        raise FileNotFoundError(f"missing_module:{proof_path}")

    cycles = int(args.cycles)
    if cycles < 1:
        raise ValueError("invalid_cycles")

    fsm = load_module(fsm_path, "failover_stress_soak_fsm")
    proof_mod = load_module(proof_path, "failover_stress_soak_proof")

    execution = execute_profiles(fsm=fsm, proof_mod=proof_mod, cycles=cycles)
    generated_at = now_iso()
    run_seed = json_dumps({
        "profiles": execution.get("profiles"),
        "summary": execution.get("summary"),
        "cycles": cycles,
    })
    run_id = "a3stress_" + hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:16]
    run_dir = output_dir / "runs" / run_id

    evidence = build_evidence(
        repo_root=repo_root,
        run_id=run_id,
        generated_at=generated_at,
        cycles=cycles,
        execution=execution,
        latest_evidence_path=latest_evidence_path,
        run_dir=run_dir,
        decision_log_path=decision_log_path,
    )

    cycle_log = {
        "object_type": "clawd.a3_failover_stress_soak_cycle_log.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "cycles": execution.get("cycles") or [],
    }

    write_json(run_dir / "evidence.json", evidence)
    write_json(run_dir / "cycle_log.json", cycle_log)
    write_json(latest_evidence_path, evidence)
    append_jsonl(
        decision_log_path,
        {
            "run_id": run_id,
            "generated_at": generated_at,
            "verdict": ((evidence.get("summary") or {}).get("overall_verdict") or "FAIL_BLOCKED"),
            "summary": dict(evidence.get("summary") or {}),
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
        },
    )

    return evidence


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministic failover/succession stress-soak harness")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root (default: auto-detected)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Stress harness artifact directory")
    ap.add_argument("--decision-log", default=DEFAULT_DECISION_LOG, help="Append-only decision log JSONL path")
    ap.add_argument(
        "--latest-evidence-path",
        default=DEFAULT_LATEST_EVIDENCE,
        help="Latest stress evidence output path",
    )
    ap.add_argument("--cycles", type=int, default=3, help="Deterministic cycles per synthetic load profile")
    ap.add_argument("--json", action="store_true", help="Print latest evidence JSON")
    return ap


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = run(args)
    except Exception as exc:
        error_payload = {
            "object_type": "clawd.a3_failover_stress_soak_evidence.v1",
            "run_id": None,
            "generated_at": now_iso(),
            "summary": {
                "overall_verdict": "FAIL_BLOCKED",
                "profile_count": 0,
                "total_cycles": 0,
                "total_ticks": 0,
                "convergence_pass_count": 0,
                "convergence_fail_count": 1,
                "blocked_cycle_count": 0,
                "synthetic_load_units_total": 0,
                "max_transition_tick_latency": 0,
                "stress_drift_detected": False,
            },
            "error": {
                "reason": str(exc),
            },
        }
        if bool(args.json):
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print(f"BLOCKER: failover stress-soak harness failed: {exc}", file=sys.stderr)
        return 2

    verdict = str((((payload or {}).get("summary") or {}).get("overall_verdict") or "FAIL_BLOCKED")).upper()
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{verdict}: failover_stress_soak_run={payload.get('run_id')}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
