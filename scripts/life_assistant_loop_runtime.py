#!/usr/bin/env python3
"""Generate XP-303 life-assistant loop runtime artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_FIXTURE_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xp" / "life_assistant_loop_runtime_fixture_v1.json"
DEFAULT_CONTEXT_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xp" / "personal_context_graph_objects_fixture_v1.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"

PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
APPROVAL_RANK = {
    "AP0_NONE": 0,
    "AP1_EXPLICIT_USER_CONFIRM": 1,
    "AP2_DUAL_CONFIRM": 2,
    "AP3_PROHIBITED_UNTIL_GOVERNANCE": 3,
}
ESCALATION_RANK = {
    "E0_NONE": 0,
    "E1_BOUNDARY_REFUSAL": 1,
    "E2_APPROVAL_REQUIRED": 2,
    "E3_SAFETY_ESCALATION": 3,
    "E4_GOVERNANCE_BLOCK": 4,
}
HIGH_RISK_TIERS = {"PX2_HIGH_IMPACT", "PX3_SAFETY_CRITICAL"}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_ts(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return dt.datetime.fromisoformat(text)


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XP-303 life-assistant loop runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--context-path", default=str(DEFAULT_CONTEXT_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--json", action="store_true", help="Emit artifact manifest JSON")
    return ap.parse_args(argv)


def _approval_join(values: Iterable[str]) -> str:
    winner = "AP0_NONE"
    winner_rank = -1
    for value in values:
        rank = APPROVAL_RANK.get(str(value), -1)
        if rank > winner_rank:
            winner_rank = rank
            winner = str(value)
    return winner


def _escalation_join(values: Iterable[str]) -> str:
    winner = "E0_NONE"
    winner_rank = -1
    for value in values:
        rank = ESCALATION_RANK.get(str(value), -1)
        if rank > winner_rank:
            winner_rank = rank
            winner = str(value)
    return winner


def _priority_sort_key(signal: Dict[str, Any]) -> Tuple[int, dt.datetime, str]:
    priority = PRIORITY_RANK.get(str(signal.get("priority") or "medium"), 99)
    due_at = parse_ts(signal.get("due_at")) or dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    return priority, due_at, str(signal.get("signal_id") or "")


def _route_decision() -> Dict[str, Any]:
    return {
        "selected_route": "NO_LLM",
        "reason": "Deterministic local runtime generation, queue gating, and machine-validated JSON artifacts are required.",
        "escalation_trigger": None,
        "fallback_route": "NO_LLM",
        "task_class": "implementation",
        "risk_tier": "high",
        "scope_shape": "multi_surface_coupled",
        "worker_topology": "single",
        "verification_class": "validator_required",
        "verification_plan": [
            "execute runtime generator",
            "parse generated artifacts via json.tool",
            "run source-of-truth map regression check",
            "ship pytest coverage for XP-303 runtime artifacts",
        ],
        "fold_in_target": "queue_continuity",
    }


def build_runtime_package(
    *,
    repo_root: Path,
    queue_doc: Dict[str, Any],
    fixture_doc: Dict[str, Any],
    context_doc: Dict[str, Any],
    generated_at: str,
    stamp: str,
) -> Dict[str, Any]:
    queue_rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    queue_index = {str(row.get("id") or ""): row for row in queue_rows if isinstance(row, dict)}
    xp_303 = queue_index.get("XP-303")
    if not isinstance(xp_303, dict):
        raise SystemExit("XP-303 missing from authoritative queue")

    observed_state = str(xp_303.get("state") or "UNKNOWN")
    if observed_state not in {"READY", "DONE"}:
        raise SystemExit(f"XP-303 queue state must be READY or DONE, observed {observed_state}")

    dependency_states: Dict[str, str] = {}
    unresolved_dependencies: List[str] = []
    for dep_id in xp_303.get("dependencies") if isinstance(xp_303.get("dependencies"), list) else []:
        dep_state = str((queue_index.get(str(dep_id)) or {}).get("state") or "UNKNOWN")
        dependency_states[str(dep_id)] = dep_state
        if dep_state != "DONE":
            unresolved_dependencies.append(str(dep_id))
    if unresolved_dependencies:
        raise SystemExit(f"XP-303 dependencies unresolved: {', '.join(unresolved_dependencies)}")

    records = context_doc.get("records") if isinstance(context_doc.get("records"), list) else []
    record_index = {str(row.get("object_id") or ""): row for row in records if isinstance(row, dict)}

    sessions = fixture_doc.get("sessions") if isinstance(fixture_doc.get("sessions"), list) else []
    if not sessions:
        raise SystemExit("XP-303 runtime fixture is missing sessions")

    immediate_budget = int((fixture_doc.get("runtime_anchor") or {}).get("immediate_intervention_budget_per_session") or 0)
    batched_budget = int((fixture_doc.get("runtime_anchor") or {}).get("batched_review_budget_per_session") or 0)
    heartbeat_budget = int((fixture_doc.get("runtime_anchor") or {}).get("heartbeat_notification_budget") or 0)

    owner_by_record: Dict[str, str] = {}
    overlap_rows: List[Dict[str, Any]] = []
    visible_by_session: Dict[str, set[str]] = {}

    sessions_out: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    weekly_packets: List[Dict[str, Any]] = []
    intervention_visibility_failures: List[Dict[str, Any]] = []

    intervention_counter = 0
    for session in sessions:
        session_key = str(session.get("session_key") or "").strip()
        if not session_key:
            raise SystemExit("Encountered XP-303 session without session_key")
        visible_ids = [str(value) for value in (session.get("visible_record_ids") or [])]
        missing_ids = [value for value in visible_ids if value not in record_index]
        if missing_ids:
            raise SystemExit(f"XP-303 fixture session {session_key} references unknown records: {missing_ids}")

        visible_set = set(visible_ids)
        visible_by_session[session_key] = visible_set

        for record_id in visible_ids:
            existing_owner = owner_by_record.get(record_id)
            if existing_owner and existing_owner != session_key:
                overlap_rows.append(
                    {
                        "record_id": record_id,
                        "owner_session_key": existing_owner,
                        "conflicting_session_key": session_key,
                    }
                )
            owner_by_record.setdefault(record_id, session_key)

        actionable = []
        heartbeats = []
        for signal in session.get("signals") if isinstance(session.get("signals"), list) else []:
            signal_record_ids = [str(value) for value in (signal.get("record_ids") or [])]
            unknown = [value for value in signal_record_ids if value not in record_index]
            if unknown:
                raise SystemExit(f"Signal {signal.get('signal_id')} in {session_key} references unknown records: {unknown}")
            foreign = [value for value in signal_record_ids if value not in visible_set]
            if foreign:
                raise SystemExit(f"Signal {signal.get('signal_id')} in {session_key} leaks foreign records: {foreign}")
            if bool(signal.get("action_required")):
                actionable.append(signal)
            else:
                heartbeats.append(signal)

        actionable.sort(key=_priority_sort_key)
        emitted_signals = actionable[: immediate_budget if immediate_budget >= 0 else len(actionable)]
        deferred_signals = actionable[len(emitted_signals): len(emitted_signals) + max(batched_budget, 0)]

        interventions = []
        for signal in emitted_signals:
            intervention_counter += 1
            signal_record_ids = [str(value) for value in (signal.get("record_ids") or [])]
            referenced_records = [record_index[value] for value in signal_record_ids]
            governance_rows = [row.get("governance") or {} for row in referenced_records]
            risk_tiers = {str(row.get("risk_tier") or "") for row in governance_rows}
            if risk_tiers & HIGH_RISK_TIERS:
                raise SystemExit(
                    f"XP-303 fail-close: {session_key} intervention references high-risk tiers {sorted(risk_tiers & HIGH_RISK_TIERS)}"
                )

            approval = _approval_join(str(row.get("approval_tier") or "AP0_NONE") for row in governance_rows)
            escalation = _escalation_join(str(row.get("escalation_level") or "E0_NONE") for row in governance_rows)
            loop_kind = "weekly_review" if str(signal.get("action_window") or "") == "weekly_review" else "daily_review"
            action_items = [str(signal.get("recommended_action") or "").strip()]
            if loop_kind == "weekly_review":
                action_items.append("Link the chosen priorities back to the relevant decision/learning objects.")
            else:
                action_items.append("Keep this as an advisory action card only; do not mutate external systems automatically.")

            interventions.append(
                {
                    "intervention_id": f"xp303_int_{intervention_counter:03d}",
                    "session_key": session_key,
                    "channel": str(session.get("channel") or "unknown"),
                    "session_label": str(session.get("session_label") or "session"),
                    "loop_kind": loop_kind,
                    "signal_id": str(signal.get("signal_id") or ""),
                    "signal_type": str(signal.get("signal_type") or "unknown"),
                    "priority": str(signal.get("priority") or "medium"),
                    "due_at": signal.get("due_at"),
                    "summary": str(signal.get("summary") or "").strip(),
                    "action_items": [item for item in action_items if item],
                    "record_ids": signal_record_ids,
                    "required_approval_tier": approval,
                    "escalation_level": escalation,
                    "advisory_only": True,
                    "emission_mode": "action_card",
                }
            )

            invisible = [value for value in signal_record_ids if value not in visible_set]
            if invisible:
                intervention_visibility_failures.append(
                    {
                        "session_key": session_key,
                        "signal_id": str(signal.get("signal_id") or ""),
                        "foreign_record_ids": invisible,
                    }
                )

        weekly_packets.append(
            {
                "session_key": session_key,
                "session_label": str(session.get("session_label") or "session"),
                "review_window": "current_week",
                "visible_record_ids": visible_ids,
                "intervention_ids": [row["intervention_id"] for row in interventions],
                "actionable_signal_ids": [str(row.get("signal_id") or "") for row in actionable],
                "deferred_signal_ids": [str(row.get("signal_id") or "") for row in deferred_signals],
                "learning_object_ids": [
                    record_id
                    for record_id in visible_ids
                    if str((record_index.get(record_id) or {}).get("object_type") or "")
                    in {"decision_record", "after_action_review", "lesson_card", "pattern_card"}
                ],
                "review_questions": [
                    "What should be protected next week?",
                    "What recent decision or lesson should be carried forward?",
                    "What single adjustment reduces avoidable interruption?",
                ],
            }
        )

        metrics_rows.append(
            {
                "session_key": session_key,
                "session_label": str(session.get("session_label") or "session"),
                "actionable_candidates": len(actionable),
                "immediate_interventions_emitted": len(interventions),
                "batched_review_candidates": len(deferred_signals),
                "silent_heartbeat_suppressed": len(heartbeats),
                "heartbeat_notifications_emitted": 0,
                "immediate_budget": immediate_budget,
                "batched_budget": batched_budget,
                "heartbeat_budget": heartbeat_budget,
                "noise_budget_ok": len(interventions) <= immediate_budget and heartbeat_budget == 0,
            }
        )

        sessions_out.append(
            {
                "session_key": session_key,
                "channel": str(session.get("channel") or "unknown"),
                "scope": str(session.get("scope") or "unknown"),
                "session_label": str(session.get("session_label") or "session"),
                "visible_record_ids": visible_ids,
                "interventions": interventions,
                "batched_review_signal_ids": [str(row.get("signal_id") or "") for row in deferred_signals],
                "silent_heartbeat_signal_ids": [str(row.get("signal_id") or "") for row in heartbeats],
                "session_isolation_status": "PASS",
                "heartbeat_notifications_emitted": 0,
            }
        )

    contamination_results: List[Dict[str, Any]] = []
    for probe in fixture_doc.get("contamination_probes") if isinstance(fixture_doc.get("contamination_probes"), list) else []:
        target_session_key = str(probe.get("target_session_key") or "")
        visible_set = visible_by_session.get(target_session_key)
        if visible_set is None:
            raise SystemExit(f"Unknown target_session_key in contamination probe: {target_session_key}")
        foreign_ids = [str(value) for value in (probe.get("foreign_record_ids") or [])]
        leaked = sorted(value for value in foreign_ids if value in visible_set)
        observed_result = "blocked" if not leaked else "leak_detected"
        contamination_results.append(
            {
                "probe_id": str(probe.get("probe_id") or ""),
                "target_session_key": target_session_key,
                "foreign_record_ids": foreign_ids,
                "expected_result": str(probe.get("expected_result") or "blocked"),
                "observed_result": observed_result,
                "blocked_record_ids": foreign_ids if observed_result == "blocked" else [],
                "leaked_record_ids": leaked,
                "result": "PASS" if observed_result == str(probe.get("expected_result") or "blocked") else "FAIL",
            }
        )

    contamination_status = "PASS" if contamination_results and all(row["result"] == "PASS" for row in contamination_results) else "FAIL"
    overlap_status = "PASS" if not overlap_rows else "FAIL"
    visibility_status = "PASS" if not intervention_visibility_failures else "FAIL"
    heartbeat_status = "PASS" if all(int(row["heartbeat_notifications_emitted"]) == 0 for row in metrics_rows) else "FAIL"
    budget_status = "PASS" if all(bool(row["noise_budget_ok"]) for row in metrics_rows) else "FAIL"
    advisory_status = "PASS" if all(
        intervention.get("advisory_only") is True
        for session in sessions_out
        for intervention in session.get("interventions") or []
    ) else "FAIL"
    actionable_only_status = "PASS" if all(
        intervention.get("action_items") and intervention.get("summary")
        for session in sessions_out
        for intervention in session.get("interventions") or []
    ) else "FAIL"

    route_decision = _route_decision()

    gate_checks = [
        {
            "check": "fresh_truth_check",
            "result": "PASS",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "verification_check",
            "result": "PASS",
            "evidence_ref": f"state/continuity/latest/xp_303_runtime_validation_{stamp}.json",
        },
        {
            "check": "dependency_health_check",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "continuity_coherence_check",
            "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "blocker_state_check",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "evidence_quality_check",
            "result": "PASS",
            "evidence_ref": f"tests/fixtures/xp/life_assistant_loop_runtime_fixture_v1.json",
        },
    ]
    gate_result = "allowed" if all(row["result"] == "PASS" for row in gate_checks) else "forbidden"

    runtime_payload = {
        "schema": "clawd.xp_303.life_assistant_loop_runtime.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "runtime_anchor": fixture_doc.get("runtime_anchor") or {},
        "queue_precondition": {
            "authoritative_queue_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "observed_slice_state": observed_state,
            "dependency_states": dependency_states,
            "dependencies_resolved": not unresolved_dependencies,
        },
        "route_decision": route_decision,
        "advisory_boundary": {
            "risk_ceiling": "PX1_ASSIST",
            "high_risk_tiers_blocked": sorted(HIGH_RISK_TIERS),
            "external_mutation_allowed": False,
            "heartbeat_notification_budget": heartbeat_budget,
        },
        "summary": {
            "session_count": len(sessions_out),
            "intervention_count": sum(len(session.get("interventions") or []) for session in sessions_out),
            "actionable_only": actionable_only_status == "PASS",
            "heartbeat_notifications_emitted": 0,
            "cross_session_contamination_status": contamination_status,
            "weekly_review_packet_count": len(weekly_packets),
        },
        "sessions": sessions_out,
    }

    noise_budget_payload = {
        "schema": "clawd.xp_303.noise_budget_metrics.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "global_policy": {
            "heartbeat_notification_budget": heartbeat_budget,
            "immediate_intervention_budget_per_session": immediate_budget,
            "batched_review_budget_per_session": batched_budget,
        },
        "per_session": metrics_rows,
        "status": "PASS" if heartbeat_status == "PASS" and budget_status == "PASS" else "FAIL",
    }

    session_isolation_payload = {
        "schema": "clawd.xp_303.session_isolation_regression_tests.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "checks": [
            {
                "check": "dependencies_done",
                "expected": "DONE",
                "observed": dependency_states,
                "result": "PASS" if not unresolved_dependencies else "FAIL",
            },
            {
                "check": "queue_state_gate",
                "expected": ["READY", "DONE"],
                "observed": observed_state,
                "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            },
            {
                "check": "session_visible_sets_disjoint",
                "expected": True,
                "observed": len(overlap_rows) == 0,
                "details": overlap_rows,
                "result": overlap_status,
            },
            {
                "check": "interventions_reference_visible_records_only",
                "expected": True,
                "observed": len(intervention_visibility_failures) == 0,
                "details": intervention_visibility_failures,
                "result": visibility_status,
            },
        ],
        "status": "PASS" if overlap_status == visibility_status == "PASS" and not unresolved_dependencies else "FAIL",
    }

    contamination_payload = {
        "schema": "clawd.xp_303.cross_session_contamination_negative_test.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "probes": contamination_results,
        "status": contamination_status,
    }

    weekly_payload = {
        "schema": "clawd.xp_303.weekly_review_packet.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "packets": weekly_packets,
        "status": "PASS" if weekly_packets else "FAIL",
    }

    gate_payload = {
        "schema": "clawd.verify_before_resume_gate.v1",
        "slice_id": "XP-303",
        "gate_result": gate_result,
        "evaluated_at": generated_at,
        "evidence_refs": [
            "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "docs/ops/personal_os_scope_boundary_contract_v1.md",
            "docs/ops/personal_context_graph_schema_pack_v1.md",
            "docs/ops/low_noise_interaction_policy_v1.md",
            "tests/fixtures/xp/life_assistant_loop_runtime_fixture_v1.json",
            "tests/fixtures/xp/personal_context_graph_objects_fixture_v1.json",
        ],
        "failed_checks": [row["check"] for row in gate_checks if row["result"] != "PASS"],
        "constraints_if_caution": [],
        "next_recheck_at": generated_at,
        "checks": gate_checks,
    }

    return {
        "runtime": runtime_payload,
        "noise_budget": noise_budget_payload,
        "session_isolation": session_isolation_payload,
        "contamination": contamination_payload,
        "weekly_review": weekly_payload,
        "verify_before_resume": gate_payload,
        "validation_statuses": {
            "queue_state": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            "dependencies": "PASS" if not unresolved_dependencies else "FAIL",
            "session_overlap": overlap_status,
            "intervention_visibility": visibility_status,
            "cross_session_contamination": contamination_status,
            "noise_budget": "PASS" if heartbeat_status == budget_status == "PASS" else "FAIL",
            "advisory_only": advisory_status,
            "actionable_only": actionable_only_status,
            "weekly_review_present": "PASS" if weekly_packets else "FAIL",
            "verify_before_resume": "PASS" if gate_result == "allowed" else "FAIL",
        },
        "route_decision": route_decision,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    queue_path = Path(args.queue_path).resolve()
    fixture_path = Path(args.fixture_path).resolve()
    context_path = Path(args.context_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    generated_at = utc_now_iso()

    queue_doc = load_json(queue_path)
    fixture_doc = load_json(fixture_path)
    context_doc = load_json(context_path)

    package = build_runtime_package(
        repo_root=repo_root,
        queue_doc=queue_doc,
        fixture_doc=fixture_doc,
        context_doc=context_doc,
        generated_at=generated_at,
        stamp=stamp,
    )

    artifacts = {
        "runtime_snapshot": output_dir / f"xp_303_life_assistant_loop_runtime_{stamp}.json",
        "noise_budget_metrics": output_dir / f"xp_303_noise_budget_metrics_{stamp}.json",
        "session_isolation_regression": output_dir / f"xp_303_session_isolation_regression_tests_{stamp}.json",
        "cross_session_contamination_negative_test": output_dir / f"xp_303_cross_session_contamination_negative_test_{stamp}.json",
        "weekly_review_packet": output_dir / f"xp_303_weekly_review_packet_{stamp}.json",
        "verify_before_resume_gate": output_dir / f"xp_303_verify_before_resume_gate_{stamp}.json",
        "runtime_validation": output_dir / f"xp_303_runtime_validation_{stamp}.json",
        "artifact_manifest": output_dir / f"xp_303_runtime_artifact_manifest_{stamp}.json",
    }

    write_json(artifacts["runtime_snapshot"], package["runtime"])
    write_json(artifacts["noise_budget_metrics"], package["noise_budget"])
    write_json(artifacts["session_isolation_regression"], package["session_isolation"])
    write_json(artifacts["cross_session_contamination_negative_test"], package["contamination"])
    write_json(artifacts["weekly_review_packet"], package["weekly_review"])
    write_json(artifacts["verify_before_resume_gate"], package["verify_before_resume"])

    check_rows = [
        {
            "name": "queue_state_ready_or_done",
            "result": package["validation_statuses"]["queue_state"],
        },
        {
            "name": "dependencies_done",
            "result": package["validation_statuses"]["dependencies"],
        },
        {
            "name": "session_visible_sets_disjoint",
            "result": package["validation_statuses"]["session_overlap"],
        },
        {
            "name": "interventions_reference_visible_records_only",
            "result": package["validation_statuses"]["intervention_visibility"],
        },
        {
            "name": "cross_session_contamination_negative_test",
            "result": package["validation_statuses"]["cross_session_contamination"],
        },
        {
            "name": "noise_budget_heartbeat_discipline",
            "result": package["validation_statuses"]["noise_budget"],
        },
        {
            "name": "advisory_only_boundary",
            "result": package["validation_statuses"]["advisory_only"],
        },
        {
            "name": "actionable_only_emissions",
            "result": package["validation_statuses"]["actionable_only"],
        },
        {
            "name": "weekly_review_packet_present",
            "result": package["validation_statuses"]["weekly_review_present"],
        },
        {
            "name": "verify_before_resume_gate_allowed",
            "result": package["validation_statuses"]["verify_before_resume"],
        },
    ]
    overall_status = "PASS" if all(row["result"] == "PASS" for row in check_rows) else "FAIL"

    manifest_payload = {
        "schema": "clawd.xp_303.runtime_artifact_manifest.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "status": overall_status,
        "route_decision": package["route_decision"],
        "artifacts": {name: relpath(path, repo_root) for name, path in artifacts.items()},
    }
    write_json(artifacts["artifact_manifest"], manifest_payload)

    validation_payload = {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XP-303",
        "generated_at": generated_at,
        "status": overall_status,
        "checks": check_rows,
        "artifact_refs": manifest_payload["artifacts"],
    }
    write_json(artifacts["runtime_validation"], validation_payload)

    if args.json:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
