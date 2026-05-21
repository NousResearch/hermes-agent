"""Kanban-first /autopilot command surface.

Autopilot is a bounded controller over the existing Kanban dispatcher.  It owns
mode/focus/policy decisions and ready-gate filtering; the dispatcher remains the
only claim/spawn/retry/accounting substrate.  `/autopilot on` uses the default
policy scope, while `/autopilot on <parent>` narrows the same bounded loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

AUTOPILOT_STATE_FILE = "gateway_autopilot_state.json"
AUTOPILOT_STATE_VERSION = 2
AUTOPILOT_USAGE = "/autopilot [status|dry-run|queue|on|pause [reason]|pause-lane <tenant> [reason]|resume-lane <tenant>|hard-stop <reason>|recover <ack>|off|stop|focus <BO-123>]"
_READ_ONLY_ACTIONS = {"status", "dry-run", "dry_run", "queue"}
_CONTROL_ACTIONS = {"on", "off", "stop", "pause", "pause-lane", "resume-lane", "hard-stop", "recover", "focus", "once"}
_MUTATIONS_ATTEMPTED: list[str] = []
_READY_GATE_REQUIREMENTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("missing_goal", ("goal:", "goal -", "objective:"), "goal"),
    ("missing_end_state", ("end-state", "end state", "output:", "deliverable"), "end-state/output"),
    ("missing_scope_non_goals", ("scope/non-goals", "non-goals", "non goals", "out of scope"), "scope/non-goals"),
    ("missing_acceptance_criteria", ("acceptance criteria", "acceptance:"), "acceptance criteria"),
    ("missing_verification_requirements", ("verification requirements", "verification:", "tests_run", "test:"), "verification requirements"),
    ("missing_authority_boundary", ("authority boundary", "authority:", "kanban"), "authority boundary"),
    ("missing_repo_lane_truth", ("repo/lane truth", "repo_full_name", "repository", "branch"), "repo/lane truth"),
    ("missing_risk_flags", ("risk flags", "risk:", "env", "secret", "prod", "customer-visible", "restart"), "risk flags"),
    ("missing_dependencies_blockers", ("dependencies/blockers", "dependencies:", "blockers:", "none"), "dependencies/blockers"),
    ("missing_review_package_expectation", ("review package", "review_ready", "changed files", "commit"), "review package expectation"),
)
_PR_URL_RE = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/\d+/?$")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
_RELEASE_ONLY_BASES = {"prod", "production"}
_CLOSED_LOOP_ALLOWED_STATES = [
    "disabled",
    "dry_run",
    "single_flight",
    "bounded_multi_tick",
    "parent_scoped",
    "lane_scoped",
    "paused",
    "hard_stopped",
    "needs_human",
]
_DEFAULT_CLOSED_LOOP_CAPS = {
    "max_active_flights": 1,
    "max_dispatches_per_tick": 1,
    "max_tasks_per_run_single_flight": 1,
    "max_tasks_per_run_early_bounded_multi_tick": 2,
    "max_new_prs_per_run": 1,
    "max_open_autopilot_prs": 2,
    "max_consecutive_failures": 1,
    "max_no_progress_ticks": 1,
    "max_same_card_retries": 1,
    "max_runtime_minutes": 60,
    "max_daily_autopilot_tasks": 3,
    "require_clean_closeout_per_task": True,
    "require_review_ready_contract_before_next_task": True,
}


def get_closed_loop_operating_contract() -> dict[str, Any]:
    """Return the BO-091 closed-loop Autopilot ADR/operating contract.

    The contract is intentionally data-shaped so later slices can validate
    policy/config files against the same invariants instead of relying on prose.
    It grants no runtime authority: dispatcher handoff and worker completion
    remain separate later slices and explicit approval gates.
    """

    return {
        "adr": "bounded_controller_not_executor",
        "authority_ceiling": "review_ready_pr",
        "state_machine": {
            "allowed_states": list(_CLOSED_LOOP_ALLOWED_STATES),
            "promotion_ladder": [
                ["dry_run", "single_flight"],
                ["single_flight", "bounded_multi_tick"],
                ["bounded_multi_tick", "parent_scoped"],
                ["bounded_multi_tick", "lane_scoped"],
            ],
            "terminal_or_human_states": ["paused", "hard_stopped", "needs_human"],
        },
        "default_caps": dict(_DEFAULT_CLOSED_LOOP_CAPS),
        "dispatcher_boundary": {
            "execution_owner": "existing_kanban_dispatcher",
            "autopilot_role": "controller_policy_evidence_layer",
            "autopilot_may_directly_claim_or_spawn": False,
            "second_dispatcher_allowed": False,
            "handoff_success_is_worker_completion": False,
            "worker_done_truth_source": "kanban_dispatcher_worker_done_evidence",
        },
        "scope_model": {
            "selectors": ["parent_public_id", "lane_tenant", "repo_project", "labels", "assignee_or_profile"],
            "scope_can_silently_widen": False,
            "scope_escape_result": "needs_human_or_activation_rejected",
        },
        "forbidden_without_current_approval": [
            "gateway_restart_reload",
            "config_env_secret_provider_billing_pricing_mutation",
            "worker_dispatch_claim_spawn_live_side_effect_before_activation_slice",
            "fork_push_or_pr_before_child_range_worker_done",
            "upstream_pr",
            "merge_release_deploy_prod_customer_visible_action",
            "canonical_main_sync_or_materialization",
        ],
        "stop_conditions": [
            "forbidden_action_requested",
            "scope_ambiguity",
            "stale_kanban_state",
            "dependency_or_blocker_detected",
            "verification_failure_threshold_exceeded",
            "worker_crash_or_timeout_repeated",
            "dispatcher_unavailable",
            "kanban_read_unavailable",
            "policy_file_invalid_or_stale",
            "worker_completion_evidence_missing",
            "budget_or_cap_exceeded",
            "pr_backlog_cap_exceeded",
            "disk_ci_runtime_safety_threshold_exceeded",
            "current_turn_approval_required_or_expired",
        ],
        "future_ralplan_required_for": [
            "merge_release_deploy_prod_customer_visible_authority",
            "gateway_restart_reload_automation",
            "config_env_secret_provider_billing_pricing_mutation",
            "replace_or_bypass_existing_dispatcher",
            "new_dispatcher_worker_lifecycle_ownership",
            "global_queue_draining_autonomy",
            "cross_repo_lane_parent_scope_expansion",
            "customer_facing_automation",
            "destructive_cleanup_authority",
            "security_auth_payment_billing_policy_mutation",
        ],
        "docs": "docs/closed-loop-kanban-autopilot.md",
    }


def validate_closed_loop_policy_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate that a proposed closed-loop policy preserves BO-091 invariants."""

    reason_codes: list[str] = []
    if contract.get("adr") != "bounded_controller_not_executor":
        reason_codes.append("adr_must_be_bounded_controller_not_executor")
    if contract.get("authority_ceiling") != "review_ready_pr":
        reason_codes.append("authority_ceiling_must_be_review_ready_pr")
    states = (contract.get("state_machine") or {}).get("allowed_states") or []
    for state in _CLOSED_LOOP_ALLOWED_STATES:
        if state not in states:
            reason_codes.append(f"missing_state_{state}")
    caps = contract.get("default_caps") or {}
    if caps.get("max_dispatches_per_tick") != 1:
        reason_codes.append("max_dispatches_per_tick_must_be_one")
    if caps.get("max_active_flights") != 1:
        reason_codes.append("max_active_flights_must_be_one")
    boundary = contract.get("dispatcher_boundary") or {}
    if boundary.get("execution_owner") != "existing_kanban_dispatcher":
        reason_codes.append("execution_owner_must_be_existing_dispatcher")
    if boundary.get("autopilot_may_directly_claim_or_spawn") is not False:
        reason_codes.append("direct_claim_or_spawn_not_allowed")
    if boundary.get("second_dispatcher_allowed") is not False:
        reason_codes.append("second_dispatcher_not_allowed")
    if boundary.get("handoff_success_is_worker_completion") is not False:
        reason_codes.append("handoff_success_must_not_equal_worker_completion")
    scope = contract.get("scope_model") or {}
    if scope.get("scope_can_silently_widen") is not False:
        reason_codes.append("scope_must_not_silently_widen")
    forbidden = set(contract.get("forbidden_without_current_approval") or [])
    for required in {"gateway_restart_reload", "config_env_secret_provider_billing_pricing_mutation"}:
        if required not in forbidden:
            reason_codes.append(f"missing_forbidden_{required}")
    future = set(contract.get("future_ralplan_required_for") or [])
    if "merge_release_deploy_prod_customer_visible_authority" not in future:
        reason_codes.append("future_ralplan_gate_missing_merge_release_prod")
    return {"ok": not reason_codes, "reason_codes": reason_codes}


def simulate_closed_loop_ticks(candidates: list[dict[str, Any]], *, max_ticks: Optional[int] = None) -> dict[str, Any]:
    """Simulate closed-loop candidate selection without side effects.

    This is BO-092's read-only simulator: it reuses the same Ready-gate and
    dispatcher-eligibility contract as the live path, but returns only
    ``would_*`` facts. It does not dispatch, claim, spawn, or mutate Kanban.
    """

    state = _read_state()
    effective = _effective_mode(str(state.get("desired_mode") or "disabled"), state)
    cap = max(1, int(max_ticks or _DEFAULT_CLOSED_LOOP_CAPS["max_tasks_per_run_early_bounded_multi_tick"]))
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    pauses: list[dict[str, Any]] = []

    if effective == "hard_stop":
        pauses.append({"reason_code": "hard_stop", "human_reason": "Autopilot hard-stop is active; no simulated handoff is allowed."})
        for candidate in candidates:
            skipped.append({
                "public_id": candidate.get("public_id"),
                "task_id": candidate.get("id"),
                "status": "rejected",
                "reason_codes": ["autopilot_hard_stop_active"],
            })
        return {
            "mode": "read_only_simulation",
            "controller_effective_mode": effective,
            "would_select": selected,
            "would_skip": skipped,
            "would_pause": pauses,
            "would_handoff": handoffs,
            "next_state": "hard_stopped",
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }

    verdict = evaluate_dispatcher_eligibility(candidates)
    for item in verdict.get("ineligible") or []:
        skipped.append({
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "status": item.get("status"),
            "reason_codes": item.get("reason_codes") or [],
            "human_reason": item.get("human_reason"),
        })
    for item in (verdict.get("eligible") or [])[:cap]:
        selected_item = {
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "reason_codes": [],
            "decision": "would_select",
        }
        selected.append(selected_item)
        handoffs.append({
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "target": "existing_kanban_dispatcher",
            "check_only": True,
            "would_dispatch": False,
            "handoff_success_is_worker_completion": False,
        })
    if len(verdict.get("eligible") or []) > cap:
        for item in (verdict.get("eligible") or [])[cap:]:
            skipped.append({
                "public_id": item.get("public_id"),
                "task_id": item.get("task_id"),
                "status": "skipped",
                "reason_codes": ["max_ticks_cap_reached"],
                "human_reason": f"simulation cap reached: max_ticks={cap}",
            })
    if not selected:
        pauses.append({"reason_code": "no_progress", "human_reason": "No eligible candidate would be selected in this simulation tick."})
    return {
        "mode": "read_only_simulation",
        "controller_effective_mode": verdict.get("controller_effective_mode", effective),
        "max_ticks": cap,
        "would_select": selected,
        "would_skip": skipped,
        "would_pause": pauses,
        "would_handoff": handoffs,
        "next_state": "continue" if selected else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }

def _candidate_active_flight(candidate: dict[str, Any]) -> Optional[dict[str, Any]]:
    status = str(candidate.get("status") or "").lower()
    claim_lock = candidate.get("claim_lock")
    worker_pid = candidate.get("worker_pid")
    current_run_id = candidate.get("current_run_id")
    if status not in {"running", "claimed", "in_progress"} and not any([claim_lock, worker_pid, current_run_id]):
        return None
    return {
        "public_id": candidate.get("public_id"),
        "task_id": candidate.get("task_id") or candidate.get("id"),
        "current_run_id": current_run_id,
        "claim_lock": claim_lock,
        "worker_pid": worker_pid,
    }


def activate_single_flight(
    candidates: list[dict[str, Any]],
    *,
    check_only_handoff: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the BO-093 single-flight activation gate without live dispatch.

    The function selects at most one eligible candidate, performs a caller-
    supplied check-only handoff probe, and returns explicit non-completion
    evidence. It never directly dispatches, claims, spawns, or mutates Kanban.
    """

    active_flights = [flight for candidate in candidates if (flight := _candidate_active_flight(candidate))]
    if active_flights:
        skipped = [
            {
                "public_id": candidate.get("public_id"),
                "task_id": candidate.get("task_id") or candidate.get("id"),
                "status": "skipped",
                "reason_codes": ["active_flight_already_present"],
                "human_reason": "single-flight activation blocked because the scoped Kanban set already has an active worker flight",
            }
            for candidate in candidates
            if not _candidate_active_flight(candidate)
        ]
        return {
            "status": "active_flight_blocked",
            "selected": None,
            "skipped": skipped,
            "handoff": None,
            "check": {"allowed": False, "reason": "active_flight_already_present"},
            "active_flights": active_flights,
            "next_state": "needs_human",
            "handoff_success_is_worker_completion": False,
            "worker_done_observed": False,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }

    simulation = simulate_closed_loop_ticks(candidates, max_ticks=1)
    selected = (simulation.get("would_select") or [])[:1]
    skipped = []
    for item in simulation.get("would_skip") or []:
        reason_codes = item.get("reason_codes") or []
        if reason_codes == ["max_ticks_cap_reached"]:
            item = {**item, "reason_codes": ["single_flight_limit_reached"], "human_reason": "single-flight activation may select at most one candidate"}
        skipped.append(item)
    for extra in (simulation.get("would_select") or [])[1:]:
        skipped.append({
            "public_id": extra.get("public_id"),
            "task_id": extra.get("task_id"),
            "status": "skipped",
            "reason_codes": ["single_flight_limit_reached"],
            "human_reason": "single-flight activation may select at most one candidate",
        })
    if not selected:
        return {
            "status": "no_candidate",
            "selected": None,
            "skipped": skipped,
            "handoff": None,
            "next_state": simulation.get("next_state", "needs_human"),
            "handoff_success_is_worker_completion": False,
            "worker_done_observed": False,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }
    candidate = selected[0]
    payload = {
        "public_id": candidate.get("public_id"),
        "task_id": candidate.get("task_id"),
        "target": "existing_kanban_dispatcher",
        "check_only": True,
        "would_dispatch": False,
        "handoff_success_is_worker_completion": False,
    }
    check = check_only_handoff(payload) if check_only_handoff else {"allowed": True, "reason": "no_check_callback_supplied"}
    allowed = bool((check or {}).get("allowed"))
    return {
        "status": "handoff_check_passed" if allowed else "handoff_check_blocked",
        "selected": candidate,
        "skipped": skipped,
        "handoff": payload,
        "check": check,
        "next_state": "pause_for_worker_observation" if allowed else "needs_human",
        "handoff_success_is_worker_completion": False,
        "worker_done_observed": False,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def dispatch_selected_once(
    candidate: dict[str, Any],
    *,
    board: Optional[str] = None,
    failure_limit: int = 1,
) -> dict[str, Any]:
    """Hand one Autopilot-selected task to the existing Kanban dispatcher.

    This intentionally reuses ``kanban_db.dispatch_once`` with a task-id filter
    so Autopilot remains a selector/policy layer, not a second dispatcher.
    """

    task_id = str(candidate.get("task_id") or candidate.get("id") or "").strip()
    if not task_id:
        return {"dispatched": False, "reason": "missing_task_id", "spawned": []}
    try:
        from hermes_cli import kanban_db as kb
    except Exception as exc:
        return {"dispatched": False, "reason": "kanban_db_unavailable", "error": str(exc), "spawned": []}
    conn = None
    try:
        conn = kb.connect(board=board)
        worker_env = {
            "HERMES_KANBAN_AUTOPILOT": "1",
            "HERMES_KANBAN_REVIEW_READY_PR_REQUIRED": "1",
        }
        repo_full_name = str(candidate.get("repo_full_name") or "").strip()
        if repo_full_name:
            worker_env["HERMES_KANBAN_EXPECTED_REPO_FULL_NAME"] = repo_full_name
        result = kb.dispatch_once(
            conn,
            max_spawn=1,
            failure_limit=failure_limit,
            board=board,
            task_ids=[task_id],
            worker_env=worker_env,
        )
    except Exception as exc:
        return {"dispatched": False, "reason": "dispatcher_error", "error": str(exc), "spawned": []}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    spawned = [
        {"task_id": tid, "assignee": who, "workspace": ws}
        for tid, who, ws in (result.spawned or [])
    ]
    return {
        "dispatched": bool(spawned),
        "target_task_id": task_id,
        "reclaimed": result.reclaimed,
        "crashed": result.crashed,
        "timed_out": result.timed_out,
        "auto_blocked": result.auto_blocked,
        "promoted": result.promoted,
        "spawned": spawned,
        "skipped_unassigned": result.skipped_unassigned,
        "skipped_nonspawnable": result.skipped_nonspawnable,
    }


def evaluate_autopilot_closeout_progress(
    evidence: dict[str, Any],
    *,
    open_autopilot_prs: int = 0,
    max_open_autopilot_prs: int = 2,
) -> dict[str, Any]:
    """Decide whether Autopilot may continue after a worker result.

    BO-094 keeps worker_done observation, review-ready/no-code evidence, and PR
    backlog caps distinct. It never grants merge/release authority.
    """

    reason_codes: list[str] = []
    worker_done = evidence.get("kanban_worker_done") is True
    if not worker_done:
        reason_codes.append("worker_done_not_observed")
    no_code = evidence.get("no_code_task") is True
    review_contract: dict[str, Any] | None = None
    review_equivalent = None
    if no_code:
        if not str(evidence.get("artifact_path") or "").strip():
            reason_codes.append("missing_no_code_artifact")
        if not str(evidence.get("verification") or "").strip():
            reason_codes.append("missing_no_code_verification")
        if evidence.get("boundaries_confirmed") is not True:
            reason_codes.append("missing_boundaries_confirmed")
        if not reason_codes:
            review_equivalent = "no_code_evidence"
    else:
        review_contract = evaluate_review_ready_contract(evidence)
        if not review_contract.get("review_ready"):
            reason_codes.append("missing_review_ready_contract")
    if open_autopilot_prs >= max_open_autopilot_prs:
        reason_codes.append("pr_backlog_cap_reached")
    return {
        "may_continue": not reason_codes,
        "reason_codes": reason_codes,
        "worker_done_observed": worker_done,
        "review_ready_contract": review_contract,
        "review_ready_equivalent": review_equivalent,
        "open_autopilot_prs": open_autopilot_prs,
        "max_open_autopilot_prs": max_open_autopilot_prs,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "next_state": "continue" if not reason_codes else "needs_human",
    }


def run_bounded_multi_tick(
    candidates: list[dict[str, Any]],
    *,
    max_tasks: int = 2,
    closeout_results: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Simulate bounded multi-tick continuation without live side effects."""

    cap = max(1, max_tasks)
    simulation = simulate_closed_loop_ticks(candidates, max_ticks=cap)
    executed: list[dict[str, Any]] = []
    pauses = list(simulation.get("would_pause") or [])
    skipped = []
    for item in simulation.get("would_skip") or []:
        reason_codes = item.get("reason_codes") or []
        if reason_codes == ["max_ticks_cap_reached"]:
            item = {**item, "reason_codes": ["max_tasks_per_run_reached"], "human_reason": f"bounded run cap reached: max_tasks={cap}"}
        skipped.append(item)
    closeouts = closeout_results or []
    for idx, selected in enumerate(simulation.get("would_select") or []):
        if idx >= cap:
            skipped.append({**selected, "status": "skipped", "reason_codes": ["max_tasks_per_run_reached"]})
            continue
        executed.append({**selected, "tick": idx + 1, "handoff": "check_only"})
        if idx < len(closeouts) and closeouts[idx].get("may_continue") is False:
            pauses.append({"reason_code": "closeout_blocked", "human_reason": ",".join(closeouts[idx].get("reason_codes") or [])})
            return {
                "mode": "bounded_multi_tick_simulation",
                "executed": executed,
                "skipped": skipped,
                "would_pause": pauses,
                "next_state": "needs_human",
                "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            }
    if not executed and not pauses:
        pauses.append({"reason_code": "no_progress", "human_reason": "No eligible work executed in bounded run."})
    return {
        "mode": "bounded_multi_tick_simulation",
        "executed": executed,
        "skipped": skipped,
        "would_pause": pauses,
        "next_state": "paused" if executed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


def filter_candidates_for_scope(candidates: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, Any]:
    """Filter candidates to an explicit parent/lane/repo/label scope."""

    in_scope: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    required_labels = set(scope.get("labels") or [])
    for candidate in candidates:
        reasons: list[str] = []
        if scope.get("parent_public_id") and candidate.get("parent_public_id") != scope.get("parent_public_id"):
            reasons.append("parent_scope_mismatch")
        if scope.get("tenant") and candidate.get("tenant") != scope.get("tenant"):
            reasons.append("lane_scope_mismatch")
        if scope.get("repo_full_name") and candidate.get("repo_full_name") != scope.get("repo_full_name"):
            reasons.append("repo_scope_mismatch")
        if required_labels and not required_labels.issubset(set(candidate.get("labels") or [])):
            reasons.append("label_scope_mismatch")
        if candidate.get("relation_type") == "dependency":
            reasons.append("hierarchy_dependency_ambiguous")
        if reasons:
            out.append({"public_id": candidate.get("public_id"), "task_id": candidate.get("id"), "reason_codes": reasons})
        else:
            in_scope.append(candidate)
    return {
        "in_scope": in_scope,
        "out_of_scope": out,
        "scope_can_silently_widen": False,
        "next_state": "continue" if in_scope else "needs_human",
    }


def generate_autopilot_run_report(run: dict[str, Any]) -> dict[str, Any]:
    """Generate a compact operator-readable Autopilot run report."""

    executed = run.get("executed") or run.get("would_select") or []
    skipped = run.get("skipped") or run.get("would_skip") or []
    blocked = run.get("would_pause") or []
    open_prs = run.get("open_prs") or []
    next_state = run.get("next_state") or "unknown"
    lines = ["Autopilot run report", f"next_state={next_state}"]
    if executed:
        lines.append("executed=" + ",".join(str(item.get("public_id") or item.get("task_id")) for item in executed))
    else:
        lines.append("zero work executed")
    if skipped:
        lines.append("skipped=" + ",".join(f"{item.get('public_id') or item.get('task_id')}:{'/'.join(item.get('reason_codes') or [])}" for item in skipped))
    if blocked:
        lines.append("blocked=" + ",".join(str(item.get("reason_code")) for item in blocked))
    if open_prs:
        lines.append("open_prs=" + ",".join(open_prs))
    if run.get("caps"):
        lines.append("caps=" + json.dumps(run.get("caps"), sort_keys=True))
    return {
        "summary": {
            "executed_count": len(executed),
            "skipped_count": len(skipped),
            "blocked_count": len(blocked),
            "open_pr_count": len(open_prs),
            "next_state": next_state,
            "zero_work": len(executed) == 0,
        },
        "text": "\n".join(lines),
    }


def build_autopilot_review_package(
    evidence: dict[str, Any],
    *,
    run_report: Optional[dict[str, Any]] = None,
    expected_repo_full_name: str = "chriskim12/hermes-agent",
) -> dict[str, Any]:
    """Build a review-ready package proof without granting live authority.

    BO-118 packages worker evidence, PR/check truth, and operator run facts in
    one machine-readable shape.  It may prove review readiness, but it never
    grants merge, release, deploy, gateway restart/reload, or worker-completion
    authority from handoff alone.
    """

    closeout = evaluate_autopilot_closeout_progress(evidence)
    review_contract = evaluate_review_ready_contract(evidence, expected_repo_full_name=expected_repo_full_name)
    report = generate_autopilot_run_report(run_report or {})
    reason_codes: list[str] = []
    for code in closeout.get("reason_codes") or []:
        if code not in reason_codes:
            reason_codes.append(code)
    for code in review_contract.get("reason_codes") or []:
        if code not in reason_codes:
            reason_codes.append(code)
    review_ready = bool(closeout.get("may_continue") and review_contract.get("review_ready"))
    pr_url = str(evidence.get("pr_url") or "").strip()
    text_lines = [
        "Autopilot review-ready PR package" if review_ready else "Autopilot review package blocked",
        f"work_id={evidence.get('work_id') or 'unknown'}",
        f"commit={evidence.get('commit') or evidence.get('head_sha') or 'missing'}",
        f"branch={evidence.get('task_branch') or evidence.get('branch') or 'missing'}",
        f"pr={pr_url or 'missing'}",
        f"run_next_state={report['summary']['next_state']}",
        "merge/release/deploy remains forbidden without current-turn approval.",
        "handoff_success_is_worker_completion=False",
    ]
    return {
        "status": "review_package_ready" if review_ready else "review_package_blocked",
        "review_ready": review_ready,
        "reason_codes": reason_codes,
        "work_id": evidence.get("work_id"),
        "commit": evidence.get("commit") or evidence.get("head_sha"),
        "task_branch": evidence.get("task_branch") or evidence.get("branch"),
        "pr": {
            "url": pr_url,
            "base": evidence.get("pr_base"),
            "head": evidence.get("pr_head"),
            "checks_passed": evidence.get("checks_passed") is True,
        },
        "run_report": report,
        "closeout_progress": closeout,
        "review_contract": review_contract,
        "authority": {
            "ceiling": "review_ready_pr",
            "merge_allowed": False,
            "release_allowed": False,
            "deploy_allowed": False,
            "gateway_restart_reload_allowed": False,
            "prod_customer_visible_allowed": False,
            "config_env_secret_mutation_allowed": False,
        },
        "worker_done_observed": evidence.get("kanban_worker_done") is True,
        "handoff_success_is_worker_completion": False,
        "next_state": "review_ready" if review_ready else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "text": "\n".join(text_lines),
    }


def evaluate_autopilot_promotion_policy(proof: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether first live pickup proof may promote to bounded check-only mode.

    BO-119 intentionally promotes only the *policy state* to a narrow,
    parent-scoped, check-only bounded mode. It does not grant live dispatch,
    worker claim/spawn, PR/push, merge, gateway restart/reload, or config/env
    mutation authority. Those remain current-turn approval gates.
    """

    reason_codes: list[str] = []
    parent_public_id = str(proof.get("parent_public_id") or "").strip()
    if not parent_public_id:
        reason_codes.append("missing_parent_scope")
    if proof.get("live_pickup_smoke_passed") is not True:
        reason_codes.append("live_pickup_smoke_missing")
    if proof.get("single_flight_guard_passed") is not True:
        reason_codes.append("single_flight_guard_missing")
    if proof.get("review_package_ready") is not True:
        reason_codes.append("review_package_not_ready")
    worker_done_children = list(proof.get("kanban_worker_done_children") or [])
    if len(worker_done_children) < 3:
        reason_codes.append("insufficient_worker_done_child_proofs")
    if int(proof.get("active_flights") or 0) > 0:
        reason_codes.append("active_flight_present")
    max_open = int(proof.get("max_open_autopilot_prs") or _DEFAULT_CLOSED_LOOP_CAPS["max_open_autopilot_prs"])
    if int(proof.get("open_autopilot_prs") or 0) >= max_open:
        reason_codes.append("pr_backlog_cap_reached")
    requested_mode = str(proof.get("requested_mode") or "bounded_multi_tick")
    allowed_requested_modes = {"bounded_multi_tick", "parent_scoped"}
    if requested_mode not in allowed_requested_modes:
        reason_codes.append("requested_mode_not_allowed")
    if proof.get("request_live_dispatch") is True:
        reason_codes.append("live_dispatch_requires_current_turn_approval")
    if proof.get("request_gateway_restart_reload") is True:
        reason_codes.append("gateway_restart_reload_requires_current_turn_approval")
    if proof.get("request_config_env_secret_mutation") is True:
        reason_codes.append("config_env_secret_mutation_requires_current_turn_approval")
    promotion_allowed = not reason_codes
    authority = {
        "ceiling": "review_ready_pr",
        "worker_dispatch_claim_spawn_allowed": False,
        "push_pr_allowed": False,
        "merge_allowed": False,
        "release_allowed": False,
        "deploy_allowed": False,
        "gateway_restart_reload_allowed": False,
        "prod_customer_visible_allowed": False,
        "config_env_secret_mutation_allowed": False,
    }
    caps = {
        "max_tasks_per_run": _DEFAULT_CLOSED_LOOP_CAPS["max_tasks_per_run_early_bounded_multi_tick"],
        "max_active_flights": _DEFAULT_CLOSED_LOOP_CAPS["max_active_flights"],
        "max_dispatches_per_tick": _DEFAULT_CLOSED_LOOP_CAPS["max_dispatches_per_tick"],
        "max_new_prs_per_run": _DEFAULT_CLOSED_LOOP_CAPS["max_new_prs_per_run"],
        "max_consecutive_failures": _DEFAULT_CLOSED_LOOP_CAPS["max_consecutive_failures"],
        "require_review_ready_contract_before_next_task": True,
    }
    return {
        "promotion_allowed": promotion_allowed,
        "promoted_mode": "bounded_multi_tick_check_only" if promotion_allowed else "blocked",
        "reason_codes": reason_codes,
        "scope": {
            "parent_public_id": parent_public_id or None,
            "scope_can_silently_widen": False,
        },
        "caps": caps,
        "authority": authority,
        "requires_current_turn_approval_for_live_dispatch": True,
        "requires_current_turn_approval_for_push_pr": True,
        "requires_current_turn_approval_for_gateway_restart_reload": True,
        "handoff_success_is_worker_completion": False,
        "worker_done_truth_source": "kanban_dispatcher_worker_done_evidence",
        "next_state": "promote_check_only" if promotion_allowed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


def harden_autopilot_policy(contract: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed policy hardening gate for closed-loop Autopilot."""

    validation = validate_closed_loop_policy_contract(contract)
    accepted = bool(validation.get("ok"))
    return {
        "accepted": accepted,
        "reason_codes": validation.get("reason_codes") or [],
        "next_state": "continue" if accepted else "hard_stopped",
        "recovery_required": not accepted,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


_RECOVERY_ACTIONS = {
    "stale_kanban_state": "pause_and_reread_kanban",
    "kanban_read_unavailable": "pause_and_reread_kanban",
    "worker_crash_or_timeout_repeated": "pause_and_require_worker_evidence",
    "worker_completion_evidence_missing": "pause_and_require_worker_evidence",
    "policy_file_invalid_or_stale": "hard_stop_and_require_recovery_ack",
    "forbidden_action_requested": "hard_stop_and_require_recovery_ack",
    "scope_ambiguity": "pause_and_require_scope_confirmation",
    "budget_or_cap_exceeded": "pause_and_report_cap",
    "pr_backlog_cap_exceeded": "pause_and_report_pr_backlog",
}


def run_autopilot_recovery_drill(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise fail-closed recovery decisions without executing side effects."""

    drills: list[dict[str, Any]] = []
    passed = True
    for scenario in scenarios:
        trigger = str(scenario.get("trigger") or "")
        action = _RECOVERY_ACTIONS.get(trigger)
        if action is None:
            passed = False
            action = "hard_stop_and_require_human_triage"
        drills.append({
            "name": scenario.get("name"),
            "trigger": trigger,
            "action": action,
            "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        })
    return {
        "passed": passed,
        "drills": drills,
        "next_state": "recovered" if passed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


@dataclass(frozen=True)
class AutopilotCommand:
    """Parsed /autopilot command shape."""

    action: str
    raw_args: str = ""
    value: str = ""

    @property
    def read_only(self) -> bool:
        return self.action in _READ_ONLY_ACTIONS


@dataclass(frozen=True)
class AutopilotResult:
    """Structured decision plus gateway/CLI friendly message."""

    ok: bool
    command: Optional[AutopilotCommand]
    message: str
    decision: dict[str, Any]
    fail_closed: bool = False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    return get_hermes_home() / AUTOPILOT_STATE_FILE


def _read_state(path: Optional[Path] = None) -> dict[str, Any]:
    state_path = path or _state_path()
    try:
        raw = state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"exists": False, "enabled": False, "desired_mode": "disabled", "path": str(state_path)}
    except OSError as exc:
        return {
            "exists": False,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": str(exc),
        }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "exists": True,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": f"invalid_json:{exc.msg}",
        }
    if not isinstance(data, dict):
        return {
            "exists": True,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": "state_not_object",
        }
    enabled = bool(data.get("enabled"))
    desired = str(data.get("desired_mode") or ("enabled" if enabled else "disabled")).strip().lower()
    return {**data, "exists": True, "enabled": enabled, "desired_mode": desired, "path": str(state_path)}


def _write_state(state: dict[str, Any], path: Optional[Path] = None) -> dict[str, Any]:
    state_path = path or _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _read_state(state_path)


def _json_object_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def _task_row_to_candidate(row: Any, *, parent_public_id: Optional[str] = None, relation_type: Optional[str] = None) -> dict[str, Any]:
    routing = _json_object_maybe(row["routing_verdict"] if "routing_verdict" in row.keys() else None)
    admission = _json_object_maybe(row["admission_snapshot"] if "admission_snapshot" in row.keys() else None)
    closeout = _json_object_maybe(row["closeout_evidence"] if "closeout_evidence" in row.keys() else None)
    skills: list[str] = []
    if "skills" in row.keys() and row["skills"]:
        try:
            parsed_skills = json.loads(row["skills"])
            if isinstance(parsed_skills, list):
                skills = [str(skill) for skill in parsed_skills if skill]
        except json.JSONDecodeError:
            skills = []
    candidate = {
        "id": row["id"],
        "task_id": row["id"],
        "public_id": row["public_id"] if "public_id" in row.keys() else None,
        "title": row["title"],
        "body": row["body"],
        "status": row["status"],
        "assignee": row["assignee"],
        "tenant": row["tenant"] if "tenant" in row.keys() else None,
        "priority": row["priority"],
        "routing_verdict": routing,
        "admission_snapshot": admission,
        "closeout_evidence": closeout,
        "parent_public_id": parent_public_id,
        "relation_type": relation_type,
        "skills": skills,
        "claim_lock": row["claim_lock"],
        "worker_pid": row["worker_pid"] if "worker_pid" in row.keys() else None,
        "current_run_id": row["current_run_id"] if "current_run_id" in row.keys() else None,
    }
    repo = admission.get("repo_full_name") or (admission.get("repo_intent") or {}).get("repo_full_name")
    if repo:
        candidate["repo_full_name"] = repo
    return candidate


def load_live_kanban_candidates(
    *,
    parent_public_id: Optional[str] = None,
    tenant: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read live Kanban candidates for Autopilot without mutating the board.

    If a parent/focus public id is supplied, only hierarchy children of that
    parent are returned.  Without an explicit parent this deliberately returns
    only ``ready`` tasks, keeping the read side narrow and side-effect free.
    """

    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return []
    with kb.connect() as conn:
        if parent_public_id:
            parent = conn.execute(
                "SELECT id, public_id FROM tasks WHERE public_id = ? OR id = ? LIMIT 1",
                (parent_public_id, parent_public_id),
            ).fetchone()
            if not parent:
                return []
            params: list[Any] = [parent["id"]]
            query = """
                SELECT c.*, l.relation_type AS relation_type, p.public_id AS parent_public_id
                FROM task_links l
                JOIN tasks p ON p.id = l.parent_id
                JOIN tasks c ON c.id = l.child_id
                WHERE l.parent_id = ? AND l.relation_type = 'hierarchy' AND c.status != 'archived'
            """
            if tenant:
                query += " AND c.tenant = ?"
                params.append(tenant)
            query += " ORDER BY c.priority DESC, c.created_at ASC LIMIT ?"
            params.append(max(1, int(limit)))
            rows = conn.execute(query, params).fetchall()
            return [
                _task_row_to_candidate(row, parent_public_id=parent["public_id"] or parent_public_id, relation_type=row["relation_type"])
                for row in rows
            ]
        params = []
        query = "SELECT * FROM tasks WHERE status = 'ready'"
        if tenant:
            query += " AND tenant = ?"
            params.append(tenant)
        query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = conn.execute(query, params).fetchall()
        return [_task_row_to_candidate(row) for row in rows]


def _candidate_scope_for(command: AutopilotCommand) -> dict[str, Any]:
    state = _read_state()
    raw_scope = str(command.value or state.get("focus") or "").strip()
    scope: dict[str, Any] = {"mode": "default_policy", "parent_public_id": None, "tenant": None}
    if raw_scope:
        scope["mode"] = "parent"
        scope["parent_public_id"] = raw_scope.upper()
    return scope


def _filter_candidates_for_scope(candidates: list[dict[str, Any]], scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply explicit focus narrowing to injected candidate lists.

    Live Kanban loads can filter in SQL.  Unit tests and gateway call sites may
    pass a preloaded candidate list; parent-scoped mode must still not escape
    that parent when candidates are injected.
    """

    parent_public_id = str(scope.get("parent_public_id") or "").strip().upper()
    if not parent_public_id:
        return candidates
    # Some call sites pass an already-scoped candidate list that predates the
    # explicit parent_public_id field.  Only enforce injected-list filtering
    # when the list carries parent metadata to compare against.
    if not any(candidate.get("parent_public_id") for candidate in candidates):
        return candidates
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("parent_public_id") or "").strip().upper() == parent_public_id
    ]


def _resolve_candidates(command: AutopilotCommand, candidates: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scope = _candidate_scope_for(command)
    if candidates is not None:
        return _filter_candidates_for_scope(candidates, scope)
    return load_live_kanban_candidates(
        parent_public_id=scope.get("parent_public_id"),
        tenant=scope.get("tenant"),
        limit=50,
    )


def parse_autopilot_args(raw_args: str) -> AutopilotCommand:
    """Parse the deliberately narrow Phase-1 command surface."""

    raw = str(raw_args or "").strip()
    if not raw:
        return AutopilotCommand(action="status", raw_args="")
    parts = raw.split(maxsplit=1)
    action = parts[0].strip().lower().replace("_", "-")
    value = parts[1].strip() if len(parts) > 1 else ""
    if action == "dry_run":
        action = "dry-run"
    if action in {"status", "dry-run", "queue", "on", "off", "stop", "pause", "pause-lane", "resume-lane", "hard-stop", "recover", "focus", "once"}:
        return AutopilotCommand(action=action, raw_args=raw, value=value)
    raise ValueError(f"unsupported /autopilot command: {raw_args!r}; usage: {AUTOPILOT_USAGE}")


def _effective_mode(desired_mode: str, state: dict[str, Any]) -> str:
    desired = str(desired_mode or "disabled").lower()
    if desired == "hard_stopped":
        return "hard_stop"
    if desired in {"stopped", "off", "disabled"}:
        return "stopped" if desired == "stopped" else "degraded"
    if desired == "paused":
        return "paused"
    # Focused `/autopilot on <parent>` narrows the executable controller loop;
    # unscoped `/autopilot on` is also executable, but only through the
    # default-policy selector.  Raw `ready` alone is still insufficient.
    if desired == "on":
        return "parent_scoped" if state.get("focus") else "default_policy_loop"
    if desired == "enabled":
        return "parent_scoped" if state.get("focus") else "blocked"
    if state.get("read_error"):
        return "degraded"
    return "degraded"


def evaluate_autopilot_ready_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a dry-run Ready-gate verdict for one Kanban candidate.

    This deliberately treats raw Kanban ``status=ready`` as insufficient.  The
    candidate must carry an executable contract before a later autopilot phase
    may even consider selection.  The function is pure/read-only: no dispatcher,
    claim, spawn, or Kanban mutation calls are made here.
    """

    body = str(candidate.get("body") or "")
    title = str(candidate.get("title") or "")
    haystack = "\n".join([title, body, json.dumps(candidate, ensure_ascii=False, sort_keys=True)]).lower()
    reason_codes: list[str] = []
    missing_labels: list[str] = []
    if str(candidate.get("status") or "").lower() != "ready":
        reason_codes.append("kanban_status_not_ready")
        missing_labels.append("Kanban status=ready")
    for code, markers, label in _READY_GATE_REQUIREMENTS:
        if not any(marker in haystack for marker in markers):
            reason_codes.append(code)
            missing_labels.append(label)
    routing = candidate.get("routing_verdict") or {}
    if isinstance(routing, str):
        try:
            routing = json.loads(routing)
        except json.JSONDecodeError:
            routing = {"raw": routing}
    verdict = str((routing or {}).get("verdict") or "").strip().lower()
    if not verdict:
        reason_codes.append("missing_routing_verdict")
        missing_labels.append("routing verdict")
    accepted = not reason_codes
    return {
        "task_id": candidate.get("id"),
        "public_id": candidate.get("public_id"),
        "autopilot_ready": accepted,
        "status": "accepted" if accepted else "rejected",
        "reason_codes": reason_codes,
        "human_reason": "ready for autopilot dry-run selection" if accepted else "Missing executable contract fields: " + ", ".join(missing_labels),
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
    }


def evaluate_review_ready_contract(
    evidence: dict[str, Any],
    *,
    expected_repo_full_name: str = "chriskim12/hermes-agent",
) -> dict[str, Any]:
    """Validate the review-ready PR package without allowing merge/release."""

    reason_codes: list[str] = []
    required = {
        "repo_full_name": evidence.get("repo_full_name"),
        "commit": evidence.get("commit") or evidence.get("head_sha"),
        "task_branch": evidence.get("task_branch") or evidence.get("branch"),
        "pr_url": evidence.get("pr_url"),
        "pr_base": evidence.get("pr_base"),
        "pr_head": evidence.get("pr_head"),
    }
    for field, value in required.items():
        if not str(value or "").strip():
            reason_codes.append(f"missing_{field}")
    bool_required = {
        "checks_passed": evidence.get("checks_passed"),
        "worktree_clean": evidence.get("worktree_clean"),
        "kanban_worker_done": evidence.get("kanban_worker_done"),
        "boundaries_confirmed": evidence.get("boundaries_confirmed"),
    }
    for field, value in bool_required.items():
        if value is not True:
            reason_codes.append(f"missing_{field}")
    commit = str(required["commit"] or "").strip()
    if commit and not _SHA_RE.fullmatch(commit):
        reason_codes.append("commit_not_sha_like")
    pr_url = str(required["pr_url"] or "").strip()
    pr_match = _PR_URL_RE.fullmatch(pr_url) if pr_url else None
    if pr_url and not pr_match:
        reason_codes.append("pr_url_not_github_pull_url")
    if pr_match:
        pr_repo = f"{pr_match.group('owner')}/{pr_match.group('repo')}".lower()
        if pr_repo != expected_repo_full_name.lower():
            reason_codes.append("pr_url_repo_mismatch")
    repo = str(required["repo_full_name"] or "").strip().lower()
    if repo and repo != expected_repo_full_name.lower():
        reason_codes.append("repo_full_name_mismatch")
    task_branch = str(required["task_branch"] or "").strip()
    pr_head = str(required["pr_head"] or "").strip()
    if task_branch and pr_head and task_branch != pr_head:
        reason_codes.append("pr_head_task_branch_mismatch")
    pr_base = str(required["pr_base"] or "").strip().lower()
    if pr_base in _RELEASE_ONLY_BASES:
        reason_codes.append("release_base_requires_separate_approval")
    return {
        "review_ready": not reason_codes,
        "status": "review_ready" if not reason_codes else "blocked",
        "reason_codes": reason_codes,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "gateway_restart_reload_allowed": False,
        "human_reason": "review-ready PR contract satisfied; merge/release still forbidden" if not reason_codes else "Review-ready PR contract missing or unsafe: " + ", ".join(reason_codes),
    }


def evaluate_dispatcher_eligibility(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Dry-run which Ready-gate-passing tasks could later reach dispatcher.

    This is a bridge to the existing Kanban dispatcher, not a second dispatcher.
    It returns handoff-shaped facts but performs no claim/spawn/DB transition.
    """

    state = _read_state()
    effective = _effective_mode(str(state.get("desired_mode") or "disabled"), state)
    paused_lanes = {str(lane).lower() for lane in (state.get("paused_lanes") or [])}
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for candidate in candidates:
        gate = evaluate_autopilot_ready_gate(candidate)
        tenant = str(candidate.get("tenant") or "").lower()
        if effective == "hard_stop":
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_hard_stop_active"],
                "human_reason": "Autopilot hard-stop is active; explicit operator recovery is required.",
            }
        elif tenant and tenant in paused_lanes:
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_lane_paused"],
                "human_reason": f"Autopilot lane is paused: {tenant}",
            }
        elif effective not in {"blocked", "default_policy_loop", "parent_scoped", "bounded_multi_tick"}:
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_effective_mode_not_blocked_pending_dispatch_gate"],
                "human_reason": "Autopilot controller is not in a dispatch-enabled mode.",
            }
        if gate["autopilot_ready"]:
            eligible.append(gate)
        else:
            ineligible.append(gate)
    return {
        "controller_effective_mode": effective,
        "paused_lanes": sorted(paused_lanes),
        "handoff_target": "existing_kanban_dispatcher",
        "second_dispatcher_created": False,
        "eligible": eligible,
        "ineligible": ineligible,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
    }


def _queue_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    candidate_list = candidates or []
    results = [evaluate_autopilot_ready_gate(candidate) for candidate in candidate_list]
    dispatcher_eligibility = evaluate_dispatcher_eligibility(candidate_list)
    return {
        "action": command.action,
        "actor": actor,
        "read_only": True,
        "status": "DRY_RUN",
        "reason": "phase3_dispatcher_eligibility_bridge_no_execution_authority",
        "candidates": results,
        "dispatcher_eligibility": dispatcher_eligibility,
        "accepted_count": sum(1 for result in results if result["autopilot_ready"]),
        "rejected_count": sum(1 for result in results if not result["autopilot_ready"]),
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "state_file_enabled_is_execution_proof": False,
    }


def _format_queue_message(decision: dict[str, Any]) -> str:
    lines = [
        "Autopilot queue dry-run: READY-GATE ONLY",
        f"accepted={decision['accepted_count']}",
        f"rejected={decision['rejected_count']}",
        "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
    ]
    for candidate in decision.get("candidates") or []:
        lines.append(
            f"- {candidate.get('public_id') or candidate.get('task_id')}: {candidate['status']} "
            f"({', '.join(candidate['reason_codes']) or 'ok'})"
        )
    return "\n".join(lines)


def _closed_loop_dry_run_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    closed_loop = simulate_closed_loop_ticks(candidates or [])
    return {
        "action": command.action,
        "actor": actor,
        "read_only": True,
        "status": "DRY_RUN",
        "reason": "closed_loop_simulator_no_execution_authority",
        "closed_loop": closed_loop,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def _format_closed_loop_dry_run_message(decision: dict[str, Any]) -> str:
    report = decision.get("closed_loop") or {}
    return "\n".join([
        "Autopilot closed-loop dry-run",
        f"would_select={len(report.get('would_select') or [])}",
        f"would_skip={len(report.get('would_skip') or [])}",
        f"would_pause={len(report.get('would_pause') or [])}",
        f"next_state={report.get('next_state')}",
        "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
    ])


def _bounded_on_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Execute one bounded Autopilot tick through the existing dispatcher."""

    state = _write_state(_controller_state_for(command, actor=actor))
    scope = _candidate_scope_for(command)
    candidate_list = candidates or []
    single_flight = activate_single_flight(candidate_list)
    dispatch_result: dict[str, Any] | None = None
    if single_flight.get("status") == "handoff_check_passed" and single_flight.get("selected"):
        dispatch_result = dispatch_selected_once(single_flight["selected"])
        handoff = dict(single_flight.get("handoff") or {})
        handoff.update({"check_only": False, "would_dispatch": True})
        single_flight = {**single_flight, "handoff": handoff}
    dispatched = bool((dispatch_result or {}).get("dispatched"))
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": state.get("desired_mode"),
        "effective_mode": _effective_mode(str(state.get("desired_mode") or "disabled"), state),
        "read_only": False,
        "status": "BOUNDED_DISPATCHED" if dispatched else "BOUNDED_BLOCKED",
        "reason": "bounded_dispatch_via_existing_dispatcher",
        "scope": scope,
        "caps": {"max_active_flights": 1, "max_dispatches_per_tick": 1},
        "single_flight": single_flight,
        "dispatch_result": dispatch_result,
        "dispatched_count": 1 if dispatched else 0,
        "dry_run_side_effects": {
            "claimed": 0,
            "spawned": len((dispatch_result or {}).get("spawned") or []),
            "mutated": 0,
            "dispatched": 1 if dispatched else 0,
        },
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "state_file_enabled_is_execution_proof": False,
    }


def autopilot_continuous_tick(*, actor: Optional[str] = None, candidates: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """Run one persisted `/autopilot on [<parent>]` controller tick.

    The gateway watcher calls this repeatedly.  It is intentionally one bounded
    tick: load either the default-policy scope or persisted parent focus, select
    at most one ready-gate-passed task, and hand only that selected task to the
    existing Kanban dispatcher.
    """

    state = _read_state()
    desired = str(state.get("desired_mode") or "disabled").lower()
    if desired != "on" or not state.get("enabled"):
        return {
            "action": "continuous-tick",
            "actor": actor,
            "desired_mode": desired,
            "effective_mode": _effective_mode(desired, state),
            "status": "IDLE",
            "reason": "autopilot_not_on",
            "scope": {"mode": "default_policy" if not state.get("focus") else "parent", "parent_public_id": state.get("focus"), "tenant": None},
            "caps": {"max_active_flights": 1, "max_dispatches_per_tick": 1},
            "dispatched_count": 0,
            "dispatch_result": None,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }
    focus = str(state.get("focus") or "").strip().upper()
    command = AutopilotCommand(action="on", raw_args=f"on {focus}" if focus else "on", value=focus)
    resolved_candidates = _resolve_candidates(command, candidates)
    return _bounded_on_decision(
        command,
        actor=actor or "autopilot-continuous-loop",
        candidates=resolved_candidates,
    )


def _once_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    single_flight = activate_single_flight(candidates or [])
    dispatch_result: dict[str, Any] | None = None
    if single_flight.get("status") == "handoff_check_passed" and single_flight.get("selected"):
        dispatch_result = dispatch_selected_once(single_flight["selected"])
        handoff = dict(single_flight.get("handoff") or {})
        handoff.update({"check_only": False, "would_dispatch": True})
        single_flight = {**single_flight, "handoff": handoff}
    if dispatch_result is not None:
        status = "DISPATCHED" if dispatch_result.get("dispatched") else "DISPATCH_BLOCKED"
    else:
        status = "CHECK_ONLY_HANDOFF_BLOCKED"
    return {
        "action": command.action,
        "actor": actor,
        "read_only": False,
        "status": status,
        "reason": "single_flight_selected_dispatch_via_existing_dispatcher" if dispatch_result is not None else "single_flight_no_dispatch_candidate",
        "single_flight": single_flight,
        "dispatch_result": dispatch_result,
        "dry_run_side_effects": {
            "claimed": 0,
            "spawned": len((dispatch_result or {}).get("spawned") or []),
            "mutated": 0,
            "dispatched": 1 if (dispatch_result or {}).get("dispatched") else 0,
        },
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def _format_once_message(decision: dict[str, Any]) -> str:
    sf = decision.get("single_flight") or {}
    selected = sf.get("selected") or {}
    return "\n".join([
        "Autopilot single-flight dispatch",
        f"status={decision.get('status')}",
        f"selected={selected.get('public_id') or selected.get('task_id')}",
        f"spawned={len(((decision.get('dispatch_result') or {}).get('spawned') or []))}",
        f"worker_done_observed={sf.get('worker_done_observed')}",
        "handoff_success_is_worker_completion=False",
        "Autopilot selected one candidate and handed only that task to the existing Kanban dispatcher.",
    ])


def _format_bounded_on_message(decision: dict[str, Any]) -> str:
    scope = decision.get("scope") or {}
    sf = decision.get("single_flight") or {}
    selected = sf.get("selected") or {}
    scope_label = scope.get("parent_public_id") or "default_policy"
    return "\n".join([
        "Autopilot bounded dispatch",
        f"status={decision.get('status')}",
        f"scope={scope_label}",
        f"selected={selected.get('public_id') or selected.get('task_id')}",
        f"dispatched={decision.get('dispatched_count')}",
        "caps=max_active_flights=1,max_dispatches_per_tick=1",
        "handoff_success_is_worker_completion=False",
    ])


def _status_decision(command: AutopilotCommand, *, actor: Optional[str] = None) -> dict[str, Any]:
    state = _read_state()
    desired_mode = str(state.get("desired_mode") or "disabled")
    effective = _effective_mode(desired_mode, state)
    reasons = []
    if effective == "parent_scoped":
        reasons.append("parent_scoped_continuous_loop_enabled")
    elif effective == "default_policy_loop":
        reasons.append("default_policy_continuous_loop_enabled")
    else:
        reasons.append("phase1_controller_state_only_no_execution_authority")
    if state.get("read_error"):
        reasons.append(str(state["read_error"]))
    if (state.get("enabled") or desired_mode in {"on", "enabled"}) and effective not in {"parent_scoped", "default_policy_loop"}:
        reasons.append("state_file_enabled_true_is_not_runtime_proof")
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": desired_mode,
        "effective_mode": effective,
        "status": effective.upper(),
        "reason": ";".join(reasons),
        "focus": state.get("focus"),
        "scope_mode": "parent" if state.get("focus") else "default_policy",
        "pause_reason": state.get("pause_reason"),
        "paused_lanes": state.get("paused_lanes") or [],
        "hard_stop_reason": state.get("hard_stop_reason"),
        "dispatch_blocked": effective in {"blocked", "paused", "hard_stop", "stopped", "degraded"},
        "operator_recovery_required": effective == "hard_stop",
        "state_file": state,
        "state_file_enabled_is_execution_proof": False,
        "read_only": command.read_only,
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "dispatch_once_called": False,
        "claim_task_called": False,
        "worker_spawn_called": False,
        "kanban_mutation_called": False,
    }


def _format_status_message(decision: dict[str, Any]) -> str:
    state = decision.get("state_file") or {}
    lines = [
        "Autopilot status: " + str(decision["status"]),
        f"desired_mode={decision['desired_mode']}",
        f"effective_mode={decision['effective_mode']}",
    ]
    if decision.get("focus"):
        lines.append(f"focus={decision['focus']}")
    if decision.get("pause_reason"):
        lines.append(f"pause_reason={decision['pause_reason']}")
    lines.append("reason=" + str(decision["reason"]))
    if decision.get("effective_mode") in {"parent_scoped", "default_policy_loop"}:
        lines.extend([
            "Continuous bounded loop is enabled for the focused parent." if decision.get("effective_mode") == "parent_scoped" else "Continuous bounded loop is enabled for the default/global policy scope.",
            "Each tick may hand at most one selected task to the existing Kanban dispatcher.",
        ])
    else:
        lines.extend(
            [
                "State file enabled=true is not execution proof.",
                "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
            ]
        )
    lines.append(f"state_file={state.get('path')}")
    return "\n".join(lines)


def _controller_state_for(command: AutopilotCommand, *, actor: Optional[str]) -> dict[str, Any]:
    previous = _read_state()
    state: dict[str, Any] = {
        "version": AUTOPILOT_STATE_VERSION,
        "enabled": bool(previous.get("enabled")),
        "desired_mode": str(previous.get("desired_mode") or ("enabled" if previous.get("enabled") else "disabled")),
        "focus": previous.get("focus"),
        "pause_reason": previous.get("pause_reason"),
        "paused_lanes": previous.get("paused_lanes") or [],
        "hard_stop_reason": previous.get("hard_stop_reason"),
        "updated_at": _iso_now(),
        "updated_by": actor or "unknown",
    }
    if command.action == "on":
        updates: dict[str, Any] = {"desired_mode": "on", "enabled": True, "pause_reason": None}
        if command.value.strip():
            updates["focus"] = command.value.strip().upper()
        else:
            updates["focus"] = None
        state.update(updates)
    elif command.action in {"off", "stop"}:
        state.update({"desired_mode": "stopped", "enabled": False, "pause_reason": None})
    elif command.action == "pause":
        state.update({"desired_mode": "paused", "enabled": False, "pause_reason": command.value or "manual_pause"})
    elif command.action == "pause-lane":
        parts = command.value.split(maxsplit=1)
        lane = parts[0].strip().lower() if parts else ""
        reason = parts[1].strip() if len(parts) > 1 else "manual_lane_pause"
        paused = {str(existing).lower() for existing in (state.get("paused_lanes") or [])}
        if lane:
            paused.add(lane)
        state.update({"paused_lanes": sorted(paused), "pause_reason": reason})
    elif command.action == "resume-lane":
        lane = command.value.split(maxsplit=1)[0].strip().lower()
        paused = {str(existing).lower() for existing in (state.get("paused_lanes") or [])}
        if lane:
            paused.discard(lane)
        state.update({"paused_lanes": sorted(paused)})
    elif command.action == "hard-stop":
        state.update({"desired_mode": "hard_stopped", "enabled": False, "hard_stop_reason": command.value or "manual_hard_stop", "pause_reason": command.value or "manual_hard_stop"})
    elif command.action == "recover":
        state.update({"desired_mode": "paused", "enabled": False, "hard_stop_reason": None, "pause_reason": command.value or "manual_recovery_acknowledged"})
    elif command.action == "focus":
        state.update({"focus": command.value.upper() if command.value else None})
    elif command.action == "once":
        state.update({"desired_mode": "paused", "enabled": False, "pause_reason": "once_not_available_until_dispatch_gate"})
    return state


def _control_decision(command: AutopilotCommand, *, actor: Optional[str] = None) -> dict[str, Any]:
    state = _write_state(_controller_state_for(command, actor=actor))
    desired_mode = str(state.get("desired_mode") or "disabled")
    effective = _effective_mode(desired_mode, state)
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": desired_mode,
        "effective_mode": effective,
        "status": effective.upper(),
        "reason": "phase1_controller_state_persisted_without_execution_authority",
        "focus": state.get("focus"),
        "scope_mode": "parent" if state.get("focus") else "default_policy",
        "pause_reason": state.get("pause_reason"),
        "paused_lanes": state.get("paused_lanes") or [],
        "hard_stop_reason": state.get("hard_stop_reason"),
        "dispatch_blocked": effective in {"blocked", "paused", "hard_stop", "stopped", "degraded"},
        "operator_recovery_required": effective == "hard_stop",
        "state_file": state,
        "state_file_enabled_is_execution_proof": False,
        "read_only": False,
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "dispatch_once_called": False,
        "claim_task_called": False,
        "worker_spawn_called": False,
        "kanban_mutation_called": False,
    }


def handle_autopilot_command(
    raw_args: str = "",
    *,
    actor: Optional[str] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
) -> AutopilotResult:
    """Handle /autopilot without execution authority in Phase 2."""

    try:
        command = parse_autopilot_args(raw_args)
    except ValueError as exc:
        return AutopilotResult(
            ok=False,
            command=None,
            message=str(exc),
            decision={"status": "BLOCKED", "reason": "parse_error", "mutations_attempted": []},
            fail_closed=True,
        )

    if command.action == "queue":
        decision = _queue_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_queue_message(decision), decision, False)

    if command.action == "dry-run":
        decision = _closed_loop_dry_run_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_closed_loop_dry_run_message(decision), decision, False)

    if command.action in {"status"}:
        decision = _status_decision(command, actor=actor)
        return AutopilotResult(True, command, _format_status_message(decision), decision, False)

    if command.action == "once":
        decision = _once_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_once_message(decision), decision, False)

    if command.action == "on" and command.value.strip():
        decision = _bounded_on_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_bounded_on_message(decision), decision, False)

    if command.action in _CONTROL_ACTIONS:
        decision = _control_decision(command, actor=actor)
        return AutopilotResult(
            ok=True,
            command=command,
            message=_format_status_message(decision),
            decision=decision,
            fail_closed=False,
        )

    decision = {"status": "BLOCKED", "reason": "unsupported_action", "mutations_attempted": []}
    return AutopilotResult(False, command, f"Unsupported /autopilot action. {AUTOPILOT_USAGE}", decision, True)
