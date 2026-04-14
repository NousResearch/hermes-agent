#!/usr/bin/env python3
"""Deterministic model-rollout operator dashboard snapshot (Wave 5/6)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_HEALTH = Path("state/continuity/model_rollout_health/latest.json")
DEFAULT_COST = Path("state/continuity/model_rollout_cost/latest.json")
DEFAULT_ROUTE_POLICY_SOAK = Path("state/continuity/model_route_policy_soak/latest.json")
DEFAULT_RING_SOAK = Path("state/continuity/model_rollout_soak/latest.json")
DEFAULT_ROUTING_DECISIONS = Path("state/continuity/session_topology_router/decisions.jsonl")
DEFAULT_LEDGER_EVENTS = Path("state/continuity/model_rollout_ledger/events.jsonl")
DEFAULT_BAKEOFF_DASHBOARD = Path("state/architecture/competitive_parity/dashboard/latest.json")
DEFAULT_BAKEOFF_POLICY = Path("state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json")
DEFAULT_OUT = Path("state/continuity/model_rollout_dashboard/latest.json")
DEFAULT_ROUTING_MAX_AGE_SEC = 21600


def now_iso(now_dt: Optional[dt.datetime] = None) -> str:
    base = now_dt or dt.datetime.now(dt.timezone.utc)
    return base.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_root / path).resolve()
    return path.resolve()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def parse_iso(value: Any) -> Optional[dt.datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def load_optional_json(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    if not path.is_file():
        return None, "not_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def resolve_reference_now(*, source_payloads: list[Optional[Mapping[str, Any]]]) -> dt.datetime:
    fixed_now = parse_iso(os.environ.get("OPENCLAW_FIXED_NOW"))
    if fixed_now is not None:
        return fixed_now

    fixed_now_ts = str(os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS") or "").strip()
    if fixed_now_ts:
        try:
            ts = int(fixed_now_ts)
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        except Exception:
            pass

    candidates: list[dt.datetime] = []
    for payload in source_payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("generated_at", "timestamp", "evaluated_at"):
            parsed = parse_iso(payload.get(key))
            if parsed is not None:
                candidates.append(parsed)

    if candidates:
        return max(candidates)
    return dt.datetime.now(dt.timezone.utc)


def load_latest_routing_decision(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str], int]:
    if not path.exists():
        return None, "missing", 0
    if not path.is_file():
        return None, "not_file", 0

    latest_row: Optional[Dict[str, Any]] = None
    latest_ts: Optional[dt.datetime] = None
    scanned = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                scanned += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("schema") or "").strip() != "clawd.session_topology_routing.decision.v1":
                    continue

                row_ts = parse_iso(row.get("evaluated_at"))
                if latest_row is None:
                    latest_row = row
                    latest_ts = row_ts
                    continue

                if row_ts is not None:
                    if latest_ts is None or row_ts >= latest_ts:
                        latest_row = row
                        latest_ts = row_ts
                elif latest_ts is None:
                    latest_row = row
    except Exception:
        return None, "unreadable", scanned

    if latest_row is None:
        return None, "no_valid_rows", scanned

    return latest_row, None, scanned


def load_latest_operator_blocked_event(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str], int]:
    if not path.exists():
        return None, "missing", 0
    if not path.is_file():
        return None, "not_file", 0

    latest_row: Optional[Dict[str, Any]] = None
    latest_ts: Optional[dt.datetime] = None
    scanned = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                scanned += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                event_type = str(row.get("event_type") or "").strip().upper()
                if event_type not in {"ACTION_BLOCKED", "PROMOTION_BLOCKED"}:
                    continue

                row_ts = parse_iso(row.get("recorded_at"))
                if latest_row is None:
                    latest_row = row
                    latest_ts = row_ts
                    continue

                if row_ts is not None:
                    if latest_ts is None or row_ts >= latest_ts:
                        latest_row = row
                        latest_ts = row_ts
                elif latest_ts is None:
                    latest_row = row
    except Exception:
        return None, "unreadable", scanned

    if latest_row is None:
        return None, "no_blocked_events", scanned

    return latest_row, None, scanned


def summarize_operator_mistake_remediation(*, repo_root: Path, events_path: Path) -> Dict[str, Any]:
    blocked_event, load_error, rows_scanned = load_latest_operator_blocked_event(events_path)
    continuity_script = (repo_root / "ops" / "openclaw" / "continuity.sh").resolve()
    ledger_path = repo_root / "state" / "continuity" / "model_rollout_ledger" / "ledger.jsonl"

    base = {
        "status": "inactive",
        "active": False,
        "events_log_path": str(events_path),
        "events_log_loaded": load_error in {None, "no_blocked_events"},
        "events_log_error": None if load_error in {None, "no_blocked_events"} else load_error,
        "events_rows_scanned": rows_scanned,
        "event_type": None,
        "event_id": None,
        "recorded_at": None,
        "action": None,
        "reason_gate": None,
        "reason_code": None,
        "operator_message": None,
        "hint": None,
        "safe_remediation_options": [],
        "safe_remediation_commands": [],
        "correction_cycle_log_ref": None,
    }

    if not isinstance(blocked_event, Mapping):
        if load_error in {"no_blocked_events", "missing"}:
            base["status"] = "none_detected"
            base["events_log_error"] = None
            base["events_log_loaded"] = True
        elif load_error:
            base["status"] = "degraded"
            base["operator_message"] = "Unable to load operator mistake history."
            base["hint"] = "Refresh controller + dashboard surfaces before retrying action-card operations."
            remediation = [
                {
                    "id": "refresh_model_rollout_controller",
                    "label": "Refresh model rollout controller projection",
                    "command": f"bash {continuity_script} model-rollout-controller --json",
                },
                {
                    "id": "refresh_model_rollout_dashboard",
                    "label": "Refresh model rollout dashboard",
                    "command": f"bash {continuity_script} model-rollout-dashboard --json",
                },
            ]
            base["safe_remediation_options"] = remediation
            base["safe_remediation_commands"] = [row.get("command") for row in remediation]
        return base

    checks = blocked_event.get("checks") if isinstance(blocked_event.get("checks"), Mapping) else {}
    state_check = checks.get("state_match") if isinstance(checks.get("state_match"), Mapping) else {}
    dwell_check = checks.get("dwell") if isinstance(checks.get("dwell"), Mapping) else {}
    health_check = checks.get("health") if isinstance(checks.get("health"), Mapping) else {}

    reason_gate = "unknown"
    reason_code = "operator_action_blocked"
    if state_check.get("ok") is False:
        reason_gate = "state_match"
        reason_code = str(state_check.get("reason") or "state_mismatch")
    elif dwell_check.get("ok") is False:
        reason_gate = "dwell"
        reason_code = str(dwell_check.get("reason") or "dwell_not_met")
    elif health_check.get("ok") is False:
        reason_gate = "health"
        reason_code = str(health_check.get("reason") or "health_not_ready")

    operator_message = "Operator action was blocked to protect rollout state integrity."
    hint = "Inspect the blocked event and run the safe remediation commands before retrying."
    if reason_gate == "state_match":
        operator_message = "The requested transition did not match the current rollout state."
        hint = "Refresh rollout ledger/controller state and retry from the displayed current state only."
    elif reason_gate == "dwell":
        operator_message = "The promotion was blocked because minimum dwell time is not yet satisfied."
        hint = "Wait for dwell to complete, then rerun controller from the same action card."
    elif reason_gate == "health":
        operator_message = "The promotion was blocked because rollout health gates are not green."
        hint = "Refresh health/dashboard surfaces and only retry after health becomes healthy."

    remediation = [
        {
            "id": "inspect_operator_correction_cycle",
            "label": "Inspect correction-cycle event log",
            "command": f"tail -n 60 {events_path}",
        },
        {
            "id": "inspect_rollout_ledger_state",
            "label": "Inspect rollout ledger state",
            "command": f"tail -n 60 {ledger_path}",
        },
        {
            "id": "refresh_model_rollout_controller",
            "label": "Refresh model rollout controller",
            "command": f"bash {continuity_script} model-rollout-controller --json",
        },
        {
            "id": "refresh_model_rollout_dashboard",
            "label": "Refresh model rollout dashboard",
            "command": f"bash {continuity_script} model-rollout-dashboard --json",
        },
    ]

    safe_commands: list[str] = []
    for row in remediation:
        cmd = str(row.get("command") or "").strip()
        if cmd and cmd not in safe_commands:
            safe_commands.append(cmd)

    correction_cycle_log_ref = str(events_path)
    if is_within(repo_root, events_path):
        correction_cycle_log_ref = str(events_path.relative_to(repo_root))

    return {
        "status": "active",
        "active": True,
        "events_log_path": str(events_path),
        "events_log_loaded": True,
        "events_log_error": None,
        "events_rows_scanned": rows_scanned,
        "event_type": str(blocked_event.get("event_type") or "").strip() or None,
        "event_id": str(blocked_event.get("event_id") or "").strip() or None,
        "recorded_at": blocked_event.get("recorded_at"),
        "action": str(blocked_event.get("action") or "").strip() or None,
        "reason_gate": reason_gate,
        "reason_code": reason_code,
        "operator_message": operator_message,
        "hint": hint,
        "safe_remediation_options": remediation,
        "safe_remediation_commands": safe_commands,
        "correction_cycle_log_ref": correction_cycle_log_ref,
    }


def summarize_routing_effective(*, repo_root: Path, decisions_path: Path, max_age_sec: int, reference_now: dt.datetime) -> Dict[str, Any]:
    decision_row, load_error, rows_scanned = load_latest_routing_decision(decisions_path)

    route = decision_row.get("route") if isinstance((decision_row or {}).get("route"), Mapping) else {}
    actionable = (
        decision_row.get("actionable_failure")
        if isinstance((decision_row or {}).get("actionable_failure"), Mapping)
        else {}
    )

    evaluated_at = (decision_row or {}).get("evaluated_at")
    evaluated_dt = parse_iso(evaluated_at)
    age_sec = None
    if evaluated_dt is not None:
        age_sec = max(0, int((reference_now - evaluated_dt).total_seconds()))

    fresh = None
    if age_sec is not None:
        fresh = True if max_age_sec <= 0 else age_sec <= max_age_sec

    decision_state = str((decision_row or {}).get("decision") or "").strip().upper() or None
    blocked = decision_state == "BLOCK"
    blocked_fresh = bool(blocked and (fresh is not False))

    actionable_commands = [
        str(cmd).strip()
        for cmd in (actionable.get("commands") if isinstance(actionable.get("commands"), list) else [])
        if str(cmd).strip()
    ]

    route_class = str(route.get("route_class") or "").strip() or None
    selected_model = str(route.get("selected_model") or "").strip() or None
    required_stage = str(route.get("required_rollout_stage") or "").strip() or None

    effective_route_class = route_class if fresh is not False else None
    effective_selected_model = selected_model if fresh is not False else None
    effective_required_stage = required_stage if fresh is not False else None

    continuity_script = (repo_root / "ops" / "openclaw" / "continuity.sh").resolve()

    return {
        "decision_log_path": str(decisions_path),
        "decision_log_present": decisions_path.exists(),
        "decision_loaded": load_error is None,
        "load_error": load_error,
        "rows_scanned": rows_scanned,
        "max_age_sec": max_age_sec,
        "latest": {
            "decision": decision_state,
            "evaluated_at": evaluated_at,
            "age_sec": age_sec,
            "fresh": fresh,
            "block_gate": (decision_row or {}).get("block_gate"),
            "block_reason": (decision_row or {}).get("block_reason"),
            "route_class": route_class,
            "selected_model": selected_model,
            "required_rollout_stage": required_stage,
            "selected_rule_id": route.get("selected_rule_id"),
            "actionable_failure": {
                "gate": actionable.get("gate"),
                "reason": actionable.get("reason"),
                "hint": actionable.get("hint"),
                "commands": actionable_commands,
            },
        },
        "effective": {
            "blocked": blocked,
            "blocked_fresh": blocked_fresh,
            "route_class": effective_route_class,
            "selected_model": effective_selected_model,
            "required_rollout_stage": effective_required_stage,
            "inspect_command": f"tail -n 60 {decisions_path}",
            "recheck_policy_command": f"bash {continuity_script} model-route-policy-lint --json",
            "blocker_guidance": {
                "hint": actionable.get("hint"),
                "first_command": actionable_commands[0] if actionable_commands else None,
            },
        },
    }


def summarize_bakeoff_governance(*, policy_pack: Optional[Mapping[str, Any]], bakeoff_dashboard: Optional[Mapping[str, Any]], routing_effective: Mapping[str, Any]) -> Dict[str, Any]:
    slice_policy = None
    if isinstance(policy_pack, Mapping):
        slices = policy_pack.get("slices")
        if isinstance(slices, Mapping):
            raw = slices.get("30")
            if isinstance(raw, Mapping):
                slice_policy = raw

    competitors = bakeoff_dashboard.get("competitors") if isinstance((bakeoff_dashboard or {}).get("competitors"), list) else []
    ranking_rows = []
    for row in competitors:
        if not isinstance(row, Mapping):
            continue
        ranking_rows.append(
            {
                "provider": str(row.get("competitor") or "").strip() or None,
                "parity_score_percent": row.get("parity_score_percent"),
                "density_score": row.get("density_score"),
                "latency_delta_ms": row.get("latency_delta_ms"),
                "component_coverage_ratio": row.get("component_coverage_ratio"),
            }
        )
    ranking_rows.sort(key=lambda row: float(row.get("parity_score_percent") or 0.0), reverse=True)

    summary = bakeoff_dashboard.get("summary") if isinstance((bakeoff_dashboard or {}).get("summary"), Mapping) else {}
    blocker_count = int(summary.get("blocker_count") or 0)
    regression_count = int(summary.get("regression_count") or 0)

    cockpit_governance = (
        slice_policy.get("cockpit_governance")
        if isinstance((slice_policy or {}).get("cockpit_governance"), Mapping)
        else {}
    )
    requires_approval = cockpit_governance.get("promotion_requires_operator_action_card_approval") is True
    requires_scorecard = cockpit_governance.get("must_attach_bakeoff_scorecard") is True

    artifacts = bakeoff_dashboard.get("artifacts") if isinstance((bakeoff_dashboard or {}).get("artifacts"), Mapping) else {}
    scorecard_ref = str(artifacts.get("scorecard_summary") or "").strip() or None

    min_provider_count = 3
    provider_count = len([row for row in ranking_rows if row.get("provider")])
    promotion_ready = (
        provider_count >= min_provider_count
        and blocker_count == 0
        and regression_count == 0
        and (not requires_scorecard or bool(scorecard_ref))
    )

    reasons: list[str] = []
    if provider_count < min_provider_count:
        reasons.append("insufficient_qualified_providers")
    if blocker_count > 0:
        reasons.append("bakeoff_blockers_present")
    if regression_count > 0:
        reasons.append("bakeoff_regression_present")
    if requires_scorecard and not scorecard_ref:
        reasons.append("bakeoff_scorecard_missing")

    routing_latest = routing_effective.get("latest") if isinstance(routing_effective.get("latest"), Mapping) else {}
    routing_actionable = routing_latest.get("actionable_failure") if isinstance(routing_latest.get("actionable_failure"), Mapping) else {}
    routing_effective_row = routing_effective.get("effective") if isinstance(routing_effective.get("effective"), Mapping) else {}

    action_prompt: Dict[str, Any] = {
        "status": "none",
        "reason": "no_action_required",
        "requires_operator_approval": False,
        "requires_scorecard_attachment": requires_scorecard,
        "scorecard_ref": scorecard_ref,
        "commands": [],
        "hint": None,
    }

    if bool(routing_effective_row.get("blocked")):
        first_command = (routing_actionable.get("commands") or [None])[0] if isinstance(routing_actionable.get("commands"), list) else None
        action_prompt = {
            "status": "blocked",
            "reason": str(routing_latest.get("block_reason") or "routing_blocked"),
            "requires_operator_approval": False,
            "requires_scorecard_attachment": requires_scorecard,
            "scorecard_ref": scorecard_ref,
            "commands": [first_command] if first_command else [],
            "hint": routing_actionable.get("hint"),
        }
    elif promotion_ready and requires_approval:
        action_prompt = {
            "status": "approval_required",
            "reason": "bakeoff_thresholds_passed_ready_for_matrix_update",
            "requires_operator_approval": True,
            "requires_scorecard_attachment": requires_scorecard,
            "scorecard_ref": scorecard_ref,
            "commands": [],
            "hint": "Review bakeoff ranking and approve routing-matrix update via cockpit action card.",
        }
    elif not promotion_ready:
        action_prompt = {
            "status": "blocked",
            "reason": reasons[0] if reasons else "bakeoff_not_ready",
            "requires_operator_approval": False,
            "requires_scorecard_attachment": requires_scorecard,
            "scorecard_ref": scorecard_ref,
            "commands": [],
            "hint": "Bakeoff governance prerequisites are incomplete; hold matrix update.",
        }

    return {
        "policy_id": slice_policy.get("policy_id") if isinstance(slice_policy, Mapping) else None,
        "policy_loaded": isinstance(slice_policy, Mapping),
        "dashboard_loaded": isinstance(bakeoff_dashboard, Mapping),
        "run_id": bakeoff_dashboard.get("run_id") if isinstance(bakeoff_dashboard, Mapping) else None,
        "provider_count": provider_count,
        "min_provider_count": min_provider_count,
        "ranking": ranking_rows,
        "summary": {
            "blocker_count": blocker_count,
            "regression_count": regression_count,
        },
        "governance": {
            "requires_operator_action_card_approval": requires_approval,
            "requires_bakeoff_scorecard": requires_scorecard,
            "scorecard_ref": scorecard_ref,
            "promotion_ready": promotion_ready,
            "reasons": reasons,
        },
        "cockpit_action_prompt": action_prompt,
    }


def derive_status(
    *,
    health: Optional[Mapping[str, Any]],
    cost: Optional[Mapping[str, Any]],
    route_soak: Optional[Mapping[str, Any]],
    ring_soak: Optional[Mapping[str, Any]],
    routing_effective: Optional[Mapping[str, Any]],
    bakeoff_projection: Optional[Mapping[str, Any]],
    operator_mistake_remediation: Optional[Mapping[str, Any]],
    errors: Dict[str, str],
) -> str:
    if errors:
        return "error"

    health_overall = str((health or {}).get("overall_status") or "").lower()
    cost_status = str((cost or {}).get("status") or "").lower()
    route_status = str((route_soak or {}).get("status") or "").lower()
    ring_status = str((ring_soak or {}).get("status") or "").lower()
    routing_blocked_fresh = bool(
        isinstance(routing_effective, Mapping)
        and bool((routing_effective.get("effective") or {}).get("blocked_fresh"))
    )
    bakeoff_prompt_status = str(
        ((bakeoff_projection or {}).get("cockpit_action_prompt") or {}).get("status") or ""
    ).strip().lower()

    if health_overall not in {"", "healthy"}:
        return "attention"
    if cost_status == "budget_exceeded":
        return "attention"
    if route_status == "policy_violations":
        return "attention"
    if routing_blocked_fresh:
        return "attention"
    if bool((operator_mistake_remediation or {}).get("active") is True):
        return "attention"
    if bakeoff_prompt_status == "approval_required":
        return "attention"
    if ring_status == "attention":
        return "attention"
    if ring_status == "soaking":
        return "soaking"
    return "ok"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic model-rollout dashboard snapshot")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--health", default=str(DEFAULT_HEALTH), help="Model rollout health snapshot JSON path")
    ap.add_argument("--cost", default=str(DEFAULT_COST), help="Model rollout cost snapshot JSON path")
    ap.add_argument("--route-policy-soak", default=str(DEFAULT_ROUTE_POLICY_SOAK), help="Route-policy soak snapshot JSON path")
    ap.add_argument("--ring-soak", default=str(DEFAULT_RING_SOAK), help="Ring-soak snapshot JSON path")
    ap.add_argument("--routing-decisions", default=str(DEFAULT_ROUTING_DECISIONS), help="Session routing decisions JSONL path")
    ap.add_argument("--ledger-events", default=str(DEFAULT_LEDGER_EVENTS), help="Model rollout ledger events JSONL path")
    ap.add_argument("--bakeoff-dashboard", default=str(DEFAULT_BAKEOFF_DASHBOARD), help="Competitive bakeoff dashboard JSON path")
    ap.add_argument("--bakeoff-policy", default=str(DEFAULT_BAKEOFF_POLICY), help="B6 provider graduation policy pack JSON path")
    ap.add_argument("--routing-max-age-sec", type=int, default=DEFAULT_ROUTING_MAX_AGE_SEC, help="Maximum routing-decision age for live blocker visibility")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output dashboard JSON path")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    health_path = resolve_path(repo_root, args.health)
    cost_path = resolve_path(repo_root, args.cost)
    route_soak_path = resolve_path(repo_root, args.route_policy_soak)
    ring_soak_path = resolve_path(repo_root, args.ring_soak)
    routing_decisions_path = resolve_path(repo_root, args.routing_decisions)
    ledger_events_path = resolve_path(repo_root, args.ledger_events)
    bakeoff_dashboard_path = resolve_path(repo_root, args.bakeoff_dashboard)
    bakeoff_policy_path = resolve_path(repo_root, args.bakeoff_policy)
    out_path = resolve_path(repo_root, args.out)

    health, health_error = load_optional_json(health_path)
    cost, cost_error = load_optional_json(cost_path)
    route_soak, route_soak_error = load_optional_json(route_soak_path)
    ring_soak, ring_soak_error = load_optional_json(ring_soak_path)
    bakeoff_dashboard, bakeoff_dashboard_error = load_optional_json(bakeoff_dashboard_path)
    bakeoff_policy, bakeoff_policy_error = load_optional_json(bakeoff_policy_path)

    reference_now = resolve_reference_now(
        source_payloads=[
            health,
            cost,
            route_soak,
            ring_soak,
            bakeoff_dashboard,
        ]
    )

    routing_max_age_sec = max(0, int(args.routing_max_age_sec or 0))
    routing_effective = summarize_routing_effective(
        repo_root=repo_root,
        decisions_path=routing_decisions_path,
        max_age_sec=routing_max_age_sec,
        reference_now=reference_now,
    )
    bakeoff_projection = summarize_bakeoff_governance(
        policy_pack=bakeoff_policy,
        bakeoff_dashboard=bakeoff_dashboard,
        routing_effective=routing_effective,
    )
    operator_mistake_remediation = summarize_operator_mistake_remediation(
        repo_root=repo_root,
        events_path=ledger_events_path,
    )

    errors: Dict[str, str] = {}
    if health_error:
        errors["health"] = health_error
    if cost_error:
        errors["cost"] = cost_error
    if route_soak_error:
        errors["route_policy_soak"] = route_soak_error
    if ring_soak_error:
        errors["ring_soak"] = ring_soak_error

    status = derive_status(
        health=health,
        cost=cost,
        route_soak=route_soak,
        ring_soak=ring_soak,
        routing_effective=routing_effective,
        bakeoff_projection=bakeoff_projection,
        operator_mistake_remediation=operator_mistake_remediation,
        errors=errors,
    )

    budget_breaches = (cost or {}).get("budget_breaches") if isinstance((cost or {}).get("budget_breaches"), list) else []
    route_violations = (route_soak or {}).get("violations") if isinstance((route_soak or {}).get("violations"), list) else []
    ring_counts = (ring_soak or {}).get("counts") if isinstance((ring_soak or {}).get("counts"), Mapping) else {}
    routing_latest = (routing_effective or {}).get("latest") if isinstance((routing_effective or {}).get("latest"), Mapping) else {}

    snapshot = {
        "schema": "clawd.model_rollout_dashboard_snapshot.v1",
        "generated_at": now_iso(reference_now),
        "status": status,
        "surfaces": {
            "health": {"path": str(health_path), "loaded": health_error is None, "error": health_error},
            "cost": {"path": str(cost_path), "loaded": cost_error is None, "error": cost_error},
            "route_policy_soak": {"path": str(route_soak_path), "loaded": route_soak_error is None, "error": route_soak_error},
            "ring_soak": {"path": str(ring_soak_path), "loaded": ring_soak_error is None, "error": ring_soak_error},
            "routing_decisions": {
                "path": str(routing_decisions_path),
                "loaded": bool(routing_effective.get("decision_loaded") is True),
                "error": routing_effective.get("load_error"),
            },
            "ledger_events": {
                "path": str(ledger_events_path),
                "loaded": bool(operator_mistake_remediation.get("events_log_loaded") is True),
                "error": operator_mistake_remediation.get("events_log_error"),
            },
            "bakeoff_dashboard": {
                "path": str(bakeoff_dashboard_path),
                "loaded": bakeoff_dashboard_error is None,
                "error": bakeoff_dashboard_error,
            },
            "bakeoff_policy": {
                "path": str(bakeoff_policy_path),
                "loaded": bakeoff_policy_error is None,
                "error": bakeoff_policy_error,
            },
        },
        "headline": {
            "health_overall_status": (health or {}).get("overall_status"),
            "cost_status": (cost or {}).get("status"),
            "ring_soak_status": (ring_soak or {}).get("status"),
            "route_policy_status": (route_soak or {}).get("status"),
            "budget_breach_count": len(budget_breaches),
            "route_policy_violation_count": len(route_violations),
            "active_ring_rollouts": int(ring_counts.get("active_rollouts") or 0),
            "ring_ready_count": int(ring_counts.get("ready") or 0),
            "ring_attention_count": int(ring_counts.get("attention") or 0),
            "routing_latest_decision": routing_latest.get("decision"),
            "routing_effective_route_class": (routing_effective.get("effective") or {}).get("route_class"),
            "routing_effective_selected_model": (routing_effective.get("effective") or {}).get("selected_model"),
            "routing_block_gate": routing_latest.get("block_gate"),
            "routing_block_reason": routing_latest.get("block_reason"),
            "routing_blocked_fresh": bool((routing_effective.get("effective") or {}).get("blocked_fresh")),
            "bakeoff_provider_count": int(bakeoff_projection.get("provider_count") or 0),
            "bakeoff_promotion_ready": bool((bakeoff_projection.get("governance") or {}).get("promotion_ready")),
            "cockpit_action_prompt_status": ((bakeoff_projection.get("cockpit_action_prompt") or {}).get("status")),
            "operator_mistake_remediation_status": operator_mistake_remediation.get("status"),
            "operator_mistake_remediation_active": bool(operator_mistake_remediation.get("active") is True),
            "operator_mistake_remediation_reason_gate": operator_mistake_remediation.get("reason_gate"),
            "operator_mistake_remediation_reason_code": operator_mistake_remediation.get("reason_code"),
            "operator_mistake_remediation_command_count": len(operator_mistake_remediation.get("safe_remediation_commands") or []),
        },
        "routing": routing_effective,
        "bakeoff": bakeoff_projection,
        "operator_mistake_remediation": operator_mistake_remediation,
        "cockpit_action_prompt": bakeoff_projection.get("cockpit_action_prompt") or {},
        "errors": errors,
    }

    if not is_within(repo_root, out_path):
        result = {
            "schema": "clawd.model_rollout_dashboard_snapshot.v1",
            "generated_at": now_iso(reference_now),
            "status": "error",
            "error": "unsafe_output_path",
            "path": str(out_path),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = dict(snapshot)
    payload["written_path"] = str(out_path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(payload))

    return 0 if status in {"ok", "soaking"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
