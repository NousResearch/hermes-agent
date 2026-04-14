#!/usr/bin/env python3
"""Deterministic model rollout cost-governance snapshot producer (v1)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from model_pool_policy_contract import load_pool_policy


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_POOL_POLICY_PATH = Path("docs/ops/model_pool_policy_v1.json")
DEFAULT_POOL_POLICY_SCHEMA = Path("docs/ops/schemas/model_pool_policy.schema.json")
DEFAULT_GATE_DECISIONS = Path("state/continuity/model_rollout_gate_runner/decisions.jsonl")
DEFAULT_ROUTING_DECISIONS = Path("state/continuity/session_topology_router/decisions.jsonl")
DEFAULT_LEDGER_EVENTS = Path("state/continuity/model_rollout_ledger/events.jsonl")
DEFAULT_OUT_PATH = Path("state/continuity/model_rollout_cost/latest.json")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _model_family_from_model_key(model_key: Optional[str]) -> Optional[str]:
    token = str(model_key or "").strip().lower()
    if not token:
        return None
    if token.startswith("openai-codex/") or "codex" in token:
        return "Codex"
    if token.startswith("deepseek/") or "deepseek" in token:
        return "DeepSeek"
    if (token.startswith("google/") and "gemini" in token) or "gemini" in token:
        return "Gemini"
    if token.startswith("moonshot/") or "kimi" in token:
        return "Kimi"
    return "Other"


def _routing_row_family(row: Mapping[str, Any]) -> Optional[str]:
    route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
    routing_telemetry = row.get("routing_telemetry") if isinstance(row.get("routing_telemetry"), Mapping) else {}

    for source in (
        route.get("selected_model_family"),
        routing_telemetry.get("selected_model_family"),
        _model_family_from_model_key(route.get("selected_model")),
    ):
        token = str(source or "").strip()
        if token:
            return token
    return None


def _task_class_from_routing_row(row: Mapping[str, Any]) -> Optional[str]:
    route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    routing_telemetry = row.get("routing_telemetry") if isinstance(row.get("routing_telemetry"), Mapping) else {}

    for source in (route.get("task_class"), request.get("task_class"), routing_telemetry.get("task_class")):
        token = str(source or "").strip()
        if token:
            return token
    return None


def _numeric_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        token = str(value or "").strip()
        if not token:
            return None
        return float(token)
    except Exception:
        return None


def _safe_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_root / path).resolve()
    return path.resolve()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            raise ValueError(f"jsonl_non_object_line:{idx}")
    return rows


def _in_window(ts: Optional[dt.datetime], window_start: dt.datetime, now_ts: dt.datetime) -> bool:
    if ts is None:
        return False
    return window_start <= ts <= now_ts


def build_snapshot(
    *,
    policy: Mapping[str, Any],
    policy_meta: Mapping[str, Any],
    gate_rows: List[Mapping[str, Any]],
    routing_rows: List[Mapping[str, Any]],
    ledger_rows: List[Mapping[str, Any]],
    now_ts: dt.datetime,
    window_hours: int,
) -> Dict[str, Any]:
    window_start = now_ts - dt.timedelta(hours=window_hours)

    route_classes_obj = policy.get("route_classes") if isinstance(policy.get("route_classes"), Mapping) else {}
    route_classes = [key for key in ("NO_LLM", "SPARK", "HEAVY") if isinstance(route_classes_obj.get(key), Mapping)]

    model_to_route: Dict[str, str] = {}
    model_to_family: Dict[str, str] = {}
    for route_class in route_classes:
        entry = route_classes_obj.get(route_class)
        if not isinstance(entry, Mapping):
            continue
        for model_key in entry.get("allowed_models") if isinstance(entry.get("allowed_models"), list) else []:
            if isinstance(model_key, str) and model_key.strip():
                key = str(model_key)
                model_to_route[key] = route_class
                family = _model_family_from_model_key(key)
                if family:
                    model_to_family[key] = family

    cost_cfg = policy.get("cost_governance") if isinstance(policy.get("cost_governance"), Mapping) else {}
    budget_cfg = cost_cfg.get("route_class_daily_budget_usd") if isinstance(cost_cfg.get("route_class_daily_budget_usd"), Mapping) else {}
    unit_cost_cfg = cost_cfg.get("model_unit_cost_usd") if isinstance(cost_cfg.get("model_unit_cost_usd"), Mapping) else {}

    unit_cost_by_model: Dict[str, float] = {}
    for model_key, raw_cost in unit_cost_cfg.items():
        if not isinstance(model_key, str):
            continue
        value = _numeric_or_none(raw_cost)
        if value is None:
            continue
        unit_cost_by_model[model_key] = float(value)

    route_class_default_unit_cost: Dict[str, float] = {}
    for route_class in route_classes:
        route_entry = route_classes_obj.get(route_class) if isinstance(route_classes_obj.get(route_class), Mapping) else {}
        allowed_models = route_entry.get("allowed_models") if isinstance(route_entry.get("allowed_models"), list) else []
        priced_values = [
            unit_cost_by_model.get(str(model_key))
            for model_key in allowed_models
            if isinstance(model_key, str) and unit_cost_by_model.get(str(model_key)) is not None
        ]
        if priced_values:
            route_class_default_unit_cost[route_class] = float(sum(priced_values) / len(priced_values))

    family_known_unit_costs: Dict[str, List[float]] = {}
    for model_key, value in unit_cost_by_model.items():
        family = model_to_family.get(model_key) or _model_family_from_model_key(model_key) or "Other"
        family_known_unit_costs.setdefault(family, []).append(float(value))

    family_default_unit_cost: Dict[str, float] = {
        family: float(sum(values) / len(values))
        for family, values in family_known_unit_costs.items()
        if values
    }
    global_default_unit_cost: Optional[float] = None
    if unit_cost_by_model:
        global_default_unit_cost = float(sum(unit_cost_by_model.values()) / len(unit_cost_by_model))

    models_summary: Dict[str, Dict[str, Any]] = {}
    route_summary: Dict[str, Dict[str, Any]] = {
        route_class: {
            "route_class": route_class,
            "gate_pass_count": 0,
            "gate_block_count": 0,
            "routing_pass_count": 0,
            "routing_block_count": 0,
            "estimated_spend_usd": 0.0,
            "budget_usd": float(budget_cfg.get(route_class)) if isinstance(budget_cfg.get(route_class), (int, float)) else None,
        }
        for route_class in route_classes
    }

    route_cost_accumulator: Dict[str, float] = {route_class: 0.0 for route_class in route_classes}

    provider_rows: Dict[str, Dict[str, Any]] = {}
    family_utilization: Dict[str, Dict[str, Any]] = {}
    task_class_tagged_count = 0
    task_class_missing_count = 0
    misrouting_signal_counts: Dict[str, int] = {}
    misrouting_signal_by_family: Dict[str, Dict[str, int]] = {}
    misrouting_signal_by_task_class: Dict[str, Dict[str, int]] = {}

    provider_pricing_totals = {
        "priced_event_count": 0,
        "exact_priced_event_count": 0,
        "fallback_priced_event_count": 0,
        "unpriced_event_count": 0,
        "estimated_spend_usd": 0.0,
    }

    gate_total = gate_pass = gate_block = 0
    for row in gate_rows:
        ts = parse_iso(row.get("evaluated_at"))
        if not _in_window(ts, window_start, now_ts):
            continue
        gate_total += 1
        decision = str(row.get("decision") or "")

        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        model_key = str(model.get("model_key") or model.get("model_ref") or "").strip()
        route_class = str(model.get("route_class") or model.get("model_family") or model_to_route.get(model_key) or "").strip()

        if decision == "PASS":
            gate_pass += 1
        else:
            gate_block += 1

        if route_class in route_summary:
            if decision == "PASS":
                route_summary[route_class]["gate_pass_count"] += 1
            else:
                route_summary[route_class]["gate_block_count"] += 1

        if model_key:
            model_row = models_summary.setdefault(
                model_key,
                {
                    "model_key": model_key,
                    "route_class": route_class or model_to_route.get(model_key),
                    "unit_cost_usd": float(unit_cost_by_model.get(model_key)) if unit_cost_by_model.get(model_key) is not None else 0.0,
                    "gate_pass_count": 0,
                    "gate_block_count": 0,
                    "routing_pass_count": 0,
                    "estimated_spend_usd": 0.0,
                },
            )
            if decision == "PASS":
                model_row["gate_pass_count"] += 1
            else:
                model_row["gate_block_count"] += 1

    routing_total = routing_pass = routing_block = 0
    for row in routing_rows:
        ts = parse_iso(row.get("evaluated_at"))
        if not _in_window(ts, window_start, now_ts):
            continue

        routing_total += 1
        decision = str(row.get("decision") or "")
        route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
        routing_telemetry = row.get("routing_telemetry") if isinstance(row.get("routing_telemetry"), Mapping) else {}
        route_class = str(route.get("route_class") or "").strip()
        selected_model = str(route.get("selected_model") or "").strip()
        selected_family = _routing_row_family(row) or "Unspecified"
        task_class = _task_class_from_routing_row(row)

        if task_class:
            task_class_tagged_count += 1
        else:
            task_class_missing_count += 1

        family_row = family_utilization.setdefault(
            selected_family,
            {
                "provider_family": selected_family,
                "routing_pass_count": 0,
                "routing_block_count": 0,
                "route_class_counts": {},
                "task_class_counts": {},
                "selected_models": {},
            },
        )

        provider_row = provider_rows.setdefault(
            selected_family,
            {
                "provider_family": selected_family,
                "routing_pass_count": 0,
                "routing_block_count": 0,
                "route_class_counts": {},
                "task_class_counts": {},
                "selected_model_counts": {},
                "pricing": {
                    "estimated_spend_usd": 0.0,
                    "priced_event_count": 0,
                    "exact_priced_event_count": 0,
                    "fallback_priced_event_count": 0,
                    "unpriced_event_count": 0,
                    "cost_basis_counts": {},
                },
                "misrouting_signal_counts": {},
            },
        )

        def _bump(counter: Dict[str, int], key: Optional[str]) -> None:
            token = str(key or "").strip() or "unspecified"
            counter[token] = int(counter.get(token) or 0) + 1

        if decision == "PASS":
            routing_pass += 1
            family_row["routing_pass_count"] += 1
            provider_row["routing_pass_count"] += 1

            _bump(family_row["route_class_counts"], route_class)
            _bump(provider_row["route_class_counts"], route_class)
            _bump(family_row["task_class_counts"], task_class)
            _bump(provider_row["task_class_counts"], task_class)
            _bump(family_row["selected_models"], selected_model)
            _bump(provider_row["selected_model_counts"], selected_model)

            if route_class in route_summary:
                route_summary[route_class]["routing_pass_count"] += 1

            if selected_model:
                model_row = models_summary.setdefault(
                    selected_model,
                    {
                        "model_key": selected_model,
                        "route_class": route_class or model_to_route.get(selected_model),
                        "unit_cost_usd": float(unit_cost_by_model.get(selected_model)) if unit_cost_by_model.get(selected_model) is not None else 0.0,
                        "gate_pass_count": 0,
                        "gate_block_count": 0,
                        "routing_pass_count": 0,
                        "estimated_spend_usd": 0.0,
                    },
                )
                model_row["routing_pass_count"] += 1

            event_cost = None
            cost_basis = "unpriced_unresolved"
            for raw_cost, basis in (
                (route.get("normalized_cost_usd"), "route_normalized_cost_usd"),
                (route.get("cost_usd"), "route_cost_usd"),
                (row.get("cost_usd"), "decision_cost_usd"),
                ((row.get("usage") if isinstance(row.get("usage"), Mapping) else {}).get("cost_usd"), "usage_cost_usd"),
            ):
                numeric = _numeric_or_none(raw_cost)
                if numeric is not None:
                    event_cost = float(numeric)
                    cost_basis = basis
                    break

            if event_cost is None and selected_model and selected_model in unit_cost_by_model:
                event_cost = float(unit_cost_by_model[selected_model])
                cost_basis = "model_unit_cost_policy"

            if event_cost is None and route_class in route_class_default_unit_cost:
                event_cost = float(route_class_default_unit_cost[route_class])
                cost_basis = "route_class_average_unit_cost_policy"

            if event_cost is None and selected_family in family_default_unit_cost:
                event_cost = float(family_default_unit_cost[selected_family])
                cost_basis = "provider_family_average_unit_cost_policy"

            if event_cost is None and global_default_unit_cost is not None:
                event_cost = float(global_default_unit_cost)
                cost_basis = "global_average_unit_cost_policy"

            pricing = provider_row["pricing"]
            pricing["cost_basis_counts"][cost_basis] = int(pricing["cost_basis_counts"].get(cost_basis) or 0) + 1

            if event_cost is None:
                pricing["unpriced_event_count"] += 1
                provider_pricing_totals["unpriced_event_count"] += 1
            else:
                event_cost = _safe_round(event_cost)
                pricing["priced_event_count"] += 1
                provider_pricing_totals["priced_event_count"] += 1
                pricing["estimated_spend_usd"] = _safe_round(float(pricing.get("estimated_spend_usd") or 0.0) + event_cost)
                provider_pricing_totals["estimated_spend_usd"] = _safe_round(
                    float(provider_pricing_totals.get("estimated_spend_usd") or 0.0) + event_cost
                )
                if cost_basis in {"route_normalized_cost_usd", "route_cost_usd", "decision_cost_usd", "usage_cost_usd", "model_unit_cost_policy"}:
                    pricing["exact_priced_event_count"] += 1
                    provider_pricing_totals["exact_priced_event_count"] += 1
                else:
                    pricing["fallback_priced_event_count"] += 1
                    provider_pricing_totals["fallback_priced_event_count"] += 1

                if route_class in route_cost_accumulator:
                    route_cost_accumulator[route_class] = float(route_cost_accumulator[route_class]) + event_cost
        else:
            routing_block += 1
            family_row["routing_block_count"] += 1
            provider_row["routing_block_count"] += 1
            if route_class in route_summary:
                route_summary[route_class]["routing_block_count"] += 1

        misrouting_signals_raw = route.get("misrouting_signals") if isinstance(route.get("misrouting_signals"), list) else routing_telemetry.get("misrouting_signals") if isinstance(routing_telemetry.get("misrouting_signals"), list) else []
        for signal in misrouting_signals_raw:
            token = str(signal or "").strip()
            if not token:
                continue
            misrouting_signal_counts[token] = int(misrouting_signal_counts.get(token) or 0) + 1
            by_family = misrouting_signal_by_family.setdefault(token, {})
            by_family[selected_family] = int(by_family.get(selected_family) or 0) + 1
            task_key = str(task_class or "").strip() or "missing_task_class"
            by_task_class = misrouting_signal_by_task_class.setdefault(token, {})
            by_task_class[task_key] = int(by_task_class.get(task_key) or 0) + 1
            provider_row["misrouting_signal_counts"][token] = int(provider_row["misrouting_signal_counts"].get(token) or 0) + 1

    ledger_total = 0
    ledger_event_counts: Dict[str, int] = {}
    for row in ledger_rows:
        ts = parse_iso(row.get("recorded_at") or row.get("evaluated_at"))
        if not _in_window(ts, window_start, now_ts):
            continue
        ledger_total += 1
        event_type = str(row.get("event_type") or "unknown")
        ledger_event_counts[event_type] = ledger_event_counts.get(event_type, 0) + 1

    for model_key, model_row in models_summary.items():
        unit = float(model_row.get("unit_cost_usd") or 0.0)
        routed = int(model_row.get("routing_pass_count") or 0)
        spend = _safe_round(unit * routed)
        model_row["estimated_spend_usd"] = spend

    for route_class in route_summary:
        route_summary[route_class]["estimated_spend_usd"] = _safe_round(route_cost_accumulator.get(route_class, 0.0))

    breaches: List[Dict[str, Any]] = []
    for route_class, row in route_summary.items():
        budget = row.get("budget_usd")
        spend = float(row.get("estimated_spend_usd") or 0.0)
        if isinstance(budget, (int, float)):
            remaining = _safe_round(float(budget) - spend)
            row["budget_remaining_usd"] = remaining
            if spend > float(budget):
                row["budget_state"] = "exceeded"
                breaches.append(
                    {
                        "route_class": route_class,
                        "budget_usd": float(budget),
                        "estimated_spend_usd": spend,
                        "overrun_usd": _safe_round(spend - float(budget)),
                    }
                )
            else:
                row["budget_state"] = "within_budget"
        else:
            row["budget_remaining_usd"] = None
            row["budget_state"] = "budget_unconfigured"

    status = "budget_exceeded" if breaches else "ok"

    provider_normalized_cost = {
        "schema": "clawd.model_rollout.provider_normalized_cost.v1",
        "normalization_method": {
            "priority": [
                "route_normalized_cost_usd",
                "route_cost_usd",
                "decision_cost_usd",
                "usage_cost_usd",
                "model_unit_cost_policy",
                "route_class_average_unit_cost_policy",
                "provider_family_average_unit_cost_policy",
                "global_average_unit_cost_policy",
                "unpriced_unresolved",
            ],
            "route_class_default_unit_cost": {
                key: _safe_round(value)
                for key, value in sorted(route_class_default_unit_cost.items())
            },
            "provider_family_default_unit_cost": {
                key: _safe_round(value)
                for key, value in sorted(family_default_unit_cost.items())
            },
            "global_default_unit_cost": _safe_round(global_default_unit_cost) if global_default_unit_cost is not None else None,
            "note": "Fallback cost defaults are applied when exact model/provider costs are unavailable to avoid zero-cost blind spots.",
        },
        "totals": {
            "routing_pass_count": routing_pass,
            "routing_block_count": routing_block,
            "task_class_tagged_count": task_class_tagged_count,
            "task_class_missing_count": task_class_missing_count,
            "priced_event_count": int(provider_pricing_totals.get("priced_event_count") or 0),
            "exact_priced_event_count": int(provider_pricing_totals.get("exact_priced_event_count") or 0),
            "fallback_priced_event_count": int(provider_pricing_totals.get("fallback_priced_event_count") or 0),
            "unpriced_event_count": int(provider_pricing_totals.get("unpriced_event_count") or 0),
            "estimated_spend_usd": _safe_round(provider_pricing_totals.get("estimated_spend_usd") or 0.0),
        },
        "providers": dict(sorted(provider_rows.items())),
        "family_utilization": dict(sorted(family_utilization.items())),
        "misrouting_incidents": {
            "total": int(sum(misrouting_signal_counts.values())) if misrouting_signal_counts else 0,
            "signal_counts": dict(sorted(misrouting_signal_counts.items())),
            "by_provider_family": {key: dict(sorted(value.items())) for key, value in sorted(misrouting_signal_by_family.items())},
            "by_task_class": {key: dict(sorted(value.items())) for key, value in sorted(misrouting_signal_by_task_class.items())},
        },
    }

    return {
        "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
        "generated_at": now_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "window_hours": window_hours,
        "window_start": window_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "policy": {
            "policy_id": policy.get("policy_id"),
            "policy_path": policy_meta.get("path"),
            "policy_schema_path": policy_meta.get("schema_path"),
        },
        "counts": {
            "gate_decisions_total": gate_total,
            "gate_pass": gate_pass,
            "gate_block": gate_block,
            "routing_decisions_total": routing_total,
            "routing_pass": routing_pass,
            "routing_block": routing_block,
            "ledger_events_total": ledger_total,
            "ledger_event_type_counts": ledger_event_counts,
        },
        "route_classes": route_summary,
        "models": dict(sorted(models_summary.items())),
        "provider_normalized_cost": provider_normalized_cost,
        "budget_breaches": breaches,
        "status": status,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic model rollout cost-governance snapshot producer (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--pool-policy", default=str(DEFAULT_POOL_POLICY_PATH), help="Unified model pool policy JSON path")
    ap.add_argument("--pool-policy-schema", default=str(DEFAULT_POOL_POLICY_SCHEMA), help="Unified model pool policy schema path")
    ap.add_argument("--gate-decisions", default=str(DEFAULT_GATE_DECISIONS), help="Gate decision JSONL path")
    ap.add_argument("--routing-decisions", default=str(DEFAULT_ROUTING_DECISIONS), help="Routing decision JSONL path")
    ap.add_argument("--ledger-events", default=str(DEFAULT_LEDGER_EVENTS), help="Rollout ledger events JSONL path")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output snapshot JSON path")
    ap.add_argument("--window-hours", type=int, default=24, help="Snapshot window in hours")
    ap.add_argument("--now", default="", help="Optional deterministic now timestamp (UTC ISO8601)")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pool_policy_path = resolve_path(repo_root, args.pool_policy)
    pool_policy_schema_path = resolve_path(repo_root, args.pool_policy_schema)
    gate_decisions_path = resolve_path(repo_root, args.gate_decisions)
    routing_decisions_path = resolve_path(repo_root, args.routing_decisions)
    ledger_events_path = resolve_path(repo_root, args.ledger_events)
    out_path = resolve_path(repo_root, args.out)

    now_ts = parse_iso(args.now) if str(args.now or "").strip() else dt.datetime.now(dt.timezone.utc)
    if now_ts is None:
        result = {
            "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "invalid_now_timestamp",
            "raw_now": args.now,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    if not isinstance(args.window_hours, int) or args.window_hours <= 0:
        result = {
            "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "invalid_window_hours",
            "window_hours": args.window_hours,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    policy_ok, reason, policy_meta, policy_doc = load_pool_policy(pool_policy_path, pool_policy_schema_path)
    if not policy_ok:
        result = {
            "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": reason,
            "details": policy_meta,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    try:
        gate_rows = load_jsonl(gate_decisions_path)
        routing_rows = load_jsonl(routing_decisions_path)
        ledger_rows = load_jsonl(ledger_events_path)
    except Exception as exc:
        result = {
            "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "input_read_failed",
            "detail": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    snapshot = build_snapshot(
        policy=policy_doc if isinstance(policy_doc, Mapping) else {},
        policy_meta=policy_meta,
        gate_rows=gate_rows,
        routing_rows=routing_rows,
        ledger_rows=ledger_rows,
        now_ts=now_ts,
        window_hours=int(args.window_hours),
    )

    if not is_within(repo_root, out_path):
        result = {
            "schema": "clawd.model_rollout_cost_governance_snapshot.v1",
            "generated_at": now_iso(),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
