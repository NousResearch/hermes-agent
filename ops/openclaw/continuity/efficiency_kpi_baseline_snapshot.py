#!/usr/bin/env python3
"""Generate XE-101 efficiency KPI baseline snapshot + validation packet."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(ts: Any) -> dt.datetime | None:
    txt = str(ts or "").strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(txt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _metric(
    *,
    metric_id: str,
    value: float | int | None,
    unit: str,
    signal_state: str,
    source: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value": value,
        "unit": unit,
        "signal_state": signal_state,
        "source": source,
        "note": note,
    }


def build_snapshot(root: Path, window_hours: int) -> Dict[str, Any]:
    generated_at = now_utc()
    window_start = generated_at - dt.timedelta(hours=window_hours)

    cost_path = root / "state" / "continuity" / "model_rollout_cost" / "latest.json"
    routing_path = root / "state" / "continuity" / "session_topology_router" / "decisions.jsonl"
    transition_path = root / "state" / "continuity" / "history" / "execution_frontier_transition_attempts.jsonl"

    cost_obj = load_json(cost_path)
    routing_rows = read_jsonl(routing_path)
    transition_rows = read_jsonl(transition_path)

    route_classes = cost_obj.get("route_classes") if isinstance(cost_obj.get("route_classes"), dict) else {}
    heavy_cost_obj = route_classes.get("HEAVY") if isinstance(route_classes.get("HEAVY"), dict) else {}
    spark_cost_obj = route_classes.get("SPARK") if isinstance(route_classes.get("SPARK"), dict) else {}
    no_llm_cost_obj = route_classes.get("NO_LLM") if isinstance(route_classes.get("NO_LLM"), dict) else {}

    heavy_estimated_cost = float(heavy_cost_obj.get("estimated_spend_usd") or 0.0)
    spark_estimated_cost = float(spark_cost_obj.get("estimated_spend_usd") or 0.0)
    no_llm_estimated_cost = float(no_llm_cost_obj.get("estimated_spend_usd") or 0.0)
    total_estimated_cost = heavy_estimated_cost + spark_estimated_cost + no_llm_estimated_cost

    routing_window_rows: List[Dict[str, Any]] = []
    for row in routing_rows:
        evaluated = parse_iso(row.get("evaluated_at"))
        if evaluated is None:
            continue
        if evaluated < window_start or evaluated > generated_at:
            continue
        routing_window_rows.append(row)

    routing_total = len(routing_window_rows)
    heavy_total = 0
    for row in routing_window_rows:
        route = row.get("route") if isinstance(row.get("route"), dict) else {}
        route_class = str(route.get("route_class") or "").strip().upper()
        if route_class == "HEAVY":
            heavy_total += 1

    heavy_rate = safe_div(float(heavy_total), float(routing_total))

    transition_window_rows: List[Dict[str, Any]] = []
    for row in transition_rows:
        recorded = parse_iso(row.get("recorded_at"))
        if recorded is None:
            continue
        if recorded < window_start or recorded > generated_at:
            continue
        transition_window_rows.append(row)

    apply_rows = [row for row in transition_window_rows if str(row.get("decision") or "").upper() == "APPLY"]
    attempts_by_candidate = Counter(str(row.get("next_candidate") or "unknown") for row in apply_rows)
    reruns_total = sum(max(0, count - 1) for count in attempts_by_candidate.values())
    reruns_per_candidate = safe_div(float(reruns_total), float(len(attempts_by_candidate)))

    cost_per_decision = safe_div(total_estimated_cost, float(routing_total))

    token_metrics: List[Dict[str, Any]] = [
        _metric(
            metric_id="tokens_per_landed_slice",
            value=None,
            unit="tokens/slice",
            signal_state="no_signal",
            source="pending_token_telemetry_surface",
            note="XE-101 baseline established; token counters are not yet emitted by runtime surfaces.",
        ),
        _metric(
            metric_id="avg_context_tokens_per_turn",
            value=None,
            unit="tokens/turn",
            signal_state="no_signal",
            source="pending_context_telemetry_surface",
            note="Context growth tracked as KPI contract placeholder until context token instrumentation lands.",
        ),
    ]

    cost_metrics: List[Dict[str, Any]] = [
        _metric(
            metric_id="estimated_cost_total_window",
            value=round4(total_estimated_cost),
            unit="usd",
            signal_state="derived",
            source="state/continuity/model_rollout_cost/latest.json",
            note="Derived from route-class spend projections in model rollout cost snapshot.",
        ),
        _metric(
            metric_id="estimated_cost_per_routing_decision",
            value=round4(cost_per_decision),
            unit="usd/decision",
            signal_state="derived" if cost_per_decision is not None else "no_signal",
            source="state/continuity/model_rollout_cost/latest.json + routing decisions",
            note="Uses 24h routing decision count; null when no routing decisions exist.",
        ),
    ]

    heavy_metrics: List[Dict[str, Any]] = [
        _metric(
            metric_id="heavy_tier_utilization_rate",
            value=round4(heavy_rate),
            unit="ratio",
            signal_state="derived" if heavy_rate is not None else "no_signal",
            source="state/continuity/session_topology_router/decisions.jsonl",
            note="Share of routing decisions classified as HEAVY within the baseline window.",
        ),
        _metric(
            metric_id="heavy_tier_decision_count",
            value=heavy_total,
            unit="count",
            signal_state="derived",
            source="state/continuity/session_topology_router/decisions.jsonl",
            note="Count of decisions with route.route_class=HEAVY in baseline window.",
        ),
    ]

    rerun_metrics: List[Dict[str, Any]] = [
        _metric(
            metric_id="reruns_total",
            value=reruns_total,
            unit="count",
            signal_state="derived",
            source="state/continuity/history/execution_frontier_transition_attempts.jsonl",
            note="Rerun proxy computed from repeated APPLY decisions for same next_candidate.",
        ),
        _metric(
            metric_id="reruns_per_slice",
            value=round4(reruns_per_candidate),
            unit="reruns/slice",
            signal_state="derived" if reruns_per_candidate is not None else "no_signal",
            source="state/continuity/history/execution_frontier_transition_attempts.jsonl",
            note="Average reruns per candidate in baseline window; proxy until explicit slice rerun ledger is emitted.",
        ),
    ]

    all_metrics: List[Dict[str, Any]] = token_metrics + cost_metrics + heavy_metrics + rerun_metrics
    no_signal_count = sum(1 for row in all_metrics if row.get("signal_state") == "no_signal")
    measured_or_derived_count = len(all_metrics) - no_signal_count

    status = "ok"
    if measured_or_derived_count == 0:
        status = "degraded"
    elif no_signal_count > 0:
        status = "partial"

    snapshot = {
        "schema": "clawd.efficiency_kpi_baseline_snapshot.v1",
        "generated_at": iso_z(generated_at),
        "window_hours": window_hours,
        "window_start": iso_z(window_start),
        "window_end": iso_z(generated_at),
        "status": status,
        "lane": "XE",
        "slice_id": "XE-101",
        "objective": "efficiency_telemetry_and_kpi_baseline_foundation",
        "summary": {
            "kpi_count": len(all_metrics),
            "measured_or_derived_count": measured_or_derived_count,
            "no_signal_count": no_signal_count,
            "routing_decision_count": routing_total,
            "heavy_tier_decision_count": heavy_total,
            "rerun_proxy_apply_count": len(apply_rows),
            "rerun_proxy_candidate_count": len(attempts_by_candidate),
        },
        "kpis": {
            "tokens": token_metrics,
            "cost": cost_metrics,
            "heavy_tier_usage": heavy_metrics,
            "reruns": rerun_metrics,
            "context_growth": [
                _metric(
                    metric_id="context_growth_rate",
                    value=None,
                    unit="ratio",
                    signal_state="no_signal",
                    source="pending_context_telemetry_surface",
                    note="Context growth trend baseline deferred until turn-level token counters are emitted.",
                )
            ],
        },
        "dashboard": {
            "cards": [
                {
                    "id": "tokens",
                    "label": "Tokens / Context",
                    "status": "attention" if no_signal_count else "ok",
                    "primary_metric": "tokens_per_landed_slice",
                },
                {
                    "id": "cost",
                    "label": "Cost",
                    "status": "ok",
                    "primary_metric": "estimated_cost_total_window",
                },
                {
                    "id": "heavy_tier",
                    "label": "Heavy Tier Usage",
                    "status": "ok" if heavy_rate is not None else "attention",
                    "primary_metric": "heavy_tier_utilization_rate",
                },
                {
                    "id": "reruns",
                    "label": "Reruns",
                    "status": "ok" if reruns_per_candidate is not None else "attention",
                    "primary_metric": "reruns_per_slice",
                },
            ],
            "operator_note": "XE-101 baseline is active. Token/context metrics are scaffolded and currently awaiting dedicated runtime counters.",
        },
        "source_refs": {
            "model_rollout_cost": "state/continuity/model_rollout_cost/latest.json",
            "routing_decisions": "state/continuity/session_topology_router/decisions.jsonl",
            "execution_frontier_transition_attempts": "state/continuity/history/execution_frontier_transition_attempts.jsonl",
            "queue_patch_contract": "state/continuity/latest/true_expanded_roadmap_efficiency_queue_patch_2026-03-28.json",
            "doctrine_pack": "reports/efficiency_layer_doctrine_pack_2026-03-28.md",
        },
    }
    return snapshot


def validate_snapshot(snapshot: Dict[str, Any], schema_path: Path) -> Tuple[bool, str, List[str]]:
    if Draft202012Validator is None:
        return False, "validator_unavailable", ["jsonschema_dependency_missing"]

    try:
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "schema_unavailable", [f"schema_load_failed:{exc}"]

    validator = Draft202012Validator(schema_obj)
    errors = sorted(validator.iter_errors(snapshot), key=lambda err: list(err.path))
    if errors:
        out = []
        for err in errors:
            path = "/".join(str(x) for x in err.path)
            out.append(f"{path or '<root>'}: {err.message}")
        return False, "schema_invalid", out
    return True, "ok", []


def atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XE-101 efficiency KPI baseline artifacts.")
    parser.add_argument("--root", default=os.environ.get("OPENCLAW_ROOT", "/home/yeqiuqiu/clawd-architect"))
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--schema",
        default="state/continuity/latest/xe101_efficiency_kpi_schema_v1.json",
        help="Schema path relative to root",
    )
    parser.add_argument(
        "--snapshot-out",
        default="state/continuity/latest/efficiency_kpi_baseline_latest.json",
        help="Snapshot output path relative to root",
    )
    parser.add_argument(
        "--validation-out",
        default="state/continuity/latest/efficiency_kpi_baseline_validation_latest.json",
        help="Validation output path relative to root",
    )
    parser.add_argument("--json", action="store_true", help="Print merged packet as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    schema_path = (root / args.schema).resolve()
    snapshot_out = (root / args.snapshot_out).resolve()
    validation_out = (root / args.validation_out).resolve()

    snapshot = build_snapshot(root, window_hours=max(1, int(args.window_hours)))
    ok, reason, errors = validate_snapshot(snapshot, schema_path)

    validation = {
        "schema": "clawd.efficiency_kpi_baseline_validation.v1",
        "generated_at": iso_z(now_utc()),
        "slice_id": "XE-101",
        "status": "pass" if ok else "fail",
        "reason": reason,
        "error_count": len(errors),
        "errors": errors,
        "validated_snapshot": str(snapshot_out.relative_to(root)),
        "schema_ref": str(schema_path.relative_to(root)),
    }

    atomic_write(snapshot_out, snapshot)
    atomic_write(validation_out, validation)

    if args.json:
        print(
            json.dumps(
                {
                    "snapshot": snapshot,
                    "validation": validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            f"XE-101 baseline status={snapshot.get('status')} validation={validation.get('status')} "
            f"kpis={((snapshot.get('summary') or {}).get('kpi_count') or 0)}"
        )

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
