"""Routing telemetry — track model usage, costs, and performance."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

# Premium multipliers per model
PREMIUM_MULTIPLIERS = {
    "gpt-5-mini": 0.0,
    "o3-mini": 0.25,
    "claude-sonnet-4.6": 1.0,
    "claude-sonnet-4": 1.0,
    "gpt-5.4": 3.0,
    "gpt-5": 3.0,
    "claude-opus-4.6": 5.0,
    "claude-opus-4.7": 5.0,
    "claude-opus-4": 5.0,
}

DEFAULT_STORE = Path.home() / ".hermes" / "router" / "telemetry.jsonl"


@dataclass
class TelemetryEvent:
    """A single routing telemetry event."""
    timestamp: str  # ISO UTC
    model: str
    provider: str
    domain: str
    decision_source: str
    turn_kind: str
    success: bool
    error_type: Optional[str]
    latency_ms: Optional[float]
    tokens_in: int
    tokens_out: int
    premium_multiplier: float
    premium_units: float


def multiplier_for(model: str) -> float:
    """Return the premium multiplier for a model. Default 1.0 if unknown."""
    return PREMIUM_MULTIPLIERS.get(model, 1.0)


def build_event(
    model: str,
    provider: str,
    domain: str,
    decision_source: str,
    turn_kind: str,
    success: bool,
    tokens_in: int,
    tokens_out: int,
    error_type: Optional[str] = None,
    latency_ms: Optional[float] = None,
    premium_multiplier: Optional[float] = None,
    premium_units: Optional[float] = None,
) -> TelemetryEvent:
    """Build a TelemetryEvent. Computes premium_units if not provided."""
    if premium_multiplier is None:
        premium_multiplier = multiplier_for(model)
    
    if premium_units is None:
        total_tokens = tokens_in + tokens_out
        premium_units = premium_multiplier * total_tokens / 1000.0
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return TelemetryEvent(
        timestamp=timestamp,
        model=model,
        provider=provider,
        domain=domain,
        decision_source=decision_source,
        turn_kind=turn_kind,
        success=success,
        error_type=error_type,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        premium_multiplier=premium_multiplier,
        premium_units=premium_units,
    )


def record_event(event: TelemetryEvent, store: Optional[Path] = None) -> Path:
    """Append a telemetry event to the JSONL store. Creates dirs if needed."""
    if store is None:
        store = DEFAULT_STORE
    
    # Create parent directory if it doesn't exist
    store.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic append: open in append mode, write JSON + newline
    with open(store, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event)) + "\n")
    
    return store


def load_events(
    store: Optional[Path] = None,
    since: Optional[datetime] = None,
) -> list[TelemetryEvent]:
    """Load events from the JSONL store. Filter by since if provided."""
    if store is None:
        store = DEFAULT_STORE
    
    if not store.exists():
        return []
    
    events = []
    with open(store, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Parse timestamp and filter by since
                event_ts = datetime.fromisoformat(data["timestamp"])
                if since is not None:
                    # Make since aware if needed
                    if since.tzinfo is None:
                        since = since.replace(tzinfo=timezone.utc)
                    if event_ts < since:
                        continue
                
                event = TelemetryEvent(**data)
                events.append(event)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip malformed lines
                continue
    
    return events


def summarize(events: list[TelemetryEvent]) -> dict:
    """Summarize a list of telemetry events.
    
    Returns dict with:
      - by_model: {model: {requests, successes, failures, success_rate,
                          avg_latency_ms, p95_latency_ms, total_premium_units,
                          fail_types: {type: count}}}
      - worst_model: model with worst success_rate (min 5 requests; None if none)
      - optimal_model: model with success_rate >= 0.95 and lowest avg_latency; None if none
      - total_requests
      - total_premium_units
      - baseline_all_opus_units: sum(tokens) * 5 / 1000
      - savings_vs_baseline: float 0..1
    """
    if not events:
        return {
            "by_model": {},
            "worst_model": None,
            "optimal_model": None,
            "total_requests": 0,
            "total_premium_units": 0.0,
            "baseline_all_opus_units": 0.0,
            "savings_vs_baseline": 0.0,
        }
    
    # Group by model
    by_model_data: dict[str, dict] = {}
    total_tokens = 0
    total_premium_units = 0.0
    
    for ev in events:
        model = ev.model
        if model not in by_model_data:
            by_model_data[model] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "latencies": [],
                "premium_units": 0.0,
                "fail_types": {},
            }
        
        d = by_model_data[model]
        d["requests"] += 1
        d["premium_units"] += ev.premium_units
        total_tokens += ev.tokens_in + ev.tokens_out
        total_premium_units += ev.premium_units
        
        if ev.success:
            d["successes"] += 1
        else:
            d["failures"] += 1
            if ev.error_type:
                d["fail_types"][ev.error_type] = d["fail_types"].get(ev.error_type, 0) + 1
        
        if ev.latency_ms is not None:
            d["latencies"].append(ev.latency_ms)
    
    # Build summary per model
    by_model = {}
    worst_model = None
    worst_success_rate = 1.0
    optimal_model = None
    optimal_latency = float("inf")
    
    for model, d in by_model_data.items():
        requests = d["requests"]
        successes = d["successes"]
        failures = d["failures"]
        success_rate = successes / requests if requests > 0 else 0.0
        
        latencies = d["latencies"]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        p95_latency = None
        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95_latency = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        
        by_model[model] = {
            "requests": requests,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "total_premium_units": d["premium_units"],
            "fail_types": d["fail_types"],
        }
        
        # Worst model: min success_rate with at least 5 requests
        if requests >= 5:
            if success_rate < worst_success_rate:
                worst_success_rate = success_rate
                worst_model = model
        
        # Optimal model: success_rate >= 0.95 and lowest avg_latency
        if success_rate >= 0.95 and avg_latency is not None:
            if avg_latency < optimal_latency:
                optimal_latency = avg_latency
                optimal_model = model
    
    baseline_all_opus_units = total_tokens * 5.0 / 1000.0
    savings_vs_baseline = (
        (baseline_all_opus_units - total_premium_units) / baseline_all_opus_units
        if baseline_all_opus_units > 0 else 0.0
    )
    
    return {
        "by_model": by_model,
        "worst_model": worst_model,
        "optimal_model": optimal_model,
        "total_requests": len(events),
        "total_premium_units": total_premium_units,
        "baseline_all_opus_units": baseline_all_opus_units,
        "savings_vs_baseline": savings_vs_baseline,
    }


def wrap_resolve_turn_route(store: Optional[Path] = None):
    """Decorator for resolve_turn_route functions.
    
    Records a telemetry event with:
      - success=True/False
      - latency measured
      - tokens_in=0, tokens_out=0
      - turn_kind inferred from result label ("smart route" vs primary fallback)
    
    The decorated function signature is:
        fn(user_message: str, routing_config: Optional[Dict], primary: Dict) -> Dict
    
    If the wrapped function raises, the exception is re-raised after recording.
    If the function returns normally (even if it fell back to primary internally),
    success=True and turn_kind is inferred from the result's label.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(user_message, routing_config, primary, *args, **kwargs):
            start = time.perf_counter()
            success = True
            error_type = None
            result = None
            try:
                result = fn(user_message, routing_config, primary, *args, **kwargs)
            except Exception as e:
                result = None
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                
                # Extract model/provider from result if available
                model = "unknown"
                provider = "unknown"
                label = None
                if isinstance(result, dict):
                    model = result.get("model", "unknown")
                    runtime = result.get("runtime", {})
                    if isinstance(runtime, dict):
                        provider = runtime.get("provider", "unknown")
                    else:
                        provider = result.get("provider", "unknown")
                    label = result.get("label")
                
                # Infer turn_kind from label
                if not success:
                    turn_kind = "error"
                elif label and "smart route" in label.lower():
                    turn_kind = "smart_route"
                elif result is not None and isinstance(result, dict) and result.get("label") is None:
                    # No label means primary was used (no routing happened)
                    turn_kind = "primary"
                else:
                    # Has a label but not "smart route" - could be primary_fallback
                    if label and "primary" in label.lower():
                        turn_kind = "primary_fallback"
                    else:
                        turn_kind = "primary"
                
                event = build_event(
                    model=model,
                    provider=provider,
                    domain="general",
                    decision_source="resolve_turn_route",
                    turn_kind=turn_kind,
                    success=success,
                    error_type=error_type,
                    latency_ms=latency_ms,
                    tokens_in=0,
                    tokens_out=0,
                )
                record_event(event, store=store)
            
            return result
        return wrapper
    return decorator
