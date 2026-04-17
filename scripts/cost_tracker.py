#!/usr/bin/env python3
"""Cost tracker CLI — analyze routing telemetry data."""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.routing_telemetry import load_events, summarize, TelemetryEvent


def parse_since(since_str: str) -> datetime:
    """Parse --since argument like '24h', '7d', '30d'."""
    if not since_str:
        return None
    
    match = re.match(r"(\d+)([hdw])", since_str.lower())
    if not match:
        raise ValueError(f"Invalid --since format: {since_str}. Use e.g. 24h, 7d, 30d")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == "h":
        delta = timedelta(hours=value)
    elif unit == "d":
        delta = timedelta(days=value)
    elif unit == "w":
        delta = timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")
    
    return datetime.now(timezone.utc) - delta


def cmd_summary(args):
    """Print summary table."""
    since = parse_since(args.since) if args.since else None
    store = Path(args.store) if args.store else None
    
    events = load_events(store=store, since=since)
    summary = summarize(events)
    
    if not summary["by_model"]:
        print("No events found.")
        return
    
    # Print ASCII table
    # Columns: model, requests, success%, avg_ms, p95_ms, premium_units, top_fail
    header = f"{'model':<25} {'requests':>10} {'success%':>10} {'avg_ms':>10} {'p95_ms':>10} {'premium_units':>15} {'top_fail':<20}"
    print(header)
    print("-" * len(header))
    
    for model, data in sorted(summary["by_model"].items()):
        requests = data["requests"]
        success_rate = data["success_rate"] * 100
        avg_ms = data["avg_latency_ms"]
        p95_ms = data["p95_latency_ms"]
        premium_units = data["total_premium_units"]
        
        # Find top failure type
        fail_types = data["fail_types"]
        top_fail = max(fail_types.items(), key=lambda x: x[1])[0] if fail_types else "-"
        
        avg_ms_str = f"{avg_ms:.1f}" if avg_ms is not None else "-"
        p95_ms_str = f"{p95_ms:.1f}" if p95_ms is not None else "-"
        
        print(f"{model:<25} {requests:>10} {success_rate:>9.1f}% {avg_ms_str:>10} {p95_ms_str:>10} {premium_units:>15.2f} {top_fail:<20}")
    
    print()
    print(f"Total requests: {summary['total_requests']}")
    print(f"Total premium units: {summary['total_premium_units']:.2f}")
    print(f"Savings vs baseline: {summary['savings_vs_baseline']*100:.1f}%")


def cmd_compare(args):
    """Compare actual costs vs baseline."""
    since = parse_since(args.since) if args.since else None
    store = Path(args.store) if args.store else None
    
    events = load_events(store=store, since=since)
    summary = summarize(events)
    
    baseline = summary["baseline_all_opus_units"]
    actual = summary["total_premium_units"]
    savings = summary["savings_vs_baseline"]
    savings_abs = baseline - actual
    
    print("=== Cost Comparison ===")
    print(f"Baseline (all Opus): {baseline:.2f} premium units")
    print(f"Actual cost:         {actual:.2f} premium units")
    print(f"Savings:             {savings*100:.1f}% ({savings_abs:.2f} units)")


def cmd_worst(args):
    """List worst-performing models by failure rate."""
    since = parse_since(args.since) if args.since else None
    store = Path(args.store) if args.store else None
    top_n = args.top
    
    events = load_events(store=store, since=since)
    summary = summarize(events)
    
    if not summary["by_model"]:
        print("No events found.")
        return
    
    # Filter models with at least 5 requests
    candidates = [
        (model, data)
        for model, data in summary["by_model"].items()
        if data["requests"] >= 5
    ]
    
    # Sort by failure rate (descending)
    candidates.sort(key=lambda x: 1 - x[1]["success_rate"], reverse=True)
    
    # Print top N
    header = f"{'model':<25} {'requests':>10} {'failure%':>10} {'top_error':<25}"
    print(header)
    print("-" * len(header))
    
    for model, data in candidates[:top_n]:
        requests = data["requests"]
        failure_rate = (1 - data["success_rate"]) * 100
        fail_types = data["fail_types"]
        top_error = max(fail_types.items(), key=lambda x: x[1])[0] if fail_types else "-"
        
        print(f"{model:<25} {requests:>10} {failure_rate:>9.1f}% {top_error:<25}")
    
    if not candidates:
        print("(No models with >= 5 requests)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze routing telemetry data",
        prog="cost_tracker",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # summary subcommand
    p_summary = subparsers.add_parser("summary", help="Print summary table")
    p_summary.add_argument("--since", help="Time window: 24h, 7d, 30d")
    p_summary.add_argument("--store", help="Path to telemetry JSONL file")
    p_summary.set_defaults(func=cmd_summary)
    
    # compare subcommand
    p_compare = subparsers.add_parser("compare", help="Compare costs vs baseline")
    p_compare.add_argument("--since", help="Time window: 24h, 7d, 30d")
    p_compare.add_argument("--store", help="Path to telemetry JSONL file")
    p_compare.set_defaults(func=cmd_compare)
    
    # worst subcommand
    p_worst = subparsers.add_parser("worst", help="List worst-performing models")
    p_worst.add_argument("--since", help="Time window: 24h, 7d, 30d")
    p_worst.add_argument("--store", help="Path to telemetry JSONL file")
    p_worst.add_argument("--top", type=int, default=5, help="Number of models to show")
    p_worst.set_defaults(func=cmd_worst)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
