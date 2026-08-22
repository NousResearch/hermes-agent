#!/usr/bin/env python3
"""debug_trajectory.py — Analyze session transcripts, cache hit rates, and token bottlenecks.

Usage:
  debug_trajectory.py analyze <transcript.jsonl> [--json]
  debug_trajectory.py cache <transcript.jsonl> [--threshold 80] [--json]
  debug_trajectory.py tools <transcript.jsonl> [--json]
  debug_trajectory.py turn <transcript.jsonl> --turn N [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_transcript_events(transcript_path: Path) -> List[Dict[str, Any]]:
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    events = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ev.setdefault("step_index", idx)
                events.append(ev)
            except Exception:
                continue
    return events


def analyze_cache_efficiency(
    events: List[Dict[str, Any]],
    threshold_pct: float = 80.0,
) -> Dict[str, Any]:
    """Calculate prompt cache hit rates across turns and detect cache-busting drops."""
    total_prompt_tokens = 0
    total_cached_tokens = 0
    turns = []
    cache_busts = []

    prev_hit_rate = 100.0

    for ev in events:
        usage = ev.get("usage") or ev.get("token_usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        cached_tokens = (
            usage.get("prompt_cache_hit_tokens")
            or usage.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        )
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        if prompt_tokens > 0:
            hit_rate = (cached_tokens / prompt_tokens) * 100.0
            total_prompt_tokens += prompt_tokens
            total_cached_tokens += cached_tokens

            turn_info = {
                "step_index": ev.get("step_index", 0),
                "type": ev.get("type", "UNKNOWN"),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "cache_hit_rate_pct": round(hit_rate, 2),
            }
            turns.append(turn_info)

            # Detect significant drop in cache hit rate indicating a prefix bust
            if prev_hit_rate >= threshold_pct and hit_rate < threshold_pct:
                cache_busts.append({
                    "step_index": ev.get("step_index", 0),
                    "previous_hit_rate": round(prev_hit_rate, 2),
                    "current_hit_rate": round(hit_rate, 2),
                    "lost_cached_tokens": prompt_tokens - cached_tokens,
                })
            prev_hit_rate = hit_rate

    overall_hit_rate = (
        round((total_cached_tokens / total_prompt_tokens) * 100.0, 2)
        if total_prompt_tokens > 0
        else 0.0
    )

    return {
        "overall_cache_hit_rate_pct": overall_hit_rate,
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_turns_analyzed": len(turns),
        "cache_busts_detected": len(cache_busts),
        "cache_busts": cache_busts,
        "turns": turns,
    }


def analyze_tool_usage(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate tool execution metrics, success vs failure, and frequencies."""
    tool_stats: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        tool_calls = ev.get("tool_calls", [])
        status = ev.get("status", "DONE")
        for tc in tool_calls:
            name = tc.get("name") or tc.get("function", {}).get("name") or "unknown_tool"
            if name not in tool_stats:
                tool_stats[name] = {"calls": 0, "errors": 0}
            tool_stats[name]["calls"] += 1
            if status == "ERROR" or ev.get("error"):
                tool_stats[name]["errors"] += 1

    return {
        "total_tools_called": sum(s["calls"] for s in tool_stats.values()),
        "tools": tool_stats,
    }


def analyze_trajectory_summary(transcript_path: Path) -> Dict[str, Any]:
    events = load_transcript_events(transcript_path)
    cache_metrics = analyze_cache_efficiency(events)
    tool_metrics = analyze_tool_usage(events)

    user_count = sum(1 for e in events if e.get("type") == "USER_INPUT")
    planner_count = sum(1 for e in events if e.get("type") == "PLANNER_RESPONSE")

    return {
        "transcript": str(transcript_path),
        "total_steps": len(events),
        "user_messages": user_count,
        "model_responses": planner_count,
        "cache_summary": {
            "overall_hit_rate_pct": cache_metrics["overall_cache_hit_rate_pct"],
            "total_prompt_tokens": cache_metrics["total_prompt_tokens"],
            "total_cached_tokens": cache_metrics["total_cached_tokens"],
            "cache_bust_drops": cache_metrics["cache_busts_detected"],
        },
        "tool_summary": tool_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Trajectory & Prompt Cache Debugger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_an = subparsers.add_parser("analyze", help="Full summary analysis of transcript.")
    p_an.add_argument("transcript", type=Path, help="Path to transcript.jsonl.")
    p_an.add_argument("--json", action="store_true", help="Output as JSON.")

    # cache
    p_ca = subparsers.add_parser("cache", help="Analyze prompt cache hit rates and drops.")
    p_ca.add_argument("transcript", type=Path, help="Path to transcript.jsonl.")
    p_ca.add_argument("--threshold", type=float, default=80.0, help="Cache drop threshold percentage.")
    p_ca.add_argument("--json", action="store_true", help="Output as JSON.")

    # tools
    p_to = subparsers.add_parser("tools", help="Analyze tool calls and failure hotspots.")
    p_to.add_argument("transcript", type=Path, help="Path to transcript.jsonl.")
    p_to.add_argument("--json", action="store_true", help="Output as JSON.")

    # turn
    p_tu = subparsers.add_parser("turn", help="Inspect a specific step or turn.")
    p_tu.add_argument("transcript", type=Path, help="Path to transcript.jsonl.")
    p_tu.add_argument("--turn", type=int, required=True, help="Step index.")
    p_tu.add_argument("--json", action="store_true", help="Output as JSON.")

    args = parser.parse_args()

    if args.command == "analyze":
        res = analyze_trajectory_summary(args.transcript)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== Trajectory Analysis Summary ===")
            print(f"Total Steps: {res['total_steps']} (User: {res['user_messages']}, Model: {res['model_responses']})")
            print(f"Cache Hit Rate: {res['cache_summary']['overall_hit_rate_pct']}%")
            print(f"Total Prompt Tokens: {res['cache_summary']['total_prompt_tokens']:,}")
            print(f"Total Cached Tokens: {res['cache_summary']['total_cached_tokens']:,}")
            print(f"Cache Bust Drops: {res['cache_summary']['cache_bust_drops']}")
            print(f"Total Tool Invocations: {res['tool_summary']['total_tools_called']}")
            for t_name, t_stat in res['tool_summary']['tools'].items():
                print(f"  - {t_name}: {t_stat['calls']} calls ({t_stat['errors']} errors)")

    elif args.command == "cache":
        events = load_transcript_events(args.transcript)
        res = analyze_cache_efficiency(events, threshold_pct=args.threshold)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== Prompt Cache Efficiency ===")
            print(f"Overall Cache Hit Rate: {res['overall_cache_hit_rate_pct']}%")
            print(f"Cached Tokens: {res['total_cached_tokens']:,} / {res['total_prompt_tokens']:,}")
            if res['cache_busts']:
                print(f"\n⚠️ Detected {len(res['cache_busts'])} Cache Bust Drops (<{args.threshold}%):")
                for cb in res['cache_busts']:
                    print(f"  Step {cb['step_index']}: Hit rate dropped from {cb['previous_hit_rate']}% to {cb['current_hit_rate']}% (-{cb['lost_cached_tokens']} cached tokens)")
            else:
                print("✓ No severe cache-busting drops detected.")

    elif args.command == "tools":
        events = load_transcript_events(args.transcript)
        res = analyze_tool_usage(events)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== Tool Usage & Reliability ===")
            print(f"Total Tool Calls: {res['total_tools_called']}")
            for t_name, t_stat in res['tools'].items():
                err_rate = (t_stat['errors'] / t_stat['calls'] * 100) if t_stat['calls'] > 0 else 0
                print(f"  - {t_name:<20}: {t_stat['calls']:>3} calls (errors: {t_stat['errors']}, error rate: {err_rate:.1f}%)")

    elif args.command == "turn":
        events = load_transcript_events(args.transcript)
        target = next((e for e in events if e.get("step_index") == args.turn), None)
        if not target:
            print(f"Step index {args.turn} not found in transcript.", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(target, indent=2))
        else:
            print(f"=== Step {args.turn} [{target.get('type')}] ===")
            if target.get("content"):
                print(f"Content:\n{target['content']}")
            if target.get("tool_calls"):
                print(f"Tool Calls: {json.dumps(target['tool_calls'], indent=2)}")
            if target.get("usage"):
                print(f"Usage: {json.dumps(target['usage'], indent=2)}")


if __name__ == "__main__":
    main()
