#!/usr/bin/env python3
"""
Smart Model Router CLI

Quick model recommendation based on task intensity.

Usage:
    python3 model-router.py --task light [--prefer-free]
    python3 model-router.py --task medium
    python3 model-router.py --task heavy
    python3 model-router.py --status
    python3 model-router.py --table
    python3 model-router.py --json
"""

import argparse
import json
import sys
from pathlib import Path


ROUTING_TABLE = {
    "light": {
        "description": "Simple lookups, confirmations, formatting, light cron jobs",
        "chain": [
            {"model": "qwen2.5-3b-instruct", "provider": "lmstudio", "cost": "free", "note": "Local, no rate limits"},
            {"model": "owl-alpha", "provider": "openrouter", "cost": "free", "note": "OpenRouter free tier"},
            {"model": "glm-4.7-free", "provider": "opencode-zen", "cost": "free", "note": "Zen free tier"},
            {"model": "claude-3-5-haiku-20241022", "provider": "anthropic", "cost": "paid", "note": "Claude Pro Haiku"},
        ],
    },
    "medium": {
        "description": "Code writing, multi-step tool use, refactoring, docs",
        "chain": [
            {"model": "claude-sonnet-4-20250514", "provider": "anthropic", "cost": "paid", "note": "Claude Sonnet — best value"},
            {"model": "glm-4.7-free", "provider": "opencode-zen", "cost": "free", "note": "Zen free tier fallback"},
            {"model": "owl-alpha", "provider": "openrouter", "cost": "free", "note": "OpenRouter free fallback"},
            {"model": "claude-3-5-haiku-20241022", "provider": "anthropic", "cost": "paid", "note": "If Sonnet rate-limited"},
        ],
    },
    "heavy": {
        "description": "Large features, architecture, complex debugging, code review",
        "chain": [
            {"model": "claude-opus-4-20250514", "provider": "anthropic", "cost": "paid", "note": "Claude Opus — best reasoning"},
            {"model": "claude-sonnet-4-20250514", "provider": "anthropic", "cost": "paid", "note": "Sonnet if Opus rate-limited"},
            {"model": "glm-4.7-free", "provider": "opencode-zen", "cost": "free", "note": "Free fallback"},
            {"model": "owl-alpha", "provider": "openrouter", "cost": "free", "note": "OpenRouter free fallback"},
        ],
    },
}


def detect_available_providers():
    """Check which providers are configured.

    Only checks config-driven providers and local service reachability.
    Does NOT read .env or check API key presence — if credentials are
    missing, Hermes failover handles 401s automatically.
    """
    avail = {
        "lmstudio": False,
    }
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2)
        avail["lmstudio"] = True
    except Exception:
        pass
    return avail


def get_recommended_model(task_intensity, prefer_free=False):
    if task_intensity not in ROUTING_TABLE:
        return {"error": f"Unknown task: {task_intensity}. Use: light, medium, heavy"}
    avail = detect_available_providers()
    chain = ROUTING_TABLE[task_intensity]["chain"]
    if prefer_free:
        free = [m for m in chain if m["cost"] == "free"]
        for m in free:
            if avail.get(m["provider"], False):
                return {"recommended": m, "reason": "Free tier preferred, provider available"}
    for m in chain:
        if avail.get(m["provider"], False):
            return {"recommended": m, "reason": f"Best available for {task_intensity} task"}
    return {
        "recommended": chain[0],
        "reason": f"Default: {chain[0]['provider']}/{chain[0]['model']}",
        "warning": "No configured credentials for any provider",
    }


def format_output(result):
    if "error" in result:
        return f"Error: {result['error']}"
    rec = result["recommended"]
    lines = [
        "+" + "-" * 58 + "+",
        "|  Smart Model Router Recommendation" + " " * 22 + "|",
        "+" + "-" * 58 + "+",
        f"|  Model:    {rec['model']:<44} |",
        f"|  Provider: {rec['provider']:<44} |",
        f"|  Cost:     {rec['cost']:<44} |",
        f"|  Note:     {rec['note']:<44} |",
        "+" + "-" * 58 + "+",
        f"|  Reason:   {result['reason']:<44} |",
        "+" + "-" * 58 + "+",
    ]
    if "warning" in result:
        lines.append(f"\nWARNING: {result['warning']}")
    return "\n".join(lines)


def print_table():
    print("\nSMART MODEL ROUTING TABLE")
    print("=" * 60)
    for tier, info in ROUTING_TABLE.items():
        print(f"\n{tier.upper()} - {info['description']}")
        print("-" * 40)
        for i, m in enumerate(info["chain"]):
            marker = "->" if i == 0 else "  "
            tag = "PAID" if m["cost"] == "paid" else "FREE"
            print(f"  {marker} [{tag}] {m['provider']}/{m['model']}")
            if m.get("note"):
                print(f"         {m['note']}")


def print_status():
    avail = detect_available_providers()
    print("\nPROVIDER STATUS")
    print("=" * 40)
    for prov, ok in avail.items():
        print(f"  [{'AVAILABLE' if ok else 'NOT AVAILABLE'}] {prov}")


def main():
    parser = argparse.ArgumentParser(description="Smart Model Router CLI")
    parser.add_argument("--task", choices=["light", "medium", "heavy"])
    parser.add_argument("--prefer-free", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.table:
        print_table()
        return
    if args.status:
        print_status()
        return
    if args.task:
        result = get_recommended_model(args.task, prefer_free=args.prefer_free)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_output(result))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
