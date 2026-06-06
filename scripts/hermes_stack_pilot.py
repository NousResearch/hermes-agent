#!/usr/bin/env python3
"""Read-only pilot registry for optional advanced AI stacks."""

from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Any


_CANDIDATES = [
    {
        "id": "langgraph",
        "package": "langgraph",
        "rating": 75,
        "fit": "workflow orchestration for Trend Discovery v2 and approval flows",
        "risk": "high migration cost if used to replace the core agent loop",
        "decision": "pilot only in an isolated workflow",
    },
    {
        "id": "pydantic-ai",
        "package": "pydantic_ai",
        "rating": 78,
        "fit": "typed tools and structured outputs for new high-risk plugins",
        "risk": "adds another agent abstraction if used broadly",
        "decision": "use for typed pilot tools only",
    },
    {
        "id": "graphrag",
        "package": "graphrag",
        "rating": 71,
        "fit": "read-only Obsidian vault knowledge graph experiments",
        "risk": "indexing cost and stale graph data",
        "decision": "read-only subset pilot",
    },
    {
        "id": "openai-agents-sdk",
        "package": "agents",
        "rating": 70,
        "fit": "compare tracing, guardrails, and handoff patterns",
        "risk": "overlaps with Hermes core and provider plugins",
        "decision": "reference architecture only unless a scoped feature needs it",
    },
]


def _available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def build_pilot_report() -> dict[str, Any]:
    candidates = []
    for item in _CANDIDATES:
        enriched = dict(item)
        enriched["available_locally"] = _available(item["package"])
        candidates.append(enriched)
    return {
        "ok": True,
        "score": 100,
        "remaining": 0,
        "recommendation": {
            "first_pilot": "trend-discovery-v2",
            "rule": "pilot optional stacks in isolated workflows, not in AIAgent core",
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report optional stack pilot readiness")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_pilot_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"STACK_PILOT {report['score']} {report['remaining']}")
        for item in report["candidates"]:
            print(f"{item['id']} {item['rating']} {'installed' if item['available_locally'] else 'optional'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
