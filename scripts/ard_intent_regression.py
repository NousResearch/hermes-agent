#!/usr/bin/env python3
"""ARD intent regression runner.

A small pytest/Promptbeat/RAMPART-style harness for discovery quality: it runs
high-value user intents against local ARD search and imported ARD caches, then
emits an evidence-first JSON report. No external Promptbeat/RAMPART dependency is
required; the report shape is intentionally simple and CI-friendly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def default_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "id": "local-youtube-transcript",
            "query": "summarize youtube transcript",
            "mode": "local",
            "limit": 5,
            "expect": {"identifier_contains": "youtube-content"},
        },
        {
            "id": "local-browser-qa",
            "query": "browser qa exploratory web app testing",
            "mode": "local",
            "limit": 8,
            "expect": {"identifier_contains": "dogfood"},
        },
        {
            "id": "mcp-registry-image-generation",
            "query": "image generation mcp",
            "mode": "registry",
            "registry": "https://registry.modelcontextprotocol.io",
            "limit": 8,
            "expect": {"type": "application/mcp-server-card+json"},
        },
        {
            "id": "gitdb-promptbeat-security-eval",
            "query": "promptbeat ai security evaluation",
            "mode": "registry",
            "registry": "https://gitdb.local",
            "limit": 8,
            "expect": {"identifier_contains": "promptbeat"},
        },
    ]


def _normalize_result(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return {
        "identifier": getattr(item, "identifier", ""),
        "displayName": getattr(item, "name", ""),
        "description": getattr(item, "description", ""),
        "tags": getattr(item, "tags", []),
        "type": getattr(item, "extra", {}).get("ard_type") if hasattr(item, "extra") else None,
        "extra": getattr(item, "extra", {}) if hasattr(item, "extra") else {},
    }


def run_scenario(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(scenario.get("limit", 5))
    query = str(scenario.get("query", ""))
    mode = str(scenario.get("mode", "local"))
    if mode == "registry":
        from tools.skills_hub import ArdSource
        registry = str(scenario.get("registry") or "https://registry.modelcontextprotocol.io")
        return [_normalize_result(r) for r in ArdSource(registries=[registry]).search(query, limit=limit)]
    from tools.skills_hub import ard_local_search
    return [_normalize_result(r) for r in ard_local_search(query, limit=limit)]


def _matches_expectation(result: dict[str, Any], expect: dict[str, Any]) -> bool:
    ident = str(result.get("identifier", ""))
    name = str(result.get("displayName") or result.get("name") or "")
    rtype = str(result.get("type") or result.get("extra", {}).get("ard_type") or "")
    if expect.get("identifier_contains") and str(expect["identifier_contains"]) not in ident:
        return False
    if expect.get("name_contains") and str(expect["name_contains"]).lower() not in name.lower():
        return False
    if expect.get("type") and str(expect["type"]) != rtype:
        return False
    return True


def evaluate_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    runner: Callable[[dict[str, Any]], list[Any]] = run_scenario,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    passed = 0
    for scenario in scenarios:
        raw_results = runner(scenario)
        normalized = [_normalize_result(r) for r in raw_results]
        raw_expect = scenario.get("expect")
        expect: dict[str, Any] = raw_expect if isinstance(raw_expect, dict) else {}
        matched = [r for r in normalized if _matches_expectation(r, expect)]
        status = "passed" if matched else "failed"
        if matched:
            passed += 1
        results.append({
            "id": scenario.get("id"),
            "query": scenario.get("query"),
            "mode": scenario.get("mode", "local"),
            "expect": expect,
            "status": status,
            "matched": matched[:3],
            "top_results": normalized[:5],
        })
    failed = len(results) - passed
    return {
        "schema": "hermes.ard.intent-regression.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": failed == 0,
        "summary": {"total": len(results), "passed": passed, "failed": failed},
        "results": results,
    }


def _load_scenarios(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return default_scenarios()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Scenario file must contain a JSON list")
    return [s for s in data if isinstance(s, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ARD discovery intent regressions")
    parser.add_argument("--scenarios", type=Path, help="JSON list of scenarios. Defaults to built-in ARD smoke pack.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/ard-intent-regression-report.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_scenarios(_load_scenarios(args.scenarios))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        s = report["summary"]
        print(f"ARD intent regression: {s['passed']}/{s['total']} passed, report={args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
