#!/usr/bin/env python3
"""Compare Hermes ARD skill search against an optional external skill-search CLI.

This is deliberately a spike/comparator, not a runtime dependency. If a
`skill-search` executable is present, the script records external results. If
not, it still writes a baseline ARD report so quality drift can be tracked in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERIES = [
    "summarize youtube transcript",
    "browser qa exploratory web app testing",
    "security scanner bug bounty",
    "mcp registry image generation",
]


def _normalize(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        normalized = dict(item)
        name = normalized.get("displayName") or normalized.get("name") or normalized.get("path") or ""
        normalized.setdefault("displayName", normalized.get("name", ""))
        if name and not normalized.get("identifier"):
            safe_name = str(name).replace("/", ":")
            agent = normalized.get("agent")
            prefix = f"external:skill-search:{agent}:" if agent else "external:skill-search:"
            normalized["identifier"] = f"{prefix}{safe_name}"
        return normalized
    return {
        "identifier": getattr(item, "identifier", ""),
        "displayName": getattr(item, "name", ""),
        "description": getattr(item, "description", ""),
        "score": getattr(item, "score", None),
        "extra": getattr(item, "extra", {}) if hasattr(item, "extra") else {},
    }


def baseline_search(query: str, limit: int) -> list[dict[str, Any]]:
    from tools.skills_hub import ard_local_search
    return [_normalize(r) for r in ard_local_search(query, limit=limit)]


def find_skill_search_command() -> list[str] | None:
    configured = os.environ.get("ARD_SKILL_SEARCH_COMMAND")
    if configured:
        parsed = shlex.split(configured)
        return parsed or None
    found = shutil.which("skill-search")
    return [found] if found else None


def external_skill_search(query: str, limit: int, command: str | list[str] | None = None) -> list[dict[str, Any]]:
    cmd = command or find_skill_search_command()
    if isinstance(cmd, str):
        base_cmd = [cmd]
    else:
        base_cmd = cmd
    if not base_cmd:
        return []
    # Support a conservative JSON interface. `skill-search-cli` (npm) accepts
    # the query as positional args and returns `{local: [...], remote: [...]}`;
    # older/prototype tools may use a `search` subcommand with `{results: [...]}`.
    attempts = [
        [*base_cmd, query, "--limit", str(limit), "--json"],
        [*base_cmd, "search", query, "--limit", str(limit), "--json"],
    ]
    data: Any = None
    for argv in attempts:
        proc = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if proc.returncode != 0:
            continue
        try:
            data = json.loads(proc.stdout)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return []
    if isinstance(data, list):
        return [_normalize(x) for x in data]
    if isinstance(data, dict):
        values = data.get("results") or data.get("items")
        if values is None and ("local" in data or "remote" in data):
            values = []
            for bucket in ("local", "remote"):
                bucket_values = data.get(bucket)
                if isinstance(bucket_values, list):
                    values.extend(bucket_values)
        if isinstance(values, list):
            return [_normalize(x) for x in values]
    return []


def compare_queries(
    queries: list[str],
    *,
    baseline_runner: Callable[[str, int], list[Any]] = baseline_search,
    external_runner: Callable[[str, int], list[Any]] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    external_available = external_runner is not None
    rows: list[dict[str, Any]] = []
    for query in queries:
        baseline = [_normalize(r) for r in baseline_runner(query, limit)]
        external = [_normalize(r) for r in external_runner(query, limit)] if external_runner else []
        rows.append({
            "query": query,
            "baseline": baseline[:limit],
            "external": external[:limit],
            "baseline_top": baseline[0].get("identifier") if baseline else None,
            "external_top": external[0].get("identifier") if external else None,
        })
    return {
        "schema": "hermes.ard.skill-search-spike.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "external_available": external_available,
        "summary": {"queries": len(rows), "limit": limit},
        "queries": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ARD local search with optional skill-search")
    parser.add_argument("--query", action="append", dest="queries", help="Query to compare. Repeatable.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ard-skill-search-spike.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    command = find_skill_search_command()
    external_runner = (lambda q, l: external_skill_search(q, l, command=command)) if command else None
    report = compare_queries(args.queries or DEFAULT_QUERIES, external_runner=external_runner, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ARD skill-search spike: queries={len(report['queries'])} external_available={report['external_available']} report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
