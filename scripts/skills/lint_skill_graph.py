#!/usr/bin/env python3
"""Lint the boring Hermes Skill Graph v0 manifest.

Default mode is warning-first and exits 0 even with warnings. Use --strict to
promote warnings to errors for CI or canary gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HERMES_HOME", str(Path.home() / ".hermes"))

from agent.skill_graph import build_skill_graph, lint_skill_graph, load_skill_graph
from hermes_constants import get_hermes_home


def _default_skills_root() -> Path:
    return get_hermes_home() / "skills"


def _default_graph_path(skills_root: Path) -> Path:
    return skills_root / ".graph" / "skill_graph.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint Hermes Skill Graph v0 manifest.")
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=_default_skills_root(),
        help="Directory containing skill folders. Defaults to $HERMES_HOME/skills.",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Graph JSON path. Defaults to <skills-root>/.graph/skill_graph.json.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build an in-memory graph from --skills-root instead of reading --graph-path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings to errors and exit non-zero on any issue.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full lint payload as JSON.",
    )

    args = parser.parse_args()
    skills_root = args.skills_root.expanduser().resolve()
    graph_path = args.graph_path or _default_graph_path(skills_root)

    if args.build:
        if not skills_root.exists():
            print(f"ERROR: skills root does not exist: {skills_root}", file=sys.stderr)
            return 2
        graph = build_skill_graph(skills_root)
        source = str(skills_root)
    else:
        graph_path = graph_path.expanduser().resolve()
        if not graph_path.exists():
            print(f"ERROR: graph path does not exist: {graph_path}", file=sys.stderr)
            return 2
        graph = load_skill_graph(graph_path)
        source = str(graph_path)

    result = lint_skill_graph(graph, strict=args.strict)
    payload = {
        "ok": result.get("ok", False),
        "strict": bool(args.strict),
        "source": source,
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
        "warning_count": len(result.get("warnings", [])),
        "error_count": len(result.get("errors", [])),
    }

    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        print(
            f"Skill graph lint: ok={payload['ok']} strict={payload['strict']} "
            f"warnings={payload['warning_count']} errors={payload['error_count']} source={source}"
        )
        for item in payload["errors"][:20]:
            print(f"ERROR {item.get('code')}: {item.get('message')}")
        for item in payload["warnings"][:20]:
            print(f"WARN {item.get('code')}: {item.get('message')}")
        remaining = payload["warning_count"] + payload["error_count"] - 20
        if remaining > 0:
            print(f"... {remaining} more issue(s); rerun with --json for full output")

    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
