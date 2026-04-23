#!/usr/bin/env python3
"""Build the boring Hermes Skill Graph v0 manifest.

Defaults to the live Hermes skills directory and writes:
  ~/.hermes/skills/.graph/skill_graph.json
  ~/.hermes/skills/.graph/skill_graph.md
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

from agent.skill_graph import build_skill_graph, run_skill_graph_canaries, write_skill_graph
from hermes_constants import get_hermes_home


def _default_skills_root() -> Path:
    return get_hermes_home() / "skills"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Hermes Skill Graph v0 manifests.")
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=_default_skills_root(),
        help="Directory containing skill folders. Defaults to $HERMES_HOME/skills.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <skills-root>/.graph/skill_graph.json.",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to <skills-root>/.graph/skill_graph.md.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Write JSON only.",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional fixed generated_at timestamp for deterministic tests.",
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run built-in internal-only canaries after building.",
    )

    args = parser.parse_args()
    skills_root = args.skills_root.expanduser().resolve()
    if not skills_root.exists():
        print(f"ERROR: skills root does not exist: {skills_root}", file=sys.stderr)
        return 2

    json_path = args.json_path or skills_root / ".graph" / "skill_graph.json"
    markdown_path = None if args.no_markdown else (args.markdown_path or skills_root / ".graph" / "skill_graph.md")

    graph = build_skill_graph(skills_root, generated_at=args.generated_at)
    write_skill_graph(graph, json_path, markdown_path)

    counts = graph.get("counts", {})
    print(
        json.dumps(
            {
                "ok": True,
                "skills_root": str(skills_root),
                "json_path": str(json_path),
                "markdown_path": str(markdown_path) if markdown_path else None,
                "nodes": counts.get("nodes", 0),
                "edges": counts.get("edges", 0),
            },
            sort_keys=True,
        )
    )

    if args.canary:
        canaries = run_skill_graph_canaries(graph)
        print(json.dumps({"canary": canaries}, sort_keys=True))
        return 0 if canaries.get("ok") else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
