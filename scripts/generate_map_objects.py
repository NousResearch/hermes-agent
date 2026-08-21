"""Deterministic generator for map/objects/ module cards.

Each card is an ICM object card with:
  - YAML frontmatter: id, kind, universe, name, summary, shape, path, interface, depends_on
  - Markdown body with Purpose, Inputs, Outputs, Dependencies
  - Live Graphify stats for the module

Generation is deterministic:
  * module order is fixed
  * Graphify is queried in a sorted, stable way
  * JSON output is sorted before serialization

Usage:
    python scripts/generate_map_objects.py
    python scripts/generate_map_objects.py --repo /path/to/hermes-agent
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_imports() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


_ensure_imports()

from agent.graphify import CodeGraphStore  # noqa: E402


MODULES = [
    {
        "name": "agent",
        "id": "agent",
        "universe": "repo",
        "path": "agent",
        "summary": "Core agent runtime with conversation orchestration, memory, caching, and provider adapters.",
        "interface": [
            "AIAgent",
            "run_conversation",
            "chat",
            "AgentLoop",
        ],
        "depends_on": [
            "repo:run_agent.py",
            "repo:model_tools.py",
            "repo:agent/graphify.py",
        ],
    },
    {
        "name": "gateway",
        "id": "gateway",
        "universe": "runtime",
        "path": "gateway",
        "summary": "Messaging gateway runner with platform adapters and async turn handling.",
        "interface": [
            "GatewayRunner",
            "start_gateway",
            "GatewaySession",
        ],
        "depends_on": [
            "repo:gateway/run.py",
            "repo:gateway/session.py",
            "repo:agent/async_utils.py",
        ],
    },
    {
        "name": "plugins",
        "id": "plugins",
        "universe": "repo",
        "path": "plugins",
        "summary": "Plugin system for memory providers, web search backends, observability, and extended capabilities.",
        "interface": [
            "PluginRegistry",
            "MemoryProvider",
            "WebSearchProvider",
            "ContextEngine",
        ],
        "depends_on": [
            "repo:plugins/__init__.py",
            "repo:tools/registry.py",
            "repo:hermes_cli/commands.py",
        ],
    },
    {
        "name": "tools",
        "id": "tools",
        "universe": "repo",
        "path": "tools",
        "summary": "Built-in tool implementations and central discovery registry for model-callable tools.",
        "interface": [
            "ToolRegistry",
            "discover_builtin_tools",
            "get_definitions",
            "dispatch",
        ],
        "depends_on": [
            "repo:tools/registry.py",
            "repo:toolsets.py",
            "repo:model_tools.py",
        ],
    },
    {
        "name": "hermes_cli",
        "id": "hermes-cli",
        "universe": "repo",
        "path": "hermes_cli",
        "summary": "CLI subcommands, setup wizard, skin engine, and slash-command registry.",
        "interface": [
            "HermesCLI",
            "COMMAND_REGISTRY",
            "setup_wizard",
            "skin_engine",
        ],
        "depends_on": [
            "repo:cli.py",
            "repo:hermes_cli/commands.py",
            "repo:hermes_cli/skin_engine.py",
        ],
    },
    {
        "name": "optional-mcps",
        "id": "optional-mcps",
        "universe": "repo",
        "path": "optional-mcps",
        "summary": "Optional MCP catalog integrations, including Graphify and third-party service adapters.",
        "interface": [
            "MCPCatalog",
            "MCPServer",
            "mcp_serve",
        ],
        "depends_on": [
            "repo:mcp_serve.py",
            "repo:optional-mcps/graphify/manifest.yaml",
            "repo:hermes_cli/subcommands/mcp.py",
        ],
    },
    {
        "name": "src-graphify",
        "id": "src-graphify",
        "universe": "repo",
        "path": "src/graphify",
        "summary": "Live Graphify index source: graph model, query engine, and JSON persistence.",
        "interface": [
            "GraphModel",
            "GraphJsonRepository",
            "GraphifyQueryEngine",
            "build_graph",
        ],
        "depends_on": [
            "repo:src/graphify/model.py",
            "repo:src/graphify/query.py",
            "repo:agent/graphify.py",
        ],
    },
]


def _graphify_stats(repo_root: Path, rel_path: str) -> dict[str, Any]:
    store = CodeGraphStore(root=repo_root)
    store.index_walk(repo_root / rel_path)
    stats = store.stats()
    return {
        "indexed_files": int(stats.get("files", 0)),
        "symbols": int(stats.get("symbols", 0)),
        "edges": int(stats.get("edges", 0)),
        "by_kind": dict(sorted(stats.get("by_kind", {}).items())),
    }


def _yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    lines = ["["]
    for item in items:
        lines.append(f"  - {item}")
    lines.append("]")
    return "\n".join(lines)


def render_frontmatter(module: dict[str, Any]) -> str:
    lines = [
        "---",
        f"id: {module['id']}",
        "kind: object",
        f"universe: {module['universe']}",
        f"name: {module['name']}",
        f"summary: >-\n  {module['summary']}",
        "aliases: []",
        "tags: []",
        "shape: object",
        f"path: {module['path']}",
        "interface:",
    ]
    for item in module.get("interface", []):
        lines.append(f"  - {item}")
    lines.append("depends_on:")
    for dep in module.get("depends_on", []):
        lines.append(f"  - {dep}")
    lines.append("---")
    return "\n".join(lines)


def render_body(module: dict[str, Any], stats: dict[str, Any]) -> str:
    lines = [
        f"# {module['name']}",
        "",
        module["summary"],
        "",
        "## Purpose",
        "",
        module["summary"],
        "",
        "## Inputs",
        "",
        f"- Repository file tree under `{module['path']}`",
        "- Python source files for Graphify symbol and edge extraction",
        "- Module docstrings and AST definitions for live indexing",
        "",
        "## Outputs",
        "",
        "- Structured symbol table: classes, functions, methods, modules",
        "- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES",
        "- JSON graph payload written to `graphify-out/graph.json`",
        "",
        "## Dependencies",
        "",
    ]
    for dep in module.get("depends_on", []):
        lines.append(f"- `{dep}`")
    lines += [
        "",
        "## Live Graphify Stats",
        "",
        "```json",
        json.dumps(stats, sort_keys=True, indent=2, ensure_ascii=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def render_card(module: dict[str, Any], stats: dict[str, Any], generated_at: str) -> str:
    return render_frontmatter(module) + "\n\n" + render_body(module, stats) + "\n"


def generate(repo_root: Path, objects_dir: Path) -> dict[str, Any]:
    objects_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "generator": "scripts/generate_map_objects.py",
        "modules": [],
    }
    for module in MODULES:
        stats = _graphify_stats(repo_root, module["path"])
        card = render_card(module, stats, generated_at)
        filename = f"{module['name']}.md"
        path = objects_dir / filename
        path.write_text(card, encoding="utf-8")
        manifest["modules"].append(
            {
                "name": module["name"],
                "id": module["id"],
                "universe": module["universe"],
                "path": module["path"],
                "file": str(path.relative_to(repo_root)),
                "stats": stats,
            }
        )
    manifest_path = objects_dir / "INDEX.md"
    manifest_path.write_text(
        _render_index(manifest, objects_dir, repo_root), encoding="utf-8"
    )
    return manifest


def _render_index(manifest: dict[str, Any], objects_dir: Path, repo_root: Path) -> str:
    lines = [
        "# Map Objects Index",
        "",
        "Deterministic manifest of generated module object cards.",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Generator: `{manifest['generator']}`",
        "",
        "## Modules",
        "",
    ]
    for entry in manifest["modules"]:
        rel = Path(entry["file"]).name
        lines.append(f"- [{entry['name']}](./{rel}) — `{entry['path']}`")
    lines += [
        "",
        "## Integrity",
        "",
        "- Manifest contents are generated deterministically from live Graphify stats.",
        "- Card order is fixed by `MODULES` sequence.",
        "- JSON payloads are sorted before serialization.",
        "- Validation tests: `tests/test_map_objects.py`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate map/objects module cards")
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="Path to hermes-agent repo root",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for cards, default <repo>/map/objects",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo).resolve()
    objects_dir = Path(args.out).resolve() if args.out else repo_root / "map" / "objects"
    manifest = generate(repo_root, objects_dir)
    print(
        json.dumps(
            {"generated": len(manifest["modules"]), "out": str(objects_dir)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
