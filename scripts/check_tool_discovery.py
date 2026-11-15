#!/usr/bin/env python3
"""Fail when a module under ``tools/`` registers a tool that discovery never imports.

``tools.registry.discover_builtin_tools`` imports only the modules ``_tool_module_candidates``
returns (``tools/*.py`` and ``tools/<pkg>/tool.py``). A module-level ``registry.register(...)``
anywhere else never runs, so its tool disappears from every session with no error. Both
predicates are the registry's own, so this check and discovery cannot disagree.

Usage: python3 scripts/check_tool_discovery.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# First on the path: an editable install of another checkout must not answer for this tree.
sys.path.insert(0, str(ROOT))
from hermes_agent.compat.paths import current_path
from tools.registry import _module_registers_tools, _tool_module_candidates


def unreached_registrations(repo: Path) -> list[str]:
    """Tracked modules under ``tools/`` that register a tool at import but are never imported."""
    tools_rel = current_path("tools")
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", tools_rel], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL, check=True,
    ).stdout
    reached = {path.resolve() for path in _tool_module_candidates(repo / tools_rel)}
    return [
        rel for rel in sorted(filter(None, listing.split("\0")))
        if rel.endswith(".py") and (repo / rel).resolve() not in reached
        and _module_registers_tools(repo / rel)
    ]


def main() -> int:
    unreached = unreached_registrations(ROOT)
    if not unreached:
        print("check_tool_discovery: OK")
        return 0
    print("These modules call registry.register() at import, but tool discovery never imports"
          " them, so their tools never load:", file=sys.stderr)
    for rel in unreached:
        print(f"  {rel}", file=sys.stderr)
    print("Register from a flat tools/<name>.py or from tools/<package>/tool.py (the package needs"
          " an __init__.py); every other module in a package is a library.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
