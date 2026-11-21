#!/usr/bin/env python3
"""Fail when a tracked module under ``tools/`` registers a tool that discovery never imports.

Inside a ``tools/`` package, discovery imports only ``<pkg>/tool.py``, so a module named anything
else that calls ``registry.register()`` at import is skipped and its tools are missing with no
error. Both questions are answered by discovery's own functions in ``tools/registry.py``.

Usage: python3 scripts/check_tool_discovery.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from hermes_agent.compat.paths import current_path
from tools.registry import _module_registers_tools, _tool_module_candidates


def unreached_registrations(tools_dir: Path) -> list[Path]:
    """Tracked modules under ``tools_dir`` that register a tool at import but discovery skips."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=tools_dir, capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    registering = {
        tools_dir / rel for rel in listed.split("\0")
        if rel and _module_registers_tools(tools_dir / rel)
    }
    # Discovery also skips a package without __init__.py (setuptools would drop it from the wheel).
    discovered = {p for p in _tool_module_candidates(tools_dir)
                  if p.parent == tools_dir or (p.parent / "__init__.py").is_file()}
    return sorted(registering - discovered)


def main() -> int:
    missed = unreached_registrations(ROOT / current_path("tools"))
    if missed:
        print("These modules call registry.register() at import, but tool discovery never imports"
              " them. Inside a package, register from <pkg>/tool.py and give the package an"
              " __init__.py:", file=sys.stderr)
        for path in missed:
            print(f"  {path.relative_to(ROOT).as_posix()}", file=sys.stderr)
        return 1
    print("check_tool_discovery: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
