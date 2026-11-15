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


def main() -> int:
    tools_dir = ROOT / current_path("tools")
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=tools_dir, capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    registering = {
        tools_dir / rel for rel in listed.split("\0")
        if rel and _module_registers_tools(tools_dir / rel)
    }
    if not registering:
        print("check_tool_discovery: found no module that registers a tool; the scan is broken",
              file=sys.stderr)
        return 1
    missed = sorted(registering - set(_tool_module_candidates(tools_dir)))
    if missed:
        print("These modules call registry.register() at import, but tool discovery never imports"
              " them. Inside a package, register from <pkg>/tool.py:", file=sys.stderr)
        for path in missed:
            print(f"  {path.relative_to(ROOT).as_posix()}", file=sys.stderr)
        return 1
    print(f"check_tool_discovery: OK ({len(registering)} registering modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
