#!/usr/bin/env python3
"""Fail when anything but ``hermes_cli/profile_provision.py`` uses ``create_profile``.

Flags any reference to the name (call, import, alias, attribute, string constant) in non-test Python;
``hermes_cli/profiles.py`` may only define it. An unparseable file fails. Suppress with
``# profile-create: ok — <why>`` on the flagged line.

Usage: python3 scripts/check_profile_create_entry.py [paths...]
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_TOP = {"tests", "tests-js", "website", "apps", "ui-tui", "web", "native", "nix"}
SKIP_ANY = {"tests", "node_modules", "__pycache__", "venv"}
ALLOWED_FILES = {ROOT / "hermes_cli" / "profile_provision.py", Path(__file__).resolve()}
SUPPRESS = "# profile-create: ok"
NAME = "create_profile"


def scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8-sig")
    if NAME not in src:
        return []
    rel = path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return [f"{rel}:{exc.lineno}: does not parse, so it cannot be checked"]
    lines = src.splitlines()
    hits: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.name.rsplit(".", 1)[-1] == NAME for a in node.names):
                hits.add(node.lineno)
        elif (isinstance(node, ast.Name) and node.id == NAME and isinstance(node.ctx, ast.Load)
              or isinstance(node, ast.Attribute) and node.attr == NAME
              or isinstance(node, ast.Constant) and node.value == NAME):
            hits.add(node.lineno)
    return [f"{rel}:{n}: uses create_profile directly" for n in sorted(hits) if SUPPRESS not in lines[n - 1]]


def main(argv: list[str]) -> int:
    targets = [(ROOT / a).resolve() for a in argv] or [ROOT]
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            for dirpath, dirnames, filenames in os.walk(t):
                top = Path(dirpath) == ROOT
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_ANY
                               and not (top and d in SKIP_TOP)]
                files.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
        elif t.suffix == ".py" and t.exists():
            files.append(t)
    problems: list[str] = []
    for f in sorted(set(files)):
        if f.resolve() not in ALLOWED_FILES:
            problems.extend(scan_file(f))
    if problems:
        print("Create profiles through hermes_cli.profile_provision.provision_profile:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"check_profile_create_entry: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
