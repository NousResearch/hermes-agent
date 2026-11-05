#!/usr/bin/env python3
"""Fail when anything but ``hermes_cli/profile_provision.py`` calls ``create_profile``.

``provision_profile(ProfileCreateRequest)`` is the one way to create a profile: the CLI, the dashboard
REST route and the Desktop/TUI RPC all build a request and render the result. Before it existed each
surface called ``hermes_cli.profiles.create_profile`` directly and added its own post-create steps, so the
same "new profile" action produced different profiles (credentials, skills, model) depending on where it
started. A new direct caller would reopen that split.

Flags any use of the name ``create_profile`` (a call, an import, an alias, or the function passed as a
value to ``run_in_threadpool`` / ``partial`` / ``to_thread``) in every Python file of the repo except
tests and the frontend trees. ``hermes_cli/profile_provision.py`` (the entry point) is exempt; in
``hermes_cli/profiles.py`` only the definition itself is allowed, so a wrapper there is flagged too.
A file that does not parse is a failure, not a pass. Suppress a true exception with
``# profile-create: ok — <why>`` on the flagged line.

Usage: python3 scripts/check_profile_create_entry.py [paths...]
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Not production Python. Top-level trees only (``plugins/web`` is production code): frontends, docs, packaging.
SKIP_TOP = {"tests", "tests-js", "website", "apps", "ui-tui", "web", "native", "nix"}
# At any depth.
SKIP_ANY = {"tests", "node_modules", "__pycache__", "venv"}
ALLOWED_FILES = {ROOT / "hermes_cli" / "profile_provision.py", Path(__file__).resolve()}
SUPPRESS = "# profile-create: ok"
NAME = "create_profile"


def scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8-sig")
    if NAME not in src:  # every flagged form spells the name; skip the parse
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
        elif isinstance(node, ast.Name) and node.id == NAME and isinstance(node.ctx, ast.Load) or isinstance(node, ast.Attribute) and node.attr == NAME:
            hits.add(node.lineno)
        elif isinstance(node, ast.Constant) and node.value == NAME:
            hits.add(node.lineno)  # getattr(profiles, "create_profile")
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
