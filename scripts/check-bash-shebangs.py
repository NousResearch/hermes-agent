#!/usr/bin/env python3
"""Find hardcoded ``#!/bin/bash`` shebangs in repository text files.

Hardcoded interpreter paths break on systems where bash is installed outside
``/bin``. The portable form is ``#!/usr/bin/env bash``.

Usage:
    python scripts/check-bash-shebangs.py
    python scripts/check-bash-shebangs.py --all
    python scripts/check-bash-shebangs.py --diff upstream/main
    python scripts/check-bash-shebangs.py path/to/file.md

The default scans staged files. ``--all`` scans the repository. ``--diff``
scans files changed from the given ref, including unstaged changes.

An intentional exception must carry ``# shebang: ok`` and a reason on the
same line. The checker is deliberately line-based so markdown code blocks,
generated-script strings, and test fixtures get the same rule.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPRESS_MARKER = re.compile(r"#\s*shebang\s*:\s*ok\b", re.IGNORECASE)
SHEBANG_RE = re.compile(r"^#![ \t]*/bin/bash(?:[ \t\n]|\Z)")

# Keep generated dependencies and build products out of the scan. Tracked
# source, docs, tests, and generated-script strings remain in scope.
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
}

EXCLUDED_FILES = {
    # This file documents the pattern it detects.
    "scripts/check-bash-shebangs.py",
}

TEXT_SUFFIXES = {
    ".bash",
    ".bat",
    ".cmd",
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".nix",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


def should_scan_file(path: Path) -> bool:
    """Return True when path is a supported source or documentation file."""
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = ""
    if rel in EXCLUDED_FILES:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if should_scan_file(path):
                yield path
            continue
        if not path.is_dir():
            continue
        for root, dirs, files in os.walk(path):
            dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
            for name in files:
                candidate = Path(root) / name
                if should_scan_file(candidate):
                    yield candidate


def git_paths(command: list[str]) -> list[Path]:
    """Return repository paths from a git command, or an empty list on error."""
    try:
        output = subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [REPO_ROOT / name for name in output.splitlines() if name.strip()]


def get_staged_files() -> list[Path]:
    return git_paths(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def get_diff_files(ref: str) -> list[Path]:
    return git_paths(["git", "diff", ref, "--name-only", "--diff-filter=ACMR"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag hardcoded /bin/bash shebangs.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific files or directories. Default: staged changes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan the full repository.",
    )
    parser.add_argument(
        "--diff",
        metavar="REF",
        help="Scan files changed from REF, including unstaged changes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.all:
        roots = [REPO_ROOT]
    elif args.diff:
        roots = get_diff_files(args.diff)
    elif args.paths:
        roots = [path.resolve() for path in args.paths]
    else:
        roots = get_staged_files()
        if not roots:
            print(
                "No staged files to scan. Pass --all, --diff REF, or paths.",
                file=sys.stderr,
            )
            return 0

    total_matches = 0
    files_scanned = 0
    for path in iter_files(roots):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
        except OSError:
            continue
        files_scanned += 1
        for lineno, line in enumerate(lines, start=1):
            if not SHEBANG_RE.match(line) or SUPPRESS_MARKER.search(line):
                continue
            try:
                display_path = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                display_path = str(path)
            print(f"{display_path}:{lineno}: [hardcoded /bin/bash shebang]")
            print(f"    {line.strip()}")
            print(
                "    - /bin/bash does not exist on every system. "
                "Use #!/usr/bin/env bash."
            )
            print("    Fix: #!/usr/bin/env bash")
            print()
            total_matches += 1

    if total_matches:
        print(
            f"\nX {total_matches} hardcoded /bin/bash shebang(s) found across "
            f"{files_scanned} file(s) scanned.",
            file=sys.stderr,
        )
        print(
            "  If intentional, add `# shebang: ok <reason>` to the same line.",
            file=sys.stderr,
        )
        return 1

    print(f"OK no hardcoded /bin/bash shebangs ({files_scanned} file(s) scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
