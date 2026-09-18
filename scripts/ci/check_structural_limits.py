#!/usr/bin/env python3
"""Structural check: physical text lines strictly < 2000 for every PR-base modified Python file.

Metric: UTF-8 physical text lines — every line in the file counts, including
blank-only, comment-only, decorator, continuation, and semicolon-containing lines.
A final unterminated line counts as one physical line via normal text iteration.
Required A2A tracked files are always checked even when not in the diff.
Modified-file discovery uses an explicit --base and HEAD with deduplication;
missing/unreadable/invalid-UTF-8 required files, unresolved base, or git failures fail closed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_LINES = 2000  # strictly < 2000; 2000 is FAIL

REQUIRED_A2A_FILES = [
    "plugins/platforms/a2a/adapter.py",
    "plugins/platforms/a2a/task_routing.py",
    "plugins/platforms/a2a/a2a_persistence.py",
    "plugins/platforms/a2a/protocol.py",
    "plugins/platforms/a2a/http_transport.py",
]

# Statuses whose paths count as modified (A/C/M/R/T/U/X/B); D is deleted.
INCLUDED_STATUSES = {"A", "C", "M", "R", "T", "U", "X", "B"}


def count_physical_lines(path: Path) -> int:
    """Count every physical UTF-8 text line in *path*.

    Raises FileNotFoundError / OSError / UnicodeDecodeError on failure;
    caller decides how to report.
    """
    count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _resolve_base(repo_root: Path, base_ref: str) -> str:
    """Resolve *base_ref* to a full SHA; fail closed on error."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", base_ref],
        capture_output=True, text=True, cwd=str(repo_root), timeout=10,
    )
    if result.returncode != 0:
        print(f"ERROR: cannot resolve base ref '{base_ref}': {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    sha = result.stdout.strip()
    # Also verify it resolves via rev-parse for display
    return sha


def _resolve_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root), timeout=10,
    )
    if result.returncode != 0:
        print(f"ERROR: cannot resolve HEAD: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def _discover_modified_python_files(repo_root: Path, base_ref: str) -> tuple[list[str], str, str]:
    """Return (sorted deduped Python paths, resolved_base_sha, head_sha).

    Uses ``git diff --name-status --diff-filter=ACMRTUXB <base>...HEAD``.
    Fails closed (exit 1) on git errors or unresolved base.
    """
    resolved_base = _resolve_base(repo_root, base_ref)
    head_sha = _resolve_head(repo_root)

    # Use three-dot diff against HEAD
    result = subprocess.run(
        ["git", "diff", "--name-status", "--diff-filter=ACMRTUXB", f"{resolved_base}...HEAD"],
        capture_output=True, text=True, cwd=str(repo_root), timeout=15,
    )
    if result.returncode != 0:
        print(f"ERROR: git diff failed for base '{base_ref}' (resolved {resolved_base}): {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # name-status lines: "<status>\\t<path>" or for renames "R...\\t<old>\\t<new>"
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0].strip()
        # status may be like "R100"; first char determines inclusion
        status_char = status[0] if status else ""
        if status_char not in INCLUDED_STATUSES:
            continue
        # For renames, the new path is the last column
        path = parts[-1].strip()
        if path.endswith(".py"):
            seen.add(path)

    return sorted(seen), resolved_base, head_sha


def main() -> int:
    parser = argparse.ArgumentParser(description="Check physical line limits (< 2000)")
    parser.add_argument("--base", default="origin/main", help="base ref or SHA for diff discovery (default: origin/main)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_arg = args.base

    # Discover modified files and resolve refs
    modified_py, resolved_base, head_sha = _discover_modified_python_files(repo_root, base_arg)

    print(f"Structural check: physical lines < {MAX_LINES} (max 1999)")
    print(f"  base: {base_arg} -> {resolved_base}")
    print(f"  HEAD: {head_sha}")
    print()

    # Build the full check set: required A2A files ∪ modified Python files
    check_set: set[str] = set(REQUIRED_A2A_FILES)
    for p in modified_py:
        check_set.add(p)

    all_ok = True

    # Report required files first, then remaining modified files
    ordered = sorted(check_set)

    print(f"── Checking {len(ordered)} file(s) (required A2A + modified Python) ──")
    for rel in ordered:
        path = repo_root / rel
        tag = "required" if rel in REQUIRED_A2A_FILES else "modified"
        # Deleted non-required modified file: report DELETED/NA
        if rel not in REQUIRED_A2A_FILES and not path.exists():
            print(f"  DELETED/NA  {rel} ({tag}, deleted)")
            continue
        try:
            count = count_physical_lines(path)
        except FileNotFoundError:
            print(f"  FAIL  {rel}: missing required file ({tag})")
            all_ok = False
            continue
        except UnicodeDecodeError as e:
            print(f"  FAIL  {rel}: invalid UTF-8 ({tag}): {e}")
            all_ok = False
            continue
        except OSError as e:
            print(f"  FAIL  {rel}: unreadable ({tag}): {e}")
            all_ok = False
            continue

        ok = 0 <= count < MAX_LINES
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {rel}: {count} physical lines (limit <{MAX_LINES}) [{tag}]")
        if not ok:
            all_ok = False

    # Also report modified files that were DELETED (non-required) for completeness
    # (already handled above as DELETED/NA, not a failure)

    print()
    if all_ok:
        print("Result: ALL OK")
        return 0
    else:
        print("Result: FAILURES DETECTED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
