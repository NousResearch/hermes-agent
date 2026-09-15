#!/usr/bin/env python3
"""Block correctness diagnostics on changed Python lines, including new files.

Run with the project's dev environment: python scripts/check_changed_python.py
--base HEAD. CI supplies the merge base; existing advisory reports retain the
full-project backlog. A missing/crashed checker or malformed report fails closed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CORRECTNESS_RULES = "E9,F63,F7,F82,B006"


def changed_lines(diff: str) -> set[int]:
    lines = set()
    for match in re.finditer(
        r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE
    ):
        start = int(match[1])
        count = int(match[2]) if match[2] is not None else 1
        lines.update(range(start, start + count))
    return lines


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def changed_python(root: Path, base: str) -> dict[str, set[int]]:
    names = git_output(
        root, "diff", "--name-only", "-z", "--diff-filter=ACMR", base, "--", "*.py"
    ).split("\0")
    untracked = git_output(
        root, "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py"
    ).split("\0")
    result = {}
    for name in sorted(set(names + untracked) - {""}):
        path = root / name
        if not path.is_file():
            continue
        if name in untracked:
            lines = set(
                range(1, len(path.read_text(encoding="utf-8").splitlines()) + 1)
            )
        else:
            lines = changed_lines(
                git_output(
                    root,
                    "diff",
                    "--no-ext-diff",
                    "--no-color",
                    "--unified=0",
                    base,
                    "--",
                    name,
                )
            )
        if lines:
            result[name] = lines
    return result


def diagnostic_location(entry: dict[str, Any], tool: str) -> tuple[str, int, int, str]:
    if tool == "ruff":
        start = (entry.get("location") or {}).get("row", 0)
        end = (entry.get("end_location") or {}).get("row", start)
        return entry.get("filename") or "", start, end, entry.get("message", "")
    location = entry.get("location") or {}
    positions = location.get("positions") or {}
    lines = location.get("lines") or {}
    start = (positions.get("begin") or {}).get("line", lines.get("begin", 0))
    end = (positions.get("end") or {}).get("line", lines.get("end", start))
    return location.get("path", ""), start, end, entry.get("description", "")


def blocking_diagnostics(
    entries: list[dict[str, Any]], tool: str, changed: dict[str, set[int]], root: Path
) -> list[str]:
    failures = []
    for entry in entries:
        if tool == "ty" and entry.get("severity") in {"info", "minor"}:
            continue
        filename, start, end, message = diagnostic_location(entry, tool)
        path = Path(filename)
        if path.is_absolute():
            try:
                filename = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
        if not filename or not start:
            # Tool-level diagnostics cannot be assigned to an unchanged line.
            failures.append(f"{tool}: {message}")
        elif any(start <= line <= end for line in changed.get(filename, set())):
            rule = entry.get("code") or entry.get("check_name") or "error"
            failures.append(f"{filename}:{start}: {tool} {rule}: {message}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="HEAD", help="Git revision to compare with (CI: merge base)"
    )
    args = parser.parse_args()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    try:
        changed = changed_python(root, args.base)
        if not changed:
            print("No changed Python lines.")
            return 0
        files = sorted(changed)
        commands = {
            "ruff": [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--extend-select",
                CORRECTNESS_RULES,
                "--output-format",
                "json",
                "--exit-zero",
                *files,
            ],
            "ty": [
                sys.executable,
                "-m",
                "ty",
                "check",
                "--python",
                sys.executable,
                "--output-format",
                "gitlab",
                "--exit-zero",
                *files,
            ],
        }
        failures = []
        for tool, command in commands.items():
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, check=True
            )
            entries = json.loads(result.stdout)
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) for entry in entries
            ):
                raise ValueError(f"Invalid {tool} report")
            failures.extend(blocking_diagnostics(entries, tool, changed, root))
        for failure in failures:
            print(failure)
        print(
            f"Changed Python check: {len(files)} files, {len(failures)} blocking diagnostics."
        )
        return 1 if failures else 0
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Correctness check could not complete: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
