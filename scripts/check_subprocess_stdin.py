#!/usr/bin/env python3
"""Check that subprocess calls in TUI-context code specify stdin=.

When Hermes runs in TUI mode, the gateway child process communicates with
the Node.js parent over a JSON-RPC protocol on stdin. Subprocess calls that
inherit this fd can cause the gateway to exit with stdin EOF during tool
execution (issue #14036, PR #39257).

This script checks that all subprocess.run() and subprocess.Popen() calls
in TUI-context files (agent/, tools/, plugins/, tui_gateway/) explicitly
set stdin= to prevent fd inheritance.

Exit codes:
  0 — all calls are safe
  1 — violations found
  2 — script error

Usage:
  python scripts/check_subprocess_stdin.py [--fix]

With --fix, prints the commands to add stdin=subprocess.DEVNULL to each
violation (does not modify files).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Directories that run inside the TUI gateway child process.
TUI_CONTEXT_DIRS = [
    "agent/",
    "tools/",
    "plugins/",
    "tui_gateway/",
]

# Files with intentional stdin= override (e.g. input= creates a pipe).
# Format: "filepath:line" or just "filepath" to skip the whole file.
KNOWN_SAFE = {
    "agent/shell_hooks.py",  # uses input=stdin_json, creates a pipe
    "plugins/security-guidance/patterns.py",  # subprocess mentions are in reminder strings, not calls
}

# Directories to skip entirely.
SKIP_DIRS = {
    "tests/",
    "scripts/",
    "skills/",
    "optional-skills/",
    "hermes_cli/",
    "gateway/",
    "cron/",
}


def find_subprocess_calls(content: str, filepath: str) -> list[dict]:
    """Find all subprocess.run/Popen calls missing stdin= in content."""
    violations = []
    lines = content.split("\n")

    # Match only actual function calls — not comments, docstrings, or prose.
    # The pattern requires an opening paren followed by an arg character
    # (quote, bracket, letter, or closing paren for empty calls).
    # This excludes ``subprocess.Popen(...)`` in docstrings and
    # subprocess.run(...) in comments.
    pattern = re.compile(r'subprocess\.(run|Popen)\s*\(["\'a-zA-Z_\[\(]')

    for i, line in enumerate(lines):
        # Skip comments.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        # Skip lines where the match is inside backticks (docstring references).
        if "``subprocess" in line:
            continue

        if not pattern.search(line):
            continue

        # Collect the full call (may span multiple lines).
        call_start = i
        paren_depth = 0
        found_open = False
        call_lines = []
        for j in range(i, min(i + 30, len(lines))):
            call_lines.append(lines[j])
            for ch in lines[j]:
                if ch == "(":
                    paren_depth += 1
                    found_open = True
                elif ch == ")":
                    paren_depth -= 1
                    if found_open and paren_depth == 0:
                        call_text = "\n".join(call_lines)

                        # Already has stdin= → safe.
                        if "stdin=" in call_text:
                            break

                        # Has input= → creates a pipe, safe.
                        if "input=" in call_text:
                            break

                        violations.append({
                            "file": filepath,
                            "line": i + 1,
                            "snippet": line.strip()[:120],
                        })
                        break
            else:
                continue
            break

    return violations


def main() -> int:
    fix_mode = "--fix" in sys.argv
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    all_violations = []

    for tui_dir in TUI_CONTEXT_DIRS:
        dirpath = repo_root / tui_dir
        if not dirpath.exists():
            continue

        for py_file in dirpath.rglob("*.py"):
            rel = str(py_file.relative_to(repo_root))

            # Skip known-safe files.
            if rel in KNOWN_SAFE:
                continue

            # Skip test files inside tools/ etc.
            parts = py_file.parts
            if any(skip.rstrip("/") in parts for skip in SKIP_DIRS):
                continue

            content = py_file.read_text()
            violations = find_subprocess_calls(content, rel)
            all_violations.extend(violations)

    if all_violations:
        print(f"❌ {len(all_violations)} subprocess calls missing stdin=:")
        for v in all_violations:
            print(f"  {v['file']}:{v['line']}: {v['snippet']}")
        if fix_mode:
            print("\nAdd stdin=subprocess.DEVNULL to each call above.")
        return 1
    else:
        print("✅ All TUI-context subprocess calls have explicit stdin=")
        return 0


if __name__ == "__main__":
    sys.exit(main())
