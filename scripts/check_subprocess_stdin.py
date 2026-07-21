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

# User plugin roots — scanned at runtime if they exist.  Plugins load from
# ``get_hermes_home() / "plugins"`` (user) and ``./.hermes/plugins/`` (project,
# gated behind ``HERMES_ENABLE_PROJECT_PLUGINS``) — see
# ``hermes_cli/plugins.py:10-12``.  The guard only checked the bundled
# ``plugins/`` dir, missing user-installed code that spawns subprocesses
# (gap reported in #67639).
#
# Import is deferred to ``main()`` (after ``os.chdir(repo_root)``) because
# this script runs as a standalone subprocess — ``hermes_constants`` isn't
# on ``sys.path`` until the repo root is added.

# subprocess and os APIs that inherit stdin by default when called without
# an explicit stdin= argument.  The original regex only covered run/Popen
# (gap #1 in #67639); call, check_output, check_call, os.system, and
# asyncio.create_subprocess_* all inherit fd 0 equally.
_SUBPROCESS_PATTERNS = [
    r"subprocess\.(run|Popen|call|check_output|check_call)\s*\([\"'a-zA-Z_\[\(]",
    r"os\.system\s*\([\"'a-zA-Z_\[\(]",
    r"asyncio\.create_subprocess_(exec|shell)\s*\([\"'a-zA-Z_\[\(]",
]

# Files with intentional stdin= override (e.g. input= creates a pipe).
# Format: "filepath:line" or just "filepath" to skip the whole file.
KNOWN_SAFE = {
    "agent/shell_hooks.py",  # uses input=stdin_json, creates a pipe
    "plugins/security-guidance/patterns.py",  # subprocess mentions are in reminder strings, not calls
}

# Inline marker that exempts a single subprocess call from this check.
# Put it in a comment on (or within) the call when the process MUST inherit
# stdin — e.g. an interactive login the user explicitly invokes. Travels with
# the line, so it survives edits that shift line numbers (unlike a pinned
# file:line entry).
EXEMPT_MARKER = "noqa: subprocess-stdin"

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
    """Find all subprocess/os/asyncio calls missing stdin= in content."""
    violations = []
    lines = content.split("\n")

    # Match only actual function calls — not comments, docstrings, or prose.
    # Multiple patterns cover subprocess.run/Popen/call/check_output/check_call,
    # os.system, and asyncio.create_subprocess_exec/shell.
    patterns = [re.compile(p) for p in _SUBPROCESS_PATTERNS]

    for i, line in enumerate(lines):
        # Skip comments.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        # Skip lines where the match is inside backticks (docstring references).
        if "``subprocess" in line:
            continue

        if not any(p.search(line) for p in patterns):
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

                        # Inline exemption marker on the call itself or within
                        # the few comment lines immediately above it → the call
                        # intentionally inherits stdin.
                        window_start = max(0, i - 4)
                        preceding = "\n".join(lines[window_start:i])
                        if EXEMPT_MARKER in call_text or EXEMPT_MARKER in preceding:
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

    # Add repo root to sys.path so we can import hermes_constants (this script
    # runs as a standalone subprocess, not as a module).
    sys.path.insert(0, str(repo_root))
    from hermes_constants import get_hermes_home

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

    # Scan user plugin directories (Gap 1: guard missed user-installed
    # plugins in get_hermes_home()/plugins/ and project plugins in
    # ./.hermes/plugins/, where code like ori/hooks.py can spawn
    # subprocesses with inherited stdin — #67639).
    plugin_roots: list[Path] = [get_hermes_home() / "plugins"]
    if os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS"):
        plugin_roots.append(Path.cwd() / ".hermes" / "plugins")
    seen_roots: set[Path] = set()
    for plugin_root in plugin_roots:
        resolved = plugin_root.resolve()
        if resolved in seen_roots or not resolved.is_dir():
            continue
        seen_roots.add(resolved)

        for py_file in resolved.rglob("*.py"):
            rel = str(py_file)
            if py_file.name in ("conftest.py",) or "/tests/" in rel:
                continue

            try:
                content = py_file.read_text()
            except Exception:
                continue
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
