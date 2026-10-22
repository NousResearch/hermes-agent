#!/usr/bin/env python3
"""Check that subprocess calls in TUI-context code specify stdin=.

When Hermes runs in TUI mode, the gateway child process communicates with
the Node.js parent over a JSON-RPC protocol on stdin. Subprocess calls that
inherit this fd can cause the gateway to exit with stdin EOF during tool
execution (issue #14036, PR #39257).

This script checks that all subprocess.run() and subprocess.Popen() calls
in TUI-context files (agent/, tools/, plugins/, tui_gateway/,
optional-skills/) explicitly set stdin= to prevent fd inheritance.

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

import ast
import os
import sys
from pathlib import Path

# Directories that run inside the TUI gateway child process.
TUI_CONTEXT_DIRS = [
    "agent/",
    "tools/",
    "plugins/",
    "tui_gateway/",
    # Optional-skill scripts can run inside the backend process (a command
    # builder calling a skill helper on a backend thread); their children would
    # inherit the JSON-RPC stdin too.
    "optional-skills/",
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
# an explicit stdin= argument, keyed by the module name the call is made on.
# Calls are found as ast.Call nodes, so arguments that start on the next line
# are checked too.
_INHERITING_CALLS = {
    "subprocess": {"run", "Popen", "call", "check_output", "check_call"},
    "os": {"system"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
}

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

# Directory names skipped at any depth below a context dir. Matching against the
# path relative to that dir keeps a skill's own ``scripts/`` folder in scope.
SKIP_DIRS = {"tests"}


def _definition_sets_stdin(tree: ast.AST, name: str) -> bool:
    """True when ``name`` is defined in the file (assignment or ``def``) and that
    definition's OWN expression/body sets ``stdin=``.

    Shared kwargs helpers (``_RUN_KW = dict(..., stdin=DEVNULL)``, ``def _run_kwargs(): return
    dict(..., stdin=DEVNULL)``) legitimately carry the guard; we only accept them when the
    definition provably sets stdin= — never on the helper's name alone, and never because an
    unrelated later call in the file happens to pass ``stdin=``.
    """
    node = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            node = n
            break
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                node = n.value if n.value is not None else n
                break
    return node is not None and _sets_stdin(node)


def _is_stdin_key(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == "stdin"


def _sets_stdin(node: ast.AST) -> bool:
    """stdin is set as a keyword (``dict(stdin=...)``), a dict-literal key (``{"stdin": ...}``),
    a subscript store (``kw["stdin"] = ...``) or ``kw.setdefault("stdin", ...)``. A ``"stdin"``
    string anywhere else (a value, a path) does not count."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.keyword) and sub.arg == "stdin":
            return True
        if isinstance(sub, ast.Dict) and any(_is_stdin_key(key) for key in sub.keys):
            return True
        if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store) and _is_stdin_key(sub.slice):
            return True
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "setdefault" and sub.args and _is_stdin_key(sub.args[0])):
            return True
    return False


def _is_inheriting_call(call: ast.Call) -> bool:
    """``subprocess.run(...)``, ``os.system(...)`` etc., including ``x.subprocess.run(...)``."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value
    owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
    return func.attr in _INHERITING_CALLS.get(owner_name, ())


def _call_is_safe(call: ast.Call, tree: ast.AST) -> bool:
    for kw in call.keywords:
        # stdin= set, or input= (creates a pipe).
        if kw.arg in ("stdin", "input"):
            return True
        if kw.arg is not None:
            continue
        # Inline splat that sets it: ``**dict(stdin=...)`` / ``**{"stdin": ...}``.
        if _sets_stdin(kw.value):
            return True
        # ``**name`` / ``**name(...)`` splat: safe when the same-file definition sets stdin=.
        value = kw.value.func if isinstance(kw.value, ast.Call) else kw.value
        if isinstance(value, ast.Name) and _definition_sets_stdin(tree, value.id):
            return True
    return False


def find_subprocess_calls(content: str, filepath: str) -> list[dict]:
    """Find all subprocess/os/asyncio calls missing stdin= in content."""
    lines = content.split("\n")
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        # Fail closed: an unparsable file cannot be shown to be safe.
        return [{"file": filepath, "line": exc.lineno or 1, "snippet": f"SyntaxError: {exc.msg}"}]

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_inheriting_call(node):
            continue
        if _call_is_safe(node, tree):
            continue
        # Inline exemption marker on the call itself or within the few comment
        # lines immediately above it → the call intentionally inherits stdin.
        start, end = node.lineno - 1, node.end_lineno or node.lineno
        if EXEMPT_MARKER in "\n".join(lines[max(0, start - 4):end]):
            continue
        violations.append({
            "file": filepath,
            "line": node.lineno,
            "snippet": lines[start].strip()[:120],
        })

    violations.sort(key=lambda v: v["line"])
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
            rel = py_file.relative_to(repo_root).as_posix()

            # Skip known-safe files.  ``relative_to`` returns a host-separated
            # path (backslashes on Windows) while KNOWN_SAFE is forward-slash.
            if py_file.relative_to(repo_root).as_posix() in KNOWN_SAFE:
                continue

            # Skip test files inside tools/ etc.
            if SKIP_DIRS & set(py_file.relative_to(dirpath).parts):
                continue

            content = py_file.read_text(encoding="utf-8-sig")
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
            if py_file.name in ("conftest.py",) or "/tests/" in py_file.as_posix():
                continue

            try:
                content = py_file.read_text(encoding="utf-8-sig")
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
