#!/usr/bin/env python3
"""Fail when profile-varying env vars are read raw instead of through the profile scope.

Under gateway multiplexing one process serves every profile off ONE shared
os.environ, into which each profile's .env is loaded with override=True. So a
plain ``os.getenv("TERMINAL_ENV")`` / ``os.environ["HERMES_WRITE_SAFE_ROOT"]``
reads whichever profile loaded last, not the profile whose turn is running - a
cross-profile leak (proven live: felix's write_file gated by jonas's
HERMES_WRITE_SAFE_ROOT). Every profile-varying read must go through the
fail-closed scope helpers instead:

  * TERMINAL_* (a global-env prefix, so get_secret would NOT isolate them) ->
    ``tools.terminal_scope.terminal_env`` / ``agent.runtime_cwd.scope_terminal_cwd``
  * HERMES_WRITE_SAFE_ROOT / HERMES_ACCEPT_HOOKS / HERMES_ALLOW_PRIVATE_URLS
    (non-global policy vars) -> ``agent.secret_scope.get_secret``

This walks tools/ and agent/ and flags a raw os.getenv / os.environ.get /
os.environ[...] whose key is (or starts with) a banned name. Legitimate raw
readers (the scope machinery itself, and terminal_tool which reads the projected
scope out of its own env) opt out with a trailing ``# scope-exempt: <reason>``
comment on the offending line.

Exit 1 with a file:line list on any hit. Run: python scripts/check_profile_env_scope.py

Scope and limits (best-effort regression nudge, NOT a complete security control):
this matches only literal-key ``os.getenv("K")`` / ``os.environ.get("K")`` /
``os.environ["K"]`` where the module is imported as ``os``. It does NOT catch an
aliased import (``import os as o``), a ``from os import getenv``, a variable/
computed key (``os.getenv(k)``), or ``os.environ.copy()``. It scans only
``tools/`` and ``agent/`` - the modules that run inside the multiplexed gateway
worker; single-profile CLI/oneshot entrypoints under ``hermes_cli/`` are out of
scope by design. The read-time scope helpers, not this guard, are the actual
isolation control; this just stops the most common verbatim regression.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("tools", "agent")

# Exact profile-varying keys that must be scoped.
BANNED_EXACT = frozenset({
    "HERMES_WRITE_SAFE_ROOT",
    "HERMES_ACCEPT_HOOKS",
    "HERMES_ALLOW_PRIVATE_URLS",
})
# Whole prefixes that are profile-varying (terminal/sandbox backend policy).
BANNED_PREFIXES = ("TERMINAL_",)

EXEMPT_MARKER = "# scope-exempt"


def _is_banned(key: str) -> bool:
    return key in BANNED_EXACT or key.startswith(BANNED_PREFIXES)


def _py_files(root: Path):
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _first_str_arg(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _raw_env_key(node: ast.AST) -> str | None:
    """Return the env key if node is a raw os.environ / os.getenv access, else None."""
    # os.getenv("KEY") / os.environ.get("KEY")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        fn = node.func
        # os.getenv(...)
        if fn.attr == "getenv" and isinstance(fn.value, ast.Name) and fn.value.id == "os":
            return _first_str_arg(node)
        # os.environ.get(...)
        if (fn.attr == "get" and isinstance(fn.value, ast.Attribute)
                and fn.value.attr == "environ"
                and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "os"):
            return _first_str_arg(node)
    # os.environ["KEY"]  (Subscript)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        attr = node.value
        if (attr.attr == "environ" and isinstance(attr.value, ast.Name) and attr.value.id == "os"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root to scan (default: this repo)")
    args = parser.parse_args()
    root = args.root.resolve()
    hits: list[str] = []
    for path in _py_files(root):
        rel = path.relative_to(root)
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            key = _raw_env_key(node)
            if key is None or not _is_banned(key):
                continue
            ln = getattr(node, "lineno", 0)
            src_line = lines[ln - 1] if 0 < ln <= len(lines) else ""
            if EXEMPT_MARKER in src_line:
                continue
            hits.append(f"{rel}:{ln}: raw os.environ read of profile-varying '{key}'")
    if hits:
        print("❌ profile-varying env vars read raw (cross-profile leak under multiplexing):")
        for h in sorted(set(hits)):
            print("  " + h)
        print(
            f"\n{len(set(hits))} site(s). Read TERMINAL_* via tools.terminal_scope.terminal_env "
            "(or agent.runtime_cwd.scope_terminal_cwd for TERMINAL_CWD), and the HERMES_* policy "
            "vars via agent.secret_scope.get_secret. A legitimate raw reader opts out with a "
            f"trailing '{EXEMPT_MARKER}: <reason>' comment."
        )
        return 1
    print(f"✅ no raw reads of profile-varying env vars in {'/, '.join(SCAN_DIRS)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
