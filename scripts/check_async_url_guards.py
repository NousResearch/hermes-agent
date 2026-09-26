#!/usr/bin/env python3
"""Flag blocking ``is_safe_url()`` calls made from inside an ``async def``.

``tools.url_safety.is_safe_url`` resolves the hostname with a synchronous
``socket.getaddrinfo``.  Calling it from a coroutine therefore holds the event
loop for the whole lookup — tens of milliseconds normally, but seconds to
minutes when the resolver stalls (a 17-minute ``getaddrinfo`` hang took the
backend down once already).  ``async_is_safe_url`` is the same check with the
DNS work moved to ``asyncio.to_thread``.

Ruff cannot see this one: ASYNC210/220/221/251 cover blocking calls written
directly in the async frame, not a blocking syscall hidden inside a helper.
Platform adapters run on the gateway event loop, so the guard applies to them
with full force.

Usage:
    python3 scripts/check_async_url_guards.py [path ...]   # default: repo root
    python3 scripts/check_async_url_guards.py --quiet

Exit code is 1 when any async-context call site is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator

BLOCKING_NAME = "is_safe_url"
SAFE_NAME = "async_is_safe_url"

# Shipped code only. Top-level trees that are not event-loop Python: the Astro
# site (`web/`), the desktop app (`apps/`), tests, and non-shipped skill docs.
# NOTE: match the *first* path component — `plugins/web/firecrawl/` is shipped
# gateway code and must stay scanned.
SKIP_TOP = {
    "tests", "skills", "optional-skills", "apps", "web", "evals",
    "node_modules", "build", "dist",
}
# Never scanned at any depth.
SKIP_ANY = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}


def _enclosing_functions(tree: ast.AST, lineno: int) -> list[tuple[int, str, bool]]:
    """Every function frame containing ``lineno``, outermost first."""
    chain: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno <= lineno <= end:
                chain.append((node.lineno, node.name, isinstance(node, ast.AsyncFunctionDef)))
    chain.sort()
    return chain


def _blocking_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to the blocking guard, e.g. ``is_safe_url as _is_safe_url``."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("url_safety"):
            for alias in node.names:
                if alias.name == BLOCKING_NAME and alias.asname:
                    aliases.add(alias.asname)
    return aliases


def iter_violations(source: str, filename: str = "<string>") -> Iterator[tuple[int, str, str]]:
    """Yield ``(lineno, function_name, source_line)`` for async-context blocking calls."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return
    aliases = _blocking_aliases(tree)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called = func.id
        elif isinstance(func, ast.Attribute):
            called = func.attr
        else:
            continue
        if called != BLOCKING_NAME and called not in aliases:
            continue
        chain = _enclosing_functions(tree, node.lineno)
        if not chain or not all(is_async for _, _, is_async in chain):
            continue  # module level or a sync helper: no event loop to stall
        text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        yield node.lineno, chain[-1][1], text


def _iter_dir(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in SKIP_TOP:
            continue
        if SKIP_ANY.intersection(rel.parts):
            continue
        yield p


def iter_source_files(root: Path, paths: list[str] | None = None) -> Iterator[Path]:
    """Every scannable .py under each requested root (default: the repo root).

    Skip rules are applied to the path *relative to the scanned root*, so a
    shipped package that happens to live under a directory called ``web``
    (``plugins/web/``) is still scanned.
    """
    if not paths:
        yield from _iter_dir(root)
        return
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from _iter_dir(p)
        elif p.suffix == ".py":
            yield p


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in argv
    root = Path(__file__).resolve().parents[1]
    found = 0
    for path in iter_source_files(root, args or None):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, func_name, text in iter_violations(source, str(path)):
            found += 1
            if not quiet:
                try:
                    shown = path.relative_to(root)
                except ValueError:
                    shown = path
                print(f"{shown}:{lineno}: async {func_name}() calls blocking {BLOCKING_NAME}() — use `await {SAFE_NAME}(...)`")
                print(f"    {text}")
    if found and not quiet:
        print(
            f"\n{found} async-context {BLOCKING_NAME}() call site(s). "
            f"That check resolves DNS on the event loop; await {SAFE_NAME}() instead."
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
