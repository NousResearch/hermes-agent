"""The hosted entry point's import graph: which modules a hosted process loads.

``reachable(repo, tree)`` walks the import statements that run at import time (module level,
including top-level ``if``/``try``/``with`` blocks, excluding ``if TYPE_CHECKING:``) from the
roots in ``config.HOSTED_ENTRY``, over one tree. Importing ``a.b.c`` loads ``a``, ``a.b`` and
``a.b.c``, and ``from a import b`` loads ``a.b`` when it is a module, so each package
``__init__`` on the way is in the graph too.

Imports inside functions are not followed: they reach almost the whole tree (CLI setup wizards,
installers) through code a hosted process never calls. A module the hosted process loads lazily
is listed as a root instead. Plugins loaded by path (``plugins/*``, ``$HERMES_HOME/plugins``)
are not Python imports and are not in the graph.
"""

from __future__ import annotations

import ast
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path

from scripts.code_health import config, gitio


def _read_all(repo: Path, tree: str | None, paths: list[str]) -> dict[str, str]:
    if tree is None:
        return {p: t for p in paths if (t := gitio.read_file(repo, None, p)) is not None}
    # One `git cat-file --batch` for the whole walk: a `git show` per module costs seconds.
    request = "".join(f"{tree}:{p}\n" for p in paths).encode("utf-8")
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input=request,
                         capture_output=True, timeout=120, check=True).stdout
    texts: dict[str, str] = {}
    pos = 0
    for path in paths:
        end = out.index(b"\n", pos)
        header = out[pos:end].split()
        pos = end + 1
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        texts[path] = out[pos:pos + size].decode("utf-8-sig", errors="replace")
        pos += size + 1
    return texts


class _Graph:
    def __init__(self, repo: Path, tree: str | None) -> None:
        self.repo, self.tree = repo, tree
        self.files = {p for p in gitio.tracked_files(repo, tree) if p.endswith(".py")}
        if tree is None:  # the working tree: untracked modules are part of the head under test
            out = gitio.git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py")
            self.files |= {p for p in out.split("\0") if p}

    def module_file(self, name: str) -> str | None:
        stem = name.replace(".", "/")
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if candidate in self.files:
                return candidate
        return None

    def walk(self, roots: tuple[str, ...]) -> frozenset[str]:
        seen: set[str] = set()
        frontier = [f for f in (self.module_file(r) for r in roots) if f]
        while frontier:
            batch = sorted(set(frontier) - seen)
            seen.update(batch)
            frontier = []
            for path, text in _read_all(self.repo, self.tree, batch).items():
                for name in _imported(path, text):
                    found = self.module_file(name)
                    if found and found not in seen:
                        frontier.append(found)
        return frozenset(seen)


def _package(path: str) -> list[str]:
    """The package a module's relative imports start from (its own, for an ``__init__``)."""
    return path[:-3].split("/")[:-1]


def _prefixes(name: str) -> Iterator[str]:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        yield ".".join(parts[:i])


def _type_checking(node: ast.If) -> bool:
    test = node.test
    name = test.id if isinstance(test, ast.Name) else test.attr if isinstance(test, ast.Attribute) else ""
    return name == "TYPE_CHECKING"


def _import_time(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    for node in body:
        if isinstance(node, ast.If):
            if not _type_checking(node):
                yield from _import_time(node.body)
            yield from _import_time(node.orelse)
        elif isinstance(node, (ast.Try, ast.With, ast.AsyncWith)):
            for field in ("body", "orelse", "finalbody"):
                yield from _import_time(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []):
                yield from _import_time(handler.body)
        else:
            yield node


def _imported(path: str, text: str) -> Iterator[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return  # the measurer reports it; an unparseable module has no edges to follow
    for node in _import_time(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield from _prefixes(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _package(path)
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            if not module:
                continue
            yield from _prefixes(module)
            for alias in node.names:
                yield f"{module}.{alias.name}"


def reachable(repo: Path, tree: str | None, roots: tuple[str, ...] | None = None) -> frozenset[str]:
    """Paths of every module *roots* (default ``config.HOSTED_ENTRY``) load at import time, in
    *tree* (``None``: the working tree)."""
    return _Graph(repo, tree).walk(config.HOSTED_ENTRY if roots is None else roots)
