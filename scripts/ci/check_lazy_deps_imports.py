#!/usr/bin/env python3
"""Fail when tracked production code imports the deleted tools.lazy_deps module.

``tools/lazy_deps.py`` was deleted by the pm migration; pm.extras
(available / ensure_import / ensure_and_bind) is the only lazy-install
surface. Any remaining import of it is an ImportError at call time, so
this guard blocks the migration from silently re-opening.

- **Tracked inventory.** The file list comes from ``git ls-files``, so
  ignored worktree/build debris is never walked. The check never passes
  by scanning nothing: an inventory that cannot be built (not a repo,
  git failure) or that contains no ``*.py`` files at all FAILS (exit 2).
- **AST import analysis.** Matches real import statements only:
  ``import tools.lazy_deps`` and dotted variants, ``from tools import
  lazy_deps``, ``from tools.lazy_deps...`` and the relative forms
  (``from . import lazy_deps``, ``from .lazy_deps import ...``) resolved
  inside the tracked ``tools/`` package. Comments, docstrings, and other
  prose mentions are ignored by construction.
- **Errors are failures.** A tracked file that cannot be read, decoded,
  or parsed is reported and fails the run (exit 2), not skipped.

tests/ files are out of scope: tests may reference the deleted module to
assert it is gone.

Exit codes: 0 = clean, 1 = offenders found (exact ``path:line`` sites on
stdout), 2 = inventory/analysis error.

Usage:
    python scripts/ci/check_lazy_deps_imports.py [repo_root]

Wired unconditionally into .github/workflows/lazy-deps-guard.yml
(blocking job); the contract tests driving this CLI against fixture
repos live in tests/pm/test_no_lazy_deps.py.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path, PurePosixPath

_DELETED_MODULE = "tools.lazy_deps"
_DELETED_PARENT = "tools"


def _tracked_python_files(root: Path) -> list[str]:
    """Tracked ``*.py`` paths (repo-relative, POSIX separators) minus tests/."""
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"inventory failed: git ls-files exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace').strip()}\n"
        )
        raise SystemExit(2)
    names = proc.stdout.decode(errors="replace").split("\0")
    deleted = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--deleted", "-z", "--", "*.py"],
        capture_output=True, check=True,
    )
    removed = set(deleted.stdout.decode(errors="replace").split("\0"))
    files = sorted(
        n.replace("\\", "/")
        for n in names
        if n.endswith(".py") and n not in removed
        and not n.replace("\\", "/").startswith("tests/")
    )
    if not files:
        sys.stderr.write(
            "inventory failed: no tracked *.py files under "
            f"{root} — refusing to pass an empty inventory\n"
        )
        raise SystemExit(2)
    return files


def _relative_target(relpath: str, node: ast.ImportFrom) -> str | None:
    """Absolute module a relative from-import resolves to for *relpath*.

    ``level`` 1 is the file's own package; each further level climbs one
    package. Returns None when the import escapes the repo root (never a
    real module, so never the deleted one).
    """
    parts = PurePosixPath(relpath).with_suffix("").parts[:-1]
    up = node.level - 1
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    mod = node.module.split(".") if node.module else []
    return ".".join([*base, *mod])


def _import_sites(tree: ast.AST, relpath: str) -> list[str]:
    """``relpath:line: <description>`` for every tools.lazy_deps import."""
    sites: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _DELETED_MODULE or alias.name.startswith(
                    _DELETED_MODULE + "."
                ):
                    sites.append(
                        f"{relpath}:{node.lineno}: import of deleted module "
                        f"'{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            kind: str | None = None
            if node.level == 0:
                if node.module == _DELETED_MODULE or (
                    node.module and node.module.startswith(_DELETED_MODULE + ".")
                ):
                    kind = f"'{node.module} ({names})'"
                elif node.module == _DELETED_PARENT and any(
                    a.name == "lazy_deps" for a in node.names
                ):
                    kind = f"'{node.module} ({names})'"
            else:
                resolved = _relative_target(relpath, node)
                if resolved == _DELETED_PARENT and any(
                    a.name == "lazy_deps" for a in node.names
                ):
                    kind = f"'{resolved} ({names})' (relative)"
                elif resolved == _DELETED_MODULE or (
                    resolved and resolved.startswith(_DELETED_MODULE + ".")
                ):
                    kind = f"'{resolved} ({names})' (relative)"
            if kind is not None:
                sites.append(
                    f"{relpath}:{node.lineno}: from-import of deleted module {kind}"
                )
    return sites


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    offenders: list[str] = []
    errors: list[str] = []
    for relpath in _tracked_python_files(root):
        path = root / relpath
        try:
            source = path.read_bytes()
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"{relpath}: inventory/read/parse failed: {exc}")
            continue
        offenders.extend(_import_sites(tree, relpath))

    if errors:
        for err in errors:
            print(f"inventory error: {err}", file=sys.stderr)
        print(
            f"{len(errors)} tracked file(s) could not be analyzed — "
            "failing the check rather than scanning a partial inventory",
            file=sys.stderr,
        )
        return 2
    if offenders:
        for site in sorted(offenders):
            print(site)
        print(
            f"{len(offenders)} import(s) of '{_DELETED_MODULE}' (deleted module — "
            "ImportError at call time). Migrate to pm.extras "
            "(available / ensure_import / ensure_and_bind).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
