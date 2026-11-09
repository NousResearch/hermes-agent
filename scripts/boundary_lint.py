#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Report-only import lint for the core/module boundary, in both directions.

* **module -> core internals**: a bundled module imports a Hermes module or name that is not on the
  published plugin API (``hermes_cli/data/plugin_api.json``), e.g. ``gateway.run``. The rule is the one
  ``hermes plugins validate`` warns with (``plugin_validate_private_imports.private_import_findings``),
  so the lint and the plugin-author warning cannot drift.
* **core -> module**: core code imports a specific module (``plugins.platforms.discord.adapter``)
  instead of reaching it through discovery and the plugin contracts.

A *module* is a directory under ``plugins/`` with a ``plugin.yaml`` manifest, or a ``plugins/<name>/``
with a ``dashboard/manifest.json``. Everything else under ``plugins/`` (``plugin_loader``,
``plugin_utils``, ``plugins/memory/__init__.py`` discovery, ...) is core plugin plumbing. *Core* is
every other file of the shipped packages (pyproject ``packages.find``) and the root ``*.py`` modules;
tests, scripts, docs and skills are neither.

Existing violations never fail anything. ``scripts/boundary_lint_baseline.json`` counts each crossing
by importing file and target (``<file> <target>``); ``--check`` (and
``tests/scripts/test_boundary_lint.py``) fails only when a crossing's count rises above it, and names
that crossing. Run ``--update`` to record an intended import, or to drop the entries a change removed.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "boundary_lint_baseline.json"
DIRECTIONS = ("module_to_core", "core_to_module")
FIX_HINT = ("Import the published plugin API (hermes_cli/data/plugin_api.json) or reach the module through"
            " discovery; if the import is intended, run `python scripts/boundary_lint.py --update` and say why"
            " in the PR.")
_SKIP_PARTS = {"tests", "__pycache__", "node_modules", "dist", ".venv", "venv"}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_agent.compat.paths import original_module, original_path
from hermes_cli.plugin_validate_private_imports import (
    _hermes_packages,
    _imported,
    _type_checking_nodes,
    private_import_findings,
)


def module_dirs(root: Path = ROOT) -> list[Path]:
    """Every bundled module directory under ``plugins/``."""
    plugins = root / "plugins"
    dirs = {p.parent for p in plugins.rglob("plugin.yaml") if not _SKIP_PARTS.intersection(p.relative_to(plugins).parts)}
    dirs |= {p.parent.parent for p in plugins.glob("*/dashboard/manifest.json")}
    return sorted(dirs)


def module_to_core(root: Path = ROOT) -> list[str]:
    """``["plugins/<module>/<rel>:<line> <target>", ...]`` for every module import of a core internal.
    An in-tree module's full-path import of itself (``plugins.web.exa.provider``) or of another module
    is not a core import, so it is left out."""
    modules = module_dirs(root)
    return [f"{d.relative_to(root).as_posix()}/{hit}" for d in modules for hit in private_import_findings(d)
            if not _in_module(hit.split(" ", 1)[1], root, modules)]


def _core_files(root: Path, modules: list[Path]):
    tops = sorted(_hermes_packages())
    for top in tops:
        package = root / top
        paths = sorted(package.rglob("*.py")) if package.is_dir() else [root / f"{top}.py"]
        for path in paths:
            rel = path.relative_to(root)
            if not path.is_file() or _SKIP_PARTS.intersection(rel.parts[:-1]) or path.name.startswith("test_"):
                continue
            if any(path.is_relative_to(d) for d in modules):
                continue
            yield path, rel


def _absolute(rel: Path, node: ast.ImportFrom) -> str:
    """The dotted module a relative ``from . import x`` in ``rel`` names."""
    package = list(rel.with_suffix("").parts[:-1])
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join(base + ([node.module] if node.module else []))


def _targets(rel: Path, node: ast.AST) -> list[list[str]]:
    """Per imported alias, the dotted spellings it could name (``pkg.name`` and ``pkg`` for a from-import)."""
    if isinstance(node, ast.ImportFrom) and node.level:
        module = _absolute(rel, node)
        return [[f"{module}.{alias.name}", module] for alias in node.names]
    return [[f"{m}.{n}", m] if n else [m] for m, n in _imported(node)]


def _in_module(target: str, root: Path, modules: list[Path]) -> bool:
    """Whether the dotted ``target`` (a module, or a name in one) is a file or package inside a module."""
    parts = target.split(".")
    if parts[0] != "plugins":
        return False
    for end in range(len(parts), 1, -1):
        path = root.joinpath(*parts[:end])
        for candidate in (path, path.with_suffix(".py")):
            if candidate.exists():
                return any(candidate.is_relative_to(d) for d in modules)
    return False


def core_to_module(root: Path = ROOT) -> list[str]:
    """``["<rel>:<line> <module path>", ...]`` for every core import that names a specific module."""
    modules = module_dirs(root)
    found: set[tuple[str, int, str]] = set()
    for path, rel in _core_files(root, modules):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError, ValueError):
            continue
        skipped = _type_checking_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            for spellings in _targets(rel, node):
                # Collapse the spellings of one alias to one target; distinct aliases count separately.
                if hits := [t for t in spellings if _in_module(t, root, modules)]:
                    found.add((rel.as_posix(), getattr(node, "lineno", 0), min(hits, key=len)))
    return [f"{rel}:{line} {target}" for rel, line, target in sorted(found)]


def collect(root: Path = ROOT) -> dict[str, list[str]]:
    return {"module_to_core": module_to_core(root), "core_to_module": core_to_module(root)}


def _key(site: str) -> str:
    """What a site couples: ``<importing file> <target>`` (repo-relative path, line dropped). Two
    branches that add the same target from different files add two different keys; a file that imports
    one target twice counts 2 under one key."""
    loc, target = site.split(" ", 1)
    return f"{original_path(loc.rsplit(':', 1)[0])} {original_module(target)}"


def tally(found: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Per direction, ``{key: count}`` with the keys of ``_key``."""
    return {d: dict(sorted(Counter(_key(site) for site in found[d]).items())) for d in DIRECTIONS}


def read_baseline(path: Path = BASELINE) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def regressions(current: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]) -> list[str]:
    """One line per key whose count rose above the baseline: a swap still fails, and the line names
    the new import."""
    return [f"{d}: {key} ({n}, baseline {baseline.get(d, {}).get(key, 0)})" for d in DIRECTIONS
            for key, n in current[d].items() if n > baseline.get(d, {}).get(key, 0)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print every violation site")
    parser.add_argument("--check", action="store_true", help="exit 1 when a count rises above the baseline")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline with the current sites")
    args = parser.parse_args(argv)
    found = collect()
    current = tally(found)
    for direction in DIRECTIONS:
        print(f"{direction}: {len(found[direction])}")
        if args.list:
            print("".join(f"  {site}\n" for site in found[direction]), end="")
    if args.update:
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0
    if args.check and (rises := regressions(current, read_baseline())):
        print("new boundary imports:\n  " + "\n  ".join(rises) + f"\n{FIX_HINT}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
