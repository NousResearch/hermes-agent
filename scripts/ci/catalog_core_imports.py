#!/usr/bin/env python3
"""Which pinned catalog plugins a change to core breaks, from a committed index of their core imports.

Catalog plugins live in their own repositories, pinned by sha, and many import Hermes modules that are
not on the published plugin API. A core PR that removes or renames such a module or name breaks the
plugin on the next release, and nothing in this repository's CI sees it (the plugin's code is not here).

``catalog_core_imports/*.json`` records, per catalog entry at its pinned sha, every Hermes import the
plugin makes (``hermes_cli.plugin_validate_private_imports.hermes_import_sites``, the parser
``hermes plugins validate`` uses). Two commands:

``build [NAME ...]``  fetches the ``.py`` files of each entry at its pin (only the entries whose pin
                      changed, unless ``--all``) and rewrites the index. Network; the scheduled
                      workflow and a maintainer run it, never a PR check.
``check``             resolves every indexed import against the base and the head tree of the change,
                      from git objects only (no network, no plugin code runs, nothing is imported), and
                      fails when an import that resolves on the base no longer resolves on the head.

An import resolves when its module exists in the tree (directly or through ``hermes_agent/compat/moves.py``)
and, for ``from module import name``, the module defines ``name`` at top level, has a submodule of that
name, or answers names dynamically (a module ``__getattr__`` or a star import). What a still-present name
DOES is not checked.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "scripts" / "ci" / "catalog_core_imports"
CATALOG = ROOT / "plugin-catalog"
OVERRIDE_LABEL = "breaks-catalog-plugin"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_agent.compat.finder import current_name
from hermes_agent.compat.paths import moves_from_source
from hermes_cli.plugin_validate_private_imports import (
    CoreImports,
    ImportBranch,
    ImportChoice,
    ImportSite,
    hermes_import_sites,
)


# ── index ────────────────────────────────────────────────────────────────────────────────────────────

def _shard(name: str) -> str:
    """Entries are split by first character so no file nears the blob-size cap and a repin edits one file."""
    first = name[:1].lower()
    return first if first.isalnum() else "_"


@dataclass(frozen=True)
class Row:
    """One catalog entry in the index: the source it was read from and its core imports, or why it could not
    be read. ``sha`` + ``subdir`` identify the code (a repo rename keeps the same commits)."""

    sha: str
    subdir: str
    imports: CoreImports
    error: str = ""

    def is_current(self, entry) -> bool:
        return not self.error and (self.sha, self.subdir) == (entry.sha, entry.subdir or "")


def _site_json(site: ImportSite) -> list:
    return [site.module, site.name, f"{site.rel}:{site.line}"]


def _site(item: list) -> ImportSite:
    module, name, where = item
    rel, _, line = where.rpartition(":")
    return ImportSite(rel, int(line), module, name)


def _branch_json(branch: ImportBranch) -> list:
    """A branch is its sites, then each nested ``try`` as ``{"try": [branch, ...]}``."""
    return [*(_site_json(site) for site in branch.sites), *({"try": _choice_json(c)} for c in branch.choices)]


def _choice_json(choice: ImportChoice) -> list:
    return [_branch_json(branch) for branch in choice.branches]


def _branch(items: list) -> ImportBranch:
    return ImportBranch(tuple(_site(i) for i in items if isinstance(i, list)),
                        tuple(_choice(i["try"]) for i in items if isinstance(i, dict)))


def _choice(branches: list) -> ImportChoice:
    return ImportChoice(tuple(_branch(b) for b in branches))


def _row_json(row: Row) -> dict:
    """The on-disk row. A plain import is kept once per (module, name), at its first site."""
    out: dict = {"sha": row.sha, **({"subdir": row.subdir} if row.subdir else {})}
    if row.error:
        return {**out, "error": row.error}
    first: dict[tuple[str, str], ImportSite] = {}
    for site in row.imports.plain:
        first.setdefault((site.module, site.name), site)
    if first:
        out["imports"] = [_site_json(site) for site in first.values()]
    if row.imports.choices:
        out["choices"] = [_choice_json(choice) for choice in row.imports.choices]
    return out


def _row(data: dict) -> Row:
    imports = CoreImports(tuple(_site(item) for item in data.get("imports", [])),
                          tuple(_choice(c) for c in data.get("choices", [])))
    return Row(data["sha"], data.get("subdir", ""), imports, data.get("error", ""))


def read_index(path: Path = INDEX) -> dict[str, Row]:
    entries: dict[str, dict] = {}
    for shard in sorted(path.glob("*.json")):
        entries.update(json.loads(shard.read_text(encoding="utf-8-sig"))["entries"])
    return {name: _row(data) for name, data in entries.items()}


def write_index(entries: dict[str, Row], path: Path = INDEX) -> None:
    """One file per shard and one entry per line, sorted, so a repin's diff is that entry's line alone."""
    path.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[str]] = {}
    for name in sorted(entries):
        shards.setdefault(_shard(name), []).append(
            f"  {json.dumps(name)}: {json.dumps(_row_json(entries[name]), separators=(',', ':'))}")
    for stale in set(p.stem for p in path.glob("*.json")) - shards.keys():
        (path / f"{stale}.json").unlink()
    for shard, lines in shards.items():
        (path / f"{shard}.json").write_text('{"entries": {\n' + ",\n".join(lines) + "\n}}\n", encoding="utf-8")


def _git(args: list[str], cwd: Path, timeout: int = 300) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=timeout,
                   stdin=subprocess.DEVNULL, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def _fetch_python_files(repo: str, sha: str, dest: Path) -> None:
    """The ``.py`` files of ``repo`` at ``sha`` in ``dest``: one shallow fetch, blobs only for ``*.py``."""
    _git(["init", "-q"], dest)
    # Parsed, never run; a symlink in the repo must not point the parser at a runner file.
    _git(["config", "core.symlinks", "false"], dest)
    _git(["remote", "add", "origin", repo], dest)
    _git(["sparse-checkout", "set", "--no-cone", "*.py"], dest)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha], dest)
    _git(["checkout", "-q", "--force", "FETCH_HEAD"], dest)


def index_entry(entry, work: Path) -> Row:
    """The index row for one catalog entry: its source and its core imports, or why they could not be read."""
    subdir = entry.subdir or ""
    dest = Path(tempfile.mkdtemp(dir=work))
    try:
        _fetch_python_files(entry.repo, entry.sha, dest)
        plugin_dir = dest / subdir if subdir else dest
        imports = hermes_import_sites(plugin_dir) if plugin_dir.is_dir() else CoreImports((), ())
        return Row(entry.sha, subdir, imports)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        error = (getattr(exc, "stderr", b"") or str(exc).encode()).decode("utf-8", "replace").strip()[-200:]
        return Row(entry.sha, subdir, CoreImports((), ()), error or type(exc).__name__)
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def build(names: list[str], rebuild_all: bool, jobs: int) -> int:
    from hermes_cli.plugin_catalog import load_catalog

    catalog = {e.name: e for e in load_catalog(CATALOG)}
    old = read_index()
    # A row is re-read when its pin or subdir moved, or when the last read failed (an outage must not drop a
    # plugin from coverage until its next repin).
    wanted = set(names) or {n for n, e in catalog.items() if rebuild_all or n not in old or not old[n].is_current(e)}
    unknown = wanted - catalog.keys()
    if unknown:
        print(f"not in the catalog: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    entries = {n: row for n, row in old.items() if n in catalog}
    with tempfile.TemporaryDirectory(prefix="catalog-imports-") as work:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for name, row in zip(sorted(wanted), pool.map(lambda n: index_entry(catalog[n], Path(work)), sorted(wanted))):
                entries[name] = row
    write_index(entries)
    failed = sorted(n for n in wanted if entries[n].error)
    print(f"indexed {len(wanted)} entr{'y' if len(wanted) == 1 else 'ies'} ({len(failed)} unreadable"
          f"{': ' + ', '.join(failed) if failed else ''}); {len(entries)} in the index")
    return 0


# ── resolving an import against a tree ───────────────────────────────────────────────────────────────

class Tree:
    """The files of one commit, read from git objects only."""

    def __init__(self, root: Path, rev: str) -> None:
        self.root, self.rev = root, rev
        listed = [p for p in self._git("ls-tree", "-r", "--name-only", "-z", rev).split("\0") if p]
        self.files = frozenset(listed)
        self.dirs = frozenset("/".join(parts[:i]) for p in listed for parts in [p.split("/")[:-1]]
                              for i in range(1, len(parts) + 1))
        path = "hermes_agent/compat/moves.py"
        self.moves = moves_from_source(self.read(path)) if path in self.files else {}

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, timeout=120,
                              stdin=subprocess.DEVNULL).stdout.decode("utf-8-sig", "replace")

    def read(self, path: str) -> str:
        return self._git("show", f"{self.rev}:{path}")


def _module_file(tree: Tree, module: str) -> str | None:
    """The file that defines ``module`` in ``tree`` ("" for a namespace package), else ``None``."""
    mapped = current_name(module, tree.moves) or module
    base = mapped.replace(".", "/")
    for path in (f"{base}.py", f"{base}/__init__.py"):
        if path in tree.files:
            return path
    return "" if base in tree.dirs else None


def _is_globals_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "globals"


def _binds_names_at_runtime(tree: ast.Module) -> bool:
    """A module that writes names through ``globals()`` binds names no parse can list: it hands the dict to a
    call (``registry.export(globals())``), mutates it (``globals().update(...)``) or assigns into it. Reading
    it (``globals().get(name)``, ``globals()[name]``) binds nothing."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(_is_globals_call(arg) for arg in [*node.args, *(k.value for k in node.keywords)]):
            return True
        if isinstance(node, ast.Attribute) and _is_globals_call(node.value) and node.attr in ("update", "setdefault", "__setitem__"):
            return True
        if isinstance(node, ast.Subscript) and _is_globals_call(node.value) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
    return False


def _top_level_names(source: str) -> tuple[set[str], bool]:
    """Names a module binds at top level (through ``if``/``try``/``with`` blocks too) and whether it
    can answer other names (a module ``__getattr__``, a star import, or a write through ``globals()``)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set(), True  # unparseable here means we cannot judge: never report it
    body = tree.body
    names: set[str] = set()
    dynamic = _binds_names_at_runtime(tree)
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            dynamic |= node.name == "__getattr__"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                dynamic |= alias.name == "*"
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            stack += [*node.body, *getattr(node, "orelse", []), *getattr(node, "finalbody", []),
                      *(s for h in getattr(node, "handlers", []) for s in h.body)]
    return names, dynamic


class Resolver:
    """Answers whether ``from module import name`` (``name`` "" for ``import module``) works on one tree."""

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self._answers: dict[tuple[str, str], bool] = {}
        self._names: dict[str, tuple[set[str], bool]] = {}

    def resolves(self, module: str, name: str) -> bool:
        key = (module, name)
        if key not in self._answers:
            self._answers[key] = self._resolves(module, name)
        return self._answers[key]

    def _resolves(self, module: str, name: str) -> bool:
        path = _module_file(self.tree, module)
        if path is None:
            return False
        if not name or name == "*" or _module_file(self.tree, f"{module}.{name}") is not None:
            return True
        if not path:
            return False  # a namespace package binds nothing but its submodules
        if path not in self._names:
            self._names[path] = _top_level_names(self.tree.read(path))
        names, dynamic = self._names[path]
        return dynamic or name in names


# ── check ────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Break:
    plugin: str
    site: ImportSite
    guarded: bool

    @property
    def target(self) -> str:
        return f"{self.site.module}.{self.site.name}" if self.site.name else self.site.module


def _runs(resolver: Resolver, branch: ImportBranch) -> bool:
    """The block runs: its imports resolve and no nested ``try`` fails as a whole."""
    return (all(resolver.resolves(site.module, site.name) for site in branch.sites)
            and all(any(_runs(resolver, b) for b in choice.branches) for choice in branch.choices))


def _runs_on_hermes(resolver: Resolver, branch: ImportBranch) -> bool:
    """The block runs AND reaches Hermes through at least one import, directly or in a nested ``try``."""
    return _runs(resolver, branch) and (bool(branch.sites) or any(
        _runs_on_hermes(resolver, b) for choice in branch.choices for b in choice.branches))


def _sites(branch: ImportBranch) -> list[ImportSite]:
    return [*branch.sites, *(s for choice in branch.choices for b in choice.branches for s in _sites(b))]


def _lost_sites(choice: ImportChoice, before: Resolver, after: Resolver) -> list[ImportSite]:
    """The sites to report when no branch of ``choice`` that reaches Hermes runs on ``after`` while one did on
    ``before``: what those branches no longer resolve. A ``try`` that still runs without Hermes (a handler
    with no Hermes import) has lost the feature, so it is reported too."""
    if any(_runs_on_hermes(after, branch) for branch in choice.branches):
        return []
    worked = [branch for branch in choice.branches if _runs_on_hermes(before, branch)]
    return [site for branch in worked for site in _sites(branch) if not after.resolves(site.module, site.name)]


def _degrades(choice: ImportChoice, after: Resolver) -> bool:
    return any(_runs(after, branch) for branch in choice.branches)


def _changed_paths(base: Tree, head: Tree) -> list[str]:
    return [p for p in base._git("diff", "--name-only", "--no-renames", "-z", base.rev, head.rev).split("\0") if p]


def _touched(module: str, trees: tuple[Tree, Tree], changed: list[str]) -> bool:
    """True when ``changed`` holds a file that decides whether ``module`` (or a name in it) resolves."""
    prefixes = {(current_name(module, t.moves) or module).replace(".", "/") for t in trees}
    return any(path == f"{p}.py" or path.startswith(f"{p}/") for p in prefixes for path in changed)


def breaks(index: dict[str, Row], base: Tree, head: Tree) -> list[Break]:
    """Every indexed import the plugin has on ``base`` and loses on ``head``.

    Only imports whose module files the change touches are resolved (all of them when it edits the
    compat table): any other import resolves on both trees or on neither."""
    changed = _changed_paths(base, head)
    every = base.moves != head.moves

    def relevant(sites) -> bool:
        return every or any(_touched(site.module, (base, head), changed) for site in sites)

    before, after = Resolver(base), Resolver(head)
    found = []
    for plugin in sorted(index):
        imports = index[plugin].imports
        for site in imports.plain:
            if relevant([site]) and before.resolves(site.module, site.name) and not after.resolves(site.module, site.name):
                found.append(Break(plugin, site, guarded=False))
        for choice in imports.choices:
            if relevant([site for branch in choice.branches for site in _sites(branch)]):
                found += [Break(plugin, site, _degrades(choice, after)) for site in _lost_sites(choice, before, after)]
    return found


def report(found: list[Break]) -> str:
    plugins = sorted({b.plugin for b in found})
    lines = [f"This change breaks {len(plugins)} pinned catalog plugin{'s' if len(plugins) != 1 else ''} "
             f"({', '.join(plugins)}): {'they import' if len(plugins) != 1 else 'it imports'} core names "
             "this change removes or renames.", ""]
    for b in found:
        how = " (inside a try: the plugin loses that feature instead of failing to load)" if b.guarded else ""
        lines.append(f"  {b.plugin} {b.site.rel}:{b.site.line} imports {b.target}{how}")
    lines += [
        "",
        "Plugins are only promised the published plugin API (hermes_cli/data/plugin_api.json, ctx), so",
        "these imports were never supported; but merging this breaks them for every user on the next release.",
        "Pick one:",
        "  1. Open a PR for an official seam first: a ctx method, a hook, or a plugin_api.json entry that",
        "     gives the plugin what it imported (plugins/AGENTS.md § Boundary lint), then tell the plugin's",
        "     maintainer to move to it.",
        "  2. Keep the old name importable: a moved module gets a key in hermes_agent/compat/moves.py,",
        "     a moved name stays where it was.",
        f"  3. If breaking the plugin is intended, add the `{OVERRIDE_LABEL}` label, say why in the PR body,",
        "     and re-run this check.",
    ]
    return "\n".join(lines)


def check(base_rev: str, head_rev: str, labels: list[str]) -> int:
    index = read_index()
    found = breaks(index, Tree(ROOT, base_rev), Tree(ROOT, head_rev))
    if not found:
        print(f"no pinned catalog plugin loses a core import ({len(index)} entries indexed)")
        return 0
    print(report(found))
    if OVERRIDE_LABEL in labels:
        print(f"\nThe `{OVERRIDE_LABEL}` label accepts these breaks.")
        return 0
    return 1


def _pull_request_labels() -> list[str]:
    """The PR's labels as they are now (a re-run replays the event that started it)."""
    repo, event = os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_EVENT_PATH", "")
    try:
        number = json.loads(Path(event).read_text(encoding="utf-8-sig"))["pull_request"]["number"] if event else None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        number = None
    if not repo or not number:
        return []
    out = subprocess.run(["gh", "api", f"repos/{repo}/pulls/{number}", "--jq", ".labels[].name"], capture_output=True,
                         text=True, encoding="utf-8", stdin=subprocess.DEVNULL, check=False, timeout=60)
    return out.stdout.split() if out.returncode == 0 else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Which pinned catalog plugins a change to core breaks.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="refresh the index from the catalog pins (network)")
    b.add_argument("names", nargs="*", help="entries to re-index (default: those whose pin changed)")
    b.add_argument("--all", action="store_true", help="re-index every entry")
    b.add_argument("--jobs", type=int, default=16)
    c = sub.add_parser("check", help="fail when the change breaks an indexed import")
    c.add_argument("--base", default="HEAD^1", help="base commit (default: the merge commit's first parent)")
    c.add_argument("--head", default="HEAD", help="head commit (default: HEAD, the merge commit)")
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.names, args.all, args.jobs)
    return check(args.base, args.head, _pull_request_labels())


if __name__ == "__main__":
    raise SystemExit(main())
