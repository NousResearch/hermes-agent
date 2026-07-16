"""Fast catch-up merge path with AST-based test selection."""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import forkdelta, gates, gitops


@dataclass(frozen=True)
class CatchupReport:
    changed_files: tuple[str, ...]
    selected_tests: tuple[str, ...]
    coverage_map: dict[str, tuple[str, ...]]
    untested: tuple[str, ...]
    gate_results: tuple[gates.GateResult, ...]


def module_name(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    parts = Path(path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _resolve_relative(module: str, level: int, package: str) -> str:
    pkg_parts = package.split(".") if package else []
    base = pkg_parts[: max(0, len(pkg_parts) - level + 1)]
    return ".".join([*base, module]) if module else ".".join(base)


def imports_for_source(path: Path, module: str) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    package = module if path.name == "__init__.py" else (module.rsplit(".", 1)[0] if "." in module else "")
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = _resolve_relative(base, node.level, package)
            if base:
                imports.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    imports.add(f"{base}.{alias.name}")
    return imports


def _module_to_path(repo: Path, module: str) -> str | None:
    rel = Path(*module.split("."))
    for candidate in (rel.with_suffix(".py"), rel / "__init__.py"):
        if (repo / candidate).exists():
            return candidate.as_posix()
    return None


def _expand_init_reexports(repo: Path, modules: set[str]) -> set[str]:
    expanded = set(modules)
    changed = True
    while changed:
        changed = False
        for mod in list(expanded):
            init_path = _module_to_path(repo, mod)
            if not init_path or not init_path.endswith("__init__.py"):
                continue
            for imported in imports_for_source(repo / init_path, mod):
                path = _module_to_path(repo, imported)
                if path and imported not in expanded:
                    expanded.add(imported)
                    changed = True
    return expanded


def build_import_index(repo: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    modules: dict[str, str] = {}
    imports: dict[str, set[str]] = {}
    for line in gitops.run_git(repo, ["ls-files", "*.py"]).stdout.splitlines():
        mod = module_name(line)
        if mod:
            modules[mod] = line
    for mod, rel in modules.items():
        imports[mod] = imports_for_source(repo / rel, mod)
    return modules, imports


def select_tests(repo: Path, changed_files: list[str], manifest_tests: list[str] | None = None) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    modules, imports = build_import_index(repo)
    reverse: dict[str, set[str]] = {}
    for mod, imported in imports.items():
        for item in imported:
            path_mod = item
            while path_mod:
                if path_mod in modules:
                    reverse.setdefault(path_mod, set()).add(mod)
                    break
                path_mod = path_mod.rsplit(".", 1)[0] if "." in path_mod else ""
    coverage: dict[str, tuple[str, ...]] = {}
    selected: set[str] = set(manifest_tests or [])
    for changed in changed_files:
        tests: set[str] = set()
        mod = module_name(changed)
        if mod:
            seeds = _expand_init_reexports(repo, {mod})
            seen = set(seeds)
            queue = deque(seeds)
            while queue:
                current = queue.popleft()
                for consumer in reverse.get(current, set()):
                    if consumer in seen:
                        continue
                    seen.add(consumer)
                    queue.append(consumer)
            for consumer in seen:
                rel = modules.get(consumer)
                if rel and rel.startswith("tests/"):
                    tests.add(rel)
        mirror = Path("tests") / changed
        if (repo / mirror).exists():
            tests.add(mirror.as_posix())
        coverage[changed] = tuple(sorted(tests))
        selected.update(tests)
    return tuple(sorted(selected)), coverage


def format_report(report: CatchupReport) -> str:
    lines = ["catchup coverage map:"]
    for path in report.changed_files:
        tests = report.coverage_map.get(path, ())
        suffix = ", ".join(tests) if tests else "UNTESTED-BY-CATCHUP"
        lines.append(f"{path}: {suffix}")
    return "\n".join(lines)


def run_catchup(repo: Path, *, max_commits: int = 50, upstream: str = "origin/main", fork_ref: str = "fork/main") -> CatchupReport:
    # Fetch the remotes the refs actually name (Greptile: no hardcoded "fork").
    for ref in (upstream, fork_ref):
        remote = ref.split("/", 1)[0]
        if remote and remote != ref:
            gitops.fetch(repo, remote)
    _ahead, behind = gitops.ahead_behind(repo, fork_ref, upstream)
    if behind > max_commits:
        raise gitops.GitError(f"catchup refuses {behind} commits behind {upstream}; use hermes_parity start")
    pre_merge = gitops.rev_parse(repo, "HEAD")
    merge = gitops.merge_no_commit(repo, upstream)
    if merge.returncode != 0:
        raise gitops.GitError(merge.stderr.strip() or merge.stdout.strip() or "merge failed")
    changed = gitops.changed_files(repo, pre_merge, "HEAD") or gitops.worktree_changed_files(repo, pre_merge)
    manifest = repo / forkdelta.DEFAULT_MANIFEST
    manifest_tests = forkdelta.manifest_nodeids(manifest) if manifest.exists() else []
    selected, coverage = select_tests(repo, changed, manifest_tests)
    results = gates.run_gates(repo, fast=True, fork_ref=fork_ref, tests=selected)
    if all(result.passed for result in results):
        results.extend(gates.run_gates(repo, stage="tests", tests=selected))
    return CatchupReport(
        changed_files=tuple(changed),
        selected_tests=selected,
        coverage_map=coverage,
        untested=tuple(path for path, tests in coverage.items() if not tests),
        gate_results=tuple(results),
    )
