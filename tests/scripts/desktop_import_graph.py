"""Print the repo modules desktop preparation's Python loads, as sorted JSON.

Usage: python desktop_import_graph.py '<entry modules JSON>' '<followed path prefixes JSON>'

Preparation imports most helpers inside functions, so importing the entry points
is not enough. For modules under the followed prefixes this also imports the
late imports of the functions that can run: every function of an entry point,
and in any other module only the names its importers take (plus whatever those
reference, and the module-level code outside an ``if __name__ == "__main__"``
guard). ``import M`` and ``from pkg import M`` take all of M. That keeps an
unrelated CLI subcommand of a helper module (setup_toolchain's archive-inputs)
out of the graph.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
ALL = "*"


def loaded() -> dict[str, str]:
    """Repo file (posix, relative) -> module name, for every loaded repo module."""
    found = {}
    for name, module in list(sys.modules.items()):
        if name == "__main__":  # this probe
            continue
        path = Path(getattr(module, "__file__", None) or "").resolve()
        if path.is_file() and path.is_relative_to(ROOT) and "site-packages" not in path.parts:
            found[path.relative_to(ROOT).as_posix()] = name
    return found


def is_repo_module(name: str | None) -> bool:
    top = (name or "").partition(".")[0]
    return not top or (ROOT / top).is_dir() or (ROOT / f"{top}.py").is_file()


def load(name: str) -> bool:
    """Import ``name``; only a missing third-party or platform module is tolerated."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if is_repo_module(error.name):
            raise
        return False  # e.g. _winapi or msvcrt on Linux
    return True


def is_main_guard(node: ast.stmt) -> bool:
    test = getattr(node, "test", None)
    return (isinstance(node, ast.If) and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "__name__")


def reachable(tree: ast.Module, wanted: set[str]) -> list[ast.stmt]:
    """The top-level nodes that can run when a caller uses ``wanted``."""
    defs = {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    live = [node for node in tree.body if node.__class__ not in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            and not is_main_guard(node)]
    queue = list(defs) if ALL in wanted else [name for name in wanted if name in defs]
    seen: set[str] = set()
    for node in live:
        queue += [ref.id for ref in ast.walk(node) if isinstance(ref, ast.Name)]
    while queue:
        name = queue.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        live.append(defs[name])
        queue += [ref.id for ref in ast.walk(defs[name]) if isinstance(ref, ast.Name)]
    return live


def requests(node: ast.AST, package: str) -> dict[str, set[str]]:
    """Module name -> the names an import statement takes from it."""
    if isinstance(node, ast.Import):
        return {alias.name: {ALL} for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return {}
    base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else str(node.module)
    taken = {base: {alias.name for alias in node.names}}
    if load(base) and hasattr(sys.modules[base], "__path__"):
        for alias in node.names:  # from pkg import submodule
            if alias.name != "*" and importlib.util.find_spec(f"{base}.{alias.name}") is not None:
                taken[f"{base}.{alias.name}"] = {ALL}
    return taken


def main() -> None:
    sys.path.insert(0, str(ROOT))
    entries, follow = json.loads(sys.argv[1]), tuple(json.loads(sys.argv[2]))
    wanted: dict[str, set[str]] = {}
    for entry in entries:
        importlib.import_module(entry)
        wanted[entry] = {ALL}
    done: dict[str, set[str]] = {}
    while True:
        modules = {name: path for path, name in loaded().items() if path.startswith(follow)}
        pending = sorted(name for name in modules if wanted.get(name, set()) - done.get(name, set()))
        if not pending:
            # Followed modules no followed import asked for came in through pm/, hermes_cli or
            # another unfollowed module: assume all of them can run.
            unasked = modules.keys() - wanted.keys()
            if not unasked:
                break
            wanted.update((name, {ALL}) for name in unasked)
            continue
        for name in pending:
            done[name] = set(wanted[name])
            package = sys.modules[name].__package__ or ""
            tree = ast.parse((ROOT / modules[name]).read_text(encoding="utf-8"))
            for node in reachable(tree, wanted[name]):
                for child in ast.walk(node):
                    for module, names in requests(child, package).items():
                        if load(module):
                            wanted.setdefault(module, set()).update(names)
    print(json.dumps(sorted(loaded())))


if __name__ == "__main__":
    main()
