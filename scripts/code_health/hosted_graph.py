"""The hosted entry point's import graph: which modules a hosted process loads.

``reachable(repo, tree)`` walks the import statements that run at import time from the roots in
``config.HOSTED_ENTRY``, over one tree. Importing ``a.b.c`` loads ``a``, ``a.b`` and ``a.b.c``,
and ``from a import b`` loads ``a.b`` when it is a module, so each package ``__init__`` on the
way is in the graph too, a root's own included.

Import time is every statement a module body runs: through ``if``/``else``, ``try``, ``with``,
``for``/``while`` loops, ``match`` cases and class bodies (nested classes included), excluding
``if TYPE_CHECKING:``. Imports inside functions are not followed: they reach almost the whole
tree (CLI setup wizards, installers) through code a hosted process never calls. A module the
hosted process loads lazily is listed as a root instead.

Two dynamic imports are modelled. ``model_tools`` calls ``tools.registry.discover_builtin_tools()``
while it imports, which imports every built-in tool module that registers a tool at module
level: when ``model_tools`` is in the graph, so is each module discovery would select in the
measured tree (``hosted_graph_tools``), with the package ``__init__`` files importing it runs. And a forwarder declared at import time with
``agent.lazy_forward`` (``switch_model = forward("agent.agent_runtime_helpers", ...)`` in the
``AIAgent`` class body) imports its module on every call, so the declaration is an edge.
Plugins loaded by path (``plugins/*``, ``$HERMES_HOME/plugins``) are not Python imports and are
not in the graph.
"""

from __future__ import annotations

import ast
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path

from scripts.code_health import config, gitio
from scripts.code_health.hosted_graph_tools import DISCOVERING_MODULE, discovered_tool_modules

_LAZY_FORWARD = "agent.lazy_forward"
_FORWARDER_FACTORIES = {"forward", "forward_static", "lazy_attr"}


def _read_all(repo: Path, tree: str | None, paths: list[str]) -> dict[str, str]:
    if tree is None:
        return {p: t for p in paths if (t := gitio.read_file(repo, None, p)) is not None}
    # One `git cat-file --batch` per frontier: a `git show` per module costs seconds.
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

    def _found(self, names: Iterator[str]) -> Iterator[str]:
        for name in names:
            found = self.module_file(name)
            if found:
                yield found

    def walk(self, roots: tuple[str, ...]) -> frozenset[str]:
        seen: set[str] = set()
        frontier = list(self._found(n for r in roots for n in _prefixes(r)))
        discovered = False
        while frontier:
            batch = sorted(set(frontier) - seen)
            seen.update(batch)
            frontier = []
            texts = _read_all(self.repo, self.tree, batch)
            for path, text in texts.items():
                frontier += [f for f in self._found(_imported(path, text)) if f not in seen]
            if not discovered and DISCOVERING_MODULE in seen:
                discovered = True
                frontier += [f for f in self._discovered() if f not in seen]
        return frozenset(seen)

    def _discovered(self) -> list[str]:
        """The built-in tool modules ``discover_builtin_tools()`` imports, in this tree, with every
        package on the way: ``importlib.import_module("tools.pkg.tool")`` runs ``tools/pkg/__init__.py``
        (and whatever it imports) first."""
        candidates = sorted(p for p in self.files if p.startswith("tools/"))
        modules = discovered_tool_modules(self.files, _read_all(self.repo, self.tree, candidates))
        return sorted(set(self._found(n for m in modules for n in _prefixes(m[:-3].replace("/", ".")))))


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
    """Every statement *body* runs: into each compound statement's blocks (loops, ``match``
    cases, ``try`` handlers, class bodies), never into a function body or ``if TYPE_CHECKING:``."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and _type_checking(node):
            yield from _import_time(node.orelse)
            continue
        yield node
        for _, value in ast.iter_fields(node):
            for item in value if isinstance(value, list) else []:
                if isinstance(item, ast.stmt):
                    yield from _import_time([item])
                elif isinstance(item, (ast.ExceptHandler, ast.match_case)):
                    yield from _import_time(item.body)


def _forwarders(tree: ast.Module) -> set[str]:
    """Local names bound to ``agent.lazy_forward``'s forwarder factories by a module-level import."""
    return {alias.asname or alias.name for node in _import_time(tree.body)
            if isinstance(node, ast.ImportFrom) and node.module == _LAZY_FORWARD
            for alias in node.names if alias.name in _FORWARDER_FACTORIES}


def _forwarded(node: ast.stmt, factories: set[str]) -> Iterator[str]:
    """``forward("agent.x", "name")`` evaluated by *node* itself (a class-body forwarder): the
    method it builds imports ``agent.x`` on every call, so the module is loaded by any process
    that calls the method; the graph treats the declaration as the import."""
    for _, value in ast.iter_fields(node):
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, ast.expr):
                continue
            for call in ast.walk(item):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id in factories and call.args
                        and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                    yield from _prefixes(call.args[0].value)


def _imported(path: str, text: str) -> Iterator[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return  # the measurer reports it; an unparseable module has no edges to follow
    factories = _forwarders(tree)
    for node in _import_time(tree.body):
        if factories:
            yield from _forwarded(node, factories)
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


def _function_import_names(tree: ast.Module) -> Iterator[str]:
    """Absolute module names imported inside a function body (relative ones are skipped: the
    audited roots use absolute imports)."""
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield from _prefixes(alias.name)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                yield from _prefixes(node.module)
                yield from (f"{node.module}.{alias.name}" for alias in node.names)


def function_imports(repo: Path, tree: str | None, root: str) -> set[str]:
    """Paths of the in-repo modules *root* imports inside a function: what the graph does not
    follow. ``config.LAZY_IMPORTS_AUDITED`` roots must place each in ``HOSTED_ENTRY`` or
    ``config.OFF_HOSTED_PATH``."""
    graph = _Graph(repo, tree)
    path = graph.module_file(root)
    text = gitio.read_file(repo, tree, path) if path else None
    if text is None:
        return set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        parsed = ast.parse(text)
    return set(graph._found(_function_import_names(parsed))) - {path}
