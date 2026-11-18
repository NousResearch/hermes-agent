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

Four dynamic imports are modelled. ``model_tools`` calls ``tools.registry.discover_builtin_tools()``
while it imports, which imports every built-in tool module that registers a tool at module
level: when ``model_tools`` is in the graph, so is each module discovery would select in the
measured tree (``hosted_graph_tools``), with the package ``__init__`` files importing it runs. And a forwarder declared at import time with
``agent.lazy_forward`` (``switch_model = forward("agent.agent_runtime_helpers", ...)`` in the
``AIAgent`` class body) imports its module on every call, so the declaration is an edge. A third:
``persistence.sqlite``'s facade imports every public module under its package on first use (its
``__getattr__``) and those import the private ones inside functions, so when the facade is in the
graph, so is every module of the package.
A fourth, by-name loaders: a module on the graph that calls ``importlib.import_module`` or ``__import__``
with a name it computes (``f"agent.transports.{name}"`` over ``_TRANSPORT_MODULES``, a table of module
names, ``f"{__name__}.{module}"``) loads each module named by a string the module spells, so each is
in the graph (``_loaded_by_name``).
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
# The built-in storage backend's facade (``persistence/sqlite/__init__.py``): its public names import the
# package's modules dynamically (``_facade_modules``). A hosted process reaches it through
# ``persistence_assembly`` and the readiness probe, so the modules it loads are on the hosted path.
_SQLITE_PACKAGE = "persistence/sqlite/"
_SQLITE_FACADE = _SQLITE_PACKAGE + "__init__.py"
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
        discovered = facade_loaded = False
        while frontier:
            batch = sorted(set(frontier) - seen)
            seen.update(batch)
            frontier = []
            texts = _read_all(self.repo, self.tree, batch)
            for path, text in texts.items():
                frontier += [f for f in self._found(_imported(path, text)) if f not in seen]
                frontier += [f for f in self._found(n for c in _loaded_by_name(path, text)
                                                    if self.module_file(c) for n in _prefixes(c)) if f not in seen]
            if not discovered and DISCOVERING_MODULE in seen:
                discovered = True
                frontier += [f for f in self._discovered() if f not in seen]
            if not facade_loaded and _SQLITE_FACADE in seen:
                facade_loaded = True
                frontier += [f for f in self._facade_modules() if f not in seen]
        return frozenset(seen)

    def _facade_modules(self) -> list[str]:
        """Every module of the ``persistence.sqlite`` package, with the packages on the way. Its facade's
        first public-name use imports every public module under the package (``__getattr__`` walks it
        with ``rglob``) and ``_backend``, and those import most private modules lazily, inside the
        functions a serving store runs (a census of one local store's open, write, read and readiness
        probe loaded 116 of them). The backend is one package with one owner, so all of it is judged
        rather than a census that drifts as its functions move imports around."""
        names = [path[:-3].replace("/", ".") for path in self.files
                 if path.startswith(_SQLITE_PACKAGE) and path.endswith(".py")]
        return sorted(set(self._found(n for m in names for n in _prefixes(m))))

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


_IMPORTERS = {"import_module", "__import__"}


def _module_name(path: str) -> str:
    stem = path[:-3]
    return stem.removesuffix("/__init__").replace("/", ".")


def _name_prefix(arg: ast.expr, path: str, constants: dict[str, str]) -> str | None:
    """The fixed part of a module name built at run time: ``f"agent.transports.{name}"`` gives
    ``agent.transports.``, ``f"{__name__}.{module}"`` this module's name plus a dot (a module-level
    string constant in place of ``__name__`` likewise), a bare variable the empty prefix; ``None``
    when the name is a literal or not of that shape."""
    if isinstance(arg, ast.Name):
        return ""
    if not isinstance(arg, ast.JoinedStr) or not arg.values:
        return None
    *head, last = arg.values
    if not isinstance(last, ast.FormattedValue):
        return None
    prefix = ""
    for part in head:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            prefix += part.value
        elif (isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
              and part.value.id in ("__name__", "__package__")):
            name = _module_name(path)
            prefix += name if part.value.id == "__name__" or path.endswith("/__init__.py") else name.rpartition(".")[0]
        elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name) and part.value.id in constants:
            prefix += constants[part.value.id]
        else:
            return None
    return prefix


def _loaded_by_name(path: str, text: str) -> Iterator[str]:
    """Modules a by-name loader in *path* can import: ``importlib.import_module(...)`` or
    ``__import__(...)`` with a computed name, anywhere in the module (a registry's discovery runs
    inside a function on first use), naming a module after a string the module itself spells. The
    transport registry (``for name in _TRANSPORT_MODULES: import_module(f"agent.transports.{name}")``),
    a table of module names a loader walks (``pm.registry``, ``agent.secret_sources.registry``), an
    adapter table a helper resolves (``gateway.run``'s ``_BUILTIN_ADAPTERS``) and a lazy facade's
    export map (``import_module(f"{__name__}.{module}")``) all have that shape. Every candidate is
    taken (the caller keeps the ones that are modules of the tree), so a name added to such a
    table is on the graph without a census; a name read from the filesystem or config is not
    modelled (no string in the module names it)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return
    constants = {node.targets[0].id: node.value.value for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    prefixes: set[str] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        func = call.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in _IMPORTERS and (prefix := _name_prefix(call.args[0], path, constants)) is not None:
            if prefix.startswith("."):  # import_module(f".{name}", __package__): relative to this package
                package = _module_name(path) if path.endswith("/__init__.py") else _module_name(path).rpartition(".")[0]
                prefix = package + prefix
            prefixes.add(prefix)
    if not prefixes:
        return
    strings = {_ref_module(node.value) for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for prefix in prefixes:
        # a bare variable names a whole module (``"pm.store"``), never a top-level one: a module
        # mentioning ``"cli"`` as a platform name does not load ``cli.py``; a prefix takes one segment
        yield from (prefix + s for s in strings if s and ("." in s if not prefix else "." not in s))


def _ref_module(value: str) -> str:
    """The module a string names, if it is spelled like a dotted module name (``"pkg.mod"``,
    ``"pkg.mod:attr"`` for a ``module:attr`` reference, ``"mod"``); else ``""``."""
    head = value.split(":", 1)[0] if value.count(":") == 1 else value
    return head if all(part.isidentifier() for part in head.split(".")) else ""


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
