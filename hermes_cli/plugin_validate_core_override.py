"""Admission lint: a catalog plugin must not rebind Hermes core at runtime.

Plugins extend Hermes through public surfaces (hooks, middleware, provider profiles, Desktop SDK
slots). A plugin that replaces a core function, method or module attribute in place — assigning
``AIAgent._replace_primary_openai_client``, ``tui_gateway.server.handle_request``, a
``gateway.run_turn`` helper, a private catalog dict — collides with every other plugin that patches
the same seam and breaks on any core release. This static pass finds those rebinds in the plugin's
Python (the import closure of ``__init__.py`` and ``dashboard/``; tests, benchmarks and scripts
excluded). Like the Desktop lint it is a review tripwire, not a sandbox.

A value is "core" when it is a Hermes module or an attribute chain off one: a name bound by an
absolute import of a Hermes top-level package, ``importlib.import_module(...)`` / ``sys.modules``
lookups, ``getattr(core, ...)``, iteration over core values, and calls to plugin functions that
return a core value. Rebinds are attribute assignment / deletion, ``setattr`` / ``delattr``,
``sys.modules[...] = ...``, ``mock.patch``, and passing a core value to a plugin helper that
``setattr``s its parameter.

In-place mutation of a core object's PRIVATE state is a rebind too (``get_plugin_manager()._hooks[
"x"].append(fn)``, ``mgr._middleware.clear()``, ``mgr._hooks[:] = []``). For these sinks a value
returned by a core function also counts as core, but only a receiver chain with a private segment
is judged, so calling a public method on a core object (``registry.add(...)``) stays clean.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_SKIP_DIRS = frozenset({"tests", "test", "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"})


@lru_cache(maxsize=1)
def _hermes_top_level() -> frozenset:
    """Top-level importable names shipped by this Hermes checkout (packages and root modules)."""
    root = Path(__file__).resolve().parents[1]
    names = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    names |= {p.stem for p in root.glob("*.py")}
    # ``hermes_plugins.*`` is where the loader imports plugins (bundled platform adapters included):
    # rebinding another plugin's module is the same collision as rebinding core.
    return frozenset((names | {"hermes_plugins"}) - {"setup", "conftest"})


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _python_files(plugin_dir: Path) -> Iterable[tuple[Path, Path]]:
    for path in sorted(plugin_dir.rglob("*.py")):
        rel = path.relative_to(plugin_dir)
        if _SKIP_DIRS.intersection(rel.parts[:-1]) or _is_test_file(rel):
            continue
        yield path, rel


def _runtime_files(plugin_dir: Path, files: dict[Path, ast.AST], local: set[str]) -> set[Path]:
    """The import closure of what Hermes loads: ``__init__.py`` (``register``) and ``dashboard/*.py``
    (dashboard plugin API). Benchmarks, CI and smoke scripts shipped beside them never run inside
    Hermes, so a stub they install into ``sys.modules`` is not a runtime override."""
    by_module: dict[str, set[Path]] = {}
    for rel in files:
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):
            by_module.setdefault(".".join(parts[start:]), set()).add(rel)

    def resolve(rel: Path, node: ast.AST) -> set[Path]:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = rel.parent.parts[: len(rel.parent.parts) - (node.level - 1)] if node.level > 1 else rel.parent.parts
                prefix = ".".join(base + tuple((node.module or "").split("."))).strip(".")
                names = [prefix] + [f"{prefix}.{a.name}".strip(".") for a in node.names]
            elif node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        found: set[Path] = set()
        for name in names:
            if name.split(".")[0] in local or isinstance(node, ast.ImportFrom) and node.level:
                found |= by_module.get(name, set())
        return found

    roots = {rel for rel in files if rel == Path("__init__.py") or (len(rel.parts) == 2 and rel.parts[0] == "dashboard")}
    seen: set[Path] = set()
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for node in ast.walk(files[rel]):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                stack.extend(resolve(rel, node) - seen)
    return seen


def _local_names(plugin_dir: Path) -> set[str]:
    """Module/package names the plugin ships itself: an import of these is local, not core."""
    names = set()
    for path in plugin_dir.rglob("*"):
        if _SKIP_DIRS.intersection(path.relative_to(plugin_dir).parts):
            continue
        if path.suffix == ".py":
            names.add(path.stem)
        elif path.is_dir():
            names.add(path.name)
    return names


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(node: ast.Call) -> str:
    return _dotted(node.func)


# Constructors whose result is a plain container, never the plugin's own helper object (unless
# the module rebinds the name: ``from .factory import make_recorder as set``).
_CONTAINER_CTORS = frozenset({"set", "dict", "list", "frozenset"})
_CONTAINER_DISPLAYS = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
# Names / attributes that reach a namespace dynamically (``globals()["rec"] = ...``, frame locals,
# ``builtins.set = ...``): a plugin using any of them gets no receiver-name exemption at all.
_NAMESPACE_ESCAPES = frozenset({
    "globals", "locals", "vars", "exec", "eval", "builtins", "__builtins__", "__dict__", "__setattr__",
    "__globals__", "_getframe", "currentframe", "f_locals", "f_globals", "ctypes"})
_TYPE_PARAMS = tuple(getattr(ast, n) for n in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, n))

# Methods that change a list / dict / set in place.
_MUTATORS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "update", "setdefault", "add", "discard", "popitem"})


def _render(node: ast.AST) -> str:
    """``get_plugin_manager()._hooks[...]``: a receiver chain as the finding names it."""
    if isinstance(node, ast.Attribute):
        return f"{_render(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_render(node.value)}[...]"
    if isinstance(node, ast.Call):
        return f"{_render(node.func)}()"
    return node.id if isinstance(node, ast.Name) else "<core>"


def _has_mutator_call(node: ast.AST) -> bool:
    """Whether the chain already holds a mutator call (``m._hooks.setdefault(k, []).append``):
    that inner call is the finding, the outer one would repeat it."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return False


def _has_private_segment(node: ast.AST) -> bool:
    """Whether an attribute in the chain (``mgr._hooks[...]``, ``m.__dict__``) is private."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return False


class _Analysis:
    def __init__(self, core_roots: frozenset, local: set[str]):
        self.core_roots = core_roots - local
        self.core_returners: set[str] = set()
        self.param_patchers: dict[str, set[int]] = {}
        self.param_mutators: dict[str, set[int]] = {}  # helpers mutating a parameter's private state
        self.aliases: dict[str, str] = {}  # ``from .compat import server as gateway_server``
        self.constants: dict[str, str] = {}  # module-level ``NAME = "module.path"``
        self.rebound_ctors: frozenset = frozenset()  # container constructors the current module rebinds
        self.containers: frozenset = frozenset()  # names every binding in the current module makes a container

    def _func_key(self, call: ast.Call) -> str:
        name = _call_name(call).split(".")[-1]
        return self.aliases.get(name, name)

    def is_container(self, node: ast.AST) -> bool:
        """A set / dict / list display or constructor: its ``add`` is never the plugin's helper."""
        if isinstance(node, _CONTAINER_DISPLAYS):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in _CONTAINER_CTORS and node.func.id not in self.rebound_ctors
        return False

    def helper_indices(self, table: dict[str, set[int]], call: ast.Call) -> set[int]:
        """Argument positions a call hands to a patching / mutating helper, matched by name. A
        method called on a container display / constructor, or on a name that EVERY binding in
        the module makes a container (``seen = set()`` and nothing else), is the container's own,
        not a same-named plugin helper."""
        func = call.func
        if isinstance(func, ast.Attribute) and (
                self.is_container(func.value) or isinstance(func.value, ast.Name) and func.value.id in self.containers):
            return set()
        return table.get(self._func_key(call), set())

    # -- what counts as a core value -------------------------------------------------------------
    def _core_module_name(self, name: object) -> bool:
        return isinstance(name, str) and name.split(".")[0] in self.core_roots

    def is_core(self, node: ast.AST, tainted: set[str], calls: bool = False) -> bool:
        """``calls``: a call to a core function returns core too (the private-state sinks only)."""
        if isinstance(node, ast.Name):
            return node.id in tainted
        if isinstance(node, ast.Attribute):
            return self.is_core(node.value, tainted, calls)
        if isinstance(node, ast.Subscript):
            if _dotted(node.value) == "sys.modules":
                return self._lookup_is_core(node.slice)
            return self.is_core(node.value, tainted, calls)
        if isinstance(node, ast.Call):
            if calls and self.is_core(node.func, tainted, calls):
                return True
            name = _call_name(node)
            if name in ("importlib.import_module", "import_module", "__import__") and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "sys.modules.get" and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "getattr" and node.args:
                return self.is_core(node.args[0], tainted)
            return self._func_key(node) in self.core_returners
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self.is_core(e, tainted, calls) for e in node.elts)
        return False

    def _lookup_is_core(self, arg: ast.AST) -> bool:
        """Whether a module-name expression names Hermes: a literal, a module-level string
        constant, or an f-string whose literal head names it (``f"hermes_plugins.{p}.adapter"``).
        A name computed some other way (the plugin locating its own package) is not judged."""
        if isinstance(arg, ast.Name):
            arg = ast.Constant(self.constants.get(arg.id))
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            head = str(arg.values[0].value)
            return "." in head and self._core_module_name(head)
        return isinstance(arg, ast.Constant) and self._core_module_name(arg.value)

    def _private_core_state(self, node: ast.AST, derived: set[str]) -> bool:
        return _has_private_segment(node) and self.is_core(node, derived, calls=True)

    def private_mutations(self, node: ast.AST, derived: set[str]) -> list[str]:
        """In-place writes to a core object's private state: a mutator call, or an item / slice
        assignment or delete, on a private receiver chain."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _MUTATORS \
                    and self._private_core_state(func.value, derived) and not _has_mutator_call(func.value):
                return [f"{_render(func)}(...)"]
            name = _call_name(node)
            return [f"{name}({_dotted(node.args[i]) or '<core>'}, ...)"
                    for i in sorted(self.helper_indices(self.param_mutators, node))
                    if i < len(node.args) and self.is_core(node.args[i], derived, calls=True)]
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else \
            [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        return [_render(t) for target in targets
                for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])
                if isinstance(t, ast.Subscript) and self._private_core_state(t, derived)]

    # -- per-scope taint --------------------------------------------------------------------------
    def scope_taint(self, nodes: list[ast.AST], inherited: set[str], calls: bool = False) -> set[str]:
        tainted = set(inherited)
        for _ in range(4):
            before = len(tainted)
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._core_module_name(alias.name):
                            tainted.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and self._core_module_name(node.module):
                    tainted.update(a.asname or a.name for a in node.names)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    if self.is_core(node.value, tainted, calls):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        tainted.update(_bound_names(targets))
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    if self.is_core(node.iter, tainted, calls):
                        tainted.update(_bound_names([node.target]))
                elif isinstance(node, ast.NamedExpr) and self.is_core(node.value, tainted, calls):
                    tainted.add(node.target.id)
            if len(tainted) == before:
                break
        return tainted

    # -- sinks ------------------------------------------------------------------------------------
    def rebinds(self, node: ast.AST, tainted: set[str], derived: set[str] | None = None) -> list[str]:
        """``derived``: the scope's taint with core-function returns included (``scope_taint(calls=True)``)."""
        hits: list[str] = []
        targets: Sequence[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                if isinstance(t, ast.Attribute) and self.is_core(t.value, tainted):
                    hits.append(_dotted(t) or t.attr)
                elif isinstance(t, ast.Subscript) and _dotted(t.value) == "sys.modules":
                    if isinstance(t.slice, ast.Constant) and self._core_module_name(t.slice.value):
                        hits.append(f"sys.modules[{t.slice.value!r}]")
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                        and self.is_core(t.value.value, tainted):
                    hits.append(f"{_dotted(t.value) or t.value.attr}[...]")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ("setattr", "delattr") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"{name}({_dotted(node.args[0]) or '<core>'}, ...)")
            elif name.endswith("patch.object") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"patch.object({_dotted(node.args[0])}, ...)")
            elif name.split(".")[-1] == "patch" and node.args and isinstance(node.args[0], ast.Constant) \
                    and self._core_module_name(node.args[0].value):
                hits.append(f"patch({node.args[0].value!r})")
            else:
                for index in sorted(self.helper_indices(self.param_patchers, node)):
                    if index < len(node.args) and self.is_core(node.args[index], tainted):
                        hits.append(f"{name}({_dotted(node.args[index]) or '<core>'}, ...)")
        return hits + [h for h in self.private_mutations(node, tainted if derived is None else derived)
                       if h not in hits]


def _def_time_parts(func: ast.AST) -> list[ast.AST]:
    """What a ``def`` statement evaluates in the enclosing scope: decorators, defaults, annotations."""
    args = func.args
    parts = [*func.decorator_list, *args.defaults, *(d for d in args.kw_defaults if d is not None)]
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    parts += [a.annotation for a in params if a is not None and a.annotation is not None]
    return parts + ([func.returns] if func.returns is not None else [])


def _walk_scope(stmt: ast.AST, class_bodies: bool = True):
    """Nodes of one scope. A nested ``def`` contributes only what runs where it is defined
    (decorators, defaults, annotations): its body is its own scope. A class body runs in place,
    but its names are class attributes (``class_bodies=False`` stops at the class statement)."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(_def_time_parts(node))
        elif isinstance(node, ast.ClassDef) and not class_bodies:
            stack.extend([*node.decorator_list, *node.bases, *node.keywords])
        else:
            stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, ast.Lambda))


def _bound_names(targets: Iterable[ast.AST]) -> set[str]:
    names = set()
    for t in targets:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            names |= _bound_names(t.elts)
    return names


def _local_bindings(nodes: list[ast.AST]) -> set[str]:
    """Every name a scope binds (any binding form): a local shadows the enclosing scope's name.
    A comprehension's target is the comprehension's own, so it shadows nothing here."""
    comp_targets = {id(n) for node in nodes if isinstance(node, ast.comprehension)
                    for n in ast.walk(node.target)}
    names = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
             and id(n) not in comp_targets}
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)) and node.name:
            names.add(node.name)
    return names


class _Scope:
    """One function scope: its own nodes, the names it binds, and what it declares
    ``global`` / ``nonlocal`` (bindings that land in the module or the enclosing function)."""

    def __init__(self, func: ast.AST, is_method: bool, parent: ast.AST | None):
        self.func, self.is_method, self.parent = func, is_method, parent
        self.nodes = [n for stmt in func.body for n in _walk_scope(stmt)]
        args = func.args
        self.params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs] + \
            [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        self.globals = {n for node in self.nodes if isinstance(node, ast.Global) for n in node.names}
        self.nonlocals = {n for node in self.nodes if isinstance(node, ast.Nonlocal) for n in node.names}
        own = [n for stmt in func.body for n in _walk_scope(stmt, class_bodies=False)]
        self.locals = (_local_bindings(own) | set(self.params)) - self.globals - self.nonlocals
        self.children: list[_Scope] = []


def _scopes(tree: ast.AST) -> list[_Scope]:
    """Every function in the module, nested ones included, parents first."""
    found: list[_Scope] = []

    def visit(node, in_class, parent):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                static = any(_dotted(d) == "staticmethod" for d in child.decorator_list)
                found.append(_Scope(child, in_class and not static, parent))
                visit(child, False, child)
            elif isinstance(child, ast.ClassDef):
                visit(child, True, parent)
            else:
                visit(child, in_class, parent)
    visit(tree, False, None)
    by_func = {scope.func: scope for scope in found}
    for scope in found:
        if scope.parent is not None:
            by_func[scope.parent].children.append(scope)
    return found


class _Module:
    """One parsed plugin file: its module-scope nodes and function scopes, built once per run."""

    def __init__(self, rel: Path, tree: ast.AST):
        self.rel, self.tree = rel, tree
        self.nodes = [n for stmt in tree.body for n in _walk_scope(stmt)]
        self.scopes = _scopes(tree)
        self.declares = any(s.globals or s.nonlocals for s in self.scopes)
        self.rebound_ctors: frozenset = _CONTAINER_CTORS  # set by ``_vet_containers``
        self.containers: frozenset = frozenset()

    def taints(self, analysis: _Analysis, calls: bool = False) -> tuple[set[str], dict]:
        """Module taint and each function's taint. A closure inherits its enclosing function's
        taint minus the names it binds itself; a core value bound to a ``global`` name taints
        the module and one bound to a ``nonlocal`` name taints the enclosing function."""
        to_module: set[str] = set()
        to_func: dict = {}
        for _ in range(4):
            module = analysis.scope_taint(self.nodes, to_module, calls)
            taints: dict = {}
            for scope in self.scopes:
                outer = taints[scope.parent] if scope.parent is not None else module
                inherited = (outer - scope.locals - scope.globals) | (module & scope.globals) \
                    | to_func.get(scope.func, set())
                taints[scope.func] = analysis.scope_taint(scope.nodes, inherited, calls)
            if not self.declares:
                return module, taints
            new_module = to_module | {n for s in self.scopes for n in taints[s.func] & s.globals}
            new_func = {k: set(v) for k, v in to_func.items()}
            for scope in self.scopes:
                if scope.parent is not None and taints[scope.func] & scope.nonlocals:
                    new_func.setdefault(scope.parent, set()).update(taints[scope.func] & scope.nonlocals)
            if new_module == to_module and new_func == to_func:
                break
            to_module, to_func = new_module, new_func
        return module, taints


def _closure_nodes(scope: _Scope, name: str) -> list[ast.AST]:
    """A scope's nodes plus those of nested functions that see ``name`` (neither rebind it nor
    declare it ``global``)."""
    nodes = list(scope.nodes)
    for child in scope.children:
        if name not in child.locals and name not in child.globals:
            nodes += _closure_nodes(child, name)
    return nodes


def _bindings(tree: ast.AST) -> list[tuple[str, ast.AST | None]]:
    """Every name binding in the module, any scope and any form, with the bound value when the
    binding is a plain ``name = value`` / ``name: T = value`` / ``(name := value)`` (else None)."""
    found: list[tuple[str, ast.AST | None]] = []
    plain: set[int] = set()
    for node in ast.walk(tree):
        pairs = [(t, node.value) for t in node.targets] if isinstance(node, ast.Assign) else \
            [(node.target, node.value)] if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value else []
        for target, value in pairs:
            if isinstance(target, ast.Name):
                plain.add(id(target))
                found.append((target.id, value))
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load) and id(node) not in plain:
            name = node.id  # unpacking, for / with / comprehension targets, augassign, del, bare annotation
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, ast.alias):
            name = (node.asname or node.name).split(".")[0]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler,
                               ast.MatchAs, ast.MatchStar, *_TYPE_PARAMS)):
            name = node.name
        elif isinstance(node, ast.MatchMapping):
            name = node.rest
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            found += [(n, None) for n in node.names]
        if name:
            found.append((name, None))
    return found


def _namespace_escapes(modules: list[_Module]) -> tuple[bool, set[str]]:
    """Whether the plugin reaches a namespace dynamically, and every attribute name it writes
    (``mod.rec = ...``, ``setattr(mod, "rec", ...)``): either can rebind a module name from outside
    its own bindings."""
    written: set[str] = set()
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Name) and node.id in _NAMESPACE_ESCAPES \
                    or isinstance(node, ast.Attribute) and node.attr in _NAMESPACE_ESCAPES \
                    or isinstance(node, ast.alias) and node.name.split(".")[0] in _NAMESPACE_ESCAPES:
                return True, written
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                written.add(node.attr)
            elif isinstance(node, ast.Call) and _call_name(node) in ("setattr", "delattr") and len(node.args) > 1:
                if not isinstance(node.args[1], ast.Constant):
                    return True, written
                written.add(str(node.args[1].value))
    return False, written


def _vet_containers(analysis: _Analysis, modules: list[_Module]) -> None:
    """Per module: the constructors it rebinds, and the receiver names exempt from helper matching.
    A name is exempt only if EVERY binding of it anywhere in the module is a container display or
    an un-rebound constructor call, the module has no ``import *``, and no plugin code writes it as
    an attribute or reaches namespaces dynamically."""
    dynamic, written = _namespace_escapes(modules)
    for module in modules:
        bindings = _bindings(module.tree)
        names = {name for name, _value in bindings}
        star = dynamic or "*" in names
        module.rebound_ctors = _CONTAINER_CTORS if star else _CONTAINER_CTORS & (names | written)
        if star:
            continue
        analysis.rebound_ctors = module.rebound_ctors
        bad = {name for name, value in bindings if value is None or not analysis.is_container(value)}
        module.containers = frozenset(names - bad - written)


def _enter_module(analysis: _Analysis, module: _Module) -> None:
    analysis.rebound_ctors, analysis.containers = module.rebound_ctors, module.containers


def _summarise_helpers(analysis: _Analysis, modules: list[_Module]) -> None:
    """Fixpoint over the whole plugin: which helpers return core values, rebind a parameter, or
    mutate a parameter's private state."""
    for _ in range(4):
        before = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()),
                  sum(len(v) for v in analysis.param_mutators.values()))
        for module in modules:
            _enter_module(analysis, module)
            _module_taint, taints = module.taints(analysis)
            for scope in module.scopes:
                func = scope.func
                if any(isinstance(n, ast.Return) and n.value is not None and analysis.is_core(n.value, taints[func])
                       for n in scope.nodes):
                    analysis.core_returners.add(func.name)
                params = [a.arg for a in func.args.posonlyargs + func.args.args]
                # A method is called as ``obj.bind(module, ...)``: its call-site args start after ``self``.
                offset = 1 if scope.is_method else 0
                for index, param in enumerate(params[offset:]):
                    nodes = _closure_nodes(scope, param) if scope.children else scope.nodes
                    if any(analysis.rebinds(n, {param}, set()) for n in nodes):
                        analysis.param_patchers.setdefault(func.name, set()).add(index)
                    if any(analysis.private_mutations(n, {param}) for n in nodes):
                        analysis.param_mutators.setdefault(func.name, set()).add(index)
        after = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()),
                 sum(len(v) for v in analysis.param_mutators.values()))
        if after == before:
            break


def core_override_findings(plugin_dir: Path) -> list[str]:
    """``["<target> (<rel>:<line>)", ...]`` for every runtime rebind of Hermes core in the plugin."""
    plugin_dir = Path(plugin_dir)
    parsed: dict[Path, ast.AST] = {}
    for path, rel in _python_files(plugin_dir):
        try:
            parsed[rel] = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (SyntaxError, ValueError):
            continue
    local = _local_names(plugin_dir)
    runtime = _runtime_files(plugin_dir, parsed, local)
    modules = [_Module(rel, tree) for rel, tree in parsed.items() if rel in runtime]
    analysis = _Analysis(_hermes_top_level(), local)
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] in local):
                analysis.aliases.update({a.asname: a.name for a in node.names if a.asname})
        for node in module.tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                analysis.constants.update({n: node.value.value for n in _bound_names(node.targets)})

    _vet_containers(analysis, modules)
    _summarise_helpers(analysis, modules)

    findings: list[str] = []
    for module in modules:
        _enter_module(analysis, module)
        module_taint, taints = module.taints(analysis)
        module_derived, derived = module.taints(analysis, calls=True)
        scopes = [(module.nodes, module_taint, module_derived)]
        scopes += [(s.nodes, taints[s.func], derived[s.func]) for s in module.scopes]
        for nodes, taint, derived_taint in scopes:
            for node in nodes:
                for target in analysis.rebinds(node, taint, derived_taint):
                    findings.append((module.rel.as_posix(), getattr(node, "lineno", 0), target))
    return [f"{target} ({rel}:{line})" for rel, line, target in sorted(set(findings))]


def check_core_override(report, plugin_dir: Path) -> None:
    """Fail the report when the plugin's Python rebinds Hermes core at runtime."""
    hits = core_override_findings(plugin_dir)
    report.add(
        "no core override", not hits,
        "rebinds Hermes core at runtime (use a public hook, middleware or provider profile): "
        + "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "no runtime rebinds of Hermes core",
    )
