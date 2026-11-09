"""Lexical bindings for HX015: which binding a name read or written in any scope refers to.

A name mutated inside a function is the module's only when no scope between them binds it: a
parameter, a local assignment (without ``global``), a nested function's local, a comprehension
variable or a class-body name each shadow the module binding there. Python decides this per
scope from every binding in that scope, so the scopes are collected in one pass before any name
is resolved.

It also lists the writes a module makes at runtime (inside a function): assignment and deletion
targets, mutating method calls and ``setattr``/``delattr``, each with the scope it runs in.

This is the one binding walk for the HX checkers: it records both the owning scope of every name
(the HX015 rules resolve through :meth:`Scope.binder`) and the import a name means
(:meth:`Scope.resolve` feeds ``py_rules.canonical_tree``, which spells import-bound names out as
their canonical dotted API), so a Python-scope rule lives here once.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field

from scripts.code_health.py_ast import _FUNCS, _call_name

MUTATORS = frozenset({
    "append", "extend", "insert", "remove", "pop", "popitem", "clear", "update", "setdefault",
    "add", "discard", "difference_update", "intersection_update", "symmetric_difference_update",
    "appendleft", "extendleft", "popleft", "rotate", "sort", "reverse", "__setitem__",
    "__delitem__", "move_to_end",
})


@dataclass(frozen=True)
class _Import:
    """What one import binds a name to: the target module path (``m.X`` for ``from m import X``),
    whether the name is an object of that module rather than a module (``from m import X`` and
    ``from . import X``, not ``import m.n as a``), whether the import is relative, and its line."""

    target: str
    obj: bool = False
    relative: bool = False
    line: int = 0

    def same(self, other: _Import) -> bool:
        return (self.target, self.obj, self.relative) == (other.target, other.obj, other.relative)


@dataclass(eq=False)
class Scope:
    kind: str  # "module" | "class" | "function" | "comprehension"
    parent: Scope | None = None
    # Names the scope binds: plain bindings here, imports in their own map, and the names a
    # ``global`` / ``nonlocal`` statement hands to an outer scope (they bind nothing here).
    bound: set[str] = field(default_factory=set)
    imports: dict[str, _Import] = field(default_factory=dict)
    declared: dict[str, str] = field(default_factory=dict)

    def root(self) -> Scope:
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def owner(self, name: str) -> Scope:
        """The scope a binding of ``name`` made here lands in: the module for a ``global``, the
        nearest enclosing function that has it for a ``nonlocal``, this scope otherwise."""
        scope = self
        while name in scope.declared:
            if scope.declared[name] == "global":
                while scope.parent is not None:
                    scope = scope.parent
                return scope
            outer = scope.parent  # nonlocal: the nearest enclosing function that has it
            while outer is not None and outer.parent is not None and (
                    outer.kind == "class" or not outer.binds(name)):
                outer = outer.parent
            if outer is None or outer.parent is None:
                return scope
            scope = outer
        return scope

    def binds(self, name: str) -> bool:
        return name in self.bound or name in self.imports or name in self.declared

    def binder(self, name: str) -> Scope:
        """The scope whose binding ``name`` means here (the module when nothing nearer binds it).
        Class bodies are invisible to the functions and comprehensions nested in them."""
        scope: Scope | None = self
        while scope is not None and scope.parent is not None:
            kind = scope.declared.get(name)
            if kind == "global":
                break
            if kind != "nonlocal" and (name in scope.bound or name in scope.imports) and (
                    scope is self or scope.kind != "class"):
                return scope
            scope = scope.parent
        return self.root()

    def resolve(self, name: str) -> _Import | None:
        """The import ``name`` means here: the innermost scope binding it decides, and a scope
        that binds it any other way (or by two imports) makes it a plain local. Class bodies are
        invisible to the functions and comprehensions nested in them."""
        scope: Scope | None = self
        while scope is not None:
            if scope is self or scope.kind != "class":
                if name in scope.declared:
                    if name in scope.bound:  # rebound before its outer binding was seen
                        return None
                    scope = scope.owner(name)
                    return scope.imports.get(name) if name not in scope.bound else None
                if name in scope.bound:
                    return None
                if name in scope.imports:
                    return scope.imports[name]
            scope = scope.parent
        return None


class Bindings(ast.NodeVisitor):
    """The scope every node is evaluated in, the scope each function/class/comprehension node
    opens, and the load scope of every name read: one walk for every HX rule that needs to know
    which binding a name means (``binder``/``resolve``) or where a node runs."""

    def __init__(self, tree: ast.Module) -> None:
        self.module = self.scope = Scope("module")
        self.scope_of: dict[int, Scope] = {}
        self.opened: dict[int, Scope] = {}
        self.loads: dict[int, Scope] = {}
        self.visit(tree)

    def visit(self, node: ast.AST) -> None:
        self.scope_of.setdefault(id(node), self.scope)
        super().visit(node)

    def at(self, node: ast.AST) -> Scope:
        return self.scope_of.get(id(node), self.module)

    def _enter(self, kind: str, owner: ast.AST, nodes: list[ast.AST]) -> None:
        outer, self.scope = self.scope, Scope(kind, self.scope)
        self.opened[id(owner)] = self.scope
        for node in nodes:
            self.visit(node)
        self.scope = outer

    def _bind(self, name: str, scope: Scope | None = None) -> None:
        (scope or self.scope).owner(name).bound.add(name)

    def _import(self, name: str, imp: _Import) -> None:
        """Bind *name* to *imp* in the scope that owns it; the first import of a name keeps its
        line, and a second import of something else makes the name ambiguous there."""
        scope = self.scope.owner(name)
        known = scope.imports.get(name)
        if known is None:
            scope.imports[name] = imp
        elif not known.same(imp):
            self._bind(name, scope)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            self._import(alias.asname or top, _Import(alias.name if alias.asname else top, line=node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # A relative import keeps its module path minus the dots: rules match the leaf.
        for alias in node.names:
            if alias.name != "*":
                target = f"{node.module}.{alias.name}" if node.module else alias.name
                self._import(alias.asname or alias.name,
                             _Import(target, obj=True, relative=bool(node.level), line=node.lineno))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads[id(node)] = self.scope
        else:
            self._bind(node.id)

    # A declaration selects the outer binding; only an assignment through it rebinds that.
    def visit_Global(self, node: ast.Global) -> None:
        self.scope.declared.update(dict.fromkeys(node.names, "global"))

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.declared.update(dict.fromkeys(node.names, "nonlocal"))

    def generic_visit(self, node: ast.AST) -> None:
        # except-as, match captures and **rest bind a plain string attribute.
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
            for attr in ("name", "rest"):
                value = getattr(node, attr, None)
                if isinstance(value, str):
                    self._bind(value)
        super().generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        """Decorators, defaults and annotations run in the enclosing scope; the body does not."""
        args = node.args
        params = [a for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
                  if a is not None]
        returns = getattr(node, "returns", None)
        for expr in (*getattr(node, "decorator_list", []), *args.defaults,
                     *(d for d in args.kw_defaults if d is not None),
                     *(a.annotation for a in params if a.annotation), *([returns] if returns else [])):
            self.visit(expr)
        if not isinstance(node, ast.Lambda):
            self._bind(node.name)
        body = node.body if isinstance(node.body, list) else [node.body]
        self._enter("function", node, [*params, *body])

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_function

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expr in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expr)
        self._bind(node.name)
        self._enter("class", node, list(node.body))

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        """The first iterable runs outside; targets, conditions and elements run inside."""
        first, *rest = node.generators
        self.visit(first.iter)
        elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        inner = [first.target, *first.ifs, *(n for g in rest for n in (g.iter, g.target, *g.ifs))]
        self._enter("comprehension", node, [*inner, *elements])

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # A walrus in a comprehension binds in the enclosing function (PEP 572).
        scope = self.scope
        while scope.kind == "comprehension" and scope.parent is not None:
            scope = scope.parent
        self._bind(node.target.id, scope)
        self.scope_of.setdefault(id(node.target), self.scope)
        self.visit(node.value)

    def binder(self, node: ast.Name) -> Scope:
        return self.at(node).binder(node.id)

    def module_name(self, node: ast.AST) -> str | None:
        """``node.id`` when *node* is a ``Name`` that means the module-level binding."""
        if isinstance(node, ast.Name) and self.binder(node) is self.module:
            return node.id
        return None

    def in_function(self, node: ast.AST) -> bool:
        """*node* runs when a function is called, not at import (class bodies run at import)."""
        scope: Scope | None = self.at(node)
        while scope is not None:
            if scope.kind == "function":
                return True
            scope = scope.parent
        return False


@dataclass(frozen=True)
class Write:
    """One runtime change. ``target`` is the expression written: rebound when ``store`` (an
    assignment or deletion of it, ``setattr`` on it when ``attr`` is set), else changed in place
    (an item store, a mutating method call)."""

    target: ast.AST
    store: bool
    line: int
    attr: str | None = None  # setattr/delattr: the attribute set on ``target`` ("" when unknown)
    aug: bool = False  # ``target += ...``: in place for a container, a rebind otherwise


def _targets(target: ast.AST, line: int, aug: bool = False) -> Iterator[Write]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _targets(elt, line)
    elif isinstance(target, ast.Starred):
        yield from _targets(target.value, line)
    elif isinstance(target, ast.Subscript):
        yield Write(target.value, False, line)
    else:
        yield Write(target, True, line, aug=aug)


_MUTATOR_NAMES = frozenset({"setattr", "delattr", "builtins.setattr", "builtins.delattr"})


def _node_writes(node: ast.AST) -> Iterator[Write]:
    if isinstance(node, (ast.Assign, ast.Delete)):
        for target in node.targets:
            yield from _targets(target, node.lineno)
    elif isinstance(node, ast.AugAssign):
        yield from _targets(node.target, node.lineno, aug=True)
    elif (isinstance(node, ast.AnnAssign) and node.value is not None
          or isinstance(node, (ast.For, ast.AsyncFor))):
        yield from _targets(node.target, node.lineno)
    elif isinstance(node, ast.NamedExpr):  # ``global X; (X := v)`` rebinds the module's X
        yield Write(node.target, True, node.lineno)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                yield from _targets(item.optional_vars, node.lineno)
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in MUTATORS:
            yield Write(func.value, False, node.lineno)
        elif _call_name(node) in _MUTATOR_NAMES and len(node.args) >= 2:
            key = node.args[1]
            name = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else ""
            yield Write(node.args[0], True, node.lineno, attr=name)


def runtime_writes(tree: ast.Module, bindings: Bindings) -> list[Write]:
    """Every write made inside a function: what runs per call, not while the module imports."""
    return [write for node in ast.walk(tree) for write in _node_writes(node)
            if bindings.in_function(node)]


def import_time(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """*body* and every statement nested in its compound statements (``if``, ``for``,
    ``while``, ``try``, ``with``, ``match``): what runs when *body* runs. Function and class
    bodies are not entered."""
    for node in body:
        yield node
        if isinstance(node, (*_FUNCS, ast.ClassDef)):
            continue
        for _, value in ast.iter_fields(node):
            for item in value if isinstance(value, list) else []:
                if isinstance(item, ast.stmt):
                    yield from import_time([item])
                elif isinstance(item, (ast.ExceptHandler, ast.match_case)):
                    yield from import_time(item.body)
