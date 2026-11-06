"""Lexical bindings for HX015: which binding a name read or written in any scope refers to.

A name mutated inside a function is the module's only when no scope between them binds it: a
parameter, a local assignment (without ``global``), a nested function's local, a comprehension
variable or a class-body name each shadow the module binding there. Python decides this per
scope from every binding in that scope, so the scopes are collected in one pass before any name
is resolved.

It also lists the writes a module makes at runtime (inside a function): assignment and deletion
targets, mutating method calls and ``setattr``/``delattr``, each with the scope it runs in.
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


@dataclass(eq=False)
class Scope:
    kind: str  # "module" | "class" | "function" | "comprehension"
    parent: Scope | None = None
    bound: set[str] = field(default_factory=set)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)

    def root(self) -> Scope:
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def binder(self, name: str) -> Scope:
        """The scope whose binding ``name`` means here (the module when nothing nearer binds it).
        Class bodies are invisible to the functions and comprehensions nested in them."""
        scope: Scope | None = self
        while scope is not None and scope.parent is not None:
            if name in scope.globals:
                break
            if name not in scope.nonlocals and name in scope.bound and (
                    scope is self or scope.kind != "class"):
                return scope
            scope = scope.parent
        return self.root()


class Bindings(ast.NodeVisitor):
    """The scope every node is evaluated in, and the scope each function/class/comprehension
    node opens."""

    def __init__(self, tree: ast.Module) -> None:
        self.module = self.scope = Scope("module")
        self.scope_of: dict[int, Scope] = {}
        self.opened: dict[int, Scope] = {}
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

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self.scope.bound.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)

    def visit_alias(self, node: ast.alias) -> None:
        if node.name != "*":
            self.scope.bound.add(node.asname or node.name.split(".")[0])

    def visit_arg(self, node: ast.arg) -> None:
        self.scope.bound.add(node.arg)

    def generic_visit(self, node: ast.AST) -> None:
        # except-as, match captures and **rest bind a plain string attribute.
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
            for attr in ("name", "rest"):
                value = getattr(node, attr, None)
                if isinstance(value, str):
                    self.scope.bound.add(value)
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
            self.scope.bound.add(node.name)
        body = node.body if isinstance(node.body, list) else [node.body]
        self._enter("function", node, [*params, *body])

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expr in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expr)
        self.scope.bound.add(node.name)
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
        scope.bound.add(node.target.id)
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
        elif _call_name(node) in ("setattr", "delattr") and len(node.args) >= 2:
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
