"""Small AST linter for names used without any visible binding.

This is intentionally narrower than a type checker. It catches the merge
incident class where a conflict resolution removes the only import/assignment
for a name while leaving later uses behind.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path


BUILTINS = set(dir(builtins))


@dataclass(frozen=True)
class UnboundIssue:
    path: str
    line: int
    column: int
    name: str


@dataclass
class Scope:
    kind: str
    parent: "Scope | None" = None
    bindings: set[str] = field(default_factory=set)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)
    uses: list[tuple[str, int, int]] = field(default_factory=list)
    children: list["Scope"] = field(default_factory=list)
    has_star_import: bool = False

    def module(self) -> "Scope":
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def can_resolve(self, name: str) -> bool:
        if name in BUILTINS:
            return True
        if name in self.globals:
            return name in self.module().bindings or self.module().has_star_import or name in BUILTINS
        if name in self.nonlocals:
            return self.parent is not None and self.parent._resolve_in_nonlocal_parents(name)
        if name in self.bindings:
            return True
        if self.parent is not None:
            return self.parent.can_resolve(name)
        return self.has_star_import

    def _resolve_in_nonlocal_parents(self, name: str) -> bool:
        scope: Scope | None = self
        while scope is not None and scope.parent is not None:
            if name in scope.bindings:
                return True
            scope = scope.parent
        return False


class _BindingCollector(ast.NodeVisitor):
    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.scope.bindings.add((alias.asname or alias.name.split(".", 1)[0]))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.scope.has_star_import = True
            else:
                self.scope.bindings.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.bindings.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.bindings.add(node.name)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_target(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.scope.bindings.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bind_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind_target(node.target)
        self.visit(node.value)

    def _bind_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.scope.bindings.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind_target(item)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)


class _ScopeBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.root = Scope("module")
        self.current = self.root

    def build(self, tree: ast.AST) -> Scope:
        _BindingCollector(self.root).visit(tree)
        self.visit(tree)
        return self.root

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.current.uses.append((node.id, node.lineno, node.col_offset))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for deco in node.decorator_list:
            self.visit(deco)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        child = Scope("function", self.current)
        self.current.children.append(child)
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        ):
            child.bindings.add(arg.arg)
        _BindingCollector(child).visit(ast.Module(body=node.body, type_ignores=[]))
        self._within(child, node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        child = Scope("class", self.current)
        self.current.children.append(child)
        _BindingCollector(child).visit(ast.Module(body=node.body, type_ignores=[]))
        self._within(child, node.body)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.elt, node.generators)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension([node.key, node.value], node.generators)

    def _visit_comprehension(self, body: ast.AST | list[ast.AST], generators: list[ast.comprehension]) -> None:
        child = Scope("comprehension", self.current)
        self.current.children.append(child)
        for gen in generators:
            _BindingCollector(child)._bind_target(gen.target)
        old = self.current
        self.current = child
        try:
            for gen in generators:
                self.visit(gen.iter)
                for condition in gen.ifs:
                    self.visit(condition)
            items = body if isinstance(body, list) else [body]
            for item in items:
                self.visit(item)
        finally:
            self.current = old

    def _within(self, scope: Scope, body: list[ast.stmt]) -> None:
        old = self.current
        self.current = scope
        try:
            for stmt in body:
                self.visit(stmt)
        finally:
            self.current = old


def _issues_for_scope(path: str, scope: Scope) -> list[UnboundIssue]:
    issues: list[UnboundIssue] = []
    if scope.has_star_import:
        return issues
    for name, line, col in scope.uses:
        if not scope.can_resolve(name):
            issues.append(UnboundIssue(path=path, line=line, column=col, name=name))
    for child in scope.children:
        issues.extend(_issues_for_scope(path, child))
    return issues


def lint_source(source: str, *, path: str = "<string>") -> list[UnboundIssue]:
    tree = ast.parse(source, filename=path)
    scope = _ScopeBuilder().build(tree)
    if scope.has_star_import:
        return []
    return _issues_for_scope(path, scope)


def lint_file(path: Path, *, repo: Path | None = None) -> list[UnboundIssue]:
    label = str(path.relative_to(repo)) if repo and path.is_relative_to(repo) else str(path)
    try:
        return lint_source(path.read_text(encoding="utf-8"), path=label)
    except SyntaxError as exc:
        return [UnboundIssue(path=label, line=exc.lineno or 0, column=exc.offset or 0, name="<syntax>")]


def lint_paths(paths: list[Path], *, repo: Path | None = None) -> list[UnboundIssue]:
    issues: list[UnboundIssue] = []
    for path in paths:
        issues.extend(lint_file(path, repo=repo))
    return issues
