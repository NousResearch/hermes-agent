"""Python structure: functions (qualname, length, nesting, body hash) and scope lookup.

Cyclomatic complexity comes from ruff (``ruff_runner``) so the number matches what
``ruff check --select C901`` prints; everything else is measured here from the AST.
"""

from __future__ import annotations

import ast
import hashlib
from bisect import bisect_right
from dataclasses import dataclass

from scripts.code_health.model import MODULE_SCOPE, FileMeasure, Unit

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
_TRY_STAR = getattr(ast, "TryStar", None)
if _TRY_STAR is not None:
    _BLOCKS = (*_BLOCKS, _TRY_STAR)


@dataclass
class Span:
    start: int
    end: int
    qualname: str


def _child_blocks(node: ast.AST):
    """Statement lists directly owned by a compound statement."""
    for name in ("body", "orelse", "finalbody"):
        yield getattr(node, name, None) or []
    for handler in getattr(node, "handlers", None) or []:
        yield handler.body
    for case in getattr(node, "cases", None) or []:
        yield case.body


def nesting_depth(stmts: list[ast.stmt], depth: int = 0) -> int:
    deepest = depth
    for stmt in stmts:
        if isinstance(stmt, (*_FUNCS, ast.ClassDef)) or not isinstance(stmt, _BLOCKS):
            continue
        for index, block in enumerate(_child_blocks(stmt)):
            # `elif` is an If alone in orelse that starts at its `if`'s column (the `elif`
            # keyword); `else:` + an indented `if` has the same AST but is nested.
            is_elif = (
                isinstance(stmt, ast.If)
                and index == 1
                and len(block) == 1
                and isinstance(block[0], ast.If)
                and block[0].col_offset == stmt.col_offset
            )
            deepest = max(deepest, nesting_depth(block, depth if is_elif else depth + 1))
    return deepest


_SCOPES = (*_FUNCS, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _own_scope(node: ast.AST):
    """Nodes in ``node``'s own scope: a nested def/lambda/class/comprehension is yielded but not
    entered (its parameters and targets bind inside it, not here)."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, _SCOPES):
            stack.extend(ast.iter_child_nodes(child))


def _bound_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return () if isinstance(node.ctx, ast.Load) else (node.id,)
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, ast.alias):
        return ((node.asname or node.name).split(".")[0],)
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return tuple(node.names)
    # def/class names, `except ... as name`, match captures and `**rest`
    found = (getattr(node, "name", None), getattr(node, "rest", None))
    return tuple(n for n in found if isinstance(n, str))


def _binds(scope: ast.AST, name: str) -> bool:
    return any(name in _bound_names(child) for child in _own_scope(scope))


def _self_references(node: ast.AST, name: str) -> list[ast.Name | ast.Attribute]:
    """References in ``node`` to its own ``name``: ``name`` loads that no nested scope rebinds,
    and ``self.name`` / ``cls.name``."""
    refs: list[ast.Name | ast.Attribute] = []
    todo = [node]
    while todo:
        for child in _own_scope(todo.pop()):
            if isinstance(child, _SCOPES):
                if not _binds(child, name):
                    todo.append(child)
            elif isinstance(child, ast.Name) and child.id == name and isinstance(child.ctx, ast.Load):
                refs.append(child)
            elif (isinstance(child, ast.Attribute) and child.attr == name
                  and isinstance(child.ctx, ast.Load)
                  and isinstance(child.value, ast.Name) and child.value.id in ("self", "cls")):
                refs.append(child)
    return refs


def body_hash(node: ast.AST) -> str:
    """Name-independent hash, so a function moved or renamed unchanged keeps its cap.

    The declared name is dropped, and so are the body's references to it (``name(...)``,
    ``self.name(...)``, ``cls.name(...)``), unless the function's own scope rebinds that name:
    a recursive function renamed together with its self-call is still the same code. A nested
    scope that binds the name (``def identity(legacy)``) only keeps its own reads.
    """
    name = getattr(node, "name", "")
    refs = _self_references(node, name) if name and not _binds(node, name) else []
    # Blank the references in place for the dump, then put them back: the tree is shared.
    saved = [(ref, ref.id if isinstance(ref, ast.Name) else ref.attr) for ref in refs]
    for ref, _ in saved:
        _rename(ref, "")
    try:
        dumped = ast.dump(node, annotate_fields=False).replace(repr(name), "''", 1)
    finally:
        for ref, original in saved:
            _rename(ref, original)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]


def _rename(ref: ast.Name | ast.Attribute, value: str) -> None:
    if isinstance(ref, ast.Name):
        ref.id = value
    else:
        ref.attr = value


def _first_line(node: ast.AST) -> int:
    """A def/class starts at its first decorator: decorator code belongs to it (and moves
    with it), not to the module."""
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])


class _UnitCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.units: dict[str, Unit] = {}
        self.spans: list[Span] = []
        self.seen: set[str] = set()
        self.funcs: list[str] = []
        self.total_lines: dict[str, int] = {}

    def _qualname(self, name: str) -> str:
        """Dotted path without ``<locals>``; a repeated name (property setter, conditional
        def) gets ``#2``, ``#3`` in source order."""
        base = ".".join([*self.stack, name])
        qual, n = base, 1
        while qual in self.seen:
            n += 1
            qual = f"{base}#{n}"
        self.seen.add(qual)
        return qual

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = self._qualname(node.name)
        # A class has no size metrics of its own, but it is a unit so that a renamed or moved
        # class keeps the violations in its body (decorators, attributes) one-to-one.
        self.units[qual] = Unit(qualname=qual, line=node.lineno, metrics={},
                                body_hash=body_hash(node), parent=self.funcs[-1] if self.funcs else None)
        self.spans.append(Span(_first_line(node), node.end_lineno or node.lineno, qual))
        self.stack.append(qual.rsplit(".", 1)[-1])
        self.generic_visit(node)
        self.stack.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual = self._qualname(node.name)
        end = node.end_lineno or node.lineno
        first = _first_line(node)
        self.total_lines[qual] = end - first + 1
        self.units[qual] = Unit(
            qualname=qual,
            line=node.lineno,
            metrics={"FUNC_LINES": end - first + 1, "NESTING": nesting_depth(node.body)},
            body_hash=body_hash(node),
            parent=self.funcs[-1] if self.funcs else None,
        )
        self.spans.append(Span(first, end, qual))
        self.stack.append(qual.rsplit(".", 1)[-1])
        self.funcs.append(qual)
        self.generic_visit(node)
        self.funcs.pop()
        self.stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func


class ScopeIndex:
    """Innermost enclosing def/class qualname for a line."""

    def __init__(self, spans: list[Span]) -> None:
        self.spans = sorted(spans, key=lambda s: (s.start, -s.end))
        self.starts = [s.start for s in self.spans]

    def scope(self, line: int) -> str:
        best = MODULE_SCOPE
        best_size = None
        for span in self.spans[: bisect_right(self.starts, line)]:
            if span.start <= line <= span.end:
                size = span.end - span.start
                if best_size is None or size <= best_size:
                    best, best_size = span.qualname, size
        return best


def measure_structure(fm: FileMeasure, tree: ast.Module) -> ScopeIndex:
    collector = _UnitCollector()
    collector.visit(tree)
    # FUNC_LINES is a function's OWN lines: a nested def is its own unit, so editing a closure
    # never counts against the function that encloses it.
    for unit in collector.units.values():
        if unit.parent is not None and unit.qualname in collector.total_lines:
            parent = collector.units[unit.parent]
            parent.metrics["FUNC_LINES"] -= collector.total_lines[unit.qualname]
    fm.units = collector.units
    fm.metrics["FILE_LINES"] = len(fm.lines)
    return ScopeIndex(collector.spans)
