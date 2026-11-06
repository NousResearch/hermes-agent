"""HX015, class attributes: class-level state a method changes at runtime.

A class attribute is shared by every instance, so a method that rebinds it through the class
(``cls.x = v``, ``ClassName.x = v``, ``type(self).x = v``, ``setattr(cls, "x", v)``), or that
mutates a class-level container in place, changes state every tenant sees. The receiver is the
method's first parameter whatever it is called (``self``, ``this``, ``cls``); a classmethod's
(and ``__new__``'s) is the class. Mutating ``self.x`` in place reaches the class attribute unless
the instance has its own copy, which counts only when ``__init__`` assigns it unconditionally (a
top-level statement of its body): a copy made on one path leaves every other instance on the
shared one.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator

from scripts.code_health.py_ast import _FUNCS, _call_name, _dotted
from scripts.code_health.py_rules_ambient_bindings import Bindings, Write

_CLASS_FIRST_PARAM = {"__new__", "__init_subclass__", "__class_getitem__"}

Attrs = dict[str, tuple[ast.AST | None, int]]  # class-body name -> (value, line)


def _receiver_kind(method: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    names = {_dotted(d) for d in method.decorator_list}
    if "staticmethod" in names:
        return None
    if "classmethod" in names or method.name in _CLASS_FIRST_PARAM:
        return "class"
    return "instance"


class _Receivers:
    """Which expressions in a class's methods mean the class, and which an instance of it."""

    def __init__(self, cls: ast.ClassDef, bindings: Bindings) -> None:
        self.cls, self.bindings = cls, bindings
        self.home = bindings.at(cls).binder(cls.name)
        self.methods = [n for n in cls.body if isinstance(n, _FUNCS)]
        self.first: dict[int, tuple[str, str]] = {}  # id(method scope) -> (param, kind)
        for method in self.methods:
            kind = _receiver_kind(method)
            args = [*method.args.posonlyargs, *method.args.args]
            scope = bindings.opened.get(id(method))
            if kind and args and scope is not None:
                self.first[id(scope)] = (args[0].arg, kind)

    def kind(self, node: ast.AST) -> str | None:
        """``"class"``, ``"instance"`` or None for an expression in one of the methods (or in a
        function nested in one, which sees the method's receiver by closure)."""
        if isinstance(node, ast.Name):
            scope = self.bindings.binder(node)
            if node.id == self.cls.name and scope is self.home:
                return "class"
            param, kind = self.first.get(id(scope), ("", None))
            return kind if param == node.id else None
        if isinstance(node, ast.Call) and _call_name(node) == "type" and len(node.args) == 1:
            return "class" if self.kind(node.args[0]) == "instance" else None
        if isinstance(node, ast.Attribute) and node.attr == "__class__":
            return "class" if self.kind(node.value) == "instance" else None
        return None

    def change(self, write: Write) -> tuple[str, str, bool] | None:
        """``(receiver kind, attribute, rebound)`` when *write* changes an attribute of the class
        or of an instance of it; an in-place change through a nested item or attribute
        (``self.x[k].append(v)``) is a change of ``x``."""
        if write.attr is not None:
            kind = self.kind(write.target)
            return (kind, write.attr, True) if kind and write.attr else None
        node = write.target
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            kind = self.kind(node.value)
            if kind and isinstance(node, ast.Attribute):
                return kind, node.attr, node is write.target and write.store
            node = node.value
        return None

    def own_copies(self) -> set[str]:
        """Attributes ``__init__`` assigns on the instance as a top-level statement of its body."""
        own: set[str] = set()
        for method in self.methods:
            if method.name != "__init__":
                continue
            for stmt in method.body:
                targets = stmt.targets if isinstance(stmt, ast.Assign) else (
                    [stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value is not None else [])
                own.update(t.attr for t in targets
                           if isinstance(t, ast.Attribute) and self.kind(t.value) == "instance")
        return own


def class_state(cls: ast.ClassDef, attrs: Attrs, writes: list[Write], bindings: Bindings,
                is_container: Callable[[ast.AST | None], bool],
                mutable: Callable[[ast.AST | None], bool],
                declared: Callable[[ast.AST | None], bool]) -> Iterator[int]:
    """Lines of class-level state of *cls* changed at runtime: the attribute's line in the class
    body, or the first write's line for an attribute the write itself creates. An attribute is
    changed in place when its value can be (a container, or an object a call built)."""
    recv = _Receivers(cls, bindings)
    own = recv.own_copies()
    rebound: dict[str, int] = {}
    mutated: set[str] = set()
    for write in writes:
        change = recv.change(write)
        if change is None:
            continue
        kind, attr, rebind = change
        in_place = not rebind or (write.aug and is_container(attrs.get(attr, (None, 0))[0]))
        if kind == "class" and not in_place:
            rebound.setdefault(attr, write.line)
        elif in_place and (kind == "class" or attr not in own):
            mutated.add(attr)
    for attr, line in rebound.items():
        value, decl_line = attrs.get(attr, (None, line))
        if not declared(value):
            yield decl_line
    for attr in mutated - set(rebound):
        value, line = attrs.get(attr, (None, 0))
        if attr in attrs and mutable(value) and not declared(value):
            yield line
