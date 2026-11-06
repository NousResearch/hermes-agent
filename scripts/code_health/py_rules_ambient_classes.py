"""HX015, class attributes: class-level state a method changes at runtime.

A class attribute is shared by every instance, so a method that rebinds it through the class
(``cls.x = v``, ``ClassName.x = v``, ``type(self).x = v``, ``setattr(cls, "x", v)``), or that
mutates a class-level container in place, changes state every tenant sees. The receiver is the
method's first parameter whatever it is called (``self``, ``this``, ``cls``); a classmethod's
(and ``__new__``'s) is the class. Mutating ``self.x`` in place reaches the class attribute unless
the instance has its own copy, which counts only when ``__init__`` (defined once, directly in the
class body) assigns it a fresh value (a ``dict``/``list``/``set`` literal, comprehension or
constructor, a ``copy``, or a call to a capitalised constructor; never an alias such as
``self.values = C.values``) as a top-level statement of its body, before anything else in that
body reads or writes the attribute, calls a method on the instance or hands the instance out, and
before any ``return``. A copy made on one path, or after the shared value was already touched,
leaves an instance on the shared one. Beyond that the copy is credited only where nothing else
in the class can replace it: every other store of the attribute, through any receiver but the
class, must be a fresh value (or an augmented assignment), so an alias, a tuple unpacking, a loop
or ``with`` target, or ``del`` of it revokes it; and no copy in the class counts if the class
uses any namespace operation at all (``setattr``, ``delattr``, ``vars``, ``eval``/``exec``, a
``getattr`` of a computed name, any dunder but the ordinary protocol ones (``__dict__``,
``__setattr__``, ``__getstate__``, ...), anything from ``builtins``, ``operator``,
``importlib``, ``gc`` or ``ctypes``), through whatever receiver, alias or string.

A method is any function the class body defines while it runs, under an ``if`` or ``try`` too.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator

from scripts.code_health.py_ast import _FUNCS, _call_name, _dotted
from scripts.code_health.py_rules_ambient_bindings import Bindings, Write, import_time

_CLASS_FIRST_PARAM = {"__new__", "__init_subclass__", "__class_getitem__"}

Attrs = dict[str, tuple[ast.AST | None, int]]  # class-body name -> (value, line)

# Names through which code can change an instance's namespace without a plain ``self.x = v``.
# Matched by text wherever they appear (a call, a reference, an alias, a string such as
# ``getattr(self, "__dict__")``) and after imports are spelled out, so ``from builtins import
# delattr as drop`` is ``builtins.delattr``. Any dunder outside ``_PLAIN_DUNDERS`` counts too
# (``__setattr__``, ``__dict__``, ``__getstate__``, ``__getattribute__``, ...), whatever its
# receiver: ``object.__setattr__``, ``o = object; o.__delattr__``.
_NAMESPACE_OPS = frozenset({"vars", "setattr", "delattr", "eval", "exec", "globals", "locals"})
# Modules whose members can reach a namespace: any use of them is treated as one.
_NAMESPACE_MODULES = frozenset({"builtins", "operator", "importlib", "gc", "ctypes"})
_PLAIN_DUNDERS = frozenset({
    "__init__", "__new__", "__init_subclass__", "__post_init__", "__class__", "__name__",
    "__qualname__", "__module__", "__doc__", "__file__", "__version__", "__all__",
    "__enter__", "__exit__", "__aenter__", "__aexit__", "__iter__", "__next__", "__aiter__",
    "__anext__", "__len__", "__contains__", "__call__", "__repr__", "__str__", "__eq__",
    "__hash__", "__bool__", "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
})


def _namespace_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    dunder = len(name) > 4 and name.startswith("__") and name.endswith("__")
    return name in _NAMESPACE_OPS or (dunder and name not in _PLAIN_DUNDERS)


def _literal_getattr_callees(nodes: list[ast.AST]) -> set[int]:
    """The ``getattr`` names that are the direct callee of a call with a literal attribute name:
    the only ``getattr`` use that is a plain read. Any other use (an alias, an argument, a
    computed name) can fetch a namespace operation."""
    return {id(n.func) for n in nodes
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
            and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
            and isinstance(n.args[1].value, str)}


def _dynamic(node: ast.AST, literal_getattr: frozenset[int] | set[int] = frozenset()) -> bool:
    """Whether *node* may change an instance's namespace by a route other than a plain
    attribute assignment or ``del`` (see ``_NAMESPACE_OPS``); a ``getattr`` whose name is not a
    literal (it may fetch any of them) counts too, and so does ``getattr`` taken as a value
    (``fetch = getattr``) rather than called with a literal name."""
    if isinstance(node, ast.Call) and _call_name(node) == "getattr":
        name = node.args[1] if len(node.args) > 1 else None
        return not (isinstance(name, ast.Constant) and isinstance(name.value, str))
    if isinstance(node, ast.Name):
        return (_namespace_name(node.id) or node.id in _NAMESPACE_MODULES
                or (node.id == "getattr" and id(node) not in literal_getattr))
    if isinstance(node, ast.Attribute):
        return _namespace_name(node.attr)
    return isinstance(node, ast.Constant) and _namespace_name(node.value)


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
        self.methods = [n for n in import_time(cls.body) if isinstance(n, _FUNCS)]
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

    def own_copies(self, fresh: Callable[[ast.AST | None], bool]) -> set[str]:
        """Attributes every instance has its own copy of once ``__init__`` returns (see the module
        docstring): assigned a *fresh* value by a top-level statement of the one ``__init__``
        written directly in the class body, ahead of any other use of that attribute or of the
        instance, and of any ``return``."""
        inits = [m for m in self.methods if m.name == "__init__"]
        if len(inits) != 1 or inits[0] not in self.cls.body:
            return set()
        own: set[str] = set()
        touched: set[str] = set()  # used before (or without) a fresh copy
        for stmt in inits[0].body:
            copies = self._fresh_copies(stmt, fresh)
            uses, leaked, returns = self._uses(stmt, copies)
            if leaked:
                break  # the instance or one of its methods may reach any shared attribute
            touched |= uses
            own |= {attr for attr in copies if attr not in touched}
            touched |= copies
            if returns:
                break
        return own

    def revoked(self, fresh: Callable[[ast.AST | None], bool]) -> set[str] | None:
        """Attributes whose own copy does not count (see the module docstring): any the class
        stores anything but a fresh value into, or deletes, through any receiver but the class;
        None, for every attribute, when the class can change a namespace dynamically."""
        nodes = list(ast.walk(self.cls))
        literal = _literal_getattr_callees(nodes)
        if any(_dynamic(node, literal) for node in nodes):
            return None
        kept: set[int] = set()  # attribute targets an assignment gives a fresh value
        for node in nodes:
            if isinstance(node, ast.Assign) and fresh(node.value):
                kept.update(id(t) for t in node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None and fresh(node.value):
                kept.add(id(node.target))
            elif isinstance(node, ast.AugAssign):
                kept.add(id(node.target))  # in place for a container: judged as a mutation
        return {node.attr for node in nodes
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                and id(node) not in kept and self.kind(node.value) != "class"}

    def _fresh_copies(self, stmt: ast.stmt, fresh: Callable[[ast.AST | None], bool]) -> set[str]:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            return set()
        if not fresh(value):
            return set()
        return {t.attr for t in targets
                if isinstance(t, ast.Attribute) and self.kind(t.value) == "instance"}

    def _uses(self, stmt: ast.stmt, copies: set[str]) -> tuple[set[str], bool, bool]:
        """``(instance attributes the statement touches, whether it uses the instance beyond an
        attribute (a method call, handing ``self`` out), whether it can return)``. The copy's own
        target is not a use."""
        uses: set[str] = set()
        leaked = returns = False
        attr_values: set[int] = set()
        targets = ({id(t) for t in stmt.targets} if isinstance(stmt, ast.Assign)
                   else {id(stmt.target)} if isinstance(stmt, ast.AnnAssign) else set())
        nodes = list(ast.walk(stmt))
        for node in nodes:
            if isinstance(node, ast.Return):
                returns = True
            if isinstance(node, ast.Attribute) and self.kind(node.value) == "instance":
                attr_values.add(id(node.value))
                if id(node) not in targets or node.attr not in copies:
                    uses.add(node.attr)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and self.kind(node.func.value) == "instance"):
                leaked = True  # ``self.setup()`` may change any attribute
        for node in nodes:
            if (isinstance(node, ast.Name) and id(node) not in attr_values
                    and self.kind(node) == "instance"):
                leaked = True  # ``register(self)``: the instance is handed out
        return uses, leaked, returns


def class_state(cls: ast.ClassDef, attrs: Attrs, writes: list[Write], bindings: Bindings,
                is_container: Callable[[ast.AST | None], bool],
                mutable: Callable[[ast.AST | None], bool],
                declared: Callable[[ast.AST | None], bool],
                fresh: Callable[[ast.AST | None], bool]) -> Iterator[int]:
    """Lines of class-level state of *cls* changed at runtime: the attribute's line in the class
    body, or the first write's line for an attribute the write itself creates. An attribute is
    changed in place when its value can be (a container, or an object a call built)."""
    recv = _Receivers(cls, bindings)
    revoked = recv.revoked(fresh)
    own = set() if revoked is None else recv.own_copies(fresh) - revoked
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
