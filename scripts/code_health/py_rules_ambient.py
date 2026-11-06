"""HX015: process-wide mutable state in a module the hosted entry point imports.

One hosted process serves many tenants' agents, so module state is shared by all of them. The
rule flags, in a module reachable from ``config.HOSTED_ENTRY`` (``hosted_graph``):

- a module-level ``dict``/``list``/``set`` (literal, comprehension or constructor) or
  ``threading.local()`` that starts empty;
- any module-level binding a function changes at runtime, whatever its initializer: rebinding it
  (``global X; X = v``, ``setattr(sys.modules[__name__], "X", v)``, ``globals()["X"] = v``), or
  changing the object in place, including through a nested item or attribute
  (``X["seed"].append(v)``, ``X.value = v``). A non-empty table the module never changes is a
  constant;
- an ``lru_cache`` / ``cache`` on a module-level function or a method;
- a class attribute changed at runtime (``py_rules_ambient_classes``);
- an object bound by ``from m import X`` that a function changes in place (``X[k] = v``,
  ``X.append(v)``, ``X.attr = v``), on the import line: ``m`` cannot see that write, so the
  importer answers for it. A write through the module itself (``m.X[k] = v``,
  ``os.environ[k] = v``) is judged as ``m``'s.

Every write is resolved to its lexical binding first (``py_rules_ambient_bindings``): a
parameter, local, nested-scope, comprehension or class-body name that shares a module name is
not the module's state. Module statements count wherever they run at import, through ``if``,
``for``, ``while``, ``try``, ``with`` and ``match`` blocks.

Two declarations pass, recognised by API identity (the canonical import resolver's dotted
target, never a leaf name): ``hermes_scope_cache.ScopedCache(...)`` (one slot per tenant and
agent) and ``hermes_scope_cache.tenant_neutral(value, reason="...")`` /
``@tenant_neutral(reason="...")`` with a non-empty literal reason (state that is the same for
every tenant). A call that only shares a declaration's name (a local ``def tenant_neutral``,
another module's) is flagged. Hits are reported on the line that creates the state, so editing
the code that mutates it is not new debt.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name, _dotted
from scripts.code_health.py_rules_ambient_bindings import Bindings, Write, import_time, runtime_writes
from scripts.code_health.py_rules_ambient_classes import class_state

_CONTAINER_CALLS = {
    "dict", "list", "set", "collections.defaultdict", "collections.OrderedDict",
    "collections.Counter", "collections.deque", "weakref.WeakKeyDictionary",
    "weakref.WeakValueDictionary", "weakref.WeakSet",
}
# Per thread, and every tenant's turns run on the same threads: state however it is created.
_ALWAYS_STATE = {"threading.local", "_thread._local"}
_CONTAINER_NODES = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
_CACHES = {"functools.lru_cache", "functools.cache", "lru_cache", "cache"}
_NEUTRAL = "hermes_scope_cache.tenant_neutral"
_SCOPED = "hermes_scope_cache.ScopedCache"
_DECLARATION_LEAVES = {"tenant_neutral", "ScopedCache"}
_DUNDER_OK = {"__all__", "__slots__", "__path__", "__match_args__"}


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _api(call: ast.Call) -> str:
    """The call's canonical dotted name when its root is an import binding, else ``""``: a local
    function or a parameter that shares a declaration's name is not the declaration."""
    root = call.func
    while isinstance(root, ast.Attribute):
        root = root.value
    return _call_name(call) if getattr(root, "imported", False) else ""


def _declared(value: ast.AST | None) -> bool:
    """``hermes_scope_cache.ScopedCache(...)`` or
    ``hermes_scope_cache.tenant_neutral(..., reason="<non-empty literal>")``."""
    if not isinstance(value, ast.Call):
        return False
    api = _api(value)
    if api == _SCOPED:
        return True
    if api != _NEUTRAL:
        return False
    reason = next((kw.value for kw in value.keywords if kw.arg == "reason"), None)
    return isinstance(reason, ast.Constant) and isinstance(reason.value, str) and bool(reason.value.strip())


def _bad_declaration(value: ast.AST | None) -> bool:
    """Named like a declaration but not one: no literal reason, or not the
    ``hermes_scope_cache`` API."""
    return (isinstance(value, ast.Call) and _leaf(_call_name(value)) in _DECLARATION_LEAVES
            and not _declared(value))


def _is_container(value: ast.AST | None) -> bool:
    if isinstance(value, _CONTAINER_NODES):
        return True
    return isinstance(value, ast.Call) and _call_name(value) in _CONTAINER_CALLS | _ALWAYS_STATE


_COPIES = {"copy.copy", "copy.deepcopy"}


def _fresh(value: ast.AST | None) -> bool:
    """A value made new where it is assigned: a container literal, comprehension or constructor
    (``dict(C.values)``), a copy (``copy.copy(v)``, ``v.copy()``), or a call to a capitalised
    constructor (``Registry()``). A name or attribute (``C.values``) is an alias."""
    if _is_container(value):
        return True
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if _call_name(value) in _COPIES or (isinstance(func, ast.Attribute) and func.attr == "copy"
                                        and not value.args):
        return True
    leaf = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return leaf[:1].isupper()


def _mutable(value: ast.AST | None) -> bool:
    """A value that can change in place: a container, a tuple holding one, or an object a call
    built (an instance, a ``threading.local``)."""
    return _is_container(value) or isinstance(value, (ast.Call, ast.Tuple))


def _is_empty(value: ast.AST | None) -> bool:
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, (ast.List, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Call):
        name = _call_name(value)
        return (not value.args and not value.keywords or name == "collections.defaultdict"
                or name in _ALWAYS_STATE)
    return False  # a comprehension is filled, not empty


def _assigned(stmts: Iterable[ast.stmt]) -> Iterator[tuple[str, ast.AST | None, int]]:
    """``(name, value, line)`` for each plain name bound by an assignment in *stmts*."""
    for node in stmts:
        pairs: list[tuple[ast.AST, ast.AST | None]] = []
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            pairs = [(node.target, None)]
        for target, value in pairs:
            if isinstance(target, ast.Name):
                yield target.id, value, node.lineno
            elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                for t, v in zip(target.elts, value.elts):
                    if isinstance(t, ast.Name):
                        yield t.id, v, node.lineno


def _module_self(node: ast.AST) -> bool:
    """``sys.modules[__name__]`` / ``globals()``: this module's own namespace."""
    if isinstance(node, ast.Subscript) and _dotted(node.value) == "sys.modules":
        return isinstance(node.slice, ast.Name) and node.slice.id == "__name__"
    return isinstance(node, ast.Call) and _call_name(node) == "globals" and not node.args


def _namespace_name(write: Write, keyed: dict[int, str]) -> str | None:
    """The module name a write through the module's own namespace rebinds:
    ``setattr(sys.modules[__name__], "X", v)``, ``sys.modules[__name__].X = v``,
    ``globals()["X"] = v``."""
    target = write.target
    if write.attr is not None:
        return (write.attr or None) if _module_self(target) else None
    if isinstance(target, ast.Attribute) and write.store and _module_self(target.value):
        return target.attr
    return keyed.get(id(target))


def _namespace_keys(tree: ast.Module) -> dict[int, str]:
    """``globals()["X"] = v`` writes ``globals()`` in place; map that node to ``"X"``."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
                and _module_self(node.value) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            out[id(node.value)] = node.slice.value
    return out


def _imported_object_writes(writes: list[Write]) -> Iterator[int]:
    """Import lines of names bound by ``from m import X`` (or ``from . import X``) that a function
    changes in place, each on the line of the import that binding resolves to
    (``X[k] = v``, ``X.append(v)``, ``X.attr = v``). ``m`` cannot see these writes, so they are
    reported here, on the import line. A write through a module (``os.environ[k] = v``,
    ``m.X[k] = v``) is that module's business, as before."""
    for write in writes:
        node = write.target
        while True:
            line = getattr(node, "from_import", None)
            if line is not None:
                if node is not write.target or not write.store or write.attr is not None:
                    yield line
                break
            if not isinstance(node, (ast.Attribute, ast.Subscript)):
                break
            node = node.value


def _module_changes(tree: ast.Module, writes: list[Write],
                    bindings: Bindings) -> tuple[dict[str, int], set[str]]:
    """Module names functions rebind (with the first line) and names they change in place. A
    function can only rebind a module name it declares ``global`` (or through the module's
    namespace), so a resolved plain-name store, ``+=`` included, is a rebind."""
    keyed = _namespace_keys(tree)
    rebound: dict[str, int] = {}
    mutated: set[str] = set()
    for write in writes:
        name = _namespace_name(write, keyed)
        if name:
            rebound.setdefault(name, write.line)
            continue
        root = write.target
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        # An imported module's attribute (``os.environ[k] = v``) is that module's state, judged there.
        name = None if getattr(root, "imported", False) else bindings.module_name(root)
        if name is None:
            continue
        if root is write.target and write.store and write.attr is None:
            rebound.setdefault(name, write.line)
        else:
            mutated.add(name)
    return rebound, mutated


def _module_state(tree: ast.Module, writes: list[Write], bindings: Bindings) -> Iterator[int]:
    rebound, mutated = _module_changes(tree, writes, bindings)
    bound: set[str] = set()
    for name, value, line in _assigned(import_time(tree.body)):
        bound.add(name)
        if _bad_declaration(value):
            yield line
        elif name in _DUNDER_OK or _declared(value):
            continue
        elif (name in rebound or (_is_container(value) and _is_empty(value))
              or (name in mutated and _mutable(value))):
            yield line
    for name, line in rebound.items():
        if name not in bound and name not in _DUNDER_OK:
            yield line  # created at runtime only, by the rebinding itself


def _decorated_cache(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if any(isinstance(d, ast.Call) and _declared(d) for d in func.decorator_list):
        return False
    names = [_call_name(d) if isinstance(d, ast.Call) else _dotted(d) for d in func.decorator_list]
    return any(n in _CACHES for n in names)


def _classes(tree: ast.Module) -> Iterator[ast.ClassDef]:
    """Classes defined while the module imports, nested classes included."""
    stack = [n for n in import_time(tree.body) if isinstance(n, ast.ClassDef)]
    while stack:
        cls = stack.pop(0)
        yield cls
        stack.extend(n for n in import_time(cls.body) if isinstance(n, ast.ClassDef))


def _cached_functions(tree: ast.Module) -> Iterator[int]:
    bodies = [tree.body, *(cls.body for cls in _classes(tree))]
    for body in bodies:
        for node in import_time(body):
            if isinstance(node, _FUNCS) and _decorated_cache(node):
                yield node.lineno


def ambient_state(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    bindings = Bindings(tree)
    writes = runtime_writes(tree, bindings)
    yield from _module_state(tree, writes, bindings)
    yield from _imported_object_writes(writes)
    yield from _cached_functions(tree)
    for cls in _classes(tree):
        attrs = {name: (value, line) for name, value, line in _assigned(import_time(cls.body))}
        yield from class_state(cls, attrs, writes, bindings, _is_container, _mutable, _declared, _fresh)
