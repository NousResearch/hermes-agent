"""HX015: process-wide mutable state in a module the hosted entry point imports.

One hosted process serves many tenants' agents, so module state is shared by all of them. The
rule flags, in a module reachable from ``config.HOSTED_ENTRY`` (``hosted_graph``):

- a module-level ``dict``/``list``/``set`` (literal, comprehension or constructor) that starts
  empty or that the module mutates (item store/delete, a mutating method, augmented assignment,
  rebinding through ``global``). A non-empty table the module never changes is a constant;
- an ``lru_cache`` / ``cache`` on a module-level function or a method;
- a class attribute changed at runtime: rebound through ``cls.x =`` / ``ClassName.x =``, or a
  class-level container mutated through ``cls``, ``ClassName`` or an instance that never sets its
  own copy.

Two declarations pass: a ``hermes_scope_cache.ScopedCache(...)`` (one slot per tenant and agent)
and ``tenant_neutral(value, reason="...")`` / ``@tenant_neutral(reason="...")`` with a non-empty
literal reason (state that is the same for every tenant). Hits are reported on the line that
creates the state, so editing the code that mutates it is not new debt.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name, _dotted

_CONTAINER_CALLS = {
    "dict", "list", "set", "collections.defaultdict", "collections.OrderedDict",
    "collections.Counter", "collections.deque", "weakref.WeakKeyDictionary",
    "weakref.WeakValueDictionary", "weakref.WeakSet",
}
_CONTAINER_NODES = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
_MUTATORS = {
    "append", "extend", "insert", "remove", "pop", "popitem", "clear", "update", "setdefault",
    "add", "discard", "difference_update", "intersection_update", "symmetric_difference_update",
    "appendleft", "extendleft", "popleft", "rotate", "sort", "reverse", "__setitem__",
    "__delitem__", "move_to_end",
}
_CACHES = {"functools.lru_cache", "functools.cache", "lru_cache", "cache"}
_NEUTRAL = "tenant_neutral"
_SCOPED = "ScopedCache"
_DUNDER_OK = {"__all__", "__slots__", "__path__", "__match_args__"}


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _declared(value: ast.AST | None) -> bool:
    """``ScopedCache(...)`` or ``tenant_neutral(..., reason="<non-empty literal>")``."""
    if not isinstance(value, ast.Call):
        return False
    leaf = _leaf(_call_name(value))
    if leaf == _SCOPED:
        return True
    if leaf != _NEUTRAL:
        return False
    reason = next((kw.value for kw in value.keywords if kw.arg == "reason"), None)
    return isinstance(reason, ast.Constant) and isinstance(reason.value, str) and bool(reason.value.strip())


def _bad_declaration(value: ast.AST | None) -> bool:
    """``tenant_neutral(...)`` without a literal, non-empty reason: it declares nothing."""
    return isinstance(value, ast.Call) and _leaf(_call_name(value)) == _NEUTRAL and not _declared(value)


def _is_container(value: ast.AST | None) -> bool:
    if isinstance(value, _CONTAINER_NODES):
        return True
    return isinstance(value, ast.Call) and _call_name(value) in _CONTAINER_CALLS


def _is_empty(value: ast.AST | None) -> bool:
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, (ast.List, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Call):
        return not value.args and not value.keywords or _call_name(value) == "collections.defaultdict"
    return False  # a comprehension is filled, not empty


def _top_level(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run at import, through top-level ``if``/``try``/``with`` blocks."""
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
            for field in ("body", "orelse", "finalbody"):
                yield from _top_level(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []):
                yield from _top_level(handler.body)


def _assigned(stmts: Iterable[ast.stmt]) -> Iterator[tuple[str, ast.AST | None, int]]:
    """``(name, value, line)`` for each plain name bound by an assignment in *stmts*."""
    for node in stmts:
        pairs: list[tuple[ast.AST, ast.AST | None]] = []
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        for target, value in pairs:
            if isinstance(target, ast.Name):
                yield target.id, value, node.lineno
            elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                for t, v in zip(target.elts, value.elts):
                    if isinstance(t, ast.Name):
                        yield t.id, v, node.lineno


def _mutated_base(node: ast.AST) -> ast.AST | None:
    """The object *node* mutates in place, if it is a mutation: ``x[k] = v`` / ``del x[k]`` /
    ``x[k] += v`` / ``x += y`` / ``x.append(...)``."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                return target.value
            if isinstance(node, ast.AugAssign):
                return target
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
        return node.func.value
    return None


def _mutations(tree: ast.Module) -> list[ast.AST]:
    out = []
    for node in ast.walk(tree):
        base = _mutated_base(node)
        if base is not None:
            out.append(base)
    return out


def _global_rebinds(tree: ast.Module) -> set[str]:
    """Module names a function rebinds through a ``global`` declaration."""
    names: set[str] = set()
    for func in ast.walk(tree):
        if not isinstance(func, _FUNCS):
            continue
        declared = {n for s in ast.walk(func) if isinstance(s, ast.Global) for n in s.names}
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in declared:
                names.add(node.id)
    return names


def _module_containers(tree: ast.Module, mutated: list[ast.AST]) -> Iterator[int]:
    mutated_names = {m.id for m in mutated if isinstance(m, ast.Name)} | _global_rebinds(tree)
    for name, value, line in _assigned(_top_level(tree.body)):
        if _bad_declaration(value):
            yield line
        elif name in _DUNDER_OK or _declared(value) or not _is_container(value):
            continue
        elif _is_empty(value) or name in mutated_names:
            yield line


def _decorated_cache(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = [_call_name(d) if isinstance(d, ast.Call) else _dotted(d) for d in func.decorator_list]
    if any(_leaf(n) == _NEUTRAL for n in names) and any(
            isinstance(d, ast.Call) and _declared(d) for d in func.decorator_list):
        return False
    return any(n in _CACHES for n in names)


def _cached_functions(tree: ast.Module) -> Iterator[int]:
    for node in _top_level(tree.body):
        if isinstance(node, _FUNCS) and _decorated_cache(node):
            yield node.lineno
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, _FUNCS) and _decorated_cache(item):
                    yield item.lineno


def _attr_target(node: ast.AST, cls: str) -> str | None:
    """``x`` for ``cls.x`` / ``ClassName.x``."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("cls", cls):
        return node.attr
    return None


def _self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _class_attributes(tree: ast.Module, mutated: list[ast.AST]) -> Iterator[int]:
    for cls in _top_level(tree.body):
        if not isinstance(cls, ast.ClassDef):
            continue
        attrs = {name: (value, line) for name, value, line in _assigned(cls.body)}
        stores: dict[str, int] = {}
        for func in (n for n in ast.walk(cls) if isinstance(n, _FUNCS)):
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    name = _attr_target(node, cls.name)
                    if name is not None:
                        stores.setdefault(name, node.lineno)
        own = {_self_attr(n) for n in ast.walk(cls) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
        yield from _rebound_attrs(attrs, stores)
        for name, (value, line) in attrs.items():
            if not _is_container(value) or _declared(value) or name in stores:
                continue
            via_class = any(_attr_target(m, cls.name) == name for m in mutated)
            via_self = name not in own and any(_self_attr(m) == name for m in mutated)
            if via_class or via_self:
                yield line


def _rebound_attrs(attrs: dict[str, tuple[ast.AST, int]], stores: dict[str, int]) -> Iterator[int]:
    for name, line in stores.items():
        value = attrs.get(name, (None, None))[0]
        if not _declared(value):
            yield attrs[name][1] if name in attrs else line


def ambient_state(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    mutated = _mutations(tree)
    yield from _module_containers(tree, mutated)
    yield from _cached_functions(tree)
    yield from _class_attributes(tree, mutated)
