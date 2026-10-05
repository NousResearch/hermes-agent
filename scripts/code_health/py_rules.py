"""Hermes-specific AST rules (the ``HX`` ids in ``config.RULES``).

Each checker is a small function ``(tree, ctx) -> iterable of line numbers``; the table at the
bottom maps rule ids to checkers. Checkers favour precision: a rule that cries wolf gets an
allow comment on every hit and stops meaning anything.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_CAPTURE_CALLS = {
    "get_hermes_home",
    "display_hermes_home",
    "load_config",
    "load_config_readonly",
    "read_raw_config",
    "get_secret",
    "getcwd",
    "expanduser",
    "_float_env",
    "_int_env",
    "_bool_env",
    "_env_float",
    "_env_int",
    "_env_bool",
}
_SYNC_CONFIG_CALLS = {"load_config", "save_config", "read_raw_config", "load_config_readonly"}
_SUBPROCESS_WAITS = {"run", "call", "check_call", "check_output"}
# health: allow HX003 -- the detector's own pattern list
_SHELL_IDENTITY = ("pgrep -f", "ps aux", "ps -ef", "ps -eo")


@dataclass
class Ctx:
    known_env: set[str] = field(default_factory=set)


@dataclass
class _Scope:
    """One lexical scope's bindings: names an import binds, and names anything else binds."""

    parent: _Scope | None = None
    kind: str = "function"  # "module" | "class" | "function" | "comprehension"
    imports: dict[str, str] = field(default_factory=dict)
    others: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> str | None:
        """Import target ``name`` means here: the innermost scope binding it decides, and a
        scope that binds it any other way (or by two imports) makes it a plain local. Class
        bodies are invisible to the functions and comprehensions nested in them."""
        scope: _Scope | None = self
        while scope is not None:
            if scope is self or scope.kind != "class":
                if name in scope.others:
                    return None
                if name in scope.imports:
                    return scope.imports[name]
            scope = scope.parent
        return None


class _Binder(ast.NodeVisitor):
    """Records each scope's bindings and the scope every loaded name is read in."""

    def __init__(self) -> None:
        self.module = self.scope = _Scope(kind="module")
        self.loads: dict[int, _Scope] = {}

    def _bind(self, name: str) -> None:
        self.scope.others.add(name)

    def _import(self, name: str, target: str) -> None:
        if self.scope.imports.get(name, target) != target:
            self._bind(name)
        self.scope.imports[name] = target

    def _enter(self, kind: str, nodes: Iterable[ast.AST]) -> None:
        outer, self.scope = self.scope, _Scope(self.scope, kind)
        for node in nodes:
            self.visit(node)
        self.scope = outer

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            self._import(alias.asname or top, alias.name if alias.asname else top)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # A relative import keeps its module path minus the dots: rules match the leaf.
        for alias in node.names:
            if alias.name != "*":
                target = f"{node.module}.{alias.name}" if node.module else alias.name
                self._import(alias.asname or alias.name, target)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads[id(node)] = self.scope
        else:
            self._bind(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.others.update(node.names)
        self.module.others.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        scope: _Scope | None = self.scope
        while scope is not None and scope.kind != "module":
            scope.others.update(node.names)
            scope = scope.parent

    def generic_visit(self, node: ast.AST) -> None:
        # except-as, match captures and **rest bind a plain string attribute.
        for attr in ("name", "rest"):
            value = getattr(node, attr, None)
            if isinstance(value, str) and not isinstance(node, ast.alias):
                self._bind(value)
        super().generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        """Decorators, defaults and annotations run in the enclosing scope; the body does not."""
        args = node.args
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        params = [a for a in every if a is not None]
        returns = getattr(node, "returns", None)
        outer = [*getattr(node, "decorator_list", []), *args.defaults,
                 *(d for d in args.kw_defaults if d is not None),
                 *(a.annotation for a in params if a.annotation), *([returns] if returns else [])]
        for expr in outer:
            self.visit(expr)
        if not isinstance(node, ast.Lambda):
            self._bind(node.name)
        self._enter("function", [*params, *(node.body if isinstance(node.body, list) else [node.body])])

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_function

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expr in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expr)
        self._bind(node.name)
        self._enter("class", node.body)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        """The first iterable runs outside; targets, conditions and elements run inside."""
        first, *rest = node.generators
        self.visit(first.iter)
        elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        inner = [first.target, *first.ifs, *(n for g in rest for n in (g.iter, g.target, *g.ifs))]
        self._enter("comprehension", [*inner, *elements])

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # A walrus in a comprehension binds in the enclosing function (PEP 572).
        scope = self.scope
        while scope.kind == "comprehension" and scope.parent is not None:
            scope = scope.parent
        scope.others.add(node.target.id)
        self.visit(node.value)


def canonical_tree(tree: ast.Module) -> ast.Module:
    """``tree`` with every import-bound name spelled out: ``sp.run`` -> ``subprocess.run``,
    ``execute`` (``from subprocess import run as execute``) -> ``subprocess.run``.

    Rules then match canonical API names, so an alias neither hides a call nor lets a local
    helper that merely shares a leaf name (``def wait_for``) pass as the real API. Each read is
    resolved in its own scope: a parameter or local of the same name shadows the import only
    in the function that binds it, and a scope that rebinds the name itself (def, class,
    assignment, parameter) leaves it alone there, since the name is ambiguous.
    """
    binder = _Binder()
    binder.visit(tree)

    class _Spell(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            scope = binder.loads.get(id(node))
            target = scope.resolve(node.id) if scope else None
            if target is None or target == node.id:
                return node
            parts = target.split(".")
            expr: ast.expr = ast.Name(parts[0], ast.Load())
            for part in parts[1:]:
                expr = ast.Attribute(expr, part, ast.Load())
            return ast.copy_location(expr, node)

    return ast.fix_missing_locations(_Spell().visit(tree))


def _dotted(node: ast.AST) -> str:
    """``os.environ.get`` for an Attribute chain, ``""`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        inner = _dotted(node.func)
        return ".".join([f"{inner}()", *reversed(parts)]) if inner else ""
    return ""


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func)


def _str_arg(call: ast.Call, index: int = 0) -> str | None:
    if len(call.args) > index:
        arg = call.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _env_read_name(node: ast.AST) -> str | None:
    """Env var name for ``os.getenv("X")`` / ``os.environ.get("X")`` / ``os.environ["X"]``."""
    if isinstance(node, ast.Call):
        if _call_name(node) in ("os.getenv", "os.environ.get", "environ.get", "getenv"):
            return _str_arg(node)
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if _dotted(node.value) in ("os.environ", "environ"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], (ast.In, ast.NotIn)):
            if _dotted(node.comparators[0]) in ("os.environ", "environ"):
                left = node.left
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    return left.value
    return None


def _env_write_names(node: ast.AST) -> list[str]:
    """Env var names a node SETS: ``os.environ["X"] = ...``, ``setdefault``, ``update``, ``putenv``."""
    if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
        key = node.slice
        if _dotted(node.value) == "os.environ" and isinstance(key, ast.Constant) and isinstance(key.value, str):
            return [key.value]
        return []
    if not isinstance(node, ast.Call):
        return []
    name = _call_name(node)
    if name in ("os.environ.setdefault", "os.putenv"):
        key = _str_arg(node)
        return [key] if key else []
    if name == "os.environ.update":
        keys = [kw.arg for kw in node.keywords if kw.arg]
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                keys += [k.value for k in arg.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        return keys
    return []


def _deferred_parts(node: ast.AST) -> list[ast.AST] | None:
    """For a node whose evaluation defers part of itself, the parts that run NOW; else None.

    A lambda/def runs only its defaults and decorators when evaluated; a generator expression
    runs only its first iterable (the element, conditions and later loops run on consumption).
    Comprehensions and class bodies run immediately, so they are not deferred.
    """
    if isinstance(node, ast.Lambda):
        return [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
    if isinstance(node, _FUNCS):
        defaults = [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
        return [*node.decorator_list, *defaults]
    if isinstance(node, ast.GeneratorExp):
        return [node.generators[0].iter]
    return None


def _eager(node: ast.AST) -> Iterator[ast.AST]:
    """Every node evaluated when ``node`` is evaluated (or a statement executes), root included."""
    stack = [node]
    while stack:
        current = stack.pop()
        parts = _deferred_parts(current)
        if parts is not None:
            stack.extend(parts)
            continue
        yield current
        stack.extend(ast.iter_child_nodes(current))


def hardcoded_home(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if (
                isinstance(node.left, ast.Call)
                and _call_name(node.left).endswith("Path.home")
                and isinstance(right, ast.Constant)
                and isinstance(right.value, str)
                and right.value.strip("/").startswith(".hermes")
            ):
                yield node.lineno
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            arg = _str_arg(node)
            if arg and arg.startswith("~/.hermes") and name.endswith(("expanduser", "Path")):
                yield node.lineno


def new_env_var(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    # Setting a variable introduces it as surely as reading one does.
    for node in ast.walk(tree):
        read = _env_read_name(node)
        names = [read] if read else _env_write_names(node)
        if any(n.startswith("HERMES_") and n not in ctx.known_env for n in names):
            yield getattr(node, "lineno", 0)


def argv_identity(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                left = node.left
                target = ast.unparse(node.comparators[0])
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    if "cmdline" in target:
                        yield node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(cmd in node.value for cmd in _SHELL_IDENTITY):
                yield node.lineno


def unscoped_secret_fallback(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        if "UnscopedSecretError" not in ast.unparse(node.type):
            continue
        for stmt in node.body:
            if any(_env_read_name(inner) for inner in ast.walk(stmt)):
                yield node.lineno
                break


def _is_capture(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name.rsplit(".", 1)[-1] in _CAPTURE_CALLS or name.endswith("Path.home"):
            return True
    return bool(_env_read_name(node))


def _capture_lines(expr: ast.AST) -> Iterator[int]:
    """Every independent capture ``expr`` evaluates: each dict entry or list item is its own
    occurrence, so one added next to an existing one is new debt. A capture nested inside
    another (``expanduser(getenv(...))``) is the same occurrence and is not entered.
    Deferred bodies (lambda, def, generator element) read at call time: that is the fix."""
    stack = [expr]
    while stack:
        node = stack.pop()
        parts = _deferred_parts(node)
        if parts is not None:
            stack.extend(parts)
        elif _is_capture(node):
            yield getattr(node, "lineno", 0)
        else:
            stack.extend(ast.iter_child_nodes(node))


def _main_guard(stmt: ast.stmt) -> str | None:
    """``"=="``/``"!="`` for ``if __name__ <op> "__main__":``, else None."""
    test = stmt.test if isinstance(stmt, ast.If) else None
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
        return None
    sides = {ast.unparse(test.left), ast.unparse(test.comparators[0])}
    if sides != {"__name__", "'__main__'"}:
        return None
    return {ast.Eq: "==", ast.NotEq: "!="}.get(type(test.ops[0]))


# Statement blocks that execute when the enclosing block does (``match`` cases via ``cases``).
_BLOCKS = ("body", "orelse", "finalbody")
_COMPOUND = (ast.If, ast.Try, ast.TryStar, ast.With, ast.For, ast.While, ast.Match)


def _import_time_blocks(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    guard = _main_guard(stmt)
    if isinstance(stmt, ast.If) and guard is not None:
        # Only the branch that runs on import: ``else`` of ``==``, body of ``!=``.
        yield stmt.orelse if guard == "==" else stmt.body
        return
    for name in _BLOCKS:
        yield getattr(stmt, name, None) or []
    for handler in getattr(stmt, "handlers", None) or []:
        yield handler.body
    for case in getattr(stmt, "cases", None) or []:
        yield case.body


def _import_time_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run at import: module/class bodies and every compound block in them."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.ClassDef):
            yield from _import_time_statements(stmt.body)
        elif isinstance(stmt, _COMPOUND):
            for block in _import_time_blocks(stmt):
                yield from _import_time_statements(block)


def _import_time_exprs(stmt: ast.stmt) -> Iterator[ast.AST]:
    """What a statement evaluates at import, besides its nested blocks (walked separately)."""
    if isinstance(stmt, ast.ClassDef):
        yield from (*stmt.decorator_list, *stmt.bases, *(kw.value for kw in stmt.keywords))
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, *_FUNCS)):
        yield stmt  # _eager keeps only a def's decorators and defaults
    elif isinstance(stmt, ast.Expr):
        # A bare call is an action, not a capture; a walrus inside it binds a module name.
        yield from (n for n in _eager(stmt.value) if isinstance(n, ast.NamedExpr))
    elif isinstance(stmt, (ast.If, ast.While)):
        yield stmt.test
    elif isinstance(stmt, ast.For):
        yield stmt.iter
    elif isinstance(stmt, ast.With):
        yield from (item.context_expr for item in stmt.items)
    elif isinstance(stmt, ast.Match):
        yield stmt.subject
        yield from (case.guard for case in stmt.cases if case.guard is not None)


def import_time_capture(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for stmt in _import_time_statements(tree.body):
        for expr in _import_time_exprs(stmt):
            yield from _capture_lines(expr)


_INFINITE = {"inf", "+inf", "infinity", "+infinity"}


def _finite(node: ast.AST | None) -> bool:
    """A deadline that is present and not statically disabled: ``None``, ``math.inf``,
    ``float("inf")`` and an overflowing literal all wait forever. An unknown expression is
    trusted (it is usually a configured value)."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return node.value is not None and node.value != float("inf")
    if _dotted(node).rpartition(".")[2] == "inf":
        return False
    if isinstance(node, ast.Call) and _call_name(node) == "float":
        arg = _str_arg(node)
        return arg is None or arg.strip().lower() not in _INFINITE
    return True


def _deadline(call: ast.Call, name: str = "timeout", position: int | None = None) -> bool:
    """True when ``call`` passes a finite deadline: ``name=<not None>``, the positional slot,
    or ``**opts`` (an unknown mapping is trusted; a literal one must name the deadline)."""
    if position is not None and len(call.args) > position:
        return _finite(call.args[position])
    for kw in call.keywords:
        if kw.arg == name:
            return _finite(kw.value)
        if kw.arg is None:
            if not isinstance(kw.value, ast.Dict):
                return True
            for key, value in zip(kw.value.keys, kw.value.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == name:
                    return _finite(value)
    return False


def _awaited_calls(body: list[ast.stmt]) -> Iterator[ast.Call]:
    """Calls awaited while ``body`` runs; a coroutine defined there runs later, unbounded."""
    for stmt in body:
        for node in _eager(stmt):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                yield node.value


_WAIT_FOR = {"asyncio.wait_for"}
# deadline context manager -> its deadline parameter (positional slot 0)
_TIMEOUT_CMS = {
    "asyncio.timeout": "delay",
    "asyncio.timeout_at": "when",
    "async_timeout.timeout": "delay",
    "async_timeout.timeout_at": "deadline",
}
_SPAWNS = {
    "subprocess.Popen": "sync",
    "asyncio.create_subprocess_exec": "async",
    "asyncio.create_subprocess_shell": "async",
}


def _bounded_calls(tree: ast.Module) -> set[int]:
    """ids of calls an asyncio deadline bounds: the awaitable passed to ``asyncio.wait_for(x,
    <finite>)`` and every call awaited directly inside ``async with asyncio.timeout(<finite>):``.
    Only the real APIs count (names are canonical); a same-named local helper bounds nothing."""
    bounded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _WAIT_FOR:
            awaitable = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg in ("fut", "aw")), None)
            if awaitable is not None and _deadline(node, "timeout", 1):
                bounded.add(id(awaitable))
        elif isinstance(node, ast.AsyncWith):
            for item in node.items:
                ctx_call = item.context_expr
                if not isinstance(ctx_call, ast.Call):
                    continue
                slot = _TIMEOUT_CMS.get(_call_name(ctx_call))
                if slot and _deadline(ctx_call, slot, 0):
                    bounded.update(id(call) for call in _awaited_calls(node.body))
    return bounded


def _spawn_kind(value: ast.AST | None) -> str | None:
    if isinstance(value, ast.Await):
        value = value.value
    return _SPAWNS.get(_call_name(value)) if isinstance(value, ast.Call) else None


def _process_handles(tree: ast.Module) -> dict[str, str]:
    """Dotted names that hold a child process (``proc``, ``self._proc``) -> ``sync``/``async``."""
    handles: dict[str, str] = {}
    for node in ast.walk(tree):
        pairs: list[tuple[ast.AST, ast.AST | None]] = []
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            pairs = [(node.target, node.value)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            pairs = [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars]
        for target, value in pairs:
            kind = _spawn_kind(value)
            if kind and _dotted(target):
                handles[_dotted(target)] = kind
    return handles


# Process methods that wait for the child, with the slot of their ``timeout`` (sync Popen);
# the asyncio Process versions take no timeout and must be bounded by asyncio.
_PROCESS_WAITS = {"communicate": 1, "wait": 0}


def _reaped_after_kill(tree: ast.Module, handles: dict[str, str]) -> set[int]:
    """ids of a sync ``proc.wait()`` that directly follows ``proc.kill()``: SIGKILL bounds it.
    Not ``communicate()`` (it reads until every grandchild holding the pipe exits) and not the
    asyncio ``Process.wait()`` (it waits for the pipe transports too); both measured to hang."""
    reaped: set[int] = set()
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            for first, second in zip(stmts, stmts[1:]):
                kill = first.value if isinstance(first, ast.Expr) else None
                wait = second.value if isinstance(second, (ast.Expr, ast.Assign)) else None
                if not (isinstance(kill, ast.Call) and isinstance(wait, ast.Call)):
                    continue
                head, _, leaf = _call_name(kill).rpartition(".")
                if (leaf == "kill" and handles.get(head) == "sync"
                        and _call_name(wait) == f"{head}.wait"):
                    reaped.add(id(wait))
    return reaped


def missing_timeout(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    handles = _process_handles(tree)
    bounded = _bounded_calls(tree) | _reaped_after_kill(tree, handles)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in bounded:
            continue
        head, _, leaf = _call_name(node).rpartition(".")
        if head == "subprocess" and leaf in _SUBPROCESS_WAITS and not _deadline(node):
            yield node.lineno
        elif leaf == "urlopen" and not _deadline(node, "timeout", 2):
            yield node.lineno
        elif leaf in _PROCESS_WAITS and head in handles:
            if handles[head] == "async" or not _deadline(node, "timeout", _PROCESS_WAITS[leaf]):
                yield node.lineno


def sync_config_in_async(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for func in ast.walk(tree):
        if not isinstance(func, ast.AsyncFunctionDef):
            continue
        for stmt in func.body:
            for node in _eager(stmt):
                if isinstance(node, ast.Call):
                    if _call_name(node).rsplit(".", 1)[-1] in _SYNC_CONFIG_CALLS:
                        yield node.lineno


def get_event_loop(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "asyncio.get_event_loop":
            yield node.lineno


def _is_exc_gather(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Await):
        node = node.value
    if not (isinstance(node, ast.Call) and _call_name(node) == "asyncio.gather"):
        return False
    return any(kw.arg == "return_exceptions" and getattr(kw.value, "value", False) is True
               for kw in node.keywords)


def _names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _from_results(node: ast.AST | None, results: set[str]) -> bool:
    """``node`` is a gather-with-exceptions call, a results name, or built from one
    (``results[i]``, ``zip(xs, results)``, ``enumerate(results)``)."""
    if node is None:
        return False
    if _is_exc_gather(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in results
    if isinstance(node, ast.Subscript):
        return _from_results(node.value, results)
    if isinstance(node, ast.Call) and _call_name(node) in ("zip", "enumerate", "list", "reversed"):
        return any(_from_results(arg, results) for arg in node.args)
    return False


def _result_names(func: ast.AST) -> set[str]:
    """Names bound to gather(return_exceptions=True) results in ``func``'s own body."""
    results: set[str] = set()
    nodes = [n for stmt in func.body for n in _eager(stmt)]
    for _ in range(3):  # results -> loop vars -> unpacked loop vars
        for node in nodes:
            if isinstance(node, ast.Assign) and _from_results(node.value, results):
                results |= set().union(*(_names(t) for t in node.targets))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _from_results(node.iter, results):
                results |= _names(node.target)
    return results


def gather_exception_check(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for func in ast.walk(tree):
        if not isinstance(func, _FUNCS):
            continue
        results = _result_names(func)
        if not results:
            continue
        for stmt in func.body:
            for node in _eager(stmt):
                if (
                    isinstance(node, ast.Call)
                    and _call_name(node) == "isinstance"
                    and len(node.args) == 2
                    and _dotted(node.args[1]) == "Exception"
                    and _from_results(node.args[0], results)
                ):
                    yield node.lineno


def bool_of_env(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "bool" and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, (ast.Call, ast.Subscript)) and _env_read_name(arg):
                yield node.lineno


def _ladder_key(test: ast.expr) -> str | None:
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], (ast.Eq, ast.In, ast.Is)):
            right = test.comparators[0]
            if isinstance(right, (ast.Constant, ast.Tuple, ast.Set, ast.List)):
                return ast.dump(test.left)
    return None


def elif_ladder(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    elifs: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or id(node) in elifs:
            continue
        keys, current = [], node
        while isinstance(current, ast.If):
            keys.append(_ladder_key(current.test))
            nxt = current.orelse
            if len(nxt) == 1 and isinstance(nxt[0], ast.If):
                elifs.add(id(nxt[0]))
                current = nxt[0]
            else:
                break
        if len(keys) >= 4 and keys[0] is not None and len(set(keys)) == 1:
            yield node.lineno


def raw_thread(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "threading.Thread":
            yield node.lineno


CHECKERS: dict[str, Callable[[ast.Module, Ctx], Iterable[int]]] = {
    "HX001": hardcoded_home,
    "HX002": new_env_var,
    "HX003": argv_identity,
    "HX004": unscoped_secret_fallback,
    "HX005": import_time_capture,
    "HX006": missing_timeout,
    "HX007": sync_config_in_async,
    "HX008": get_event_loop,
    "HX009": gather_exception_check,
    "HX010": bool_of_env,
    "HX011": elif_ladder,
    "HX012": raw_thread,
}
