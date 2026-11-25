"""HX013: a wall-clock sleep in a test (registered in ``py_rules.CHECKERS``).

A fixed delay is a race against the runner's load. The one sleep that is not is the interval of
a loop that polls a condition under a bound, so the poll-shape analysis lives here too.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name, _dotted, _eager

# Sleeps that block on the wall clock -> keyword names of their delay (positional slot 0).
_SLEEPS = {
    "time.sleep": ("secs", "seconds"),
    "asyncio.sleep": ("delay",),
    "anyio.sleep": ("delay",),
    "trio.sleep": ("seconds",),
    # a fresh Event nobody can set is a sleep spelled differently
    "threading.Event().wait": ("timeout",),
}
_LOOPS = (ast.While, ast.For, ast.AsyncFor)


def _delay(call: ast.Call, names: tuple[str, ...]) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return next((kw.value for kw in call.keywords if kw.arg in names), None)


def _zero(node: ast.expr | None) -> bool:
    """A literal ``0`` / ``0.0``: a scheduler yield, not a delay (``False`` is not a delay)."""
    return (isinstance(node, ast.Constant) and type(node.value) in (int, float)
            and node.value == 0)


_CLOCKS = {"time.monotonic", "time.time", "time.perf_counter", "time.monotonic_ns", "time.time_ns"}


def _reads_clock(node: ast.AST, clocks: frozenset[str] = frozenset()) -> bool:
    """A wall-clock read: ``time.monotonic()`` and friends, an event loop's ``.time()``, or a
    local in ``clocks`` (every binding of it is derived from one: ``end = time.monotonic() + 5``)."""
    return any((isinstance(inner, ast.Name) and inner.id in clocks)
               or (isinstance(inner, ast.Call) and _clock_call(inner)) for inner in _eager(node))


def _clock_call(call: ast.Call) -> bool:
    name = _call_name(call)
    return name in _CLOCKS or name.endswith(("loop().time", "loop.time"))


def _duration(node: ast.AST) -> bool:
    """A plain number or a name for one (``timeout``, ``self.timeout``), or arithmetic on them."""
    if isinstance(node, ast.BinOp):
        return _duration(node.left) and _duration(node.right)
    if isinstance(node, ast.UnaryOp):
        return _duration(node.operand)
    if isinstance(node, ast.Attribute):
        name = _dotted(node)  # ``self.timeout``, not ``path.stat().st_mtime``
        return bool(name) and "()" not in name
    return isinstance(node, (ast.Constant, ast.Name))


def _clock_value(node: ast.AST, clocks: frozenset[str]) -> bool:
    """A time value: a clock call, a clock local, or one plus/minus a duration or another time
    (``time.monotonic() + timeout``). ``time.time() - path.stat().st_mtime`` is an age, not a time."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _clock_value(node.left, clocks), _clock_value(node.right, clocks)
        return (left or right) and (left or _duration(node.left)) and (right or _duration(node.right))
    if isinstance(node, ast.Name):
        return node.id in clocks
    return isinstance(node, ast.Call) and _clock_call(node)


def _clock_only(leaf: ast.expr, clocks: frozenset[str]) -> bool:
    """A test operand that only watches the clock: it reads one and calls nothing else
    (``time.monotonic() < self.end``, ``time.time() - t["at"] < 0.5``). A call to anything but a
    clock is observed state (``time.time() - path.stat().st_mtime > 5``). Fails closed: an
    attribute compared with the clock (``obj.cooldown_until <= time.monotonic()``) stays clock-only,
    since a stored timestamp looks the same; poll such state through a predicate instead."""
    if not _reads_clock(leaf, clocks):
        return False
    stack: list[ast.AST] = [leaf]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and _clock_call(node):
            continue  # ``asyncio.get_event_loop().time()`` is one clock read
        if isinstance(node, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom)):
            return False
        stack.extend(ast.iter_child_nodes(node))
    return True


def _own(scope: ast.AST) -> Iterator[ast.AST]:
    """Nodes of ``scope`` outside the defs, lambdas and classes nested in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (*_FUNCS, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _clock_names(scope: ast.AST) -> frozenset[str]:
    """Locals of ``scope`` whose every binding assigns a time value (``_clock_value``; a
    constant start value aside): ``end = time.monotonic() + 5``, ``now =
    loop.time()``. A name also bound any other way (parameter, loop target, tuple unpacking)
    is not; ``+=`` keeps what it was."""
    plain: dict[int, ast.AST] = {}
    nodes = list(_own(scope))
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                plain[id(target)] = node.value
    augmented = {id(node.target) for node in nodes if isinstance(node, ast.AugAssign)}
    values: dict[str, list[ast.AST | None]] = {}
    for node in nodes:
        if isinstance(node, ast.arg):
            values.setdefault(node.arg, []).append(None)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in augmented:
            values.setdefault(node.id, []).append(plain.get(id(node)))
    clocks: frozenset[str] = frozenset()
    while True:
        grown = frozenset(name for name, found in values.items() if all(
            isinstance(v, ast.Constant) or (v is not None and _clock_value(v, clocks)) for v in found)
            and not all(isinstance(v, ast.Constant) for v in found))
        if grown == clocks:
            return clocks
        clocks = grown


def _turns_on_condition(test: ast.expr, value: bool, clocks: frozenset[str]) -> bool:
    """Whether ``test`` can become ``value`` on something other than the clock.

    Only an operand that is neither a constant nor clock-only (``_clock_only``) is a condition, so
    ``time.monotonic() < a and time.monotonic() < b`` only waits. ``or`` becomes true when any
    operand does but false only when all do (``and`` the reverse), so a ``while not done() or
    time.monotonic() < end`` test still waits out the clock.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _turns_on_condition(test.operand, not value, clocks)
    if isinstance(test, ast.BoolOp):
        parts = [_turns_on_condition(operand, value, clocks) for operand in test.values]
        return any(parts) if isinstance(test.op, ast.Or) == value else all(parts)
    return not isinstance(test, ast.Constant) and not _clock_only(test, clocks)


def _blocks(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    """The statement lists a non-loop, non-``if`` compound statement runs in its own frame."""
    for name in ("body", "orelse", "finalbody"):
        yield getattr(stmt, name, None) or []
    for part in (*getattr(stmt, "handlers", ()), *getattr(stmt, "cases", ())):
        yield part.body


def _exits_on_condition(body: list[ast.stmt], clocks: frozenset[str], conditioned: bool = False,
                        own: bool = True) -> bool:
    """A ``break`` that ends THIS loop, or a ``return``, in a branch taken on a non-clock
    condition. A ``break`` in a nested loop's body ends that loop (``own=False``), while one in
    its ``else:`` is ours again; nested defs and classes run elsewhere and are skipped."""
    for stmt in body:
        if isinstance(stmt, ast.Return) or (own and isinstance(stmt, ast.Break)):
            if conditioned:
                return True
        elif isinstance(stmt, ast.If):
            branches = ((stmt.body, True), (stmt.orelse, False))
            if any(_exits_on_condition(block, clocks, conditioned or _turns_on_condition(stmt.test, taken, clocks),
                                       own) for block, taken in branches):
                return True
        elif isinstance(stmt, _LOOPS):
            if (_exits_on_condition(stmt.body, clocks, conditioned, own=False)
                    or _exits_on_condition(stmt.orelse, clocks, conditioned, own)):
                return True
        elif isinstance(stmt, (ast.Try, ast.TryStar)) and stmt.handlers:
            # the try body's exit is taken only when nothing raised: `try: return probe()` retries
            probes = any(isinstance(node, ast.Call) and _call_name(node) not in _SLEEPS
                         for part in stmt.body for node in _eager(part))
            if (_exits_on_condition(stmt.body, clocks, conditioned or probes, own)
                    or any(_exits_on_condition(block, clocks, conditioned, own) for block in _blocks(stmt))):
                return True
        elif not isinstance(stmt, (*_FUNCS, ast.ClassDef)):
            if any(_exits_on_condition(block, clocks, conditioned, own) for block in _blocks(stmt)):
                return True
    return False


def _deadline_poll(loop: ast.AST, clocks: frozenset[str]) -> bool:
    """A loop polling a condition under a bound: a ``while`` that reads a clock and can end on
    the condition, or ``for _ in range(n)`` that can (n rounds bound it; load only stretches
    them). ``while time.monotonic() < end: sleep(.1)`` alone is a fixed delay."""
    if isinstance(loop, ast.For):
        bounded = isinstance(loop.iter, ast.Call) and _call_name(loop.iter) == "range"
        return bounded and _exits_on_condition(loop.body, clocks)
    if not isinstance(loop, ast.While):
        return False
    clock = _reads_clock(loop.test, clocks) or any(_reads_clock(stmt, clocks) for stmt in loop.body)
    return clock and (_turns_on_condition(loop.test, False, clocks) or _exits_on_condition(loop.body, clocks))


def _sleep_loops(tree: ast.Module) -> Iterator[tuple[ast.Call, ast.AST | None, ast.AST]]:
    """Each call in ``tree`` with the loop whose BODY runs it in the same function (or None),
    and that function (or the module). A loop's ``else:``, ``for`` iterable and ``while`` test
    run outside its body, so they belong to the enclosing loop."""
    stack: list[tuple[ast.AST, ast.AST | None, ast.AST]] = [(tree, None, tree)]
    while stack:
        node, loop, scope = stack.pop()
        body = {id(stmt) for stmt in node.body} if isinstance(node, _LOOPS) else set()
        for child in ast.iter_child_nodes(node):
            here = node if id(child) in body else loop
            if isinstance(child, ast.Call):
                yield child, here, scope
            if isinstance(child, (*_FUNCS, ast.Lambda)):
                stack.append((child, None, child))
            else:
                stack.append((child, here, scope))


def wall_clock_sleep(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    """``time.sleep(x)`` / ``asyncio.sleep(x)`` (any alias) with ``x`` not a literal zero,
    unless it is the interval of a loop polling a condition under a deadline or round bound.

    Every other non-zero delay counts, however small: the sub-10 ms ones on main are poll
    intervals without a deadline and timestamp-separation hacks, which a Windows runner's
    ~15.6 ms clock already breaks. A non-literal delay counts too.
    """
    clocks: dict[int, frozenset[str]] = {}
    for node, loop, scope in _sleep_loops(tree):
        names = _SLEEPS.get(_call_name(node))
        if names is None:
            continue
        delay = _delay(node, names)
        if names == ("timeout",) and (delay is None or (
                isinstance(delay, ast.Constant) and delay.value is None)):
            continue  # an Event wait without a deadline hangs; it does not sleep
        if _zero(delay):
            continue
        if loop is not None:
            if id(scope) not in clocks:
                clocks[id(scope)] = _clock_names(scope)
            if _deadline_poll(loop, clocks[id(scope)]):
                continue
        yield node.lineno
