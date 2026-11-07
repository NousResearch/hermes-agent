#!/usr/bin/env python3
"""Check that the SQLite backend runs the legacy store's SQL, statement by statement.

Every SQL statement the backend executes, and every function that builds SQL text for one, maps
to its source in a legacy tree through per-family manifests, one JSON file per family in this
directory (``persistence/sqlite/_sqlmap/<family>.json``). The backend is the ``persistence/sqlite``
package and the top-level ``sqlite_support.py`` it shares with the other SQLite stores::

    {
      "family": "conversation",
      "legacy_modules": ["hermes_state_messages.py", "gateway/x.py::Owner.method"],
      "callers": ["gateway/x_owner.py"],
      "wrappers": {"execute_write": "write", "read_ctx": "read"},
      "units": [
        {"backend": "persistence/sqlite/conversation/messages.py::Messages.append",
         "legacy": "hermes_state_messages.py::MessagesMixin.append_message#0-2",
         "names": {"self._conn": "conn"},
         "exception": {"ruling": "read context", "aspects": ["transaction"], "reason": "..."}}
      ],
      "builders": [{"backend": "persistence/sqlite/x.py::_where", "legacy": "hermes_state_x.py::_where"}],
      "not_moved": [{"legacy": "hermes_state_messages.py::Mixin.f#3", "reason": "...", "ruling": "checker limit"}]
    }

A reference is ``<path>::<qualname>``, optionally ``#<selection>``. The path is relative to its
tree's root. The qualname names a module-level function or a method (``Class.method``); nested
functions belong to the function that defines them, and ``<module>`` holds module-level
statements. The selection picks statements by their 0-based position in source order (``#2``,
``#0-2``, ``#0,3``); without one the reference means all of the function's statements. A list
of references is their statements concatenated in the order given. ``scan`` prints every
statement with its reference.

- ``legacy_modules``: legacy files, or single functions, whose SQL the family moves. Every
  statement and builder in them must be the legacy side of a unit or builder entry (in any
  manifest) or be listed in ``not_moved``.
- ``callers``: legacy files read only for their calls: their calls count for inherited
  transactions and for the transactions a handed callable runs in, but none of their statements or
  functions is a SQL source, and no unit, builder or ``legacy_modules`` entry may name them.
- ``wrappers``: callable names that run their body in a transaction, mapped to a kind label. They
  extend the built-in table (``_execute_write``, ``_execute_transcript_write``: write;
  ``_read_ctx``, ``_read_retrying_ioerr``: read) and apply to both trees. An entry may instead be
  ``{"kind": "write", "begin": "immediate", "default": false}``: a call's label is then the kind with
  only its begin mode, the named keyword's value at the call or ``default`` without it
  (``write(immediate=True)``), every other argument (the file, the connection opener) left out; with
  no ``begin`` the label is the bare kind. A ``with`` block whose first statement executes
  ``BEGIN IMMEDIATE`` begins immediate whatever the keyword says. A key ``<path>::<name>`` applies to
  calls in that file only and wins over the bare name. ``partial(W, f, ...)`` and
  ``asyncio.to_thread(W, f, ...)`` of a wrapper ``W`` are the call ``W(f, ...)``.
- ``units``: backend statements paired position by position with legacy statements. ``legacy``
  is null for statements with no legacy source, which need a ``source`` exception. ``names``
  renames legacy names (exact dotted names) to the backend's before expressions are compared, in the
  transaction labels the statements inherit from their callers too.
- ``exception``: a ruled difference. Differences in the listed aspects are reported as ruled
  exceptions instead of failures; an exception that excuses nothing fails. Aspects: source,
  count, order, method, text, binding, builder, transaction. Its ``ruling`` is one of the
  categories below and its ``reason`` says how the difference keeps main's behaviour.
- ``inherits``: on a unit of one backend and one legacy function, the transaction label the legacy
  function inherits from callers that have not moved yet, its arguments written with or without
  ``self.``. The backend function inherits it on top of its own callers' kinds, and so do the
  functions it calls. It fails unless it is the legacy function's inherited label, some legacy
  caller is outside every unit, and the backend's own callers, under the other declarations, do
  not give it already. ``check`` lists every declaration it accepted.
- ``builders``: backend functions whose return value feeds SQL text, paired with the legacy
  function that builds the same text; ``names`` and ``exception`` work as in units. A function feeds
  SQL text when the text, its builder slice or another such function calls it: a bare name calls the
  function its enclosing scopes or imports bind (no function for a builtin or a function outside the
  tree, every function of that name when the name is bound otherwise), a method call the functions of
  the inheritance group it is looked up in when that group defines the name (as below), any other call
  every function of its name. An entry whose backend function is not found to feed SQL text fails.
- ``not_moved``: legacy statements or builders of the family that stay behind, with a reason and
  optionally a ``ruling`` category. A reference naming a builder covers the builder only, never the
  statements of its function.

Rulings name one of these categories; any other value makes the manifest unusable:

* ``boundary translation``: the verb takes or returns typed values at the storage boundary (a typed
  record, a value the caller computes or validates, a parameter where main bound a literal, several of
  main's statements as one parametrized statement); the SQL binds the values main bound.
* ``no caller code in a transaction``: main ran caller code (a probe, a hook, a rule that needs process
  or gateway state, work after the commit) inside the store's transaction; the backend takes its
  outcome as a parameter, or returns the data the caller judges outside the transaction.
* ``read context``: main read outside the store's transactions (a connection of its own, a deferred
  transaction that never takes the write lock, autocommit statements); the backend reads in its read
  context, with main's lock wait where main set one.
* ``schema step``: DDL main ran in a constructor or inside a later transaction runs in the backend's
  schema step for that file.
* ``fewer callers``: a copy of a main helper that only some of main's callers reach, so it inherits
  only their transaction labels.
* ``stated difference``: a behaviour difference the pull request states.
* ``checker limit``: the difference lies in what this checker infers (calls matched by name, calls it
  does not follow, builder detection that over-includes, a builder slice holding arguments that are
  not SQL), not in the SQL the store runs or the transaction it runs in.

Each pair of statements is compared on the execute method; the normalized SQL text (whitespace and
comments outside literals collapsed, each dynamic part shown as the expression filling it); the
binding expression (a local assigned once is inlined); the builder slice (the local statements that
build dynamic text or a built parameter list, with their enclosing conditions); and the
transaction: the kind and arguments of the innermost enclosing wrapper, or, when no wrapper
encloses the statement, the kinds its function is called under, over caller chains of up to six
calls (a function also called outside any transaction reports only the transaction kinds). Within
a unit, statements sharing one legacy transaction must share one backend transaction, and the
reverse. Execution order is source order,
so a reordered statement fails as ``order``. SQL handed to a helper that executes its own ``sql``
parameter, directly or inside a nested function or lambda it hands to a wrapper, is also extracted
at every call of that helper, under the helper's transaction, with the helper's binding and execute
method (``(conn.executemany if many else conn.execute)`` too) read with the call's arguments in
place of the helper's parameters. For a helper's SQL and for inherited transactions, a call counts
as a call of a function when it names the function and its arguments fit the signature; a bare name
reaches only the function its scopes or imports bind, as for builders; a call on
``self``, ``cls`` or a class name reaches only the functions of that class's inheritance group (the
classes it derives from or that derive from it) when the group defines that name; a call on any
other receiver (a parameter such as ``holder``, an attribute, a call's result) is looked up in the
group of the tree's store class (``StoreHolder`` in the backend, ``SessionDB`` in the legacy tree),
so a second class defining the store's read primitives is not a target of the store's calls. A
bare name never reaches a method, and a method called on a builtin value (a local bound only to what
an execute call returns, or a module-level name bound only to dict, list or set displays) reaches
none of them.
A callable handed to a call as an argument runs in the transaction its callee calls that parameter
in: when the callee calls the parameter inside a wrapper of its own, a lambda or nested function passed
there (directly, or through a local it is assigned to) runs under that wrapper's label, and so does a
function passed by its bare name or as ``partial(f, ...)``. A callee that passes the parameter on as a
call's argument runs it where that call does; a callee that returns the parameter, or a nested
function calling it outside any wrapper (a guard factory), runs it where the factory call's result is
handed on as an argument. ``partial(f, ...)`` hands its arguments to ``f``'s parameters, the value
handed on may be one branch of a conditional or of ``and``/``or``, and a nested function that its
enclosing function returns, and neither calls nor hands on itself, runs where that function's call
results are handed on.
Names in SQL text resolve through imports, also from the same tree's modules outside the checked
files (read for their names only). An attribute of a parameter whose class the checker cannot see
(``db._COLUMNS``) resolves as an attribute of the store class.

    check.py check --legacy M1_TREE --branch BRANCH_TREE [--manifests DIR] [--json]
    check.py scan --tree TREE PATH... [--wrapper NAME=KIND] [--json]
    check.py draft --legacy M1_TREE --branch BRANCH_TREE --family NAME --legacy-module REF... PATH...

``check`` exits 0 when everything maps, 1 on any failure, 2 on unusable input.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EXECUTE_METHODS = ("execute", "executemany", "executescript")
DEFAULT_WRAPPERS = {
    "_execute_write": "write",
    "_execute_transcript_write": "write",
    "_read_ctx": "read",
    "_read_retrying_ioerr": "read",
}
TRANSPARENT_CALLS = ("partial",)
FORWARDING_CALLS = ("partial", "to_thread")
ASPECTS = ("source", "count", "order", "method", "text", "binding", "builder", "transaction")
COMPARED = ("method", "text", "binding", "builder", "transaction")
NOT_BUILDERS = ("append", "extend", "insert", "update", "add", "join", "format")
MUTATORS = NOT_BUILDERS[:5]
BACKEND_ROOT = "persistence/sqlite"
BACKEND_FILES = ("sqlite_support.py",)
MANIFEST_DIR = "persistence/sqlite/_sqlmap"
LEGACY_GLOB = "hermes_state*.py"
MODULE_SCOPE = "<module>"
_MAX_DEPTH = 6
RULINGS = tuple(re.findall(r"^\* ``([^`]+)``:", __doc__, re.MULTILINE))
_TEXT_DEPTH = 60
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)

_SQL_TOKEN = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|--[^\n]*|/\*.*?\*/|\s+|[(),;]|[^\s(),;'\"`]+|.",
    re.DOTALL,
)
_SLOT = "\x00"
_REF = re.compile(r"(?P<path>[^:]+)::(?P<qual>[^#]+)(?:#(?P<sel>[0-9,\- ]+))?")


class ManifestError(Exception):
    """A manifest or reference that cannot be used."""


def normalize_sql(text: str) -> str:
    """Collapse whitespace and drop comments outside literals; keep everything else."""
    out: list[str] = []
    for token in _SQL_TOKEN.findall(text):
        if token.isspace() or token.startswith(("--", "/*")):
            continue
        if out and token not in ",;)" and out[-1] != "(":
            out.append(" ")
        out.append(token)
    return re.sub(_SLOT + r"(\d+)" + _SLOT, r"{\1}", "".join(out))


def terminal(node: ast.AST) -> str:
    """Last name of a call target (``a.b.c`` -> ``c``)."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


class _Substitute(ast.NodeTransformer):
    """Replace names by expressions (a helper's parameters by a call's arguments)."""

    def __init__(self, values: dict[str, ast.AST]):
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return self.values.get(node.id, node)


class _Renamer(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]):
        self.names = names

    def _swap(self, node: ast.AST) -> ast.AST:
        name = dotted(node)
        if name in self.names:
            return ast.parse(self.names[name], mode="eval").body
        return self.generic_visit(node)

    visit_Name = _swap
    visit_Attribute = _swap


# The receivers a moved method's statements reach their store through: ``self`` and ``cls`` in the legacy
# store's methods, ``db`` in the legacy module functions that take the store first, ``holder`` in the backend
# functions copied from either, which take the store holder first.
_RECEIVERS = frozenset({"self", "cls", "db", "holder"})
# The store class of each tree: a receiver whose class the checker cannot see is an instance of it.
STORE_CLASSES = ("StoreHolder", "SessionDB")


def _is_receiver(node: ast.AST) -> bool:
    value = node.value if isinstance(node, ast.keyword) else node
    return isinstance(value, ast.Name) and value.id in _RECEIVERS and getattr(node, "arg", value.id) in _RECEIVERS


class _Receiverless(ast.NodeTransformer):
    """An expression without its receiver, so a method and its function copy read alike: ``r.name`` is
    ``name``, a call's leading receiver argument and receiver keyword are dropped, and ``partial(f, r)``
    is ``f``, for ``r`` in ``_RECEIVERS``."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in _RECEIVERS:
            return ast.Name(node.attr, node.ctx)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if dotted(node.func) == "partial" and len(node.args) == 2 and _is_receiver(node.args[1]):
            return self.visit(node.args[0])
        if node.args and _is_receiver(node.args[0]):
            node.args = node.args[1:]
        node.keywords = [k for k in node.keywords if not _is_receiver(k)]
        return self.generic_visit(node)


def _txn_args(args: list) -> list:
    """A transaction wrapper's arguments without a receiver passed to it."""
    return [a for a in args if not _is_receiver(a)]


def render(node: ast.AST | None, names: dict[str, str] | None = None) -> str:
    if node is None:
        return ""
    if names:
        node = _Renamer(names).visit(copy.deepcopy(node))
    node = _Receiverless().visit(copy.deepcopy(node))
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}"
    return ast.unparse(node)


def _slice_line(conditions: list, stmt: ast.AST, names: dict[str, str] | None) -> str:
    prefix = "".join(f"[{tag} {render(n, names)}] " for tag, n in conditions)
    return prefix + render(stmt, names)


@dataclass
class Site:
    """One executed statement: where it runs, its text, binding, builder slice and transaction."""

    file: str
    func: str
    line: int
    col: int
    method: str
    template: str
    slots: list = field(default_factory=list)
    binding: ast.AST | None = None
    slice: list = field(default_factory=list)
    txn_kind: str = "none"
    txn_args: list = field(default_factory=list)
    group: str | None = None
    calls: set = field(default_factory=set)
    ordinal: int = 0

    @property
    def ref(self) -> str:
        return f"{self.file}::{self.func}#{self.ordinal}"

    def view(self, names: dict[str, str] | None = None) -> dict:
        slots = [render(slot, names) for slot in self.slots]
        text = re.sub(r"\{(\d+)\}", lambda m: "{" + slots[int(m[1])] + "}", self.template)
        args = ", ".join(render(a, names) for a in _txn_args(self.txn_args))
        kind = receiverless_label(self.txn_kind, names) if self.group is None else self.txn_kind
        return {
            "method": self.method,
            "text": text,
            "binding": render(self.binding, names),
            "builder": [_slice_line(c, s, names) for c, s in self.slice],
            "transaction": f"{kind}({args})" if args else kind,
        }


def _module_candidates(rel: str, module: str | None, level: int) -> list[str]:
    """Files an import may name, relative to the tree root."""
    parts = (module or "").split(".") if module else []
    if level:
        base = Path(rel).parent
        for _ in range(level - 1):
            base = base.parent
        stem = base.joinpath(*parts) if parts else base
    else:
        stem = Path(*parts)
    return [f"{stem.as_posix()}.py", f"{stem.as_posix()}/__init__.py"]


class Module:
    """One parsed file: parents, module constants, imports, classes and functions."""

    def __init__(self, root: Path, rel: str):
        self.rel = rel
        self.tree = ast.parse((root / rel).read_text(encoding="utf-8"), rel)
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.consts: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.outer: dict[ast.AST, tuple[str, ast.ClassDef | None]] = {}
        self.imports: dict[str, tuple[list[str], str]] = {}
        self.aliases: dict[str, list[str]] = {}
        self._index_body(self.tree.body, "", None)
        for node in ast.walk(self.tree):
            self._index_import(node)

    def _index_body(self, body: list, prefix: str, cls: ast.ClassDef | None) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                self.classes.setdefault(node.name, node)
                self._index_body(node.body, f"{prefix}{node.name}.", node)
            elif isinstance(node, _DEFS):
                self.outer[node] = (prefix + node.name, cls)
            elif isinstance(node, ast.Assign) and not prefix:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.consts[target.id] = node.value

    def _index_import(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.aliases[alias.asname or alias.name] = _module_candidates(self.rel, alias.name, 0)
        elif isinstance(node, ast.ImportFrom):
            source = _module_candidates(self.rel, node.module, node.level)
            for alias in node.names:
                local = alias.asname or alias.name
                self.imports[local] = (source, alias.name)
                sub = f"{node.module}.{alias.name}" if node.module else alias.name
                self.aliases[local] = _module_candidates(self.rel, sub, node.level)

    def outer_of(self, node: ast.AST) -> ast.AST | None:
        while node in self.parents:
            node = self.parents[node]
            if node in self.outer:
                return node
        return None

    def qualname(self, node: ast.AST) -> str:
        outer = self.outer_of(node)
        return self.outer[outer][0] if outer is not None else MODULE_SCOPE

    def innermost_def(self, node: ast.AST) -> ast.AST | None:
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, _DEFS):
                return node
        return None

    def def_qualname(self, node: ast.AST) -> str:
        names = [node.name]
        current = node
        while current in self.parents:
            current = self.parents[current]
            if isinstance(current, (*_DEFS, ast.ClassDef)):
                names.append(current.name)
        return ".".join(reversed(names))


def _targets(node: ast.AST) -> list:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in node.items if item.optional_vars]
    return [node.target]


_ASSIGNING = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
              ast.NamedExpr, ast.comprehension)


def _scope_of(module: Module, node: ast.AST) -> ast.AST | None:
    """The function or class whose body binds the names a node binds; None for the module."""
    while node in module.parents:
        node = module.parents[node]
        if isinstance(node, (*_DEFS, ast.ClassDef)):
            return node
    return None


def _locals_of(func: ast.AST | None) -> tuple[dict[str, list], set[str]]:
    """Values assigned per local name (None marks a non-simple assignment), and all parameters."""
    assigned: dict[str, list] = {}
    params: set[str] = set()
    for node in ast.walk(func) if func is not None else []:
        if isinstance(node, (*_DEFS, ast.Lambda)):
            args = node.args
            params.update(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)
            params.update(a.arg for a in (args.vararg, args.kwarg) if a)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _simple_target(node):
            if node.value is not None:
                assigned.setdefault(_targets(node)[0].id, []).append(node.value)
        elif isinstance(node, _ASSIGNING):
            for target in _targets(node):
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        assigned.setdefault(name.id, []).append(None)
    return assigned, params


def _simple_target(node: ast.AST) -> bool:
    targets = _targets(node)
    return len(targets) == 1 and isinstance(targets[0], ast.Name)


class Resolver:
    """Static text of a SQL expression inside one function; unresolved parts become slots."""

    def __init__(self, index: Tree, module: Module, func: ast.AST | None):
        self.index, self.module, self.func = index, module, func
        self.cls = module.outer[func][1] if func in module.outer else None
        self.assigned, self.params = _locals_of(func)
        self.slots: list[ast.AST] = []
        self.dynamic: set[str] = set()
        self.calls: set[str] = set()
        self.depth = 0

    def template(self, node: ast.AST) -> str:
        parts = self.text(node)
        return normalize_sql("".join(p if isinstance(p, str) else f"{_SLOT}{p}{_SLOT}" for p in parts))

    def slot(self, node: ast.AST) -> list:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                self.calls.update(self.index.call_names(self.module, sub))
            elif isinstance(sub, ast.Name) and (sub.id in self.assigned or sub.id in self.params):
                self.dynamic.add(sub.id)
        self.slots.append(node)
        return [len(self.slots) - 1]

    def text(self, node: ast.AST) -> list:
        handler = _TEXT_HANDLERS.get(type(node))
        if handler is None or self.depth > _TEXT_DEPTH:
            return self.slot(node)
        self.depth += 1
        try:
            return handler(self, node)
        finally:
            self.depth -= 1

    def _constant(self, node: ast.Constant) -> list:
        return [node.value] if isinstance(node.value, str) else self.slot(node)

    def _joined(self, node: ast.JoinedStr) -> list:
        parts: list = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            elif value.conversion == -1 and value.format_spec is None:
                parts.extend(self.text(value.value))
            else:
                parts.extend(self.slot(ast.JoinedStr(values=[value])))
        return parts

    def _binop(self, node: ast.BinOp) -> list:
        if isinstance(node.op, ast.Add):
            return self.text(node.left) + self.text(node.right)
        return self.slot(node)

    def _name(self, node: ast.Name) -> list:
        if node.id in self.params:
            return self.slot(node)
        values = self.assigned.get(node.id)
        if values is None:
            return self._global(self.module, node.id, node)
        if len(values) == 1 and values[0] is not None and not self._in_loop(values[0]):
            return self.text(values[0])
        return self.slot(node)

    def _in_loop(self, value: ast.AST) -> bool:
        node = value
        while node in self.module.parents and node is not self.func:
            node = self.module.parents[node]
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                return True
        return False

    def _global(self, module: Module, name: str, node: ast.AST) -> list:
        if name in module.consts:
            return self._foreign(module, module.consts[name], node)
        target = self.index.imported(module, name)
        return self._foreign(target[0], target[1], node) if target else self.slot(node)

    def _foreign(self, module: Module, value: ast.AST, node: ast.AST) -> list:
        """A module-level value resolved in its own module; only fully static text is kept."""
        inner = Resolver(self.index, module, None)
        inner.depth = self.depth
        parts = inner.text(value)
        return self.slot(node) if any(isinstance(p, int) for p in parts) else parts

    def _attribute(self, node: ast.Attribute) -> list:
        owner = dotted(node.value)
        found = None
        if owner in ("self", "cls"):
            found = self.index.class_attr(self.module, self.cls, node.attr) if self.cls is not None else None
            found = found or self.index.only_class_attr(node.attr)
        elif owner:
            cls = self.index.find_class(self.module, owner)
            found = self.index.class_attr(cls[0], cls[1], node.attr) if cls else None
            found = found or self.index.module_const(self.module, owner, node.attr)
            if not found and owner in self.params and (store := self.index.store_class()):
                found = self.index.class_attr(store[0], store[1], node.attr)
        return self._foreign(found[0], found[1], node) if found else self.slot(node)

    def _call(self, node: ast.Call) -> list:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format" and all(k.arg for k in node.keywords):
            mark = len(self.slots)
            receiver = self.text(func.value)
            args = ["".join(map(str, self.text(a))) for a in node.args]
            kwargs = {k.arg: "".join(map(str, self.text(k.value))) for k in node.keywords}
            if len(self.slots) == mark:
                return ["".join(receiver).format(*args, **kwargs)]
            del self.slots[mark:]
        return self.slot(node)


_TEXT_HANDLERS = {
    ast.Constant: Resolver._constant,
    ast.JoinedStr: Resolver._joined,
    ast.BinOp: Resolver._binop,
    ast.Name: Resolver._name,
    ast.Attribute: Resolver._attribute,
    ast.Call: Resolver._call,
}


def _conditions(module: Module, node: ast.AST, stop: ast.AST) -> list:
    """Enclosing ``if``/``for``/``while`` headers of a statement, outermost first."""
    out = []
    child = node
    while child in module.parents and child is not stop:
        parent = module.parents[child]
        if isinstance(parent, ast.If) and child is not parent.test:
            out.append(("if" if child in parent.body else "not", parent.test))
        elif isinstance(parent, (ast.For, ast.AsyncFor)) and child in parent.body:
            out.append(("for", parent))
        elif isinstance(parent, ast.While) and child in parent.body:
            out.append(("while", parent.test))
        child = parent
    return out[::-1]


def _builds(stmt: ast.AST, name: str) -> bool:
    """Whether a statement assigns, extends or mutates the local ``name``."""
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor)):
        return any(isinstance(n, ast.Name) and n.id == name for t in _targets(stmt) for n in ast.walk(t))
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        return isinstance(func, ast.Attribute) and func.attr in MUTATORS and dotted(func.value) == name
    return False


def builder_slice(module: Module, func: ast.AST, roots: set[str], assigned: dict) -> list:
    """Local statements that build the root names, transitively, in source order."""
    stmts = [n for n in ast.walk(func) if isinstance(n, ast.stmt) and n is not func]
    seen: set[str] = set()
    picked: dict[int, ast.AST] = {}
    work = [r for r in roots if r in assigned]
    while work:
        name = work.pop()
        if name in seen:
            continue
        seen.add(name)
        for stmt in stmts:
            if _builds(stmt, name):
                picked[id(stmt)] = stmt
                work.extend(_used_names(stmt.iter if isinstance(stmt, (ast.For, ast.AsyncFor)) else stmt, assigned))
    ordered = sorted(picked.values(), key=lambda s: (s.lineno, s.col_offset))
    return [(_conditions(module, s, func), s) for s in ordered]


def _used_names(node: ast.AST, assigned: dict) -> list[str]:
    """Locals a statement reads, leaving out call receivers (``conn`` in ``conn.execute(...)``)."""
    receivers = {id(c.func.value) for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
    return [n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in assigned and id(n) not in receivers]


def _method(func: ast.AST, bound: dict | None = None) -> str | None:
    """The execute method a call target names; ``(c.executemany if many else c.execute)`` picks its
    branch when ``bound`` gives the test a constant, else reads ``executemany if many else execute``."""
    if isinstance(func, ast.Attribute):
        return func.attr if func.attr in EXECUTE_METHODS else None
    if not isinstance(func, ast.IfExp):
        return None
    body, orelse = _method(func.body), _method(func.orelse)
    if body is None or orelse is None:
        return None
    test = (bound or {}).get(dotted(func.test), func.test)
    if isinstance(test, ast.Constant):
        return body if test.value else orelse
    return f"{body} if {render(test)} else {orelse}"


def _binding_arg(call: ast.Call) -> ast.AST | None:
    if len(call.args) > 1:
        return call.args[1]
    for keyword in call.keywords:
        if keyword.arg in ("parameters", "params", "seq_of_parameters"):
            return keyword.value
    return None


def _param_names(func: ast.AST) -> list[str]:
    args = func.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _bind_call(func: ast.AST, call: ast.Call) -> dict[str, ast.AST]:
    """Arguments of a call by the callee's parameter names, defaults filled in."""
    args = func.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    bound = dict(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    bound.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None})
    if positional and positional[0] in ("self", "cls") and isinstance(call.func, ast.Attribute):
        positional = positional[1:]
    bound.update({n: a for n, a in zip(positional, call.args) if not isinstance(a, ast.Starred)})
    bound.update({k.arg: k.value for k in call.keywords if k.arg})
    return bound


def _is_wrapper(value) -> bool:
    """A wrapper entry: a kind label, or a kind with the keyword naming its begin mode and that keyword's default."""
    if isinstance(value, str):
        return True
    return (isinstance(value, dict) and not set(value) - {"kind", "begin", "default"} and isinstance(value.get("kind"), str)
            and isinstance(value.get("begin", ""), str) and isinstance(value.get("default", False), bool))


def _begins_immediate(stmt: ast.AST) -> bool:
    """Whether a statement is an execute call of ``BEGIN IMMEDIATE``."""
    call = stmt.value if isinstance(stmt, ast.Expr) else None
    return (isinstance(call, ast.Call) and terminal(call.func) in EXECUTE_METHODS and len(call.args) >= 1
            and isinstance(call.args[0], ast.Constant) and str(call.args[0].value).strip().upper() == "BEGIN IMMEDIATE")


def _holds(arg: ast.AST, child: ast.AST) -> bool:
    """Whether a wrapper argument carries the closure (directly or through a partial)."""
    value = arg.value if isinstance(arg, ast.keyword) else arg
    return any(n is child for n in ast.walk(value))


class Tree:
    """Every SQL statement and SQL-building function in a set of files of one tree."""

    def __init__(self, root: Path, rels, wrappers: dict[str, str | dict], inherits: dict | None = None,
                 callers=()):
        self.root, self.wrappers, self.inherits = root, wrappers, inherits or {}
        self.callers = frozenset(callers)
        self.modules = {rel: Module(root, rel) for rel in sorted(set(rels) | self.callers) if (root / rel).is_file()}
        self.context: dict[str, Module | None] = {}
        self.calls_by_name: dict[str, list] = {}
        self.defs: dict[str, list] = {}
        for module in self.modules.values():
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Call):
                    self.calls_by_name.setdefault(terminal(node.func), []).append((module, node))
                elif isinstance(node, _DEFS):
                    self.defs.setdefault(node.name, []).append((module, node))
        self._inherited: dict[tuple[ast.AST, int], tuple] = {}
        self._builders: dict | None = None
        self._scopes: dict[tuple[str, int], dict[str, list]] = {}
        self._hooks: dict[ast.AST, dict[str, list]] = {}
        self._partials: dict[ast.Call, ast.Call] = {}
        self._locals: dict[ast.AST | None, dict[str, list]] = {}
        self._handed: dict[str, list] | None = None
        self._groups: dict[tuple[str, str], tuple[str, str]] | None = None
        self.sites: dict[tuple[str, str], list[Site]] = {}
        self._extract()

    # -- name lookup --------------------------------------------------------------

    def _module(self, candidates: list[str]) -> Module | None:
        """The module an import names: one of the tree's files, else the same tree's file on disk, read
        only to resolve its names (its statements are not extracted)."""
        found = next((self.modules[c] for c in candidates if c in self.modules), None)
        for rel in candidates if found is None else []:
            if rel not in self.context:
                self.context[rel] = Module(self.root, rel) if (self.root / rel).is_file() else None
            found = found or self.context[rel]
        return found

    def imported(self, module: Module, name: str) -> tuple[Module, ast.AST] | None:
        if name not in module.imports:
            return None
        candidates, original = module.imports[name]
        target = self._module(candidates)
        return (target, target.consts[original]) if target and original in target.consts else None

    def module_const(self, module: Module, alias: str, name: str) -> tuple[Module, ast.AST] | None:
        target = self._module(module.aliases.get(alias, []))
        return (target, target.consts[name]) if target and name in target.consts else None

    def find_class(self, module: Module, name: str) -> tuple[Module, ast.ClassDef] | None:
        if name in module.classes:
            return module, module.classes[name]
        if name not in module.imports:
            return None
        candidates, original = module.imports[name]
        target = self._module(candidates)
        return (target, target.classes[original]) if target and original in target.classes else None

    def only_class_attr(self, attr: str) -> tuple[Module, ast.AST] | None:
        """The class attribute of that name when exactly one class in the tree assigns it."""
        hits = [(module, node.value) for module in self.sources() for cls in module.classes.values()
                for node in cls.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets)]
        return hits[0] if len(hits) == 1 else None

    def class_attr(self, module: Module, cls: ast.ClassDef, attr: str, depth: int = 0):
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets):
                return module, node.value
        for base in cls.bases if depth < _MAX_DEPTH else []:
            found = self.find_class(module, terminal(base))
            hit = self.class_attr(found[0], found[1], attr, depth + 1) if found else None
            if hit:
                return hit
        return None

    def groups(self) -> dict[tuple[str, str], tuple[str, str]]:
        """Each class's inheritance group: classes joined through the bases they name, transitively."""
        if self._groups is None:
            parent: dict[tuple[str, str], tuple[str, str]] = {}

            def find(key):
                while parent.setdefault(key, key) != key:
                    key = parent[key]
                return key

            for module in self.modules.values():
                for cls in module.classes.values():
                    for base in cls.bases:
                        found = self.find_class(module, terminal(base))
                        if found:
                            parent[find((module.rel, cls.name))] = find((found[0].rel, found[1].name))
                    find((module.rel, cls.name))
            self._groups = {key: find(key) for key in parent}
        return self._groups

    def receiver_group(self, module: Module, call: ast.Call) -> tuple[str, str] | None:
        """The inheritance group a method call is looked up in: the caller's class for ``self`` and
        ``cls``, the class a receiver name refers to, or else the store class."""
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        if receiver in ("self", "cls"):
            outer = module.outer_of(call)
            cls = module.outer[outer][1] if outer is not None else None
            found = (module, cls) if cls is not None else None
        else:
            found = (self.find_class(module, receiver) if receiver else None) or self.store_class()
        return self.groups().get((found[0].rel, found[1].name)) if found else None

    def store_class(self) -> tuple[Module, ast.ClassDef] | None:
        hits = [(m, m.classes[name]) for m in self.modules.values() for name in STORE_CLASSES if name in m.classes]
        return hits[0] if len(hits) == 1 else None

    def _def_group(self, module: Module, func: ast.AST) -> tuple[str, str] | None:
        cls = module.outer.get(func, (None, None))[1]
        return self.groups().get((module.rel, cls.name)) if cls is not None else None

    # -- extraction ---------------------------------------------------------------

    def sources(self) -> list[Module]:
        """The modules whose statements and functions are SQL sources: every module but the caller files."""
        return [m for m in self.modules.values() if m.rel not in self.callers]

    def _extract(self) -> None:
        relays: dict[tuple, tuple] = {}
        for module in self.sources():
            for node in ast.walk(module.tree):
                method = _method(node.func) if isinstance(node, ast.Call) else None
                if method is not None:
                    sql = node.args[0] if node.args else None
                    site = self._site(module, node, sql, _binding_arg(node), method)
                    self._note_relay(relays, module, node, sql, site)
        done: set[tuple] = set()
        while len(done) < len(relays):
            fresh = [key for key in relays if key not in done]
            done.update(fresh)
            for key in fresh:
                for module, call in self.calls_by_name.get(relays[key][0].name, []):
                    if self.may_call(module, call, relays[key][4], relays[key][0]):
                        self._relay_site(relays, relays[key], module, call)
        for sites in self.sites.values():
            sites.sort(key=lambda s: (s.line, s.col))
            for i, site in enumerate(sites):
                site.ordinal = i

    def _site(self, module: Module, call: ast.Call, sql, binding, method: str) -> Site:
        func = module.outer_of(call)
        resolver = Resolver(self, module, func)
        template = resolver.template(sql) if sql is not None else ""
        binding = self._inline_binding(resolver, binding)
        built = {binding.id} if isinstance(binding, ast.Name) else set()
        site = Site(module.rel, module.qualname(call), call.lineno, call.col_offset, method, template,
                    slots=resolver.slots, binding=binding)
        text_slice = builder_slice(module, func, resolver.dynamic, resolver.assigned) if func else []
        site.slice = builder_slice(module, func, resolver.dynamic | built, resolver.assigned) if func else []
        site.calls = (resolver.calls | self.calls_in(module, (s for _, s in text_slice))) - set(NOT_BUILDERS)
        site.txn_kind, site.txn_args, site.group = self.transaction(module, call)
        self.sites.setdefault((site.file, site.func), []).append(site)
        return site

    @staticmethod
    def _inline_binding(resolver: Resolver, binding: ast.AST | None) -> ast.AST | None:
        if not isinstance(binding, ast.Name) or resolver.func is None:
            return binding
        values = resolver.assigned.get(binding.id) or []
        mutated = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and dotted(n.func.value) == binding.id
            for n in ast.walk(resolver.func)
        )
        return values[0] if len(values) == 1 and values[0] is not None and not mutated else binding

    def _note_relay(self, relays: dict, module: Module, call: ast.Call, sql, site: Site) -> None:
        """Record a function that executes SQL it receives as a parameter, directly or inside a nested
        function or lambda it defines."""
        func = _sql_owner(module, call, sql)
        if func is not None:
            relays.setdefault((id(func), sql.id), (func, sql.id, call, site, module))

    def may_call(self, module: Module, call: ast.Call, home: Module, func: ast.AST) -> bool:
        """Whether a call of ``func``'s name can run it: the arguments fit its signature, a bare name
        runs no method and only the function its scopes or imports bind (any of that name when bound
        otherwise), an attribute of an imported module of the tree runs only that module's function, and a
        call looked up in an inheritance group that defines the name runs only that group's definition."""
        if not _accepts(func, call) or _builtin_receiver(module, call):
            return False
        if isinstance(call.func, ast.Name):
            if isinstance(home.parents.get(func), ast.ClassDef):
                return False
            refs = self._bare(module, call, call.func.id, 0)
            return refs is None or f"{home.rel}::{home.def_qualname(func)}" in refs
        receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
        if isinstance(receiver, ast.Name):
            imported = [f for f in module.aliases.get(receiver.id, ()) if f in self.modules]
            if imported:
                return home.rel in imported and not isinstance(home.parents.get(func), ast.ClassDef)
        grouped = self._group_defs(module, call, func.name)
        return not grouped or any(d is func for _, d in grouped)

    def _group_defs(self, module: Module, call: ast.Call, name: str) -> list:
        """The functions named ``name`` in the inheritance group a method call is looked up in; empty when
        the call is no method call or that group defines no such function."""
        group = self.receiver_group(module, call)
        return [(m, d) for m, d in self.defs.get(name, []) if group is not None and self._def_group(m, d) == group]

    def _relay_site(self, relays: dict, relay: tuple, module: Module, call: ast.Call) -> None:
        """A helper's statement at one call of the helper: the call's SQL, and the helper's binding,
        execute method and transaction with the call's arguments in place of its parameters; explicit
        arguments none of those uses join the transaction's arguments."""
        func, sql_param, inner_call, inner, home = relay
        if (module is home and func.lineno <= call.lineno <= func.end_lineno or func.name in EXECUTE_METHODS
                or module.rel in self.callers):
            return
        args = _bind_call(func, call)
        if sql_param not in args:
            return
        shadowed = _shadowed(home, inner_call, func)
        bound = {k: v for k, v in args.items() if k not in shadowed}
        chooser = inner_call.func.test if isinstance(inner_call.func, ast.IfExp) else None
        parts = [p for p in (inner.binding, chooser, *inner.txn_args) if p is not None]
        used = {n.id for p in parts for n in ast.walk(p) if isinstance(n, ast.Name)} - shadowed
        swap = _Substitute(bound)
        binding = swap.visit(copy.deepcopy(inner.binding)) if inner.binding is not None else None
        site = self._site(module, call, args[sql_param], binding, _method(inner_call.func, bound) or inner.method)
        extra = [ast.keyword(arg=k, value=v) for k, v in args.items() if k != sql_param and k not in used
                 and any(v is a for a in [*call.args, *(kw.value for kw in call.keywords)])]
        site.txn_kind = inner.txn_kind
        site.txn_args = [swap.visit(copy.deepcopy(a)) for a in inner.txn_args] + extra
        site.group = f"{module.rel}:{call.lineno}:{call.col_offset}"
        self._note_relay(relays, module, call, args[sql_param], site)

    # -- transactions -------------------------------------------------------------

    def transaction(self, module: Module, node: ast.AST, depth: int = 0) -> tuple[str, list, str | None]:
        """Kind, arguments and group of the innermost transaction wrapper around a node."""
        found = self._lexical(module, node)
        if found is not None:
            return found
        outer = module.outer_of(node)
        if outer is None or depth >= _MAX_DEPTH:
            return "none", [], None
        return self._inherit(module, outer, depth + 1)

    def _lexical(self, module: Module, node: ast.AST, stop: ast.AST | None = None):
        child = node
        while child in module.parents:
            parent = module.parents[child]
            wrapper = self._wrapper_call(module, parent, child)
            if wrapper is not None:
                kind, args = self._wrapper_label(module, wrapper, child, parent)
                return kind, args, f"{module.rel}:{wrapper.lineno}:{wrapper.col_offset}"
            hooked = self._hooked(module, parent, child)
            if hooked is not None:
                return hooked
            if parent in module.outer:
                return None
            if parent is stop:
                return None
            if isinstance(parent, _DEFS):
                child = _use_site(module, parent, parent.name)
                if child is None:
                    return self._factory_label(module, parent)
            elif (isinstance(parent, ast.Assign) and child is parent.value and isinstance(child, (ast.Lambda, ast.IfExp))
                  and _simple_target(parent)):
                child = _use_site(module, parent, parent.targets[0].id)
            else:
                child = parent
                continue
            if child is None:
                return None
        return None

    def _hooked(self, module: Module, parent: ast.AST, child: ast.AST) -> tuple[str, list, str | None] | None:
        """The label of the hook parameter a closure is handed to as a call argument."""
        if isinstance(parent, ast.keyword) and isinstance(module.parents.get(parent), ast.Call):
            call = module.parents[parent]
        elif isinstance(parent, ast.Call) and any(child is a for a in parent.args):
            call = parent
        else:
            return None
        labels = self.hook_labels(module, call, child) if isinstance(child, (ast.Lambda, ast.Name, ast.IfExp)) else []
        return _one_label(labels)

    def _factory_label(self, module: Module, inner: ast.AST) -> tuple[str, list, str | None] | None:
        """The label of a nested function its enclosing function returns: where the calls of the
        enclosing function hand their results on."""
        func = module.innermost_def(inner)
        if not _returned(module, func, inner):
            return None
        return _one_label([label for caller, call in self.calls_by_name.get(func.name, [])
                           if self.may_call(caller, call, module, func) for label in self._result_labels(caller, call)])

    def hook_labels(self, module: Module, call: ast.Call, arg: ast.AST) -> list[tuple[str, list, str | None]]:
        """The transactions a callable passed to a call as ``arg`` runs in: for each function the call can
        run, the wrappers around that function's own calls of the parameter ``arg`` binds, the labels of
        the calls it hands that parameter on to, and, when it returns the parameter or a guard calling it,
        the labels of the call it hands its result to."""
        labels = []
        if terminal(call.func) == "partial" and call.args and arg is not call.args[0]:
            call = self._partial_call(module, call)
        for home, func in self.defs.get(terminal(call.func), []):
            if self.may_call(module, call, home, func):
                param = next((p for p, v in _bind_call(func, call).items() if v is arg), None)
                for label in self._hook_params(home, func).get(param, []):
                    labels.extend(self._result_labels(module, call) if label is _RETURNED else [label])
        return labels

    def _partial_call(self, module: Module, call: ast.Call) -> ast.Call:
        """``partial(f, *args, **kwargs)`` as the call ``f(*args, **kwargs)``, placed where the partial is."""
        if call not in self._partials:
            bound = ast.Call(func=call.args[0], args=call.args[1:], keywords=call.keywords)
            module.parents[bound] = module.parents.get(call)
            self._partials[call] = bound
        return self._partials[call]

    def _result_labels(self, module: Module, call: ast.Call) -> list[tuple[str, list, str | None]]:
        """The transactions a call's result runs in when the call, or a conditional or ``and``/``or``
        expression it is a branch of, is another call's argument."""
        value, parent = call, module.parents.get(call)
        while isinstance(parent, ast.BoolOp) or isinstance(parent, ast.IfExp) and value is not parent.test:
            value, parent = parent, module.parents.get(parent)
        outer = module.parents.get(parent) if isinstance(parent, ast.keyword) else parent
        if not isinstance(outer, ast.Call) or value is outer.func:
            return []
        return self.hook_labels(module, outer, value)

    def _hook_params(self, home: Module, func: ast.AST) -> dict[str, list]:
        """What a function does with each parameter it may run: the labels of the wrappers it calls the
        parameter inside, the labels of the calls it passes the parameter to, and ``_RETURNED`` when it
        returns the parameter or a nested function that calls it outside any wrapper."""
        if func not in self._hooks:
            hooks = self._hooks[func] = {}
            params = set(_param_names(func)) - set(self._assigned(func))
            for node in ast.walk(func):
                if (not isinstance(node, ast.Name) or node.id not in params or not isinstance(node.ctx, ast.Load)
                        or node.id in _shadowed(home, node, func)):
                    continue
                parent = home.parents[node]
                if isinstance(parent, ast.Call) and node is parent.func:
                    inner = home.innermost_def(parent)
                    factory = _returned(home, func, inner)
                    label = self._lexical(home, parent, inner if factory else None)
                    labels = [label] if label is not None else [_RETURNED] if factory else []
                elif isinstance(parent, ast.Return) and home.innermost_def(parent) is func:
                    labels = [_RETURNED]
                else:
                    call = home.parents.get(parent) if isinstance(parent, ast.keyword) else parent
                    labels = self.hook_labels(home, call, node) if isinstance(call, ast.Call) and node is not call.func else []
                hooks.setdefault(node.id, []).extend(labels)
        return self._hooks[func]

    def _assigned(self, func: ast.AST | None) -> dict[str, list]:
        if func not in self._locals:
            self._locals[func] = _locals_of(func)[0]
        return self._locals[func]

    def handed(self) -> dict[str, list]:
        """Names handed to a call as an argument, bare or as ``partial(f, ...)``, directly or through a local
        the calling function assigns: name -> (module, call, argument, the name's node)."""
        if self._handed is None:
            self._handed = {}
            for module in self.modules.values():
                for call in (n for n in ast.walk(module.tree) if isinstance(n, ast.Call)):
                    assigned = self._assigned(module.innermost_def(call))
                    for arg in [*call.args, *(k.value for k in call.keywords)]:
                        local = isinstance(arg, ast.Name) and arg.id in assigned
                        values = [v for v in assigned[arg.id] if v is not None] if local else [arg]
                        for ref in (ref for value in values for ref in _callable_names(value)):
                            self._handed.setdefault(ref.id, []).append((module, call, arg, ref))
        return self._handed

    def wrapper(self, module: Module, call: ast.Call) -> str | dict | None:
        """The wrapper entry a call names, or the wrapper a ``partial`` or ``to_thread`` call hands its first
        argument; the entry scoped to the calling file first, then the bare name."""
        name = terminal(call.args[0] if self._forwards(module, call) else call.func)
        return self.wrappers.get(f"{module.rel}::{name}", self.wrappers.get(name))

    def _forwards(self, module: Module, call: ast.Call) -> bool:
        """Whether a call is ``partial(W, ...)`` or ``to_thread(W, ...)`` of a wrapper ``W``."""
        if terminal(call.func) not in FORWARDING_CALLS or not call.args:
            return False
        name = terminal(call.args[0])
        return f"{module.rel}::{name}" in self.wrappers or name in self.wrappers

    def _wrapper_label(self, module: Module, wrapper: ast.Call, child: ast.AST, parent: ast.AST) -> tuple[str, list]:
        """Kind and label arguments of a wrapper call; a begin-mode entry keeps only the begin keyword."""
        entry = self.wrapper(module, wrapper)
        handed = wrapper.args[1:] if self._forwards(module, wrapper) else wrapper.args
        if isinstance(entry, str):
            return entry, [a for a in [*handed, *wrapper.keywords] if not _holds(a, child)]
        begin = entry.get("begin")
        if begin is None:
            return entry["kind"], []
        given = next((k.value for k in wrapper.keywords if k.arg == begin), None)
        if isinstance(parent, (ast.With, ast.AsyncWith)) and _begins_immediate(parent.body[0]):
            given = ast.Constant(True)
        return entry["kind"], [ast.keyword(arg=begin, value=given or ast.Constant(entry.get("default", False)))]

    def _wrapper_call(self, module: Module, parent: ast.AST, child: ast.AST) -> ast.Call | None:
        """The known wrapper a closure or ``with`` block hands its body to, seen from ``child``."""
        call = None
        if isinstance(parent, ast.Call) and isinstance(child, (ast.Lambda, ast.Name)) and child is not parent.func:
            call = parent
        elif isinstance(parent, (ast.With, ast.AsyncWith)) and child in parent.body:
            calls = [i.context_expr for i in parent.items if isinstance(i.context_expr, ast.Call)]
            call = next((c for c in calls if self.wrapper(module, c) is not None), None)
        while (call is not None and not self._forwards(module, call) and terminal(call.func) in TRANSPARENT_CALLS
               and isinstance(module.parents.get(call), ast.Call)):
            call = module.parents[call]
        return call if call is not None and self.wrapper(module, call) is not None else None

    def _inherit(self, home: Module, func: ast.AST, depth: int) -> tuple[str, list, str | None]:
        """The kinds a function is called under, over the calls that can run it within the depth bound,
        their arguments as written (``Site.view`` renders them). Kept per depth: a caller chain counts the
        same whichever statement asks first."""
        if (func, depth) in self._inherited:
            return self._inherited[func, depth]
        kinds = set()
        for module, call in self.calls_by_name.get(func.name, []):
            if not self.may_call(module, call, home, func):
                continue
            kind, args, _ = self.transaction(module, call, depth)
            kinds.update(_kinds(kind) if kind.startswith("mixed(") else {_kind_text(kind, args)})
        target = f"{home.rel}::{home.def_qualname(func)}"
        for module, call, arg, ref in self.handed().get(func.name, []):
            if target in (self._bare(module, ref, ref.id, 0) or ()):
                kinds.update(_kind_text(kind, args) for kind, args, _ in self.hook_labels(module, call, arg))
        declared = self.inherits.get((home.rel, home.outer.get(func, ("",))[0]))
        kinds.update(_kinds(declared) if declared else ())
        self._inherited[func, depth] = (_label(kinds), [], None)
        return self._inherited[func, depth]

    def without(self, key: tuple[str, str]) -> Tree:
        """The same tree with one declared inherited transaction dropped."""
        other = copy.copy(self)
        other.inherits = {k: v for k, v in self.inherits.items() if k != key}
        other._inherited = {}
        return other

    def inherited(self, ref: str) -> str:
        """The transaction label a referenced function inherits from its callers."""
        module, func = self.find_def(ref)
        return receiverless_label(self._inherit(module, func, 1)[0])

    # -- builders -----------------------------------------------------------------

    def builders(self) -> dict[str, tuple[Module, ast.AST]]:
        """Functions in the tree whose return value feeds SQL text, transitively, by reference."""
        if self._builders is not None:
            return self._builders
        found: dict[str, tuple[Module, ast.AST]] = {}
        by_ref = {f"{m.rel}::{m.def_qualname(n)}": (m, n) for defs in self.defs.values() for m, n in defs}
        work = sorted({name for sites in self.sites.values() for site in sites for name in site.calls})
        while work:
            name = work.pop()
            for module, node in [by_ref[name]] if name in by_ref else self.defs.get(name, []):
                ref = f"{module.rel}::{module.def_qualname(node)}"
                if (ref in found or module.rel in self.callers
                        or not any(isinstance(r, ast.Return) and r.value is not None for r in ast.walk(node))):
                    continue
                found[ref] = (module, node)
                work.extend(sorted(self.fingerprint(module, node)[1]))
        self._builders = found
        return found

    def fingerprint(self, module: Module, node: ast.AST) -> tuple[list, set[str]]:
        """The returned templates, their slots and the builder slice of a function, and its calls."""
        resolver = Resolver(self, module, node)
        returns = [r for r in ast.walk(node) if isinstance(r, ast.Return) and r.value is not None
                   and module.innermost_def(r) is node]
        items: list = [("return", resolver.template(r.value)) for r in returns]
        items.extend(("slot", s) for s in resolver.slots)
        sliced = builder_slice(module, node, set(resolver.dynamic), resolver.assigned)
        items.extend(("slice", c, s) for c, s in sliced)
        return items, (resolver.calls | self.calls_in(module, (s for _, s in sliced))) - set(NOT_BUILDERS)

    def calls_in(self, module: Module, nodes) -> set[str]:
        return {name for n in nodes for c in ast.walk(n) if isinstance(c, ast.Call) for name in self.call_names(module, c)}

    def call_names(self, module: Module, call: ast.Call) -> set[str]:
        """What a call in SQL-building code may run: the ``path::qualname`` of each tree function a bare
        name is bound to (none for a builtin or a function outside the tree) or of each function of its
        name in the inheritance group a method call is looked up in, else the terminal name."""
        if isinstance(call.func, ast.Name):
            refs = self._bare(module, call, call.func.id, 0)
            if refs is not None:
                return refs
        name = terminal(call.func)
        grouped = self._group_defs(module, call, name) if name not in NOT_BUILDERS else []
        return {f"{m.rel}::{m.def_qualname(d)}" for m, d in grouped} or {name}

    def _bare(self, module: Module, node: ast.AST | None, name: str, depth: int) -> set[str] | None:
        """The tree functions a bare name at ``node`` (None: the module scope, chasing an import) is bound
        to, through the enclosing function scopes and imports; None when a binding is no def or import."""
        if depth > _MAX_DEPTH:
            return None
        child = node
        while child is not None and child in module.parents and not isinstance(child, _DEFS):
            child = module.parents[child]
            if isinstance(child, ast.Lambda) and name in {a.arg for a in ast.walk(child.args) if isinstance(a, ast.arg)}:
                return None
        scope = module.innermost_def(node) if node is not None else None
        while True:
            bindings = self._scope_names(module, scope).get(name)
            if bindings is not None:
                break
            if scope is None:
                return set() if node is not None else None
            scope = module.innermost_def(scope)
        refs: set[str] = set()
        for binding in bindings:
            if isinstance(binding, _DEFS):
                refs.add(f"{module.rel}::{module.def_qualname(binding)}")
            elif isinstance(binding, ast.ImportFrom):
                original = next(a.name for a in binding.names if (a.asname or a.name) == name)
                target = self._module(_module_candidates(module.rel, binding.module, binding.level))
                found = self._bare(target, None, original, depth + 1) if target and target.rel in self.modules else set()
                if found is None:
                    return None
                refs |= found
            elif not isinstance(binding, ast.ClassDef):
                return None
        return refs

    def _scope_names(self, module: Module, scope: ast.AST | None) -> dict[str, list]:
        """The nodes binding each name directly in a function scope (None: the module scope): defs, classes,
        imports, parameters and assigned names."""
        key = (module.rel, id(scope))
        if key not in self._scopes:
            names: dict[str, list] = {}
            body = module.tree if scope is None else scope
            if scope is not None:
                for arg in ast.walk(scope.args):
                    if isinstance(arg, ast.arg):
                        names.setdefault(arg.arg, []).append(arg)
            for node in ast.walk(body):
                if node is body or _scope_of(module, node) is not scope:
                    continue
                if isinstance(node, (*_DEFS, ast.ClassDef)):
                    names.setdefault(node.name, []).append(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        names.setdefault(alias.asname or alias.name.split(".")[0], []).append(node)
                elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    names.setdefault(node.id, []).append(node)
                elif isinstance(node, (ast.Global, ast.Nonlocal)):
                    for alias in node.names:
                        names.setdefault(alias, []).append(node)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    names.setdefault(node.name, []).append(node)
            self._scopes[key] = names
        return self._scopes[key]

    def find_def(self, ref: str) -> tuple[Module, ast.AST]:
        path, qual, _ = parse_ref(ref)
        module = self.modules.get(path)
        for node in ast.walk(module.tree) if module else []:
            if isinstance(node, _DEFS) and module.def_qualname(node) == qual:
                return module, node
        raise ManifestError(f"{ref}: no such function")

    def select(self, ref: str) -> list[Site]:
        path, qual, selection = parse_ref(ref)
        sites = self.sites.get((path, qual))
        if sites is None:
            if path not in self.modules:
                raise ManifestError(f"{ref}: no such file")
            raise ManifestError(f"{ref}: no SQL statements there")
        if selection is None:
            return list(sites)
        picked = []
        for i in selection:
            if i >= len(sites):
                raise ManifestError(f"{ref}: only {len(sites)} statements")
            picked.append(sites[i])
        return picked

    def sites_in(self, ref: str) -> list[Site]:
        """Statements of a ``legacy_modules`` entry: a whole file or one function."""
        if "::" in ref:
            return self.select(ref)
        if ref not in self.modules:
            raise ManifestError(f"{ref}: no such file")
        return [s for (path, _), sites in sorted(self.sites.items()) if path == ref for s in sites]


_RETURNED = ("returned", [], None)


def _returned(module: Module, func: ast.AST, inner: ast.AST | None) -> bool:
    """Whether ``inner`` is a function nested directly in ``func`` that ``func`` returns by name."""
    return (func is not None and inner is not None and inner is not func and module.innermost_def(inner) is func
            and any(isinstance(r, ast.Return) and isinstance(r.value, ast.Name) and r.value.id == inner.name
                    and module.innermost_def(r) is func for r in ast.walk(func)))


def _callable_names(value: ast.AST) -> list[ast.Name]:
    """The names a value hands on as a callable: a bare name, ``partial(f, ...)``'s ``f``, in either branch of
    a conditional."""
    names = []
    for ref in (value.body, value.orelse) if isinstance(value, ast.IfExp) else (value,):
        if isinstance(ref, ast.Call) and terminal(ref.func) == "partial" and ref.args:
            ref = ref.args[0]
        if isinstance(ref, ast.Name):
            names.append(ref)
    return names


def _one_label(labels: list) -> tuple[str, list, str | None] | None:
    """One transaction label for several: the label itself when they agree, else their mixed kinds."""
    kinds = {_kind_text(kind, args) for kind, args, _ in labels}
    if len(kinds) > 1:
        return _label(kinds), [], None
    return labels[0] if labels else None


def _kind_text(kind: str, args: list) -> str:
    """One kind of a transaction label, its arguments as written."""
    args = _txn_args(args)
    return f"{kind}({', '.join(ast.unparse(a) for a in args)})" if args else kind


def _kinds(label: str) -> set[str]:
    """The kinds of a transaction label (``mixed(a | b)`` holds several)."""
    return set(label[len("mixed("):-1].split(" | ")) if label.startswith("mixed(") else {label}


def _label(kinds: set[str]) -> str:
    """The label of a set of kinds: ``none`` only when no other kind is there."""
    kinds = kinds - {"none"} if len(kinds) > 1 else kinds
    return "none" if not kinds else next(iter(kinds)) if len(kinds) == 1 else "mixed(" + " | ".join(sorted(kinds)) + ")"


def receiverless_label(label: str, names: dict[str, str] | None = None) -> str:
    """A transaction label with each kind's arguments rendered as the checker renders them."""
    return _label({render(ast.parse(kind, mode="eval").body, names) for kind in _kinds(label)})


def _accepts(func: ast.AST, call: ast.Call) -> bool:
    """Whether a call's arguments fit a function's signature, with or without a bound ``self``/``cls``
    (a starred argument fits anything)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return True
    args = func.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    required = positional[:len(positional) - len(args.defaults)]
    named = {k.arg for k in call.keywords}
    kwonly = {a.arg for a in args.kwonlyargs}
    missing_kwonly = {a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None} - named
    bound = (1, 0) if positional and positional[0] in ("self", "cls") else (0,)
    for skip in bound:
        free = positional[skip:]
        given = len(call.args)
        if given > len(free) and args.vararg is None or missing_kwonly:
            continue
        if named - set(free[given:]) - kwonly and args.kwarg is None:
            continue
        if set(required[skip:][given:]) - named:
            continue
        return True
    return False


_CONTAINERS = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)


def _builtin_receiver(module: Module, call: ast.Call) -> bool:
    """Whether a method call's receiver is a builtin value, whose methods are never the tree's functions: a
    local bound only to what an execute call returns (a cursor), or a module-level name bound only to dict,
    list or set displays and not rebound in the calling function."""
    func = call.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    name = func.value.id
    assigned, params = _locals_of(module.innermost_def(call))
    if name in params:
        return False
    if name in assigned:
        return all(isinstance(v, ast.Call) and _method(v.func) in EXECUTE_METHODS for v in assigned[name])
    values = [node.value for node in module.tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
              and any(isinstance(t, ast.Name) and t.id == name for t in _targets(node))]
    return bool(values) and all(isinstance(v, _CONTAINERS) for v in values)


def _sql_owner(module: Module, call: ast.AST, sql) -> ast.AST | None:
    """The function whose parameter a call's SQL is: the innermost function, or an enclosing one when
    the nested functions and lambdas in between do not bind the name."""
    if not isinstance(sql, ast.Name):
        return None
    node = call
    while node in module.parents:
        node = module.parents[node]
        if not isinstance(node, (*_DEFS, ast.Lambda)):
            continue
        if isinstance(node, _DEFS) and sql.id in _param_names(node):
            return node
        assigned, params = _locals_of(node)
        if sql.id in assigned or sql.id in params or node in module.outer:
            return None
    return None


def _shadowed(module: Module, node: ast.AST, func: ast.AST) -> set[str]:
    """Names bound by the nested functions and lambdas between a node and its enclosing ``func``."""
    names: set[str] = set()
    while node in module.parents and node is not func:
        node = module.parents[node]
        if isinstance(node, (*_DEFS, ast.Lambda)) and node is not func:
            assigned, params = _locals_of(node)
            names.update(assigned, params)
    return names


def _use_site(module: Module, func: ast.AST, name: str) -> ast.AST | None:
    """The first place the nested function or local closure ``name`` defined at ``func`` is called or passed
    on, inside its enclosing scope."""
    scope = module.parents.get(func)
    while scope is not None and not isinstance(scope, (*_DEFS, ast.Module)):
        scope = module.parents.get(scope)
    for node in ast.walk(scope) if scope is not None else []:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            return node
        for arg in [*node.args, *(k.value for k in node.keywords)]:
            for ref in _callable_names(arg):
                if ref.id == name:
                    return ref
    return None


def builder_view(items: list, names: dict[str, str] | None) -> list[str]:
    out = []
    for item in items:
        if item[0] == "return":
            out.append("return " + item[1])
        elif item[0] == "slot":
            out.append("slot " + render(item[1], names))
        else:
            out.append(_slice_line(item[1], item[2], names))
    return out


def parse_ref(ref: str) -> tuple[str, str, list[int] | None]:
    match = _REF.fullmatch(ref.strip()) if isinstance(ref, str) else None
    if match is None:
        raise ManifestError(f"{ref!r}: not a reference (path::qualname[#selection])")
    selection = None
    if match["sel"]:
        selection = []
        for part in match["sel"].replace(" ", "").split(","):
            low, _, high = part.partition("-")
            selection.extend(range(int(low), int(high or low) + 1))
    return match["path"], match["qual"], selection


def _refs(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


# -- manifests -------------------------------------------------------------------

_TOP_KEYS = {"family", "legacy_modules", "callers", "wrappers", "units", "builders", "not_moved"}
_ENTRY_KEYS = {"backend", "legacy", "names", "exception", "note", "inherits"}
_NOT_MOVED_KEYS = {"legacy", "reason", "ruling"}


def _is_refs(value) -> bool:
    return isinstance(value, str) or isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


def _check_entry(where: str, entry) -> None:
    if not isinstance(entry, dict) or set(entry) - _ENTRY_KEYS or "backend" not in entry or "legacy" not in entry:
        raise ManifestError(f"{where}: needs backend and legacy, allows {sorted(_ENTRY_KEYS)}")
    if not _is_refs(entry["backend"]) or not (entry["legacy"] is None or _is_refs(entry["legacy"])):
        raise ManifestError(f"{where}: backend and legacy are references (legacy may be null)")
    names = entry.get("names", {})
    if not isinstance(names, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()):
        raise ManifestError(f"{where}: names maps legacy names to backend names")
    exception = entry.get("exception")
    if exception is not None:
        aspects = exception.get("aspects") if isinstance(exception, dict) else None
        if set(exception) != {"ruling", "aspects", "reason"} or not aspects or set(aspects) - set(ASPECTS):
            raise ManifestError(f"{where}: exception needs ruling, reason and aspects from {list(ASPECTS)}")
        if exception["ruling"] not in RULINGS:
            raise ManifestError(f"{where}: ruling {exception['ruling']!r} is not one of {list(RULINGS)}")
    if "inherits" in entry and not all(isinstance(entry[k], str) for k in ("inherits", "backend", "legacy")):
        raise ManifestError(f"{where}: inherits is a transaction label on a unit of one backend and one legacy function")
    if entry["legacy"] is None and "source" not in (exception or {}).get("aspects", []):
        raise ManifestError(f"{where}: a null legacy source needs a source exception")


def validate(name: str, data) -> None:
    if not isinstance(data, dict) or set(data) - _TOP_KEYS or not isinstance(data.get("family"), str):
        raise ManifestError(f"{name}: needs family, allows {sorted(_TOP_KEYS)}")
    if not all(isinstance(m, str) for m in data.get("legacy_modules", [])):
        raise ManifestError(f"{name}: legacy_modules lists paths or references")
    if not all(isinstance(m, str) and "::" not in m for m in data.get("callers", [])):
        raise ManifestError(f"{name}: callers lists file paths")
    if not all(isinstance(k, str) and _is_wrapper(v) for k, v in data.get("wrappers", {}).items()):
        raise ManifestError(f"{name}: wrappers maps callable names to kinds or to kind, begin and default")
    for key in ("units", "builders"):
        for i, entry in enumerate(data.get(key, [])):
            _check_entry(f"{name} {key}[{i}]", entry)
    for i, entry in enumerate(data.get("not_moved", [])):
        if not isinstance(entry, dict) or set(entry) - _NOT_MOVED_KEYS or not _is_refs(entry.get("legacy")) \
                or not isinstance(entry.get("reason"), str):
            raise ManifestError(f"{name} not_moved[{i}]: needs legacy and reason, allows ruling")
        if "ruling" in entry and entry["ruling"] not in RULINGS:
            raise ManifestError(f"{name} not_moved[{i}]: ruling {entry['ruling']!r} is not one of {list(RULINGS)}")


def load_manifests(directory: Path) -> list[tuple[str, dict]]:
    if not directory.is_dir():
        raise ManifestError(f"{directory}: no manifest directory")
    manifests = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{path.name}: {exc}") from exc
        validate(path.name, data)
        manifests.append((path.name, data))
    return manifests


# -- comparison ------------------------------------------------------------------


def _diff(a, b) -> str:
    a, b = str(a), str(b)
    i = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, i - 30)
    return f"backend ...{a[start:i + 50]!r} legacy ...{b[start:i + 50]!r}"


def compare_sites(back: list[Site], legacy: list[Site], names: dict[str, str]) -> list[tuple[str, str]]:
    """Differences between backend statements and their legacy statements, by aspect."""
    bviews = [s.view() for s in back]
    lviews = [s.view(names) for s in legacy]
    out: list[tuple[str, str]] = []
    if len(back) != len(legacy):
        out.append(("count", f"{len(back)} backend statements, {len(legacy)} legacy"))
    btexts = [(v["method"], v["text"]) for v in bviews]
    ltexts = [(v["method"], v["text"]) for v in lviews]
    if btexts != ltexts and sorted(btexts) == sorted(ltexts):
        order = ", ".join(s.ref for s in back)
        return out + [("order", f"backend {order} runs the legacy statements in another order")]
    for bsite, lsite, bview, lview in zip(back, legacy, bviews, lviews):
        for aspect in COMPARED:
            if bview[aspect] != lview[aspect]:
                out.append((aspect, f"{bsite.ref} vs {lsite.ref}: {_diff(bview[aspect], lview[aspect])}"))
    return out + _grouping(back, legacy)


def _grouping(back: list[Site], legacy: list[Site]) -> list[tuple[str, str]]:
    """Statements of one transaction on one side must be one transaction on the other."""
    forward: dict[str, set] = {}
    backward: dict[str, set] = {}
    for bsite, lsite in zip(back, legacy):
        if bsite.group and lsite.group:
            forward.setdefault(lsite.group, set()).add(bsite.group)
            backward.setdefault(bsite.group, set()).add(lsite.group)
    out = [("transaction", f"legacy transaction at {g} is split over backend transactions {sorted(s)}")
           for g, s in forward.items() if len(s) > 1]
    return out + [("transaction", f"backend transaction at {g} merges legacy transactions {sorted(s)}")
                  for g, s in backward.items() if len(s) > 1]


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    not_moved: list[str] = field(default_factory=list)
    inherited: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def judge(self, where: str, entry: dict, differences: list[tuple[str, str]]) -> None:
        exception = entry.get("exception") or {}
        aspects = set(exception.get("aspects", []))
        for aspect, detail in differences:
            if aspect in aspects:
                self.exceptions.append(f"{where} [{exception['ruling']}] {aspect}: {detail} ({exception['reason']})")
            else:
                self.failures.append(f"{where} {aspect}: {detail}")
        unused = aspects - {aspect for aspect, _ in differences}
        if unused:
            self.failures.append(f"{where}: exception {exception['ruling']} excuses no {', '.join(sorted(unused))} difference")


class Checker:
    """Run every manifest of a branch against the legacy tree."""

    def __init__(self, legacy: Path, branch: Path, manifests: list[tuple[str, dict]], backend_root: str = BACKEND_ROOT):
        self.manifests = manifests
        wrappers = dict(DEFAULT_WRAPPERS)
        for _, data in manifests:
            wrappers.update(data.get("wrappers", {}))
        backend = branch / backend_root
        skip = (branch / MANIFEST_DIR).resolve()
        rels = [p.relative_to(branch).as_posix() for p in sorted(backend.rglob("*.py")) if skip not in p.resolve().parents]
        rels.extend(rel for rel in BACKEND_FILES if (branch / rel).is_file())
        inherits = self._declared_inherits()
        self.backend = Tree(branch, rels, wrappers, inherits)
        callers = {path for _, data in manifests for path in data.get("callers", [])}
        sources = self._legacy_files(legacy)
        named = callers & set(sources)
        if named:
            raise ManifestError(f"caller files named as SQL sources: {sorted(named)}")
        self.legacy = Tree(legacy, sources, wrappers, callers=callers)
        self.moved = {parse_ref(r)[:2] for _, data in manifests for entry in data.get("units", [])
                      for r in _refs(entry["legacy"])}
        self.report = Report()
        self.back_cover: Counter = Counter()
        self.legacy_cover: set[str] = set()
        self.builder_cover: set[str] = set()
        self.legacy_builder_cover: set[str] = set()

    def _declared_inherits(self) -> dict[tuple[str, str], str]:
        """(backend file, qualname) -> the transaction label a unit declares the function inherits."""
        declared: dict[tuple[str, str], str] = {}
        for name, data in self.manifests:
            for entry in data.get("units", []):
                if "inherits" in entry:
                    key = parse_ref(entry["backend"])[:2]
                    if key in declared:
                        raise ManifestError(f"{name}: {entry['backend']} declares its inherited transaction twice")
                    declared[key] = receiverless_label(entry["inherits"])
        return declared

    def _legacy_files(self, legacy: Path) -> list[str]:
        rels = {p.name for p in legacy.glob(LEGACY_GLOB)}
        for _, data in self.manifests:
            refs = list(data.get("legacy_modules", []))
            for key in ("units", "builders", "not_moved"):
                refs.extend(r for entry in data.get(key, []) for r in _refs(entry.get("legacy")))
            rels.update(r.split("::")[0] for r in refs)
        return sorted(rels)

    def run(self) -> Report:
        for name, data in self.manifests:
            for entry in data.get("units", []):
                self._unit(name, entry)
            for entry in data.get("builders", []):
                self._builder(name, entry)
            for entry in data.get("not_moved", []):
                self._not_moved(name, entry)
        self._backend_coverage()
        self._legacy_coverage()
        return self.report

    def _unit(self, name: str, entry: dict) -> None:
        where = f"{name}: {' + '.join(_refs(entry['backend']))}"
        try:
            back = [s for r in _refs(entry["backend"]) for s in self.backend.select(r)]
            legacy = [s for r in _refs(entry["legacy"]) for s in self.legacy.select(r)]
        except ManifestError as exc:
            self.report.failures.append(f"{name}: {exc}")
            return
        self.back_cover.update(s.ref for s in back)
        self.legacy_cover.update(s.ref for s in legacy)
        if entry["legacy"] is None:
            differences = [("source", f"{s.ref} has no legacy source: {s.view()['text']}") for s in back]
        else:
            differences = compare_sites(back, legacy, entry.get("names", {}))
        if "inherits" in entry:
            self._check_inherits(where, entry)
        self.report.judge(where, entry, differences)

    def _check_inherits(self, where: str, entry: dict) -> None:
        """A declared transaction stands in for legacy callers that have not moved yet."""
        declared = receiverless_label(entry["inherits"])
        legacy = self.legacy.inherited(entry["legacy"])
        home, func = self.legacy.find_def(entry["legacy"])
        unmoved = sorted({f"{m.rel}::{m.qualname(c)}" for m, c in self.legacy.calls_by_name.get(func.name, [])
                          if self.legacy.may_call(m, c, home, func) and (m.rel, m.qualname(c)) not in self.moved})
        if declared != legacy:
            problem = f"the legacy function inherits {legacy!r}"
        elif not unmoved:
            problem = "every legacy caller has moved"
        elif self.backend.without(parse_ref(entry["backend"])[:2]).inherited(entry["backend"]) == declared:
            problem = "its backend callers and the other declarations give it already"
        else:
            self.report.inherited.append(f"{where}: {declared} from {len(unmoved)} legacy callers, first {unmoved[0]}")
            return
        self.report.failures.append(f"{where} transaction: declares inherited {declared!r}, but {problem}")

    def _builder(self, name: str, entry: dict) -> None:
        where = f"{name}: builder {' + '.join(_refs(entry['backend']))}"
        try:
            back = [self.backend.find_def(r) for r in _refs(entry["backend"])]
            legacy = [self.legacy.find_def(r) for r in _refs(entry["legacy"])]
        except ManifestError as exc:
            self.report.failures.append(f"{name}: {exc}")
            return
        self.builder_cover.update(_refs(entry["backend"]))
        self.legacy_builder_cover.update(_refs(entry["legacy"]))
        for ref in _refs(entry["backend"]):
            if ref not in self.backend.builders():
                self.report.failures.append(f"{where}: stale entry, {ref} feeds no SQL text")
        bview = [line for m, n in back for line in builder_view(self.backend.fingerprint(m, n)[0], None)]
        lview = [line for m, n in legacy for line in builder_view(self.legacy.fingerprint(m, n)[0], entry.get("names", {}))]
        if entry["legacy"] is None:
            differences = [("source", f"no legacy source: {bview}")]
        else:
            differences = [] if bview == lview else [("builder", _diff(bview, lview))]
        self.report.judge(where, entry, differences)

    def _not_moved(self, name: str, entry: dict) -> None:
        for ref in _refs(entry["legacy"]):
            try:
                if ref in self.legacy.builders():
                    self.legacy_builder_cover.add(ref)
                else:
                    self.legacy_cover.update(s.ref for s in self.legacy.select(ref))
            except ManifestError as exc:
                self.report.failures.append(f"{name}: not_moved {exc}")
                continue
            ruling = f" [{entry['ruling']}]" if entry.get("ruling") else ""
            self.report.not_moved.append(f"{name}: {ref}{ruling} ({entry['reason']})")

    def _backend_coverage(self) -> None:
        sites = [s for group in self.backend.sites.values() for s in group]
        for site in sites:
            count = self.back_cover[site.ref]
            if count != 1:
                state = "has no manifest entry" if not count else f"is mapped {count} times"
                self.report.failures.append(f"backend statement {site.ref} (line {site.line}) {state}: {site.view()['text']}")
        builders = self.backend.builders()
        for ref in sorted(set(builders) - self.builder_cover):
            self.report.failures.append(f"backend builder {ref} has no manifest entry")
        self.report.counts.update(backend_statements=len(sites), backend_builders=len(builders))

    def _legacy_coverage(self) -> None:
        refs = [r for _, data in self.manifests for r in data.get("legacy_modules", [])]
        checked = 0
        builders = self.legacy.builders()
        for ref in refs:
            try:
                sites = self.legacy.sites_in(ref)
            except ManifestError as exc:
                self.report.failures.append(f"legacy_modules {exc}")
                continue
            checked += len(sites)
            for site in sites:
                if site.ref not in self.legacy_cover:
                    self.report.failures.append(f"legacy statement {site.ref} (line {site.line}) is unmapped: {site.view()['text']}")
            for bref in sorted(set(builders) - self.legacy_builder_cover):
                if bref.split("::")[0] == ref or bref == ref or bref.startswith(ref + "."):
                    self.report.failures.append(f"legacy builder {bref} is unmapped")
        self.report.counts.update(legacy_statements=checked, manifests=len(self.manifests))


# -- commands --------------------------------------------------------------------


def _print_report(report: Report, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": not report.failures, **report.__dict__}, indent=2))
        return
    for title, lines in (("FAIL", report.failures), ("RULED EXCEPTIONS", report.exceptions), ("NOT MOVED", report.not_moved),
                         ("CALLERS NOT MOVED", report.inherited)):
        print(f"{title} ({len(lines)})")
        for line in lines:
            print(f"  {line}")
    counts = ", ".join(f"{k} {v}" for k, v in report.counts.items())
    print(f"{'OK' if not report.failures else 'FAILED'}: {counts}")


def cmd_check(args) -> int:
    branch = Path(args.branch)
    try:
        manifests = load_manifests(Path(args.manifests) if args.manifests else branch / MANIFEST_DIR)
        report = Checker(Path(args.legacy), branch, manifests, args.backend_root).run()
    except ManifestError as exc:
        print(f"unusable input: {exc}", file=sys.stderr)
        return 2
    _print_report(report, args.json)
    return 1 if report.failures else 0


def _wrapper_args(pairs: list[str]) -> dict[str, str]:
    wrappers = dict(DEFAULT_WRAPPERS)
    wrappers.update(dict(pair.split("=", 1) for pair in pairs))
    return wrappers


def _expand(root: Path, paths: list[str]) -> list[str]:
    rels = []
    for path in paths:
        target = root / path
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        rels.extend(p.relative_to(root).as_posix() for p in found)
    return rels


def _context_tree(root: Path, paths: list[str], wrappers: dict[str, str]) -> tuple[Tree, set[str]]:
    """A tree over the named files plus the backend package and legacy modules beside them, so
    imported SQL text resolves; returns it with the named files."""
    wanted = set(_expand(root, paths))
    context = {p.name for p in root.glob(LEGACY_GLOB)}
    if (root / BACKEND_ROOT).is_dir():
        context.update(_expand(root, [BACKEND_ROOT]))
    return Tree(root, sorted(wanted | context), wrappers), wanted


def cmd_scan(args) -> int:
    tree, wanted = _context_tree(Path(args.tree), args.paths, _wrapper_args(args.wrapper))
    sites = [s for key in sorted(tree.sites) if key[0] in wanted for s in tree.sites[key]]
    builders = {ref: builder_view(tree.fingerprint(m, n)[0], None) for ref, (m, n) in sorted(tree.builders().items())
                if ref.split("::")[0] in wanted}
    if args.json:
        print(json.dumps({"statements": [{"ref": s.ref, "line": s.line, **s.view()} for s in sites],
                          "builders": builders}, indent=2))
        return 0
    for site in sites:
        view = site.view()
        print(f"{site.ref}  line {site.line}  {view['method']}  {view['transaction']}")
        print(f"    {view['text']}")
        if view["binding"]:
            print(f"    binding: {view['binding']}")
        for line in view["builder"]:
            print(f"    builds: {line}")
    for ref, lines in builders.items():
        print(f"builder {ref}")
        for line in lines:
            print(f"    {line}")
    return 0


def _draft_unit(func_sites: list[Site], candidates: dict[tuple, list[Site]]) -> dict:
    """Pair a backend function with the legacy function holding most of its statements."""
    votes: Counter = Counter()
    for site in func_sites:
        view = site.view()
        votes.update({(s.file, s.func) for s in candidates.get((view["method"], view["text"]), [])})
    first = func_sites[0]
    backend = f"{first.file}::{first.func}"
    if not votes:
        return {"backend": backend, "legacy": None, "note": "no legacy statement with the same text"}
    short = first.func.rsplit(".", 1)[-1]
    best = max(votes, key=lambda k: (votes[k], k[1].rsplit(".", 1)[-1] == short, Path(k[0]).stem == Path(first.file).stem, k))
    used, picked = set(), []
    for site in func_sites:
        view = site.view()
        match = next((s for s in candidates.get((view["method"], view["text"]), [])
                      if (s.file, s.func) == best and s.ordinal not in used), None)
        if match is not None:
            used.add(match.ordinal)
            picked.append(match.ordinal)
    unit = {"backend": backend, "legacy": f"{best[0]}::{best[1]}#{','.join(map(str, picked))}"}
    if len(picked) != len(func_sites):
        unit["note"] = f"{len(func_sites) - len(picked)} statements unmatched"
    return unit


def _draft_builder(ref: str, legacy: dict) -> str | None:
    """The legacy builder of the same name, preferring the same qualname, then the same file."""
    path, qual = ref.split("::", 1)
    same = [r for r in sorted(legacy) if r.rsplit(".", 1)[-1].split("::")[-1] == qual.rsplit(".", 1)[-1]]
    return max(same, key=lambda r: (r.split("::", 1)[1] == qual, Path(r.split("::")[0]).stem == Path(path).stem),
               default=None)


def cmd_draft(args) -> int:
    legacy_root, branch = Path(args.legacy), Path(args.branch)
    wrappers = _wrapper_args(args.wrapper)
    legacy_files = sorted({p.name for p in legacy_root.glob(LEGACY_GLOB)} | {m.split("::")[0] for m in args.legacy_module})
    ltree = Tree(legacy_root, legacy_files, wrappers)
    btree, wanted = _context_tree(branch, args.paths, wrappers)
    candidates: dict[tuple, list[Site]] = {}
    for ref in args.legacy_module:
        for site in ltree.sites_in(ref):
            view = site.view()
            candidates.setdefault((view["method"], view["text"]), []).append(site)
    units = [_draft_unit(btree.sites[key], candidates) for key in sorted(btree.sites) if key[0] in wanted]
    builders = [{"backend": ref, "legacy": _draft_builder(ref, ltree.builders())} for ref in sorted(btree.builders())
                if ref.split("::")[0] in wanted]
    print(json.dumps({"family": args.family, "legacy_modules": args.legacy_module, "units": units,
                      "builders": builders, "not_moved": []}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="prove every backend statement against the legacy tree")
    check.add_argument("--legacy", required=True, help="root of the legacy tree")
    check.add_argument("--branch", required=True, help="root of the branch tree")
    check.add_argument("--manifests", help=f"manifest directory (default: BRANCH/{MANIFEST_DIR})")
    check.add_argument("--backend-root", default=BACKEND_ROOT, help="backend package, relative to the branch")
    check.add_argument("--json", action="store_true")
    check.set_defaults(run=cmd_check)
    scan = sub.add_parser("scan", help="print the statements and builders found in files of a tree")
    scan.add_argument("--tree", required=True)
    scan.add_argument("--wrapper", action="append", default=[], metavar="NAME=KIND")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("paths", nargs="+")
    scan.set_defaults(run=cmd_scan)
    draft = sub.add_parser("draft", help="propose a family manifest by matching statement text")
    draft.add_argument("--legacy", required=True)
    draft.add_argument("--branch", required=True)
    draft.add_argument("--family", required=True)
    draft.add_argument("--legacy-module", action="append", required=True, metavar="REF")
    draft.add_argument("--wrapper", action="append", default=[], metavar="NAME=KIND")
    draft.add_argument("paths", nargs="+")
    draft.set_defaults(run=cmd_draft)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
