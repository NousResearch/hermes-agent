"""Where a move may rewrite a string: the places that take a module name (decision D47), found before any rewrite.

A sink is a place the code hands a value over as a module name or a module path: ``import_module``,
``__import__``, ``find_spec``, ``importorskip``, ``resolve_name`` and functions that pass a parameter
straight to one of them; ``mock.patch``/``monkeypatch`` string targets; ``sys.modules`` keys; logger
names (``getLogger``, caplog filters); ``-m <module>`` argv pairs and commands given to a process;
Python code given to an interpreter. A string reaches a sink directly, through a name bound to it in
the same function, or as an element of a literal table that only feeds sinks. Every other string is
context the rewrite uses to decide what to flag: tables it cannot prove, path pieces, i18n keys,
docstrings.
"""
from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, field

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import GlobalScope, ParentNodeProvider, ScopeProvider

MODULE, TARGET, SYS_MODULES, MODULE_OBJECT, LOGGER, ARGV, CODE, COMMAND = (
    "module-name", "patch-target", "sys-modules", "module-object", "logger", "argv", "code", "command")
STRINGS = (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString)
_SYS_MODULES = re.compile(r"(?:(?:^|\.)_*sys_*\.)?modules$")
_MODULE_CALLS = {"import_module": MODULE, "__import__": MODULE, "find_spec": MODULE, "importorskip": MODULE,
                 "run_module": MODULE,
                 "patch": TARGET, "setattr": TARGET, "delattr": TARGET, "resolve_name": TARGET}
_ALIASABLE = {"import_module", "find_spec", "importorskip", "resolve_name", "patch"}
_LOG_LEVEL_CALLS = {"set_level", "at_level"}
_SYS_MODULES_METHODS = {"pop", "get", "setdefault"}
_SYS_MODULES_REMOVERS = {"pop", "clear", "popitem"}
_CODE_CALLS = {"exec", "eval", "compile"}
_PROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output", "system", "popen", "getoutput",
                  "getstatusoutput", "create_subprocess_shell"}
_I18N_CALLS = {"t", "_t", "_gt", "tn", "gettext", "ngettext"}
_PATH_CALLS = {"join", "joinpath", "Path", "PurePath", "PurePosixPath", "PureWindowsPath"}
_TEXT_WRAPPERS = {"dedent", "cleandoc"}
_INTERPRETER_KEYWORDS = {"argv0", "python", "interpreter", "python_executable", "executable"}
_ROOT_EXPRESSION = re.compile(r"__file__|\.parents?\b|project_root|repo_root|checkout|install_root|source_root|"
                              r"\b_?(?:REPO|PROJECT|CHECKOUT|SOURCE)(?:_ROOT|_DIR)?\b|\bROOT\b|\bREPO_ROOT\b")
_HELPER_SINKS = {"import_module", "__import__", "find_spec", "importorskip"}
FREE_HELPERS: dict[str, int] = {"_tools_mod": 0, "_lazy": 0}
"""Helpers that ``tui_gateway.method_ctx.bind_module`` puts on a split module's globals at run time, so the file
that calls them shows no binding: name -> index of the parameter that takes a module name."""
NAMED_HELPERS: dict[str, dict[int, str]] = {"hermes_cli.web_deps.LateState": {1: "module"}}
"""Callables (old dotted name) whose argument reaches an import through state the helper scan does not follow: the
class keeps ``module`` on the instance and resolves it through ``_server`` at each use."""
_MODULE_OBJECT_CALLS = {"ModuleType", "spec_from_file_location", "spec_from_loader", "ModuleSpec"}
_NAMED_ARGUMENT_SINKS: dict[str, tuple[str, str]] = {
    "getLogger": ("name", LOGGER), "LogRecord": ("name", LOGGER), "makeLogRecord": ("name", LOGGER),
    "set_level": ("logger", LOGGER), "at_level": ("logger", LOGGER), "assertLogs": ("logger", LOGGER),
    "assertNoLogs": ("logger", LOGGER), **{name: ("name", MODULE_OBJECT) for name in _MODULE_OBJECT_CALLS}}
"""Call name -> (keyword that may carry the value instead of the first argument, sink kind)."""
_MESSAGE_CALLS = {"print", "debug", "info", "warning", "error", "exception", "critical", "log", "fail", "skip",
                  "xfail", "echo", "write"}
_MESSAGE_PARTS = (cst.FormattedStringExpression, cst.FormattedString, cst.ConcatenatedString, cst.BinaryOperation,
                  cst.Arg, cst.Tuple, cst.Element)


@dataclass(frozen=True)
class Sink:
    kind: str
    table: bool = False
    """Reached through a table or a name, not written at the sink itself."""
    loose: bool = False
    """A patch target whose attribute may be missing (``raising=False``): only its module must resolve."""


@dataclass
class _Pending:
    """Values that reach a sink through a name or a table, rewritten once every use of them is known to feed a sink."""
    kind: str
    depth: int
    values: list
    binding: object = None
    table: object = None
    loop: cst.CSTNode | None = None


@dataclass
class Findings:
    sinks: dict[int, Sink] = field(default_factory=dict)
    tables: dict[int, str] = field(default_factory=dict)
    """String -> why the table it sits in cannot be proven to feed only sinks."""
    module_elements: set[int] = field(default_factory=set)
    """Strings in a container that also holds a moved module name, or the module of a (module, attr) pair."""
    path_pieces: set[int] = field(default_factory=set)
    i18n: set[int] = field(default_factory=set)
    prose: set[int] = field(default_factory=set)
    joined: set[int] = field(default_factory=set)
    """Parts of an implicitly concatenated string: judged once, as the whole text."""
    matchers: dict[int, str] = field(default_factory=dict)
    keep_module_form: set[int] = field(default_factory=set)
    """``import a`` statements whose rewrite keeps ``import new.a as a``: the from-form changes behaviour there."""
    chains: dict[int, tuple[int, str]] = field(default_factory=dict)
    """``__import__("a.b").b.X`` call -> (attributes to drop, first attribute after the module)."""
    collapse: dict[int, int] = field(default_factory=dict)
    """Attribute node -> the ``__import__`` call whose chain it belongs to."""
    import_flags: dict[int, str] = field(default_factory=dict)


def helper_parameters(source: str) -> dict[str, dict[int, str]]:
    """Module-level functions that pass a parameter straight to ``import_module``/``__import__``/``find_spec``, or to
    another such function of the same module: name -> {parameter index: parameter name}. Callers anywhere pass module
    names through them."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out: dict[str, dict[int, str]] = {}
    while True:
        found = {node.name: params for node in functions if (params := _sink_params(node, out))}
        if found == out:
            return out
        out = found


def _sink_params(node: ast.FunctionDef | ast.AsyncFunctionDef, helpers: dict[str, dict[int, str]]) -> dict[int, str]:
    params = [a.arg for a in node.args.posonlyargs + node.args.args]
    assigned = {t.id for n in ast.walk(node) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    sinks = {name: {0} for name in _HELPER_SINKS} | {name: set(found) for name, found in helpers.items()}
    imports = {"__import__"}
    for inner in ast.walk(node):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = inner.args.posonlyargs + inner.args.args
            for arg, default in zip(arguments[len(arguments) - len(inner.args.defaults):], inner.args.defaults):
                if _callee_name(default) in _HELPER_SINKS:
                    sinks[arg.arg] = {0}
                    if _callee_name(default) == "__import__":
                        imports.add(arg.arg)
    found, fed = {}, {id(n) for n in _sys_modules_lookups(node)}
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if _callee_name(call.func) in imports and not _ast_fromlist_names(call):
            continue  # it returns the top package: a rewritten name would change what the helper returns
        for index in sinks.get(_callee_name(call.func), ()):
            arg = call.args[index] if index < len(call.args) else None
            if isinstance(arg, ast.Name) and arg.id in params and arg.id not in assigned:
                found[params.index(arg.id)] = arg.id
                fed.add(id(arg))
    return {index: name for index, name in found.items() if _only_fed(node, name, fed)}


def _sys_modules_lookups(node: ast.AST):
    """Names looked up in ``sys.modules`` (``sys.modules.get(n)``, ``sys.modules[n]``, ``n in sys.modules``): module
    names too, so they do not keep a parameter from being a helper's module-name parameter."""
    def is_modules(expr: ast.AST) -> bool:
        return isinstance(expr, ast.Attribute) and expr.attr == "modules" and getattr(expr.value, "id", "") == "sys"

    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in ("get", "pop") \
                and is_modules(inner.func.value) and inner.args and isinstance(inner.args[0], ast.Name):
            yield inner.args[0]
        elif isinstance(inner, ast.Subscript) and is_modules(inner.value) and isinstance(inner.slice, ast.Name):
            yield inner.slice
        elif isinstance(inner, ast.Compare) and isinstance(inner.left, ast.Name) and len(inner.ops) == 1 \
                and isinstance(inner.ops[0], (ast.In, ast.NotIn)) and is_modules(inner.comparators[0]):
            yield inner.left


def _only_fed(node: ast.AST, name: str, fed: set[int]) -> bool:
    """Every read of ``name`` under ``node`` is a sink argument or sits in an f-string or a log/print call."""
    messages = {id(inner) for call in ast.walk(node) if isinstance(call, ast.Call)
                and _callee_name(call.func) in _MESSAGE_CALLS for arg in call.args for inner in ast.walk(arg)}
    messages |= {id(inner) for text in ast.walk(node) if isinstance(text, ast.JoinedStr) for inner in ast.walk(text)}
    return all(id(use) in fed or id(use) in messages for use in ast.walk(node)
               if isinstance(use, ast.Name) and use.id == name and isinstance(use.ctx, ast.Load))


def _callee_name(node: ast.AST) -> str:
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def _ast_fromlist_names(call: ast.Call) -> bool:
    """An ``__import__`` call whose ``fromlist`` (keyword or fourth argument) is a non-empty literal."""
    value = next((k.value for k in call.keywords if k.arg == "fromlist"), call.args[3] if len(call.args) > 3 else None)
    return isinstance(value, (ast.List, ast.Tuple)) and bool(value.elts)


def _fromlist_names(call: cst.Call) -> bool:
    """``_ast_fromlist_names`` for a libcst call."""
    positional = [arg.value for arg in call.args if arg.keyword is None and not arg.star]
    value = next((arg.value for arg in call.args if arg.keyword is not None and arg.keyword.value == "fromlist"),
                 positional[3] if len(positional) > 3 else None)
    return isinstance(value, (cst.List, cst.Tuple)) and bool(value.elements)


def is_sys_modules(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value == "sys.modules"
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Call) and node.attr.value == "modules":
        call = node.value
        return get_full_name_for_node(call.func) == "__import__" and len(call.args) == 1 \
            and isinstance(call.args[0].value, cst.SimpleString) and call.args[0].value.evaluated_value == "sys"
    return _SYS_MODULES.search(get_full_name_for_node(node) or "") is not None


def string_parts(node: cst.BaseExpression):
    if isinstance(node, cst.ConcatenatedString):
        yield from string_parts(node.left)
        yield from string_parts(node.right)
    else:
        yield node


def _callee(node: cst.Call) -> tuple[str, cst.BaseExpression | None]:
    if isinstance(node.func, cst.Attribute):
        return node.func.attr.value, node.func.value
    return (node.func.value if isinstance(node.func, cst.Name) else ""), None


class SinkFinder(cst.CSTVisitor):
    """One pass over a module that records every sink and the context the rewrite needs; nothing changes."""

    METADATA_DEPENDENCIES = (ParentNodeProvider, ScopeProvider)

    def __init__(self, plan, path: str) -> None:
        super().__init__()
        self.plan, self.path = plan, path
        self.module = path[:-3].removesuffix("/__init__").replace("/", ".") if path.endswith(".py") else ""
        self.found = Findings()
        self.caught: set[str] = set()
        self.aliases: dict[str, str] = {}
        """Local name -> sink function it is bound to (``from importlib import import_module as _im``)."""
        self.imported: dict[str, str] = {}
        """Local name -> dotted name it was imported as (for helpers defined in other modules)."""
        self.functions: list[cst.FunctionDef] = []
        self.all_functions: list[cst.FunctionDef] = []
        self.imports_in: dict[int, int] = {}
        self.removing: set[int] = set()
        self.param_sinks: dict[int, dict[str, str]] = {}
        self.local_calls: list[tuple[cst.Name, list[cst.Arg]]] = []
        self.fed: set[int] = set()
        """Name, subscript and ``.get`` nodes that sit in a sink."""
        self.pending: list[_Pending] = []
        self.scopes: set | None = None

    # Recording.

    def _mark(self, node: cst.BaseExpression | None, kind: str, table: bool = False, loose: bool = False) -> None:
        if node is None:
            return
        known = self.found.sinks.get(id(node))
        if known is None or (known.kind == COMMAND and kind == CODE):
            self.found.sinks[id(node)] = Sink(kind, table, loose)

    def _parent(self, node: cst.CSTNode) -> cst.CSTNode | None:
        return self.get_metadata(ParentNodeProvider, node, None)

    def _function(self) -> cst.FunctionDef | None:
        return self.functions[-1] if self.functions else None

    # Values that reach a sink.

    def feed(self, expr: cst.BaseExpression | None, kind: str, table: bool = False, depth: int = 0) -> None:
        """Mark every string ``expr`` provably evaluates to as a ``kind`` sink. Values reached through a name or a
        table are proven once the whole module is read (``leave_Module``)."""
        if expr is None or depth > 4:
            return
        expr = self._unwrap(expr)
        if isinstance(expr, STRINGS):
            self._mark(expr, kind, table)
            if kind in (MODULE, TARGET, SYS_MODULES, LOGGER) and isinstance(expr, cst.FormattedString):
                self._feed_prefix(expr, depth)
        elif isinstance(expr, cst.Name):
            self.fed.add(id(expr))
            self._feed_name(expr, kind, depth)
        elif isinstance(expr, cst.Subscript) or (isinstance(expr, cst.Call) and _callee(expr)[0] == "get"):
            self.fed.add(id(expr))
            owner = expr.value if isinstance(expr, cst.Subscript) else expr.func.value
            self._feed_table(owner, [], kind, depth, method="values")
        elif isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.Add) and kind == MODULE:
            self.feed(expr.left, MODULE, True, depth + 1)

    def _feed_prefix(self, expr: cst.FormattedString, depth: int) -> None:
        """``f"{PACKAGE}.{name}"``: the name bound to PACKAGE is a package name."""
        first = expr.parts[0] if expr.parts else None
        second = expr.parts[1] if len(expr.parts) > 1 else None
        if isinstance(first, cst.FormattedStringExpression) and isinstance(second, cst.FormattedStringText) \
                and second.value.startswith("."):
            self.feed(first.expression, MODULE, True, depth + 1)

    def _feed_name(self, name: cst.Name, kind: str, depth: int) -> None:
        for referent in self._referents(name) or ():
            node = getattr(referent, "node", None)
            if isinstance(node, cst.Param):
                owner = self._parent(self._parent(node))
                if isinstance(owner, cst.FunctionDef):
                    self.param_sinks.setdefault(id(owner), {})[node.name.value] = kind
                continue
            if not isinstance(node, cst.Name):
                continue
            path, current = [], node
            parent = self._parent(current)
            while isinstance(parent, cst.Element) and isinstance(self._parent(parent), (cst.Tuple, cst.List)):
                container = self._parent(parent)
                path.insert(0, list(container.elements).index(parent))
                current, parent = container, self._parent(container)
            if isinstance(parent, cst.AssignTarget) and isinstance(assign := self._parent(parent), cst.Assign) \
                    and len(assign.targets) == 1:
                self.pending.append(_Pending(kind, depth, list(self._pick(assign.value, path)), binding=referent))
            elif isinstance(parent, (cst.For, cst.CompFor)) and parent.target is current:
                self._feed_table(parent.iter, path, kind, depth, loop=parent)

    def _feed_table(self, iterable: cst.BaseExpression, path: list[int], kind: str, depth: int,
                    loop: cst.CSTNode | None = None, method: str | None = None) -> None:
        """Elements of a literal table (inline, or bound to one name), each picked at ``path``, as a pending proof."""
        if isinstance(iterable, cst.Call) and isinstance(iterable.func, cst.Attribute) and not iterable.args:
            method, iterable = iterable.func.attr.value, iterable.func.value
        table = None
        if isinstance(iterable, cst.Name):
            iterable, table = self._table_binding(iterable)
        if not isinstance(iterable, (cst.Tuple, cst.List, cst.Set, cst.Dict)):
            return
        values = [v for element in _elements(iterable, method) for v in self._pick(element, path)]
        self.pending.append(_Pending(kind, depth, values, table=table, loop=loop))

    def _table_binding(self, name: cst.Name):
        """The literal a table name is bound to, with that binding; (None, None) when it is not one literal."""
        referents = list(self._referents(name) or ())
        if len(referents) != 1 or not isinstance(getattr(referents[0], "node", None), cst.Name):
            return None, None
        target = self._parent(referents[0].node)
        assign = self._parent(target) if isinstance(target, cst.AssignTarget) else None
        if not isinstance(assign, cst.Assign) or len(assign.targets) != 1:
            return None, None
        return assign.value, referents[0]

    def _referents(self, name: cst.Name):
        scope = self.get_metadata(ScopeProvider, name, None)
        if scope is None:
            return None
        for access in scope.accesses:
            if access.node is name:
                return access.referents
        return None

    def _free(self, name: str) -> bool:
        """``name`` is bound nowhere in this file."""
        return not any(name in scope.assignments for scope in self._scopes())

    def _scopes(self):
        if self.scopes is None:
            self.scopes = {scope for scope in self.metadata[ScopeProvider].values() if scope is not None}
        return self.scopes

    def _assignments(self, name: cst.Name):
        scope = self.get_metadata(ScopeProvider, name, None)
        return list(scope.assignments[name.value]) if scope is not None else []

    def _pick(self, value: cst.BaseExpression, path: list[int]):
        for position in path:
            value = self._unwrap(value)
            if not isinstance(value, (cst.Tuple, cst.List)) or position >= len(value.elements):
                return
            value = value.elements[position].value
        yield value

    def _strings_in(self, node: cst.BaseExpression):
        node = self._unwrap(node)
        if isinstance(node, cst.SimpleString):
            yield node
        elif isinstance(node, (cst.Tuple, cst.List, cst.Set)):
            for element in node.elements:
                yield from self._strings_in(element.value)

    @staticmethod
    def _unwrap(node: cst.BaseExpression) -> cst.BaseExpression:
        while isinstance(node, cst.Call) and _callee(node)[0] in _TEXT_WRAPPERS and len(node.args) == 1:
            node = node.args[0].value
        return node

    # Proofs, once every sink of the module is known.

    def _prove(self) -> None:
        for _ in range(8):
            pending, self.pending = self.pending, []
            if not pending:
                return
            for item in pending:
                reason = self._unproven(item)
                for value in item.values:
                    if reason is None:
                        self.feed(value, item.kind, True, item.depth + 1)
                    else:
                        self.found.tables.update((id(s), reason) for s in self._strings_in(value))

    def _unproven(self, item: _Pending) -> str | None:
        """Why the values of ``item`` cannot be rewritten; None when every use of them feeds a sink."""
        if item.kind in (CODE, COMMAND):
            return None
        binding = item.binding or item.table
        if binding is not None and isinstance(binding.scope, GlobalScope) and not binding.name.startswith("_") \
                and not self.path.startswith("tests/"):
            return "public module-level name; other modules may read it"
        if item.loop is not None and not self._loop_feeds(item.loop):
            return "the loop variable is also used outside module-name sinks"
        if item.binding is not None and not all(self._fed_or_message(a.node) for a in item.binding.references):
            return "the name is also used outside module-name sinks"
        if item.table is not None and not all(self._table_use_feeds(a.node) for a in item.table.references):
            return "the table is also used outside module-name sinks"
        return None

    def _loop_feeds(self, loop: cst.CSTNode) -> bool:
        for target in _names_in(loop.target):
            scope = self.get_metadata(ScopeProvider, target, None)
            if scope is None:
                return False
            for assignment in scope.assignments[target.value]:
                if getattr(assignment, "node", None) is target and \
                        not all(self._fed_or_message(a.node) for a in assignment.references):
                    return False
        return True

    def _table_use_feeds(self, use: cst.CSTNode) -> bool:
        parent = self._parent(use)
        if isinstance(parent, (cst.For, cst.CompFor)) and parent.iter is use:
            return self._loop_feeds(parent)
        if isinstance(parent, cst.Subscript):
            return id(parent) in self.fed
        if isinstance(parent, cst.Attribute) and isinstance(call := self._parent(parent), cst.Call):
            if parent.attr.value == "get":
                return id(call) in self.fed
            loop = self._parent(call)
            return isinstance(loop, (cst.For, cst.CompFor)) and loop.iter is call and self._loop_feeds(loop)
        return False

    def _fed_or_message(self, node: cst.CSTNode) -> bool:
        """A use that feeds a sink, or one that only goes into a log line, a print or an assert message."""
        if id(node) in self.fed:
            return True
        child, parent = node, self._parent(node)
        for _ in range(6):
            if isinstance(parent, cst.Assert):
                return parent.msg is child
            if isinstance(parent, cst.Call):
                return _callee(parent)[0] in _MESSAGE_CALLS and child is not parent.func
            if not isinstance(parent, _MESSAGE_PARTS):
                return False
            child, parent = parent, self._parent(parent)
        return False

    # Imports and names bound to sink functions.

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = get_full_name_for_node(node.module) if node.module is not None else ""
        if node.relative:
            package = self.path.split("/")[:-1]
            package = package[:len(package) - (len(node.relative) - 1)]
            module = ".".join([*package, module] if module else package)
        if not module or isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            name = get_full_name_for_node(alias.name)
            bound = alias.asname.name.value if alias.asname and isinstance(alias.asname.name, cst.Name) else name
            if name in _ALIASABLE:
                self.aliases[bound] = name
            self.imported[bound] = f"{module}.{name}"

    def visit_Import(self, node: cst.Import) -> None:
        function = self._function()
        self.imports_in[id(node)] = id(function) if function is not None else 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.functions.append(node)
        self.all_functions.append(node)

    def leave_FunctionDef(self, original: cst.FunctionDef) -> None:
        self.functions.pop()

    def visit_Try(self, node: cst.Try) -> None:
        """``import a`` under a handler for ModuleNotFoundError, or one that reads ``exc.name``, keeps its form."""
        if not any(self._tells_modules_apart(handler) for handler in node.handlers):
            return
        for statement in node.body.body:
            for small in statement.body if isinstance(statement, cst.SimpleStatementLine) else ():
                if isinstance(small, cst.Import):
                    self.found.keep_module_form.add(id(small))

    @staticmethod
    def _tells_modules_apart(handler: cst.ExceptHandler) -> bool:
        types = handler.type.elements if isinstance(handler.type, cst.Tuple) else [handler.type]
        names = {get_full_name_for_node(t.value if isinstance(t, cst.Element) else t) for t in types if t is not None}
        if "ModuleNotFoundError" in names:
            return True
        if handler.name is None or not isinstance(handler.name.name, cst.Name):
            return False
        caught = handler.name.name.value
        code = cst.Module([]).code_for_node(handler.body)
        return re.search(rf"\b{re.escape(caught)}\.name\b", code) is not None

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        if node.name is not None and isinstance(node.name.name, cst.Name):
            self.caught.add(node.name.name.value)

    def visit_Del(self, node: cst.Del) -> None:
        if isinstance(node.target, cst.Subscript) and is_sys_modules(node.target.value):
            self._removes_modules()

    def _removes_modules(self) -> None:
        function = self._function()
        if function is not None:
            self.removing.add(id(function))

    # Strings: their context.

    def visit_Expr(self, node: cst.Expr) -> None:
        if isinstance(node.value, (cst.SimpleString, cst.ConcatenatedString)):
            self.found.prose.update(id(n) for n in string_parts(node.value))

    def visit_ConcatenatedString(self, node: cst.ConcatenatedString) -> None:
        if isinstance(self._parent(node), cst.ConcatenatedString):
            self.found.joined.add(id(node))
        else:
            self.found.joined.update(id(n) for n in string_parts(node))

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        """``<checkout root> / "tools"``: a path into the source tree, built from a folder name."""
        if isinstance(node.operator, cst.Divide) and isinstance(node.right, cst.SimpleString) \
                and _ROOT_EXPRESSION.search(_source(node.left)):
            self.found.path_pieces.add(id(node.right))

    def visit_Tuple(self, node: cst.Tuple) -> None:
        self._container(node.elements)
        self._mark_argv(node.elements)

    def visit_List(self, node: cst.List) -> None:
        self._container(node.elements)
        self._mark_argv(node.elements)

    def visit_Set(self, node: cst.Set) -> None:
        self._container(node.elements)

    def visit_Dict(self, node: cst.Dict) -> None:
        self._container([e for e in node.elements if isinstance(e, cst.DictElement)], keys=True)

    def _container(self, elements, keys: bool = False) -> None:
        values = [e.key for e in elements] + [e.value for e in elements] if keys else [e.value for e in elements]
        strings = [v for v in values if isinstance(v, cst.SimpleString) and isinstance(v.evaluated_value, str)]
        texts = [s.evaluated_value for s in strings]
        if any(self.plan.module_text(t) and not self.plan.is_word(t) for t in texts):
            self.found.module_elements.update(id(s) for s in strings)
        elif not keys and len(strings) == 2 == len(values) and texts[1] in self.plan.defined.get(texts[0], ()):
            self.found.module_elements.add(id(strings[0]))

    def _mark_argv(self, elements) -> None:
        """``"-m", "<module>"`` and ``"-c", "<code>"`` pairs: the child process gets them without the compat finder."""
        for flag, value in itertools.pairwise(elements):
            if isinstance(flag.value, cst.SimpleString) and flag.value.evaluated_value in ("-m", "-c"):
                self.feed(value.value, ARGV if flag.value.evaluated_value == "-m" else CODE)

    def visit_Subscript(self, node: cst.Subscript) -> None:
        if is_sys_modules(node.value) and len(node.slice) == 1 and isinstance(node.slice[0].slice, cst.Index):
            self.feed(node.slice[0].slice.value, SYS_MODULES)

    def visit_Comparison(self, node: cst.Comparison) -> None:
        self._visit_matcher(node)
        target = node.comparisons[0]
        if len(node.comparisons) != 1:
            return
        if isinstance(target.operator, (cst.In, cst.NotIn)) and is_sys_modules(target.comparator):
            self.feed(node.left, SYS_MODULES)
        elif isinstance(target.operator, (cst.Equal, cst.NotEqual)):
            for side, other in ((node.left, target.comparator), (target.comparator, node.left)):
                if self._is_record_name(other):
                    self.feed(side, LOGGER)
                elif self._is_module_key(other):
                    self.feed(side, SYS_MODULES)
        elif isinstance(target.operator, (cst.In, cst.NotIn)) and self._is_record_names(target.comparator):
            self.feed(node.left, LOGGER)

    def _visit_matcher(self, node: cst.Comparison) -> None:
        sides = [node.left, *(target.comparator for target in node.comparisons)]
        exception_name = any(isinstance(side, cst.Attribute) and side.attr.value == "name"
                             and get_full_name_for_node(side.value) in self.caught for side in sides)
        for side in sides:
            if exception_name:
                self._hold_matcher(side, "ImportError.name literal")
            elif isinstance(side, (cst.List, cst.Tuple)) and isinstance(node.comparisons[0].operator,
                                                                         (cst.Equal, cst.NotEqual)):
                for element in side.elements:
                    self._hold_matcher(element.value, "command-line matcher")

    def _hold_matcher(self, node: cst.BaseExpression, reason: str) -> None:
        for string in string_parts(node):
            if isinstance(string, cst.SimpleString):
                self.found.matchers[id(string)] = reason

    def _is_record_name(self, node: cst.BaseExpression) -> bool:
        """``record.name`` of a log record caplog captured (``for record in caplog.records``)."""
        if not (isinstance(node, cst.Attribute) and node.attr.value == "name"):
            return False
        base = node.value
        if isinstance(base, cst.Subscript):
            return _records(base.value)
        if not isinstance(base, cst.Name):
            return False
        for referent in self._referents(base) or ():
            bound = getattr(referent, "node", None)
            loop = self._parent(bound) if isinstance(bound, cst.Name) else None
            if isinstance(loop, (cst.For, cst.CompFor)) and _records(loop.iter):
                return True
        return False

    def _is_module_key(self, node: cst.BaseExpression) -> bool:
        """A name that iterates over ``sys.modules`` keys (``for name in list(sys.modules)``)."""
        if not isinstance(node, cst.Name):
            return False
        for referent in self._referents(node) or ():
            bound = getattr(referent, "node", None)
            loop = self._parent(bound) if isinstance(bound, cst.Name) else None
            if isinstance(loop, (cst.For, cst.CompFor)) and loop.target is bound and _over_modules(loop.iter):
                return True
        return False

    def _is_record_names(self, node: cst.BaseExpression) -> bool:
        return isinstance(node, (cst.ListComp, cst.SetComp, cst.GeneratorExp)) and self._is_record_name(node.elt)

    # Calls.

    def visit_Call(self, node: cst.Call) -> None:
        name, owner = _callee(node)
        positional = [arg.value for arg in node.args if arg.keyword is None and not arg.star]
        keywords = {arg.keyword.value: arg.value for arg in node.args if arg.keyword is not None}
        first = positional[0] if positional else None
        if not _foreign_interpreter(keywords):
            self._mark_argv(node.args)
        if isinstance(node.func, cst.Name):
            self.local_calls.append((node.func, list(node.args)))
        if name == "partial" and positional and isinstance(positional[0], cst.Name):
            self.local_calls.append((positional[0], list(node.args[1:])))
        if name in _I18N_CALLS and first is not None:
            self.found.i18n.update(id(n) for n in string_parts(first))
        elif name in _PATH_CALLS and any(_ROOT_EXPRESSION.search(_source(arg)) for arg in (owner, *positional)
                                         if arg is not None and not isinstance(arg, cst.SimpleString)):
            self.found.path_pieces.update(id(arg) for arg in positional if isinstance(arg, cst.SimpleString))
        self._call_sink(node, name, owner, positional, keywords)

    def _call_sink(self, node, name, owner, positional, keywords) -> None:
        first = positional[0] if positional else None
        named = _NAMED_ARGUMENT_SINKS.get(name)
        if named is not None:
            keyword, kind = named
            self.feed(keywords.get(keyword) if keyword == "logger" and name in _LOG_LEVEL_CALLS
                      else first or keywords.get(keyword), kind)
        elif name == "startswith" and isinstance(node.func, cst.Attribute):
            self._visit_startswith(node, positional)
        elif owner is not None and is_sys_modules(owner):
            self._sys_modules_method(name, first)
        elif first is not None and is_sys_modules(first):
            self._sys_modules_call(name, positional, keywords)
        elif name in ("multiple", "dict") and owner is not None and get_full_name_for_node(owner).endswith("patch"):
            self.feed(first, MODULE if name == "multiple" else TARGET)
        elif name in _MODULE_CALLS or name in self.aliases:
            self._module_call(node, self.aliases.get(name, name), first, owner, keywords)
        else:
            self._program_or_helper(node, name, owner, positional)

    def _program_or_helper(self, node, name, owner, positional) -> None:
        first = positional[0] if positional else None
        if name in _CODE_CALLS and owner is None:
            self.feed(first, CODE)
        elif name in _PROCESS_CALLS and first is not None and not isinstance(first, (cst.List, cst.Tuple)):
            self.feed(first, COMMAND)
        elif name == "partial" and isinstance(first, (cst.Name, cst.Attribute)):
            self._helper_call(node.args[1:], first.value if isinstance(first, cst.Name) else first.attr.value,
                              first.value if isinstance(first, cst.Attribute) else None)
        else:
            self._helper_call(node.args, name, owner)

    def _visit_startswith(self, node: cst.Call, positional) -> None:
        if any(f"{name}.name" in cst.Module([]).code_for_node(arg) for arg in positional for name in self.caught):
            self._hold_matcher(node.func.value, "ImportError.name literal")
        elif positional and (self._is_record_name(node.func.value) or self._is_module_key(node.func.value)):
            kind = LOGGER if self._is_record_name(node.func.value) else SYS_MODULES
            prefixes = positional[0]
            for prefix in prefixes.elements if isinstance(prefixes, cst.Tuple) else [cst.Element(prefixes)]:
                self.feed(prefix.value, kind)

    def _sys_modules_method(self, name: str, first) -> None:
        if name in _SYS_MODULES_REMOVERS:
            self._removes_modules()
        if name in _SYS_MODULES_METHODS:
            self.feed(first, SYS_MODULES)

    def _sys_modules_call(self, name: str, positional, keywords) -> None:
        if name == "dict" and len(positional) > 1 and isinstance(positional[1], cst.Dict):
            for element in positional[1].elements:
                if isinstance(element, cst.DictElement):
                    self.feed(element.key, SYS_MODULES)
        elif name in ("setitem", "delitem") and len(positional) > 1:
            if not (len(positional) > 2 and self._aliases_moved_module(positional[1], positional[2])):
                self.feed(positional[1], SYS_MODULES)
            if name == "delitem":
                self._removes_modules()
        if name == "dict" and isinstance(keywords.get("clear"), cst.Name) and keywords["clear"].value == "True":
            self._removes_modules()

    def _aliases_moved_module(self, key: cst.BaseExpression, value: cst.BaseExpression) -> bool:
        """``setitem(sys.modules, "<old name>", mod)`` where ``mod`` is already imported by the new name of that same
        module: the old name is kept on purpose (code that still imports it, a frozen old updater, reads it)."""
        if not (isinstance(key, cst.SimpleString) and isinstance(value, cst.Name)
                and isinstance(old := key.evaluated_value, str) and (new := self.plan.new(old)) is not None):
            return False
        for referent in self._referents(value) or ():
            node = getattr(referent, "node", None)
            if isinstance(node, cst.ImportFrom) and node.module is not None and not node.relative \
                    and not isinstance(node.names, cst.ImportStar):
                package = get_full_name_for_node(node.module)
                if any(f"{package}.{get_full_name_for_node(alias.name)}" == new
                       and (alias.asname.name.value if alias.asname else get_full_name_for_node(alias.name)) == value.value
                       for alias in node.names):
                    return True
        return False

    def _module_call(self, node: cst.Call, name: str, first, owner, keywords) -> None:
        kind = _MODULE_CALLS[name]
        if kind == TARGET and (first is None or not isinstance(self._unwrap(first), STRINGS + (cst.Name,))):
            return
        raising, create = keywords.get("raising"), keywords.get("create")
        if kind == TARGET and isinstance(first, cst.SimpleString) and (
                isinstance(raising, cst.Name) and raising.value == "False"
                or isinstance(create, cst.Name) and create.value == "True"):
            self._mark(first, kind, loose=True)
        if name == "__import__" and owner is None and not self._import_value_kept(node, first):
            return
        self.feed(first, kind)
        if name == "__import__" and owner is None and isinstance(first, cst.SimpleString):
            self._import_chain(node, first)

    def _import_value_kept(self, node: cst.Call, first) -> bool:
        """``__import__`` returns the top package unless ``fromlist`` names something, so a rewritten name keeps the
        call's value only for a bare literal (``_import_chain`` follows or flags it), a non-empty ``fromlist``, or a
        discarded result. Anything else keeps the old name, which the compat finder still resolves."""
        return (isinstance(first, cst.SimpleString) and len(node.args) == 1) or _fromlist_names(node) \
            or isinstance(self._parent(node), cst.Expr)

    def _import_chain(self, node: cst.Call, first: cst.SimpleString) -> None:
        """``__import__("a.b")`` returns ``a``: after the move it returns ``hermes_agent``, so the chain changes."""
        if len(node.args) != 1 or not isinstance(first.evaluated_value, str):
            return
        parts = first.evaluated_value.split(".")
        attributes, current = [], node
        while isinstance(parent := self._parent(current), cst.Attribute) and parent.value is current:
            attributes.append(parent)
            current = parent
        if not attributes:
            if not isinstance(self._parent(node), cst.Expr):
                self.found.import_flags[id(first)] = "__import__ result used without an attribute chain"
            return
        names = [a.attr.value for a in attributes]
        drop = len(parts) - 1
        if names[:drop] != parts[1:] or len(names) <= drop:
            self.found.import_flags[id(first)] = "__import__ chain does not follow the module name"
            return
        self.found.chains[id(node)] = (drop, names[drop])
        self.found.collapse.update((id(a), id(node)) for a in attributes[:drop])

    def _helper_call(self, args, name: str, owner) -> None:
        """A call to a function another module defines, whose parameter is a module name."""
        if owner is None and name in FREE_HELPERS and self._free(name):
            index = FREE_HELPERS[name]
            if index < len(args) and args[index].keyword is None and not args[index].star:
                self.feed(args[index].value, MODULE)
            return
        dotted = self.imported.get(name) if owner is None else None
        if dotted is None and owner is None and self.module:
            dotted = f"{self.module}.{name}"
        if owner is not None and (base := get_full_name_for_node(owner)) is not None:
            dotted = f"{self.imported.get(base, base)}.{name}"
        if dotted is None:
            return
        module, _, function = dotted.rpartition(".")
        params = self.plan.helper(module, function)
        for index in params:
            arg = args[index] if index < len(args) else None
            if arg is not None and arg.keyword is None and not arg.star:
                self.feed(arg.value, MODULE)

    # After the walk: parameters that reach a sink.

    def leave_Module(self, original: cst.Module) -> None:
        """Calls of this file's own helpers and pytest parametrize tables of its tests feed the parameters
        that reach a sink; imports in a function that removes modules from sys.modules keep their form."""
        self._prove()
        for function in self.all_functions:
            sinks = self.param_sinks.get(id(function))
            if not sinks:
                continue
            hold = {name: None if self._param_feeds(function, name) else
                    "the parameter it is passed to is also used outside module-name sinks" for name in sinks}
            self._parametrize(function, sinks, hold)
            self._local_callers(function, sinks, hold)
        for node_id, function_id in self.imports_in.items():
            if function_id and function_id in self.removing:
                self.found.keep_module_form.add(node_id)
        self._prove()

    def _param_feeds(self, function: cst.FunctionDef, name: str) -> bool:
        """Every use of parameter ``name`` in ``function`` feeds a sink or only a message."""
        for param in function.params.posonly_params + function.params.params + function.params.kwonly_params:
            if param.name.value != name:
                continue
            for assignment in self._assignments(param.name):
                if getattr(assignment, "node", None) is param:
                    return all(self._fed_or_message(access.node) for access in assignment.references)
        return False

    def _hold_or_feed(self, value, kind: str, hold: str | None) -> None:
        if hold is None:
            self.feed(value, kind, True)
        elif value is not None:
            self.found.tables.update((id(string), hold) for string in self._strings_in(value))

    def _parametrize(self, function: cst.FunctionDef, sinks: dict[str, str], hold: dict[str, str | None]) -> None:
        for decorator in function.decorators:
            call = decorator.decorator
            if not (isinstance(call, cst.Call) and _callee(call)[0] == "parametrize" and len(call.args) >= 2):
                continue
            names = call.args[0].value
            if not isinstance(names, cst.SimpleString) or not isinstance(names.evaluated_value, str):
                continue
            argnames = [n.strip() for n in names.evaluated_value.split(",") if n.strip()]
            rows = call.args[1].value
            for position, argname in enumerate(argnames):
                if argname in sinks:
                    self._feed_rows(rows, position, len(argnames), sinks[argname], hold[argname])

    def _feed_rows(self, rows, position: int, width: int, kind: str, hold: str | None) -> None:
        if not isinstance(rows, (cst.List, cst.Tuple)):
            return
        for row in rows.elements:
            value = row.value
            if isinstance(value, cst.Call) and _callee(value)[0] == "param":
                value = cst.Tuple([cst.Element(a.value) for a in value.args if a.keyword is None])
                if width == 1:
                    value = value.elements[0].value if value.elements else None
            if width == 1:
                self._hold_or_feed(value, kind, hold)
            elif isinstance(value, (cst.Tuple, cst.List)) and position < len(value.elements):
                self._hold_or_feed(value.elements[position].value, kind, hold)

    def _local_callers(self, function: cst.FunctionDef, sinks: dict[str, str], hold: dict[str, str | None]) -> None:
        params = [p.name.value for p in function.params.posonly_params + function.params.params]
        for func, args in self.local_calls:
            if func.value != function.name.value:
                continue
            referents = self._referents(func) or self._assignments(func)
            if not referents or not all(getattr(r, "node", None) is function for r in referents):
                continue
            for index, arg in enumerate(args):
                name = arg.keyword.value if arg.keyword is not None else params[index] if index < len(params) else None
                if name in sinks and not arg.star:
                    self._hold_or_feed(arg.value, sinks[name], hold[name])


def _elements(node: cst.BaseExpression | None, method: str | None):
    if isinstance(node, cst.Dict):
        for element in node.elements:
            if not isinstance(element, cst.DictElement):
                continue
            if method == "items":
                yield cst.Tuple([cst.Element(element.key), cst.Element(element.value)])
            else:
                yield element.value if method == "values" else element.key
    elif isinstance(node, (cst.Tuple, cst.List, cst.Set)) and method in (None, "values"):
        for element in node.elements:
            if not isinstance(element, cst.StarredElement):
                yield element.value


def _foreign_interpreter(keywords: dict[str, cst.BaseExpression]) -> bool:
    """The call names the interpreter its ``-m``/``-c`` arguments go to, and it is not this tree's own: the program
    may run in another install (an earlier release), where the old names are the right ones."""
    return any(get_full_name_for_node(value) not in ("sys.executable", "None") for keyword, value in keywords.items()
               if keyword in _INTERPRETER_KEYWORDS)


def _names_in(node: cst.CSTNode):
    if isinstance(node, cst.Name):
        yield node
    elif isinstance(node, (cst.Tuple, cst.List)):
        for element in node.elements:
            yield from _names_in(element.value)


def _source(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)


def _over_modules(node: cst.BaseExpression) -> bool:
    while isinstance(node, cst.Call):
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "keys" and not node.args:
            node = node.func.value
        elif isinstance(node.func, cst.Name) and node.func.value in ("list", "sorted", "tuple", "set") and node.args:
            node = node.args[0].value
        else:
            return False
    return is_sys_modules(node)


def _records(node: cst.BaseExpression) -> bool:
    code = cst.Module([]).code_for_node(node)
    return code.endswith(".records") or "get_records(" in code or code.endswith("record_tuples")

