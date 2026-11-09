"""Rewrite Python sources from old module names to the names a move commit gave them.

Usage: python scripts/move/rewrite_imports.py --from-commit <A> [--paths P ...] [--check]

Rewritten: ``import`` and ``from`` statements, attribute chains of plain ``import`` bindings,
``"-m", "<module>"`` argv pairs and ``-m <module>`` in strings, ``sys.modules`` keys and logger names
(the strict compat finder in tests cannot see those). Every other site that names a moved module or path
is printed as a flag, one line each: ``path:line  <class>  <old name>  <reason>``. The new old-name keys
go into the compat MOVES table. Exit status 1 when a file fails to parse, a flag remains, or (with
--check) a file would change.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider, ScopeProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renames import current_name, git, module_renames, moved_modules, renamed_paths

DYNAMIC, STRING_REF, BY_PATH, FILE_MATH, OTHER = (
    "dynamic-import", "string-module-ref", "by-path", "file-math", "other")
_DASH_M = re.compile(r"(?<![\w-])(-m(?:\s+|[\"'],\s*[\"']))([A-Za-z_][\w.]*)")
_CODE_IMPORT = re.compile(r"(?<![\w.])((?:from|import)\s+)([A-Za-z_][\w.]*)(?=\s+import\b|\s*(?:$|[\n,;)]|\\n| as\b))",
                          re.MULTILINE)
_PYTHON_SHEBANG = re.compile(r"#!.*\bpython")
_SYS_MODULES = re.compile(r"(?:(?:^|\.)_*sys_*\.)?modules$")
_ASSERT_LOGS = {"assertLogs", "assertNoLogs"}
_SYS_MODULES_METHODS = {"pop", "get", "setdefault"}
_ATTRIBUTE_TARGETS = {"patch", "setattr", "delattr", "resolve_name"}
_MODULE_TARGETS = {"import_module", "__import__", "find_spec", "importorskip"}


@dataclass
class Plan:
    renames: dict[str, str]
    paths: list[tuple[str, str]]
    names: dict[str, set[str]]
    """Every moved module's old name -> the names its top level defines."""

    @property
    def moved_files(self) -> set[str]:
        return {new for _, new in self.paths}

    def old_paths(self) -> list[tuple[str, str]]:
        """Old -> new path text a by-path literal can carry: each moved file and each moved directory."""
        pairs = {old: new for old, new in self.paths if old.endswith(".py")}
        for old, new in self.paths:
            old_dir, new_dir = str(PurePosixPath(old).parent), str(PurePosixPath(new).parent)
            if old_dir not in (".", new_dir):
                pairs[old_dir + "/"] = new_dir + "/"
        return sorted(pairs.items(), key=lambda item: -len(item[0]))

    def resolves(self, text: str, attribute_allowed: bool) -> bool:
        """``text`` starts with a moved module; what follows it is an attribute only when ``attribute_allowed``."""
        name, colon, _ = text.partition(":")
        parts = name.split(".")
        for size in range(len(parts), 0, -1):
            if ".".join(parts[:size]) in self.names:
                return size == len(parts) or (attribute_allowed and not colon)
        return False

    def names_module(self, text: str) -> bool:
        """``text`` is a moved module, ``module:attr`` or ``module.attr`` of a name that module defines."""
        module, colon, _ = text.partition(":")
        if colon:
            return module in self.names
        parts = text.split(".")
        for size in range(len(parts), 0, -1):
            prefix = ".".join(parts[:size])
            if prefix in self.names:
                return size == len(parts) or parts[size] in self.names[prefix]
        return False


@dataclass
class Result:
    path: str
    source: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _Sink:
    kind: str
    attribute_allowed: bool | None = None
    """None: rewrite any literal; otherwise rewrite only a name that resolves in the moved tree."""


class _Rewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plan: Plan, path: str, accesses: dict[cst.CSTNode, str | None]) -> None:
        super().__init__()
        self.plan, self.path, self.accesses = plan, path, accesses
        self.counts: dict[str, int] = {}
        self.flags: list[str] = []
        self.sinks: dict[int, _Sink] = {}
        self.prose: set[int] = set()
        self.blocked: dict[int, str] = {}
        self.moved = path in plan.moved_files
        self.tops = {old.split(".")[0] for old in plan.renames}

    def _count(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _flag(self, node: cst.CSTNode, kind: str, name: str, reason: str) -> None:
        line = self.get_metadata(PositionProvider, node).start.line
        self.flags.append(f"{self.path}:{line}  {kind}  {name}  {reason}")

    def _new(self, dotted: str | None) -> str | None:
        return current_name(dotted, self.plan.renames) if dotted else None

    # Imports and the names they bind.

    def leave_Import(self, original: cst.Import, updated: cst.Import) -> cst.Import:
        names = []
        for alias in updated.names:
            new = self._new(get_full_name_for_node(alias.name))
            if new is not None:
                alias = alias.with_changes(name=cst.parse_expression(new))
                self._count("import")
            names.append(alias)
        return updated.with_changes(names=names)

    def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if updated.relative or updated.module is None:
            return updated
        module = get_full_name_for_node(updated.module)
        new_module = self._new(module)
        if not isinstance(updated.names, cst.ImportStar):
            split = self._split_parent(original, updated, module, new_module or module)
            if split is not None:
                return split
        if new_module is None:
            return updated
        self._count("import")
        return updated.with_changes(module=cst.parse_expression(new_module))

    def _split_parent(self, original: cst.ImportFrom, node: cst.ImportFrom, module: str,
                      new_module: str) -> cst.ImportFrom | None:
        """``from pkg import child`` where ``pkg.child`` moved apart from ``pkg``."""
        targets = [self._new(f"{module}.{alias.name.value}") for alias in node.names]
        apart = [t for t, alias in zip(targets, node.names)
                 if t is not None and t != f"{new_module}.{alias.name.value}"]
        if not apart:
            return None
        parents = {t.rpartition(".")[0] for t in apart}
        if len(apart) != len(node.names) or len(parents) != 1:
            self._flag(original, OTHER, module, "from-import mixes children that moved apart; split it by hand")
            return None
        names = []
        for alias, target in zip(node.names, targets):
            leaf = target.rpartition(".")[2]
            bound = alias.asname.name.value if alias.asname else alias.name.value
            asname = None if leaf == bound else cst.AsName(name=cst.Name(bound))
            names.append(alias.with_changes(name=cst.Name(leaf), asname=asname))
        self._count("import")
        return node.with_changes(module=cst.parse_expression(parents.pop()), names=names)

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.BaseExpression:
        if self.moved and original.value == "__file__":
            self._flag(original, FILE_MATH, "__file__", "this file moved; check the path arithmetic")
        return self._access(original, updated)

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        return self._access(original, updated)

    def visit_Del(self, node: cst.Del) -> None:
        self.blocked.update((id(n), "del of a name bound by a moved import; fix by hand")
                            for n in _names_in(node.target))

    def visit_MatchAs(self, node: cst.MatchAs) -> None:
        if node.name is not None:
            self.blocked[id(node.name)] = "pattern rebinds a name bound by a moved import; fix by hand"

    def _access(self, original, updated):
        if original not in self.accesses:
            return updated
        new, reason = self.accesses[original]
        if id(original) in self.blocked:
            new, reason = None, self.blocked[id(original)]
        if new is None:
            self._flag(original, OTHER, get_full_name_for_node(original), reason)
            return updated
        self._count("attribute")
        return cst.parse_expression(new)

    # Which string nodes are sinks: decided top-down, before their children are left.

    def visit_Subscript(self, node: cst.Subscript) -> None:
        if _is_sys_modules(node.value) and len(node.slice) == 1 \
                and isinstance(node.slice[0].slice, cst.Index):
            self._mark(node.slice[0].slice.value, "sys-modules")

    def visit_Comparison(self, node: cst.Comparison) -> None:
        target = node.comparisons[0]
        if len(node.comparisons) == 1 and isinstance(target.operator, (cst.In, cst.NotIn)) \
                and _is_sys_modules(target.comparator):
            self._mark(node.left, "sys-modules")

    def visit_Call(self, node: cst.Call) -> None:
        func = get_full_name_for_node(node.func) or ""
        owner_node = node.func.value if isinstance(node.func, cst.Attribute) else None
        method = node.func.attr.value if owner_node is not None else func
        positional = [arg.value for arg in node.args if arg.keyword is None and not arg.star]
        keywords = {arg.keyword.value: arg.value for arg in node.args if arg.keyword is not None}
        if method == "getLogger" and (positional or "name" in keywords):
            self._mark(positional[0] if positional else keywords["name"], "logger")
        elif method in ("set_level", "at_level") and "logger" in keywords:
            self._mark(keywords["logger"], "logger")
        elif method in _ASSERT_LOGS and (positional or "logger" in keywords):
            self._mark(positional[0] if positional else keywords["logger"], "logger")
        elif owner_node is not None and _is_sys_modules(owner_node) and method in _SYS_MODULES_METHODS and positional:
            self._mark(positional[0], "sys-modules")
        elif positional and _is_sys_modules(positional[0]):
            if method == "dict" and len(positional) > 1 and isinstance(positional[1], cst.Dict):
                for element in positional[1].elements:
                    if isinstance(element, cst.DictElement):
                        self._mark(element.key, "sys-modules")
            elif method in ("setitem", "delitem") and len(positional) > 1:
                self._mark(positional[1], "sys-modules")
        elif positional and method in _ATTRIBUTE_TARGETS | _MODULE_TARGETS:
            self._mark(positional[0], STRING_REF, method in _ATTRIBUTE_TARGETS)

    def visit_Expr(self, node: cst.Expr) -> None:
        if isinstance(node.value, (cst.SimpleString, cst.ConcatenatedString)):
            self.prose.update(id(n) for n in _strings(node.value))

    def visit_List(self, node: cst.List) -> None:
        self._mark_argv(node.elements)

    def visit_Tuple(self, node: cst.Tuple) -> None:
        self._mark_argv(node.elements)

    def _mark_argv(self, elements) -> None:
        """``"-m", "<module>"`` element pairs: the child process runs without the compat finder."""
        for flag, module in itertools.pairwise(elements):
            if isinstance(flag.value, cst.SimpleString) and flag.value.evaluated_value == "-m":
                self._mark(module.value, "argv")

    def _mark(self, node: cst.BaseExpression, kind: str, attribute_allowed: bool | None = None) -> None:
        self.sinks[id(node)] = _Sink(kind, attribute_allowed)

    # Strings, at the string node.

    def on_leave(self, original, updated):
        updated = super().on_leave(original, updated)
        sink = self.sinks.get(id(original))
        if sink is not None:
            return self._leave_sink(original, updated, sink)
        if isinstance(original, cst.SimpleString):
            return self._leave_string(original, updated)
        if isinstance(original, cst.FormattedString):
            return self._leave_fstring(original, updated)
        return updated

    def _leave_sink(self, original, updated, sink: _Sink):
        if isinstance(updated, cst.SimpleString) and isinstance(text := updated.evaluated_value, str):
            if self._new(text.partition(":")[0]) is None:
                return updated
            if sink.attribute_allowed is not None and not self.plan.resolves(text, sink.attribute_allowed):
                self._flag(original, sink.kind, text, "string target does not resolve in the moved tree; fix by hand")
                return updated
            rewritten = _with_text(updated, self._whole)
            if rewritten is None:
                self._flag(original, OTHER, text, f"{sink.kind} literal spells the name with escapes; fix by hand")
                return updated
            self._count(sink.kind)
            return rewritten
        if isinstance(original, (cst.FormattedString, cst.BinaryOperation, cst.ConcatenatedString)) \
                and _mentions_old(original, self.plan.renames):
            self._flag(original, DYNAMIC, _code(original), f"computed {sink.kind} name; check by hand")
        return updated

    def _whole(self, text: str) -> str:
        module, colon, attr = text.partition(":")
        return (self._new(module) or module) + colon + attr

    def _leave_string(self, original: cst.SimpleString, updated: cst.SimpleString):
        text = original.evaluated_value
        if not isinstance(text, str):
            return updated
        rewritten = self._text(text, original)
        if self.plan.names_module(rewritten):
            self._flag(original, OTHER, text, "string names a moved module; rewrite or confirm by hand")
        else:
            for old, new in self.plan.old_paths():
                if re.search(rf"(?<![\w/.]){re.escape(old)}", rewritten):
                    where = "docstring prose" if id(original) in self.prose else "string"
                    self._flag(original, BY_PATH, old, f"{where} names a moved path (now {new})")
                    break
        if rewritten == text:
            return updated
        dashed = _DASH_M.sub(self._dash_m, text)
        replaced = _with_text(updated, lambda raw: _DASH_M.sub(self._dash_m, raw)) if rewritten == dashed \
            else _with_value(updated, rewritten)
        if replaced is None:
            self._flag(original, OTHER, text[:80], "code in a string spells the name with escapes; fix by hand")
            return updated
        self._count("string-code")
        return replaced

    def _text(self, text: str, node: cst.CSTNode | None = None) -> str:
        """``-m <module>`` arguments, and the Python program when the text parses as one: code a child process runs."""
        text = _DASH_M.sub(self._dash_m, text)
        if node is None or "import" not in text or not any(top in text for top in self.tops):
            return text
        try:
            program = rewrite_source(text, "<string>", self.plan)
        except RecursionError:
            program = Result("<string>", error="too deep")
        if program.error:
            for match in _CODE_IMPORT.finditer(text):
                if self._new(match.group(2)) is not None:
                    self._flag(node, OTHER, match.group(0), "code in a string imports a moved module but does not parse; fix by hand")
            return text
        for flag in program.flags:
            _, kind, name, reason = flag.split("  ", 3)
            self._flag(node, kind, name, reason)
        return program.source if program.source is not None else text

    def _dash_m(self, match: re.Match) -> str:
        new = self._new(match.group(2))
        return match.group(0) if new is None else match.group(1) + new

    def _leave_fstring(self, original: cst.FormattedString, updated: cst.FormattedString):
        parts = []
        for part in updated.parts:
            if isinstance(part, cst.FormattedStringText):
                text = _DASH_M.sub(self._dash_m, part.value)
                if text != part.value:
                    self._count("string-code")
                    part = part.with_changes(value=text)
            parts.append(part)
        updated = updated.with_changes(parts=parts)
        for match in _CODE_IMPORT.finditer("".join(_literal_text(part) for part in parts)):
            if self._new(match.group(2)) is not None:
                self._flag(original, OTHER, match.group(0), "f-string builds code that imports a moved module; fix by hand")
        if _imports_old(original, self.plan.renames):
            self._flag(original, DYNAMIC, _code(original), "f-string builds a moved module name; check by hand")
        return updated


def _strings(node: cst.BaseExpression):
    if isinstance(node, cst.SimpleString):
        yield node
    elif isinstance(node, cst.ConcatenatedString):
        yield from _strings(node.left)
        yield from _strings(node.right)


def _is_sys_modules(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value == "sys.modules"
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Call) and node.attr.value == "modules":
        call = node.value
        return get_full_name_for_node(call.func) == "__import__" and len(call.args) == 1 \
            and isinstance(call.args[0].value, cst.SimpleString) and call.args[0].value.evaluated_value == "sys"
    return _SYS_MODULES.search(get_full_name_for_node(node) or "") is not None


def _with_text(node: cst.SimpleString, transform) -> cst.SimpleString | None:
    """``node`` with ``transform`` applied to its source text; None when that does not give the transformed value.

    The check catches literals that spell a module name with escapes, which a source-text edit cannot see.
    """
    head = len(node.prefix) + len(node.quote)
    raw = node.value[head:-len(node.quote)]
    candidate = node.with_changes(value=node.value[:head] + transform(raw) + node.quote)
    return candidate if candidate.evaluated_value == transform(node.evaluated_value) else None


def _with_value(node: cst.SimpleString, value: str) -> cst.SimpleString | None:
    """``node`` spelling ``value``; None when its escapes cannot be kept (``\\x``, ``\\u``, a raw prefix with backslashes)."""
    head = len(node.prefix) + len(node.quote)
    raw = node.value[head:-len(node.quote)]
    if raw != node.evaluated_value:
        if "\\" in re.sub(r"\\[nt\\'\"]", "", raw) or "r" in node.prefix.lower():
            return None
        quote = node.quote[0]
        raw = value.replace("\\", "\\\\").replace(quote, "\\" + quote)
        if len(node.quote) == 1:
            raw = raw.replace("\n", "\\n").replace("\t", "\\t")
    else:
        raw = value
    candidate = node.with_changes(value=node.value[:head] + raw + node.quote)
    return candidate if candidate.evaluated_value == value else None


def _code(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node).replace("\n", " ")[:80]


def _literal_text(node: cst.CSTNode) -> str:
    if isinstance(node, cst.FormattedStringText):
        return node.value
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value if isinstance(node.evaluated_value, str) else ""
    return "".join(_literal_text(child) for child in node.children)


def _mentions_old(node: cst.CSTNode, renames: dict[str, str]) -> bool:
    text = _literal_text(node)
    return any(re.search(rf"(?<![\w./-]){re.escape(old)}(?![\w/-])", text) for old in renames)


def _imports_old(node: cst.FormattedString, renames: dict[str, str]) -> bool:
    """An f-string whose text starts with a moved module and continues with ``.{...}``."""
    first = node.parts[0] if node.parts else None
    if not isinstance(first, cst.FormattedStringText):
        return False
    return any(first.value == old + "." or first.value.startswith(old + ".") and first.value.endswith(".")
               for old in renames) and len(node.parts) > 1


def _binding_accesses(wrapper: MetadataWrapper, renames: dict[str, str]):
    """Each use of a name bound by a plain ``import``, as (new dotted spelling, None) or (None, flag reason).

    A use is rewritten only when every binding it can see is an import of a moved module.
    """
    found = {}
    parents = wrapper.resolve(ParentNodeProvider)
    for scope in set(wrapper.resolve(ScopeProvider).values()):
        if scope is None:
            continue
        for access in scope.accesses:
            if not isinstance(access.node, (cst.Name, cst.Attribute)):
                continue
            referents = list(access.referents)
            imports = [r for r in referents if isinstance(getattr(r, "node", None), cst.Import)
                       and _plain_alias(r.node, r.name)]
            moved = [r for r in imports if _imports_moved(r.node, r.name, renames)]
            new = current_name(get_full_name_for_node(access.node), renames)
            if imports and not moved:
                chain = _chain(access.node, parents)
                if current_name(get_full_name_for_node(chain), renames) is not None:
                    found[access.node] = (None, "uses a moved submodule through an unmoved import; fix by hand")
                continue
            if moved and len(moved) == len(referents):
                found[access.node] = (new, None) if new is not None else \
                    (None, "name bound by a rewritten import does not map; fix by hand")
            elif moved and new is not None:
                found[access.node] = (None, "name also bound by something other than a moved import; fix by hand")
    return found


def _chain(node: cst.CSTNode, parents) -> cst.CSTNode:
    """The longest ``a.b.c`` attribute chain that starts at ``node``."""
    while isinstance(parent := parents.get(node), cst.Attribute) and parent.value is node:
        node = parent
    return node


def _names_in(node: cst.CSTNode):
    if isinstance(node, (cst.Name, cst.Attribute)):
        yield node
    elif isinstance(node, (cst.Tuple, cst.List)):
        for element in node.elements:
            yield from _names_in(element.value)


def _imports_moved(node: cst.Import, name: str, renames: dict[str, str]) -> bool:
    return any(current_name(full, renames) is not None for alias in node.names
               if (full := get_full_name_for_node(alias.name)) and (full == name or full.startswith(name + ".")))


def _plain_alias(node: cst.Import, name: str) -> bool:
    """``import a.b`` binds ``a`` with no alias; an access through ``a`` resolves from the root."""
    return any(alias.asname is None and (full := get_full_name_for_node(alias.name)) is not None
               and (full == name or full.startswith(name + ".")) for alias in node.names)


def rewrite_source(source: str, path: str, plan: Plan) -> Result:
    try:
        wrapper = MetadataWrapper(cst.parse_module(source))
    except cst.ParserSyntaxError as error:
        return Result(path, error=f"{path}: parse error: {str(error).splitlines()[0]}")
    rewriter = _Rewriter(plan, path, _binding_accesses(wrapper, plan.renames))
    new = wrapper.visit(rewriter).code
    return Result(path, new if new != source else None, rewriter.counts, rewriter.flags)


def top_level_names(source: str) -> set[str]:
    """Names a module's top level binds, including inside its top-level ``if``/``try``/``with``/``for``."""
    names = set()
    pending = list(ast.parse(source).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            pending += node.body + getattr(node, "orelse", []) + getattr(node, "finalbody", []) + \
                [s for h in getattr(node, "handlers", []) for s in h.body]
            if isinstance(node, ast.For):
                names |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _work(item: tuple[Path, str, Plan]) -> Result:
    root, path, plan = item
    return rewrite_source((root / path).read_text(encoding="utf-8-sig"), path, plan)


def _candidates(root: Path, paths: list[str], plan: Plan) -> list[str]:
    listed = git(root, "ls-files", "--", *(paths or ["."])).splitlines()
    words = {old.split(".")[0] for old in plan.renames} | {old.rstrip("/") for old, _ in plan.old_paths()}
    needle = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)))
    out = []
    for path in listed:
        if PurePosixPath(path).parts[-2:] == ("compat", "moves.py"):
            continue
        file = root / path
        if not file.is_file() or (PurePosixPath(path).suffix not in ("", ".py")):
            continue
        text = file.read_text(encoding="utf-8-sig", errors="replace")
        if (path.endswith(".py") or _PYTHON_SHEBANG.match(text)) and (path in plan.moved_files or needle.search(text)):
            out.append(path)
    return out


def update_moves(root: Path, renames: dict[str, str], write: bool) -> int:
    """Add this move's old names to the compat MOVES table; earlier keys follow their target's new name."""
    tracked = git(root, "ls-files", "*hermes_agent/compat/moves.py").split()
    if not tracked:
        return 0
    file = root / tracked[0]
    source = file.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.AnnAssign)
                and getattr(n.target, "id", None) == "MOVES")
    moves = ast.literal_eval(node.value)
    table = {old: current_name(new, renames) or new for old, new in moves.items()} | renames
    if table == moves:
        return 0
    body = "".join(f'    "{old}": "{new}",\n' for old, new in sorted(table.items()))
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [f"MOVES: dict[str, str] = {{\n{body}}}\n"]
    if write:
        file.write_text("".join(lines), encoding="utf-8")
    return sum(1 for old in table if moves.get(old) != table[old])


def build_plan(root: Path, commit: str) -> Plan:
    names = {old: top_level_names((root / path).read_text(encoding="utf-8-sig"))
             for old, (_, path) in moved_modules(root, commit).items()}
    return Plan(module_renames(root, commit), renamed_paths(root, commit), names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-commit", required=True, help="the move commit (git mv only)")
    parser.add_argument("--paths", nargs="*", default=[], help="limit to these files or directories")
    parser.add_argument("--check", action="store_true", help="write nothing; fail when anything would change")
    args = parser.parse_args(argv)
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    plan = build_plan(root, args.from_commit)
    if not plan.renames:
        print("rewritten=0 flagged=0 failed=0")
        return 0
    files = _candidates(root, args.paths, plan)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_work, [(root, f, plan) for f in files], chunksize=8))
    counts: dict[str, int] = {}
    flags, failed, changed = [], [], []
    for result in results:
        if result.error:
            failed.append(result.error)
            continue
        flags += result.flags
        for kind, n in result.counts.items():
            counts[kind] = counts.get(kind, 0) + n
        if result.source is not None:
            changed.append(result.path)
            if not args.check:
                (root / result.path).write_text(result.source, encoding="utf-8")
    moves = update_moves(root, plan.renames, write=not args.check)
    for line in failed + flags:
        print(line)
    kinds = " ".join(f"{kind}={n}" for kind, n in sorted(counts.items()))
    print(f"files={len(changed)} moves={moves} {kinds}".rstrip())
    print(f"rewritten={sum(counts.values())} flagged={len(flags)} failed={len(failed)}")
    return 1 if failed or flags or (args.check and (changed or moves)) else 0


if __name__ == "__main__":
    sys.exit(main())
