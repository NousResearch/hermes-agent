"""Rewrite Python sources from old module names to the names a move commit gave them.

Usage: python scripts/move/rewrite_imports.py --from-commit <A> [--paths P ...] [--check]

Sites listed in ``exclusions.py`` are left as they are and printed once per reason as ``excluded <reason>``;
sites listed as by-hand are flagged instead of rewritten. A row whose anchor is no longer in its file is flagged.
A UTF-8 BOM is kept; a file that is not UTF-8 is skipped with a flag.

Rewritten: ``import`` and ``from`` statements (``import a`` keeps its bound name: ``import new.a as a``),
attribute chains of plain ``import a.b`` bindings, ``"-m", "<module>"`` argv pairs and ``-m <module>`` in
strings, ``sys.modules`` keys and logger names (the strict compat finder in tests cannot see those), a string
passed to any call whose whole value is a moved module or ``module.attr`` of a name it defines, the literal
head of an f-string target, and ``pyproject.toml`` entry points and package-data keys. Config keys
(``DEFAULT_CONFIG`` paths such as ``agent.reasoning_effort``) never change. Every other site that names a
moved module or path is printed as a flag, one line each: ``path:line  <class>  <old name>  <reason>``.
The new old-name keys go into the compat MOVES table. Exit status 1 when a file fails to parse, a flag
remains, or (with --check) a file would change.
"""
from __future__ import annotations

import argparse
import ast
import functools
import itertools
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import libcst as cst
from libcst.helpers import get_full_name_for_node
from libcst.metadata import GlobalScope, MetadataWrapper, ParentNodeProvider, PositionProvider, ScopeProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusions import BY_HAND, EXCLUDED, IDENTITY, anchor_spans
from renames import current_name, git, module_renames, moved_modules, renamed_paths
from rewrite_toml import rewrite_pyproject

DYNAMIC, STRING_REF, BY_PATH, FILE_MATH, BY_HAND_KIND, OTHER, CALL_ARG = (
    "dynamic-import", "string-module-ref", "by-path", "file-math", "by-hand", "other", "call-arg")
_DOTTED_NAME = re.compile(r"[A-Za-z_][\w.]*(?::[A-Za-z_][\w.]*)?")
GENERATE_HINT = ("next: after `just move-docs`, run `just move-generate`: generated files embed code and docstrings "
                 "this move rewrote (apps/shared/src/gateway-contract.*, tests/hermes_cli/serve_handler_ledger.json)")
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
    moves: dict[str, str] = field(default_factory=dict)
    """The compat MOVES table of the tree: earlier chunks' old names, so exclusion paths match after any move."""
    excluded: tuple[tuple[str, str, str], ...] = EXCLUDED
    by_hand: tuple[tuple[str, str], ...] = BY_HAND
    config_keys: frozenset[str] = frozenset()
    """Dotted ``DEFAULT_CONFIG`` keys: ``agent.reasoning_effort`` is a config key even where it is also a module."""

    @property
    def moved_files(self) -> set[str]:
        return {new for _, new in self.paths}

    def aliases(self, path: str) -> set[str]:
        """``path`` and where it lives after this move and the earlier ones in MOVES."""
        out = {path} | {new for old, new in self.paths if old == path}
        if path.endswith(".py"):
            module = path[:-3].removesuffix("/__init__").replace("/", ".")
            new = current_name(module, {**self.moves, **self.renames})
            if new is not None and current_name(new, self.renames) is not None:
                new = current_name(new, self.renames)
            if new is not None:
                stem = new.replace(".", "/")
                out |= {f"{stem}.py", f"{stem}/__init__.py"} if path.endswith("__init__.py") else {f"{stem}.py"}
        return out

    @functools.cached_property
    def rows_by_path(self) -> dict[str, tuple[list[tuple[str, str]], list[str]]]:
        """Each file path (old or new) -> its excluded (anchor, reason) rows and by-hand anchors."""
        rows: dict[str, tuple[list, list]] = {}
        for path, anchor, reason in self.excluded:
            for alias in self.aliases(path):
                rows.setdefault(alias, ([], []))[0].append((anchor, reason))
        for path, anchor in self.by_hand:
            for alias in self.aliases(path):
                rows.setdefault(alias, ([], []))[1].append(anchor)
        return rows

    @functools.cached_property
    def path_pattern(self) -> re.Pattern | None:
        """One pattern for every old path text (compiled once: per-literal compiles dominated chunk 11's run time)."""
        olds = [old for old, _ in self.old_paths()]
        return re.compile(r"(?<![\w/.])(" + "|".join(map(re.escape, olds)) + ")") if olds else None

    @functools.cached_property
    def new_path(self) -> dict[str, str]:
        return dict(self.old_paths())

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
    excluded: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _Sink:
    kind: str
    attribute_allowed: bool | None = None
    by_hand: bool = False
    """None: rewrite any literal; otherwise rewrite only a name that resolves in the moved tree."""
    pair: cst.BaseExpression | None = None
    """For a call argument: the positional argument after it (``_lazy("mod", "attr")``)."""


class _Rewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plan: Plan, path: str, accesses: dict[cst.CSTNode, str | None],
                 blocked_imports: dict[cst.CSTNode, str] | None = None) -> None:
        super().__init__()
        self.plan, self.path, self.accesses = plan, path, accesses
        self.blocked_imports = blocked_imports or {}
        self.counts: dict[str, int] = {}
        self.flags: list[str] = []
        self.excluded: list[str] = []
        self.ranges: list[tuple[int, int, str]] = []
        self.by_hand_spans: list[tuple[int, int]] = []
        self.stale: list[str] = []
        self.sinks: dict[int, _Sink] = {}
        self.prose: set[int] = set()
        self.blocked: dict[int, str] = {}
        self.matchers: dict[int, str] = {}
        self.caught: set[str] = set()
        self.moved = path in plan.moved_files
        self.tops = {old.split(".")[0] for old in plan.renames}

    def _count(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _flag(self, node: cst.CSTNode, kind: str, name: str, reason: str) -> None:
        if self._in_range(node):
            return
        line = self.get_metadata(PositionProvider, node).start.line
        self.flags.append(f"{self.path}:{line}  {kind}  {name}  {reason}")

    def anchor(self, source: str) -> None:
        """Turn this file's exclusion rows into line spans of ``source``; a row whose anchor is gone is flagged."""
        excluded, by_hand = self.plan.rows_by_path.get(self.path, ([], []))
        for anchor, reason in excluded:
            spans = anchor_spans(source, anchor)
            self.ranges += [(first, last, reason) for first, last in spans]
            if not spans:
                self.stale.append(f"{self.path}:1  stale-exclusion  {anchor[:60]}  excluded site ({reason}) not found; "
                                  "check the file by hand and fix the row in exclusions.py")
        for anchor in by_hand:
            spans = anchor_spans(source, anchor)
            self.by_hand_spans += spans
            if not spans:
                self.stale.append(f"{self.path}:1  stale-exclusion  {anchor[:60]}  by-hand site not found; "
                                  "check the file by hand and fix the row in exclusions.py")

    def _lines(self, node: cst.CSTNode) -> tuple[int, int]:
        position = self.get_metadata(PositionProvider, node)
        return position.start.line, position.end.line

    def _in_range(self, node: cst.CSTNode) -> bool:
        start, end = self._lines(node)
        return any(first <= end and start <= last for first, last, _ in self.ranges)

    def _held(self, node: cst.CSTNode, name: str = "") -> bool:
        """True when ``node`` sits at an excluded site (not touched, not flagged) or a by-hand one (flagged)."""
        start, end = self._lines(node)
        for first, last, reason in self.ranges:
            if first <= end and start <= last:
                self.excluded.append(reason)
                return True
        if any(first <= end and start <= last for first, last in self.by_hand_spans):
            self.flags.append(f"{self.path}:{start}  {BY_HAND_KIND}  {name or _code(node)}  {IDENTITY}; edit by hand")
            return True
        return False

    def _new(self, dotted: str | None) -> str | None:
        return current_name(dotted, self.plan.renames) if dotted else None

    # Imports and the names they bind.

    def leave_Import(self, original: cst.Import, updated: cst.Import) -> cst.Import:
        """``import a`` becomes ``import new.a as a``: the bound name stays, so its uses keep their spelling.

        ``import a.b`` binds the new root, so its uses are respelled; one that would shadow another use of
        that root in the same function (UnboundLocalError) is flagged instead.
        """
        if original in self.blocked_imports:
            self._flag(original, OTHER, _code(original), self.blocked_imports[original])
            return updated
        names = []
        for alias in updated.names:
            old = get_full_name_for_node(alias.name)
            new = self._new(old)
            if new is not None and not self._held(original, old):
                keep = {} if alias.asname is not None or "." in old else {"asname": cst.AsName(name=cst.Name(old))}
                alias = alias.with_changes(name=cst.parse_expression(new), **keep)
                self._count("import")
            names.append(alias)
        return updated.with_changes(names=names)

    def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if updated.relative or updated.module is None:
            return updated
        module = get_full_name_for_node(updated.module)
        new_module = self._new(module)
        if self._held(original, module):
            return original
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
        if self._held(original, get_full_name_for_node(original)):
            return updated
        self._count("attribute")
        return cst.parse_expression(new)

    # Which string nodes are sinks: decided top-down, before their children are left.

    def visit_Subscript(self, node: cst.Subscript) -> None:
        if _is_sys_modules(node.value) and len(node.slice) == 1 \
                and isinstance(node.slice[0].slice, cst.Index):
            self._mark(node.slice[0].slice.value, "sys-modules")

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        if node.name is not None and isinstance(node.name.name, cst.Name):
            self.caught.add(node.name.name.value)

    def _hold_matcher(self, node: cst.BaseExpression, reason: str) -> None:
        self.matchers.update((id(string), reason) for string in _strings(node))

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

    def visit_Comparison(self, node: cst.Comparison) -> None:
        self._visit_matcher(node)
        target = node.comparisons[0]
        if len(node.comparisons) == 1 and isinstance(target.operator, (cst.In, cst.NotIn)) \
                and _is_sys_modules(target.comparator):
            self._mark(node.left, "sys-modules")

    def visit_Call(self, node: cst.Call) -> None:
        func = get_full_name_for_node(node.func) or ""
        if isinstance(node.func, cst.Attribute) and node.func.attr.value == "startswith" \
                and any(f"{name}.name" in _code(arg.value) for arg in node.args for name in self.caught):
            self._hold_matcher(node.func.value, "ImportError.name literal")
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
        self._mark_argv(node.args)
        for arg, after in itertools.zip_longest(node.args, node.args[1:]):
            if not arg.star and id(arg.value) not in self.sinks \
                    and isinstance(arg.value, (cst.SimpleString, cst.FormattedString)):
                pair = after.value if after is not None and after.keyword is None and not after.star else None
                self.sinks[id(arg.value)] = _Sink(CALL_ARG, pair=pair)

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
        if id(original) in self.matchers:
            return self._leave_matcher(original, updated)
        if sink is not None:
            return self._leave_sink(original, updated, sink)
        if isinstance(original, cst.SimpleString):
            return self._leave_string(original, updated)
        if isinstance(original, cst.FormattedString):
            return self._leave_fstring(original, updated)
        return updated

    def _leave_sink(self, original, updated, sink: _Sink):
        if id(original) in self.matchers:
            return self._leave_matcher(original, updated)
        if isinstance(original, cst.FormattedString) and (head := self._fstring_head(original, updated)) is not None:
            return head
        if sink.kind == CALL_ARG:
            return self._leave_call_arg(original, updated, sink)
        if isinstance(updated, cst.SimpleString) and isinstance(text := updated.evaluated_value, str):
            if self._new(text.partition(":")[0]) is None or self._held(original, text):
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

    def _leave_call_arg(self, original, updated, sink: _Sink):
        """A string passed to a call is rewritten when its whole value names a moved module, ``module.attr`` or
        ``module:attr`` of a name that module defines; a bare top-level name needs ``(module, attr)`` evidence.
        A dotted name that only starts with a moved module is flagged; config keys are left alone."""
        if isinstance(original, cst.FormattedString):
            return self._leave_fstring(original, updated)
        text = updated.evaluated_value if isinstance(updated, cst.SimpleString) else None
        if not isinstance(text, str) or not _DOTTED_NAME.fullmatch(text) or text in self.plan.config_keys \
                or self._new(text.partition(":")[0]) is None:
            return self._leave_string(original, updated) if isinstance(original, cst.SimpleString) else updated
        dotted = "." in text or ":" in text
        paired = isinstance(sink.pair, cst.SimpleString) and sink.pair.evaluated_value in self.plan.names.get(text, ())
        if not self.plan.names_module(text) or not (dotted or paired):
            if not dotted or self.plan.path_pattern is not None and self.plan.path_pattern.search(text):
                return self._leave_string(original, updated)
            self._flag(original, CALL_ARG, text, "call argument starts with a moved module but does not resolve "
                                                 "in the moved tree; fix by hand")
            return updated
        if self._held(original, text):
            return updated
        rewritten = _with_text(updated, self._whole)
        if rewritten is None:
            self._flag(original, OTHER, text, "call-arg literal spells the name with escapes; fix by hand")
            return updated
        self._count(CALL_ARG)
        return rewritten

    def _fstring_head(self, original: cst.FormattedString, node: cst.FormattedString) -> cst.FormattedString | None:
        """``f"pkg.mod.{name}"`` in a sink: its literal head names a moved module, so only the head changes."""
        head = node.parts[0] if node.parts else None
        if not isinstance(head, cst.FormattedStringText) or len(node.parts) < 2 or not head.value.endswith(".") \
                or "\\" in head.value or not self.plan.names_module(module := head.value[:-1]) \
                or self._held(original, module):
            return None
        self._count("f-string")
        return node.with_changes(parts=[head.with_changes(value=self._new(module) + "."), *node.parts[1:]])

    def _whole(self, text: str) -> str:
        module, colon, attr = text.partition(":")
        return (self._new(module) or module) + colon + attr

    def _leave_matcher(self, original, updated):
        text = original.evaluated_value if isinstance(original, cst.SimpleString) else ""
        if isinstance(text, str) and self._new(text.partition(":")[0]) is not None:
            self._flag(original, BY_HAND_KIND, text, f"{self.matchers[id(original)]}; edit by hand")
        return updated

    def _leave_string(self, original: cst.SimpleString, updated: cst.SimpleString):
        text = original.evaluated_value
        if not isinstance(text, str):
            return updated
        if id(original) in self.matchers:
            return self._leave_matcher(original, updated)
        rewritten = self._text(text, original)
        if self.plan.names_module(rewritten) and rewritten not in self.plan.config_keys:
            self._flag(original, OTHER, text, "string names a moved module; rewrite or confirm by hand")
        elif self.plan.path_pattern is not None and (match := self.plan.path_pattern.search(rewritten)):
            old = match.group(1)
            where = "docstring prose" if id(original) in self.prose else "string"
            self._flag(original, BY_PATH, old, f"{where} names a moved path (now {self.plan.new_path[old]})")
        if rewritten == text or self._held(original, text):
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
                if text != part.value and not self._held(original, text):
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
    """Uses of names bound by a plain dotted ``import a.b``, as (new dotted spelling, None) or (None, flag reason),
    and the function-local dotted imports whose new root binding would shadow another use of that root.

    ``import a`` keeps its bound name (``import new.a as a``), so its uses are not touched. A use is rewritten
    only when every binding it can see is an import of a moved module.
    """
    found, local, uses = {}, {}, []
    parents = wrapper.resolve(ParentNodeProvider)
    for scope in set(wrapper.resolve(ScopeProvider).values()):
        if scope is None:
            continue
        for assignment in () if isinstance(scope, GlobalScope) else scope.assignments:
            node = getattr(assignment, "node", None)
            if isinstance(node, cst.Import) and _plain_alias(node, assignment.name) \
                    and _imports_moved(node, assignment.name, renames) and not _keeps_name(node, assignment.name):
                local[node] = scope
        for access in scope.accesses:
            if not isinstance(access.node, (cst.Name, cst.Attribute)):
                continue
            referents = list(access.referents)
            uses.append((access.node, scope, {getattr(r, "node", None) for r in referents}))
            imports = [r for r in referents if isinstance(getattr(r, "node", None), cst.Import)
                       and _plain_alias(r.node, r.name)]
            moved = [r for r in imports if _imports_moved(r.node, r.name, renames)]
            if moved and all(_keeps_name(r.node, r.name) for r in moved):
                continue
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
    return found, _shadowing_imports(found, local, uses, {new.split(".")[0] for new in renames.values()})


def _shadowing_imports(found, local, uses, roots: set[str]) -> dict[cst.CSTNode, str]:
    """Function-local ``import a.b`` imports whose rewrite would bind a new root another use there reads globally.

    Their uses are dropped from ``found`` so the import and its uses stay as they were, flagged.
    """
    by_scope: dict = {}
    for node, scope in local.items():
        by_scope.setdefault(scope, set()).add(node)
    blocked = {}
    for scope, imports in by_scope.items():
        for node, use_scope, referents in uses:
            spelled = (found.get(node) or (None,))[0] or get_full_name_for_node(node) or ""
            root = spelled.split(".")[0]
            if root in roots and _within(use_scope, scope) and not referents <= imports:
                reason = (f"rewrite would bind {root} in this function, where {root} is also read from outside it "
                          "(UnboundLocalError); fix by hand")
                blocked.update((imp, reason) for imp in imports)
                break
    for node, _, referents in uses:
        if referents & blocked.keys():
            found.pop(node, None)
    return blocked


def _within(scope, outer) -> bool:
    while scope is not outer:
        if isinstance(scope, GlobalScope) or scope.parent is scope:
            return False
        scope = scope.parent
    return True


def _keeps_name(node: cst.Import, name: str) -> bool:
    """``import a`` (no alias, no dot): the rewrite binds ``a`` again, so uses of ``a`` stay as written."""
    return "." not in name and any(alias.asname is None and get_full_name_for_node(alias.name) == name
                                   for alias in node.names)


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
    rewriter = _Rewriter(plan, path, *_binding_accesses(wrapper, plan.renames))
    if path != "<string>":
        rewriter.anchor(source)
    new = wrapper.visit(rewriter).code
    return Result(path, new if new != source else None, rewriter.counts, rewriter.stale + rewriter.flags,
                  rewriter.excluded)


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


_BOM = "\ufeff"
_PLAN: Plan | None = None


def _init(plan: Plan) -> None:
    global _PLAN
    _PLAN = plan


def _work(item: tuple[Path, str]) -> Result:
    root, path = item
    try:
        source = (root / path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        return Result(path, flags=[f"{path}:1  undecodable  -  not UTF-8 ({error.reason} at byte {error.start}); "
                                   "skipped, rewrite its imports by hand"])
    bom = source.startswith(_BOM)
    result = rewrite_source(source.removeprefix(_BOM), path, _PLAN)
    if bom and result.source is not None:
        result.source = _BOM + result.source
    return result


def _candidates(root: Path, paths: list[str], plan: Plan) -> list[str]:
    listed = git(root, "ls-files", "--", *(paths or ["."])).splitlines()
    words = {old.split(".")[0] for old in plan.renames} | {old.rstrip("/") for old, _ in plan.old_paths()}
    needle = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)))
    out = []
    for path in listed:
        if PurePosixPath(path).parts[-2:] == ("compat", "moves.py") or path.startswith("scripts/move/"):
            continue  # old names as data: the MOVES table, and the exclusion anchors and move tools themselves
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


def _module_names(file: Path) -> set[str] | None:
    """Top-level names of a moved module in the working tree; None when the branch deleted it."""
    if not file.is_file():
        return None
    try:
        return top_level_names(file.read_bytes().decode("utf-8-sig", errors="replace"))
    except SyntaxError:
        return set()


def _moves_table(root: Path) -> dict[str, str]:
    tracked = git(root, "ls-files", "*hermes_agent/compat/moves.py").split()
    if not tracked or not (root / tracked[0]).is_file():
        return {}
    node = next((n for n in ast.parse((root / tracked[0]).read_text(encoding="utf-8-sig")).body
                 if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "MOVES"), None)
    return ast.literal_eval(node.value) if node is not None else {}


def _config_keys(root: Path) -> frozenset[str]:
    """Every dotted key path of ``DEFAULT_CONFIG`` (``agent``, ``agent.reasoning_effort``, ...)."""
    keys: set[str] = set()
    for path in git(root, "ls-files", "*hermes_cli/config_defaults.py").split():
        body = ast.parse((root / path).read_text(encoding="utf-8-sig")).body
        pending = [("", node.value) for node in body if isinstance(node, ast.Assign)
                   and any(getattr(target, "id", None) == "DEFAULT_CONFIG" for target in node.targets)]
        while pending:
            prefix, value = pending.pop()
            for key, item in zip(getattr(value, "keys", ()), getattr(value, "values", ())):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(prefix + key.value)
                    pending.append((f"{prefix}{key.value}.", item))
    return frozenset(keys)


def build_plan(root: Path, commit: str) -> Plan:
    names = {old: found for old, (_, path) in moved_modules(root, commit).items()
             if (found := _module_names(root / path)) is not None}
    return Plan(module_renames(root, commit), renamed_paths(root, commit), names, _moves_table(root),
                config_keys=_config_keys(root))


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
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(plan,)) as pool:
        results = list(pool.map(_work, [(root, f) for f in files], chunksize=4))
    counts: dict[str, int] = {}
    flags, failed, changed, excluded = [], [], [], []
    for result in results:
        if result.error:
            failed.append(result.error)
            continue
        flags += result.flags
        excluded += result.excluded
        for kind, n in result.counts.items():
            counts[kind] = counts.get(kind, 0) + n
        if result.source is not None:
            changed.append(result.path)
            if not args.check:
                (root / result.path).write_bytes(result.source.encode("utf-8"))
    moves = update_moves(root, plan.renames, write=not args.check)
    if not args.paths or "pyproject.toml" in args.paths:
        toml_count, toml_flags = rewrite_pyproject(root, plan, write=not args.check)
        flags += toml_flags
        if toml_count:
            counts["toml"] = toml_count
            changed.append("pyproject.toml")
    for line in failed + flags:
        print(line)
    for reason in sorted(set(excluded)):
        print(f"excluded {reason}  ({excluded.count(reason)} sites)")
    kinds = " ".join(f"{kind}={n}" for kind, n in sorted(counts.items()))
    print(f"files={len(changed)} moves={moves} {kinds}".rstrip())
    print(f"rewritten={sum(counts.values())} flagged={len(flags)} failed={len(failed)}")
    if changed and not args.check:
        print(GENERATE_HINT)
    return 1 if failed or flags or (args.check and (changed or moves)) else 0


if __name__ == "__main__":
    sys.exit(main())
