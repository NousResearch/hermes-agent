"""Rewrite Python sources from old module names to the names a move commit gave them.

Usage: python scripts/move/rewrite_imports.py --from-commit <A> [--paths P ...] [--check]

Imports: ``import a`` becomes ``from hermes_agent import a`` (same bound name; the lint rejects
``import hermes_agent.a as a``), except in a ``try`` whose handler catches ModuleNotFoundError or reads
``exc.name``, and in a function that removes modules from ``sys.modules``: there the from-form changes
the exception or skips the re-import, so ``import hermes_agent.a as a  # noqa: PLR0402`` is written.
``import a.b`` binds the new root and its uses are respelled. ``from a import x`` changes its module.

Strings change only where the code takes a module name (decision D47, see ``sinks.py``): import
functions and the helpers that wrap them, validated patch targets, ``sys.modules`` keys, logger names,
``-m`` argv pairs, commands given to a process, Python code given to an interpreter, and
``pyproject.toml`` entry points. A string anywhere else that names a moved module or path is printed
as a flag, ``path:line  <class>  <name>  <reason>``, and left as it is.

Sites listed in ``exclusions.py`` are left alone and printed once per reason as ``excluded <reason>``;
by-hand sites are flagged. A row whose anchor is gone is flagged. A UTF-8 BOM is kept; a file that is not
UTF-8 is skipped with a flag. The new old-name keys go into the compat MOVES table. Exit status 1 when a
file fails to parse, a flag remains, or (with --check) a file would change.
"""
from __future__ import annotations

import argparse
import ast
import functools
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
from exclusions import BY_HAND, EXCLUDED, IDENTITY, KEPT_NAMES, anchor_spans
from renames import current_name, git, module_renames, moved_modules, renamed_paths
from pieces import respell, text_of
from rewrite_toml import rewrite_pyproject
from sinks import ARGV, CODE, COMMAND, LOGGER, MODULE, STRINGS, SYS_MODULES, TARGET, Findings, Sink, \
    SinkFinder, helper_parameters, string_parts

DYNAMIC, STRING_REF, BY_PATH, FILE_MATH, BY_HAND_KIND, MODULE_STRING, MODULE_TABLE, CODE_STRING, KEPT_NAME, \
    IMPORT, OTHER = ("dynamic-import", "string-module-ref", "by-path", "file-math", "by-hand", "module-string",
                     "module-table", "code-string", "kept-name", "import", "other")
GENERATE_HINT = ("next: after `just move-docs`, run `just move-generate`: generated files embed code and docstrings "
                 "this move rewrote (apps/shared/src/gateway-contract.*, tests/hermes_cli/serve_handler_ledger.json)")
NOQA = "PLR0402"
_DASH_M = re.compile(r"(?<![\w-])(-m(?:\s+|[\"'],\s*[\"']))([A-Za-z_][\w.-]*)")
_DASH_C = re.compile(r"(?<![\w-])-c\s+([\"'])(.+?)(?<!\\)\1", re.DOTALL)
_CODE_IMPORT = re.compile(r"(?<![\w.])((?:from|import)\s+)([A-Za-z_][\w.]*)(?=\s+import\b|\s*(?:$|[\n,;)#]|\\n| as\b))",
                          re.MULTILINE)
_PYTHON_SHEBANG = re.compile(r"#!.*\bpython")
_WORD = re.compile(r"[a-z]+")
_MODULE_ATTRIBUTES = {"__file__", "__name__", "__path__", "__spec__", "__loader__", "__dict__", "__all__", "__doc__"}
_HELPER_SOURCES = re.compile(r"import_module|__import__|find_spec|importorskip")


@dataclass
class Plan:
    renames: dict[str, str]
    paths: list[tuple[str, str]]
    names: dict[str, set[str]]
    """Every moved module's old name -> the names its top level binds (imports included)."""
    moves: dict[str, str] = field(default_factory=dict)
    """The compat MOVES table of the tree: earlier chunks' old names, so exclusion paths match after any move."""
    excluded: tuple[tuple[str, str, str], ...] = EXCLUDED
    by_hand: tuple[tuple[str, str], ...] = BY_HAND
    config_keys: frozenset[str] = frozenset()
    """Dotted ``DEFAULT_CONFIG`` keys: ``agent.reasoning_effort`` is a config key even where it is also a module."""
    defined: dict[str, set[str]] = field(default_factory=dict)
    """Every moved module's old name -> the names its top level defines (``def``, ``class``, assignment)."""
    helpers: dict[str, dict[str, dict[int, str]]] = field(default_factory=dict)
    """Module -> function -> parameters it passes straight to an import function."""
    kept_names: frozenset[str] = frozenset()
    """Moved module names that excluded sites keep under their old name (persisted command lines)."""

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
        """Path-shaped text that starts with a top-level name this move took away; ``find_path`` checks it."""
        tops = sorted({PurePosixPath(old).parts[0] for old, _ in self.paths}, key=len, reverse=True)
        return re.compile(r"(?<![\w/.-])((?:" + "|".join(map(re.escape, tops)) + r")(?:/[\w.-]+)*/?)") if tops else None

    @functools.cached_property
    def old_tree(self) -> frozenset[str]:
        """Every old file path this move renamed and every folder above one."""
        out = set()
        for old, _ in self.paths:
            parts = PurePosixPath(old).parts
            out |= {"/".join(parts[:size]) for size in range(1, len(parts) + 1)}
        return frozenset(out)

    def find_path(self, text: str) -> tuple[str, str] | None:
        """(old, new) for the first moved file or folder ``text`` names: a top-level file, a folder followed by a file
        or folder that moved with it (``tools/registry.py``, ``tools/environments/``; never ``tools/list``), or the
        whole string being a moved folder (``"tools/"``)."""
        if self.path_pattern is None:
            return None
        if text.endswith("/") and text[:-1] in self.old_tree and "/" not in text[:-1]:
            return text, self.new_path.get(text, "")
        for match in self.path_pattern.finditer(text):
            parts = match.group(1).rstrip("/").split("/")
            for size in range(len(parts), 0, -1):
                old = "/".join(parts[:size])
                if old in self.old_tree and (size > 1 or old.endswith(".py")):
                    folder = not old.endswith(".py")
                    return old + "/" * folder, self.new_path.get(old + "/" * folder, "")
        return None

    @functools.cached_property
    def new_path(self) -> dict[str, str]:
        return dict(self.old_paths())

    @functools.cached_property
    def folders(self) -> frozenset[str]:
        """Top-level folder and file names this move took away: ``tools``, ``model_tools.py``."""
        return frozenset(PurePosixPath(old).parts[0] for old, _ in self.paths)

    def old_paths(self) -> list[tuple[str, str]]:
        """Old -> new path text a by-path literal can carry: each moved file and each moved directory."""
        pairs = {old: new for old, new in self.paths if old.endswith(".py")}
        for old, new in self.paths:
            old_dir, new_dir = str(PurePosixPath(old).parent), str(PurePosixPath(new).parent)
            if old_dir not in (".", new_dir):
                pairs[old_dir + "/"] = new_dir + "/"
        return sorted(pairs.items(), key=lambda item: -len(item[0]))

    def new(self, name: str) -> str | None:
        return current_name(name, self.renames)

    def is_module(self, text: str) -> bool:
        """``text`` is exactly a moved module, package or namespace folder."""
        return text in self.names

    def is_target(self, text: str) -> bool:
        """``module``, ``module.attr`` or ``module:attr``, where ``attr`` is a name the module binds."""
        module, colon, attr = text.partition(":")
        if colon:
            return module in self.names and attr.split(".")[0] in self.names[module]
        parts = text.split(".")
        for size in range(len(parts), 0, -1):
            prefix = ".".join(parts[:size])
            if prefix in self.names:
                return size == len(parts) or parts[size] in _MODULE_ATTRIBUTES \
                    or bool({parts[size], "__getattr__"} & self.names[prefix])
        return False

    def has_module_prefix(self, text: str) -> bool:
        parts = text.partition(":")[0].split(".")
        return any(".".join(parts[:size]) in self.names for size in range(len(parts), 0, -1))

    def module_text(self, text: str) -> bool:
        """A string that reads as a moved module: the module, or ``module.attr``/``module:attr`` of a name it defines."""
        module, colon, attr = text.partition(":")
        if colon:
            return module in self.defined and attr in self.defined[module]
        if text in self.names:
            return True
        prefix, _, attr = text.rpartition(".")
        return prefix in self.defined and attr in self.defined[prefix]

    def is_word(self, text: str) -> bool:
        """An ordinary lower-case word (``tools``, ``agent``): read as English unless the context says module."""
        return _WORD.fullmatch(text) is not None

    def helper(self, module: str, function: str) -> dict[int, str]:
        for name in (module, self.new(module)):
            if name is not None and function in self.helpers.get(name, {}):
                return self.helpers[name][function]
        return {}


@dataclass
class Result:
    path: str
    source: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    error: str | None = None


class _Rewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plan: Plan, path: str, found: Findings, accesses: dict[cst.CSTNode, tuple],
                 blocked_imports: dict[cst.CSTNode, str]) -> None:
        super().__init__()
        self.plan, self.path, self.found, self.accesses = plan, path, found, accesses
        self.blocked_imports = blocked_imports
        self.counts: dict[str, int] = {}
        self.flags: list[str] = []
        self.excluded: list[str] = []
        self.ranges: list[tuple[int, int, str]] = []
        self.by_hand_spans: list[tuple[int, int]] = []
        self.stale: list[str] = []
        self.blocked: dict[int, str] = {}
        self.noqa: set[int] = set()
        self.split: set[int] = set()
        self.chains_done: set[int] = set()
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

    # Imports and the names they bind.

    def leave_Import(self, original: cst.Import, updated: cst.Import):
        """``import a`` -> ``from hermes_agent import a``; ``import a.b`` -> ``import hermes_agent.a.b``."""
        if original in self.blocked_imports:
            self._flag(original, IMPORT, _code(original), self.blocked_imports[original])
            return updated
        keep_form = id(original) in self.found.keep_module_form
        kept, froms = [], []
        for alias in updated.names:
            old = get_full_name_for_node(alias.name)
            new = self.plan.new(old)
            if new is None or self._held(original, old):
                kept.append(alias)
                continue
            self._count("import")
            bound = alias.asname.name.value if alias.asname else None if "." in old else old
            parent, _, leaf = new.rpartition(".")
            if bound is None:
                kept.append(alias.with_changes(name=cst.parse_expression(new)))
            elif leaf == bound and parent and not keep_form:
                froms.append(cst.ImportFrom(module=cst.parse_expression(parent), names=[cst.ImportAlias(cst.Name(leaf))]))
            else:
                kept.append(alias.with_changes(name=cst.parse_expression(new), asname=cst.AsName(name=cst.Name(bound))))
                if leaf == bound and parent:
                    self.noqa.add(id(original))
        if not froms:
            return updated.with_changes(names=_commas(kept))
        statements = ([updated.with_changes(names=_commas(kept))] if kept else []) + froms
        if len(statements) == 1:
            return statements[0]
        self.split.add(id(original))
        return cst.FlattenSentinel(statements)

    def leave_SimpleStatementLine(self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine):
        ids = {id(statement) for statement in original.body}
        if ids & self.noqa:
            updated = updated.with_changes(trailing_whitespace=_with_noqa(updated.trailing_whitespace))
        if len(original.body) == 1 and ids & self.split:
            lines = []
            for index, statement in enumerate(updated.body):
                lines.append(updated.with_changes(
                    body=[statement.with_changes(semicolon=cst.MaybeSentinel.DEFAULT)],
                    leading_lines=updated.leading_lines if index == 0 else (),
                    trailing_whitespace=updated.trailing_whitespace if index == 0 else cst.TrailingWhitespace()))
            return cst.FlattenSentinel(lines)
        return updated

    def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if updated.relative or updated.module is None:
            return updated
        module = get_full_name_for_node(updated.module)
        new_module = self.plan.new(module)
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
        targets = [self.plan.new(f"{module}.{alias.name.value}") for alias in node.names]
        apart = [t for t, alias in zip(targets, node.names)
                 if t is not None and t != f"{new_module}.{alias.name.value}"]
        if not apart:
            return None
        parents = {t.rpartition(".")[0] for t in apart}
        if len(apart) != len(node.names) or len(parents) != 1:
            self._flag(original, IMPORT, module, "from-import mixes children that moved apart; split it by hand")
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
        if self.found.collapse.get(id(original)) in self.chains_done:
            return updated.value
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
            self._flag(original, IMPORT, get_full_name_for_node(original), reason)
            return updated
        if self._held(original, get_full_name_for_node(original)):
            return updated
        self._count("attribute")
        return cst.parse_expression(new)

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        """``__import__("a.b").b.X`` -> ``__import__("hermes_agent.a.b", fromlist=["X"]).X``."""
        chain = self.found.chains.get(id(original))
        if chain is None or updated.args[0].value.deep_equals(original.args[0].value):
            return updated
        self.chains_done.add(id(original))
        fromlist = cst.Arg(keyword=cst.Name("fromlist"), value=cst.List([cst.Element(cst.SimpleString(f'"{chain[1]}"'))]),
                           equal=cst.AssignEqual(cst.SimpleWhitespace(""), cst.SimpleWhitespace("")))
        first = updated.args[0].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
        return updated.with_changes(args=[first, fromlist])

    # Strings.

    def on_leave(self, original, updated):
        updated = super().on_leave(original, updated)
        if not isinstance(original, STRINGS):
            return updated
        key = id(original)
        if key in self.found.matchers:
            return self._leave_matcher(original, updated)
        sink = self.found.sinks.get(key)
        if sink is not None:
            return self._leave_sink(original, updated, sink)
        if key in self.found.joined:
            return updated
        if isinstance(original, cst.ConcatenatedString):
            found = text_of(original)
            self._judge(original, found[0] if found else "".join(_literal_text(p) for p in string_parts(original)))
        elif isinstance(original, cst.SimpleString) and isinstance(original.evaluated_value, str):
            self._judge(original, original.evaluated_value)
        elif isinstance(original, cst.FormattedString):
            self._judge_fstring(original)
        return updated

    def _leave_matcher(self, original, updated):
        text = original.evaluated_value if isinstance(original, cst.SimpleString) else ""
        if isinstance(text, str) and self.plan.new(text.partition(":")[0]) is not None:
            self._flag(original, BY_HAND_KIND, text, f"{self.found.matchers[id(original)]}; edit by hand")
        return updated

    def _leave_sink(self, original, updated, sink: Sink):
        if sink.kind in (CODE, COMMAND):
            return self._leave_program(original, updated, sink)
        if isinstance(original, cst.FormattedString):
            return self._leave_fstring_sink(original, updated, sink)
        if not isinstance(original, cst.SimpleString) or not isinstance(text := original.evaluated_value, str):
            if _mentions_old(original, self.plan.renames):
                self._flag(original, DYNAMIC, _code(original), f"computed {sink.kind} name; check by hand")
            return updated
        return self._leave_name(original, updated, sink, text)

    def _leave_name(self, original: cst.SimpleString, updated: cst.SimpleString, sink: Sink, text: str):
        new = self.plan.new(text.partition(":")[0])
        if new is None or self._held(original, text):
            return updated
        if (reason := self.found.import_flags.get(id(original))) is not None:
            self._flag(original, DYNAMIC, text, f"{reason}; fix by hand")
            return updated
        if sink.kind == ARGV and text in self.plan.kept_names and self.path.startswith("tests/"):
            self._flag(original, KEPT_NAME, text, "excluded production code keeps this module name in persisted "
                                                  "command lines; check whether this test mirrors it")
            return updated
        valid = {MODULE: self.plan.is_module, ARGV: self.plan.is_module,
                 TARGET: self.plan.has_module_prefix if sink.loose else self.plan.is_target}.get(sink.kind)
        if valid is not None and not valid(text):
            self._flag(original, STRING_REF, text, f"{sink.kind} string does not resolve in the moved tree; fix by hand")
            return updated
        rewritten = _with_text(updated, self._whole)
        if rewritten is None:
            self._flag(original, OTHER, text, f"{sink.kind} literal spells the name with escapes; fix by hand")
            return updated
        self._count(f"{sink.kind}-table" if sink.table else sink.kind)
        return rewritten

    def _whole(self, text: str) -> str:
        module, colon, attr = text.partition(":")
        return (self.plan.new(module) or module) + colon + attr

    def _leave_fstring_sink(self, original: cst.FormattedString, updated: cst.FormattedString, sink: Sink):
        """``f"pkg.mod.{name}"`` in a sink: its literal head names a moved module, so only the head changes."""
        head = updated.parts[0] if updated.parts else None
        valid = self.plan.is_target if sink.kind == TARGET else self.plan.is_module
        if isinstance(head, cst.FormattedStringText) and len(updated.parts) > 1 and head.value.endswith(".") \
                and "\\" not in head.value and valid(module := head.value[:-1]):
            if self._held(original, module):
                return updated
            self._count(f"{sink.kind}-f-string")
            return updated.with_changes(parts=[head.with_changes(value=self.plan.new(module) + "."), *updated.parts[1:]])
        if _imports_old(original, self.plan.renames) or _mentions_old(original, self.plan.renames):
            self._flag(original, DYNAMIC, _code(original), f"computed {sink.kind} name; check by hand")
        return updated

    def _leave_program(self, original, updated, sink: Sink):
        """Code an interpreter runs, or a command a process runs: rewrite what it imports and runs with -m."""
        found = text_of(updated)
        if found is None:
            return updated
        text, pieces = found
        new = self._program(original, text, sink.kind)
        if new == text or self._held(original, text[:60]):
            return updated
        rewritten = respell(updated, pieces, new)
        if rewritten is None:
            self._flag(original, CODE_STRING, text[:80].replace("\n", " "), "the rewrite crosses string pieces or "
                                                                             "needs an escape; fix by hand")
            return updated
        self._count(f"{sink.kind}-table" if sink.table else sink.kind)
        return rewritten

    def _program(self, node: cst.CSTNode, text: str, kind: str) -> str:
        if kind == CODE and (program := self._python(node, text)) is not None:
            return program
        text = _DASH_M.sub(self._dash_m, text)
        return _DASH_C.sub(lambda match: self._dash_c(node, match), text)

    def _python(self, node: cst.CSTNode, text: str) -> str | None:
        """``text`` rewritten as a Python program; None when it does not parse as one."""
        if not any(top in text for top in self.tops):
            return text
        try:
            program = rewrite_source(text, "<string>", self.plan)
        except RecursionError:
            return None
        if program.error:
            return None
        for flag in program.flags:
            _, kind, name, reason = flag.split("  ", 3)
            self._flag(node, kind, name, reason)
        return program.source if program.source is not None else text

    def _dash_m(self, match: re.Match) -> str:
        new = self.plan.new(match.group(2))
        return match.group(0) if new is None or not self.plan.is_module(match.group(2)) else match.group(1) + new

    def _dash_c(self, node: cst.CSTNode, match: re.Match) -> str:
        quote, code = match.group(1), match.group(2)
        program = self._python(node, code.replace("\\" + quote, quote))
        if program is None:
            return match.group(0)
        return match.group(0).replace(code, program.replace(quote, "\\" + quote), 1)

    def _judge(self, node: cst.CSTNode, text: str) -> None:
        """Flags for a string no sink receives: it is left as it is."""
        key = id(node)
        if key in self.found.i18n or key in self.found.prose:
            return
        if self._code_flags(node, text):
            return
        if (reason := self.found.tables.get(key)) is not None:
            if self.plan.module_text(text):
                self._flag(node, MODULE_TABLE, text, f"module name in a table: {reason}; rewrite or confirm by hand")
            return
        if self.plan.module_text(text) and text not in self.plan.config_keys:
            self._judge_module(node, text, key)
        elif text.endswith(".") and self.plan.is_module(text[:-1]):
            self._flag(node, MODULE_STRING, text, "module-name prefix outside any import; rewrite or confirm by hand")
        elif key in self.found.path_pieces and text in self.plan.folders:
            self._flag(node, BY_PATH, text, "path built from a moved folder name; check the path arithmetic")
        elif (found := self.plan.find_path(text)) is not None:
            self._flag(node, BY_PATH, found[0], f"string names a moved path (now {found[1]})")

    def _judge_module(self, node: cst.CSTNode, text: str, key: int) -> None:
        if key in self.found.path_pieces:
            self._flag(node, BY_PATH, text, "path built from a moved folder name; check the path arithmetic")
        elif key in self.found.module_elements:
            self._flag(node, MODULE_TABLE, text, "module name in a table no import reads here; rewrite or confirm "
                                                 "by hand")
        elif not self.plan.is_word(text):
            self._flag(node, MODULE_STRING, text, "string names a moved module outside any import; rewrite or "
                                                  "confirm by hand")

    def _code_flags(self, node: cst.CSTNode, text: str) -> bool:
        found = False
        for match in _DASH_M.finditer(text):
            if self.plan.new(match.group(2)) is not None:
                self._flag(node, CODE_STRING, match.group(0), "`-m` names a moved module in a string no process "
                                                              "receives here; rewrite or confirm by hand")
                found = True
        if "import" in text:
            for match in _CODE_IMPORT.finditer(text):
                if self.plan.new(match.group(2)) is not None:
                    self._flag(node, CODE_STRING, match.group(0).strip(), "code in a string no interpreter receives "
                                                                          "here imports a moved module")
                    found = True
        return found

    def _judge_fstring(self, original: cst.FormattedString) -> None:
        if id(original) in self.found.i18n or id(original) in self.found.prose:
            return
        found = text_of(original)
        text = found[0] if found else "".join(_literal_text(part) for part in original.parts)
        if self._code_flags(original, text):
            return
        if _imports_old(original, self.plan.renames):
            self._flag(original, MODULE_STRING, _code(original), "f-string builds a moved module name outside any "
                                                                 "import; check by hand")


def _commas(aliases: list[cst.ImportAlias]) -> list[cst.ImportAlias]:
    if aliases:
        aliases[-1] = aliases[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return aliases


def _with_noqa(trailing: cst.TrailingWhitespace) -> cst.TrailingWhitespace:
    comment = trailing.comment.value if trailing.comment is not None else ""
    if NOQA in comment:
        return trailing
    if "noqa:" in comment:
        value = re.sub(r"(noqa:\s*[\w, ]*\w)", rf"\1, {NOQA}", comment, count=1)
    else:
        value = f"{comment}  # noqa: {NOQA}" if comment else f"# noqa: {NOQA}"
    whitespace = trailing.whitespace if comment else cst.SimpleWhitespace("  ")
    return trailing.with_changes(whitespace=whitespace, comment=cst.Comment(value))


def _with_text(node: cst.SimpleString, transform) -> cst.SimpleString | None:
    """``node`` with ``transform`` applied to its source text; None when that does not give the transformed value.

    The check catches literals that spell a module name with escapes, which a source-text edit cannot see.
    """
    head = len(node.prefix) + len(node.quote)
    raw = node.value[head:-len(node.quote)]
    candidate = node.with_changes(value=node.value[:head] + transform(raw) + node.quote)
    return candidate if candidate.evaluated_value == transform(node.evaluated_value) else None


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

    ``import a`` keeps its bound name, so its uses are not touched. A use is rewritten only when every binding it
    can see is an import of a moved module.
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
    finder = SinkFinder(plan, path)
    wrapper.visit(finder)
    rewriter = _Rewriter(plan, path, finder.found, *_binding_accesses(wrapper, plan.renames))
    if path != "<string>":
        rewriter.anchor(source)
    new = wrapper.visit(rewriter).code
    return Result(path, new if new != source else None, rewriter.counts, rewriter.stale + rewriter.flags,
                  rewriter.excluded)


def top_level_names(source: str) -> tuple[set[str], set[str]]:
    """Names a module's top level defines and the names it imports, including inside top-level
    ``if``/``try``/``with``/``for``."""
    defined, imported = set(), set()
    pending = list(ast.parse(source).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            pending += node.body + getattr(node, "orelse", []) + getattr(node, "finalbody", []) + \
                [s for h in getattr(node, "handlers", []) for s in h.body]
            if isinstance(node, ast.For):
                defined |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defined |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return defined, imported


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


def _module_names(file: Path) -> tuple[set[str], set[str]] | None:
    """(defined, imported) top-level names of a moved module in the working tree; None when the branch deleted it."""
    if not file.is_file():
        return None
    try:
        return top_level_names(file.read_bytes().decode("utf-8-sig", errors="replace"))
    except SyntaxError:
        return set(), set()


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


def _moved_names(root: Path, commit: str, renames: dict[str, str]) -> dict[str, tuple[set[str], set[str]]]:
    """Old dotted name -> (defined, imported) names of every moved module, by its import name and by its path from
    the repository root, so a module under a folder without ``__init__.py`` (``plugins.platforms.x``) is known, and
    each folder on the way is known as a namespace package."""
    out: dict[str, tuple[set[str], set[str]]] = {}
    for old, (_, path) in moved_modules(root, commit).items():
        if (found := _module_names(root / path)) is not None:
            out[old] = found
    for old_path, new_path in renamed_paths(root, commit):
        if not old_path.endswith(".py") or (found := _module_names(root / new_path)) is None:
            continue
        parts = old_path[:-3].removesuffix("/__init__").split("/")
        dotted = ".".join(parts)
        if current_name(dotted, renames) is not None:
            out.setdefault(dotted, found)
        for size in range(1, len(parts)):
            prefix = ".".join(parts[:size])
            if current_name(prefix, renames) is not None:
                out.setdefault(prefix, (set(), set()))
    return {name: found for name, found in out.items() if current_name(name, renames) is not None}


def _helpers(root: Path, files: list[str]) -> dict[str, dict[str, dict[int, str]]]:
    """Module -> function -> parameter: functions that pass a parameter straight to an import function."""
    out = {}
    for path in files:
        if not path.endswith(".py"):
            continue
        text = (root / path).read_text(encoding="utf-8-sig", errors="replace")
        if _HELPER_SOURCES.search(text) and (found := helper_parameters(text)):
            out[path[:-3].removesuffix("/__init__").replace("/", ".")] = found
    return out


def _kept_names(root: Path, plan: Plan) -> frozenset[str]:
    """Moved module names written inside excluded sites kept for persisted or shipped text."""
    out = set()
    dotted = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+")
    for path, anchor, reason in plan.excluded:
        if reason not in KEPT_NAMES:
            continue
        for alias in plan.aliases(path):
            file = root / alias
            if not file.is_file():
                continue
            source = file.read_text(encoding="utf-8-sig", errors="replace")
            lines = source.splitlines()
            for first, last in anchor_spans(source, anchor):
                out |= {m for m in dotted.findall("\n".join(lines[first - 1:last])) if plan.new(m) is not None}
    return frozenset(out)


def build_plan(root: Path, commit: str, files: list[str] | None = None) -> Plan:
    renames = module_renames(root, commit)
    found = _moved_names(root, commit, renames)
    names = {old: defined | imported for old, (defined, imported) in found.items()}
    defined = {old: names_defined for old, (names_defined, _) in found.items()}
    listed = files if files is not None else git(root, "ls-files", "*.py").split()
    plan = Plan(renames, renamed_paths(root, commit), names, _moves_table(root), config_keys=_config_keys(root),
                defined=defined, helpers=_helpers(root, listed))
    plan.kept_names = _kept_names(root, plan)
    return plan


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
