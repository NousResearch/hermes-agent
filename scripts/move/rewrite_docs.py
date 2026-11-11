"""Rewrite moved paths and module names in docs and in Python comments and docstrings.

Usage: python scripts/move/rewrite_docs.py (--from-commit <A> | --synthetic-move <path> ...)
           [--paths P ...] [--check] [--samples N] [--stats-json FILE]

Runs per move chunk, for that chunk's names only (the rename table of ``--from-commit``, the same one
``rewrite_imports.py`` uses; ``--synthetic-move`` builds one for a rehearsal before the move exists).

Text it reads: ``*.md``, ``*.mdx``, ``*.txt`` whole; ``*.yaml``/``*.yml``/``*.yaml.example`` comments only;
``*.py`` comments and docstrings only (never code). Rewrites:

1. Path tokens: a moved file (``tui_gateway/entry.py``), a moved directory with a trailing slash
   (``tui_gateway/``), a bare stem (``tools/cronjob_tools``), a ``.symbol`` or sentence-final ``.`` after
   either, a glob under a moved directory (``agent/turn_*.py``), a unique root file (``run_agent.py``),
   and the same path behind ``github.com/<org>/hermes-agent/blob/<ref>/``. ``::symbol`` and ``:line`` stay.
2. Dotted module names (``tui_gateway.entry``). A dotted name is a module reference only when it resolves: a
   moved module, or a moved module plus an attribute that module defines. A config key (``DEFAULT_CONFIG`` path,
   ``plugins.enabled``, ``providers.<id>.api_key``), an i18n key and a name whose root is ambiguous stay.
3. ``from M import ...``, ``import M`` and ``-m M``: the only places a bare top-level word (``providers``,
   ``tools``, ``agent``) is rewritten, and only as code: inside a code span, a code fence or a ``>>>`` line, or as
   an import statement that parses as Python and starts its line. ``from providers like Daloopa`` is English:
   it does not parse, so it stays. A bare word alone (backticked or not) is never rewritten.

Left alone, and counted as ``skip-*``: a path behind any ``/`` (``~/.hermes/cron/``, ``$HERMES_HOME/...``,
``./x``); bare directories that also name HERMES_HOME dirs or other projects' dirs (``tools/``, ``cron/``), and a
glob that takes every file of one (``cron/*``, ``plugins/**``);
a name whose root is a moved *file* that other projects share (``utils.x``, ``cli.x``); config keys
(``agent.reasoning_effort``, ``plugins.enabled``, ``gateway.platforms.*`` outside an import); i18n keys
(``t("gateway.status.*")``); tree-diagram lines; and the files in
``EXCLUDED`` (frozen surfaces, generated pages, third-party skill docs, locale values) and the shims
and byte copies ``generate_shims.py`` writes at old module paths.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import fnmatch
import functools
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renames import config_keys, current_name, git, module_renames, renamed_paths

#: Files and trees the rewrite never touches. Frozen surfaces keep the old names on purpose, generated pages
#: are regenerated, third-party docs name other projects' modules, locale values are runtime strings
#: (by hand), and ``hermes_agent/compat`` and the move scripts hold the old names as data.
EXCLUDED = (
    "tests/compat/**", "hermes_agent/compat/**", "plugin-catalog/**", "evals/**/results/**", "**/SCORECARD*",
    "tests/install/KNOWN_FAILURES.md", "scripts/update-test/HANDOFF.md", "hermes_cli/observability/schemas/**",
    "**/vendor/conformance/**", "website/docs/user-guide/skills/bundled/**",
    "website/docs/user-guide/skills/optional/**", "website/docs/reference/*skills-catalog.md",
    "website/docs/developer-guide/plugins/public-api.md", "website/docs/developer-guide/gateway-command-parity.md",
    "optional-skills/mlops/**", "skills/mlops/**", "locales/**", "contributors/**",
    "scripts/move/**", "tests/scripts/test_move_*", "node_modules/**", "**/node_modules/**",
)
#: Root names that also mean something else (HERMES_HOME dirs, MCP methods, other projects' packages).
#: A bare ``name/`` or ``name.attr`` on one of them is by hand; it needs a second segment that is a module.
AMBIGUOUS_ROOTS = frozenset({
    "agent", "tools", "plugins", "cron", "gateway", "providers", "skills", "cli", "utils", "config",
    "state", "memory", "hooks", "logs", "sessions", "profiles", "cache", "web", "ui", "api",
})
#: Config paths that are also module paths. Matched by name, never by the words around them.
CONFIG_DENY = re.compile(
    r"^(?:agent\.(?:reasoning_effort|reasoning_overrides|system_prompt)|gateway\.platforms\.)")
FILE_EXT = {"py", "md", "mdx", "txt", "yaml", "yml", "json", "toml", "ts", "tsx", "js", "jsx", "mjs", "cjs",
            "sh", "ps1", "cfg", "ini", "lock", "html", "css", "png", "svg", "example", "db", "log", "jsonl"}
TEXT_SUFFIXES = (".md", ".mdx", ".txt")
YAML_SUFFIXES = (".yaml", ".yml", ".yaml.example", ".yml.example")
_TOKEN = re.compile(r"(?<![\w.\-~$@%#])[A-Za-z_][\w.\-]*(?:/[\w.\-*<>{}\[\]]*)*")
_BLOB = re.compile(r"github\.com/[\w.\-]+/hermes-agent/blob/main/")  # a commit or tag permalink keeps its old path
_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_IMPORT = re.compile(r"(?:^|(?<=[\s`'\"(]))(from\s+|import\s+|-m\s+|-m[\"'],\s*[\"'])([A-Za-z_][\w.]*)(?![\w/\-])")
_CODE_SPAN = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)")
_FENCE = re.compile(r"\s*(?:```|~~~)")
_NAME = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\s+as\s+[A-Za-z_]\w*)?"
_STATEMENT = re.compile(
    rf"(?:from\s+[A-Za-z_][\w.]*\s+import\s+(?:\*|\(|\.\.\.|{_NAME}(?:\s*,\s*{_NAME})*)"
    rf"|import\s+{_NAME}(?:\s*,\s*{_NAME})*)")
_STATEMENT_START = re.compile(r"\s*(?:(?:[-*+]|\d+[.)])\s+)?(?:>>>\s*|\.\.\.\s*|#\s*|\$\s*)?")
_I18N_CALL = re.compile(r"(?<![\w.])_?(?:g?t|tn)\(\s*f?[\"']?$")
_TREE_LINE = re.compile(r"[├└│─]")


@dataclass
class Plan:
    renames: dict[str, str]
    """Old dotted module name -> new, a moved package once by its prefix."""
    paths: list[tuple[str, str]]
    """Old path -> new path of every moved file."""
    tree: list[str]
    """Every tracked path before the move."""
    config_keys: frozenset[str] = frozenset()
    i18n_keys: frozenset[str] = frozenset()
    """Every leaf key of ``locales/en.yaml``: ``cli.queue.queued`` is a message key, not a module path."""
    root: Path | None = None
    """The tree whose moved files are read for the attributes a module defines."""
    attributes: dict[str, frozenset[str] | None] = field(default_factory=dict, init=False)
    files: dict[str, str] = field(init=False)
    dirs: dict[str, str] = field(init=False)
    modules: set[str] = field(init=False)
    packages: set[str] = field(init=False)
    """Every dotted prefix of a module name, so a namespace package without ``__init__.py`` counts too."""
    unique_root: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.files = dict(self.paths)
        moved = set(self.files)
        under: dict[str, list[str]] = {}
        for path in self.tree:
            for parent in PurePosixPath(path).parents:
                if str(parent) != ".":
                    under.setdefault(str(parent), []).append(path)
        self.dirs = {}
        for old, new in self.paths:
            old_parent, new_parent = PurePosixPath(old).parent, PurePosixPath(new).parent
            while str(old_parent) != "." and old_parent.name == new_parent.name:
                key = str(old_parent)
                if key not in self.dirs and all(p in moved for p in under.get(key, [])):
                    self.dirs[key] = str(new_parent)
                old_parent, new_parent = old_parent.parent, new_parent.parent
        self.modules = set()
        for path in self.tree:
            if path.endswith(".py"):
                parts = list(PurePosixPath(path).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                self.modules.add(".".join(parts))
        self.packages = {".".join(m.split(".")[:k]) for m in self.modules for k in range(1, len(m.split(".")) + 1)}
        names = Counter(PurePosixPath(p).name for p in self.tree)
        self.unique_root = {old for old in self.files
                            if "/" not in old and names[old] == 1 and old not in AMBIGUOUS_FILES}


    def module_attributes(self, module: str) -> frozenset[str] | None:
        """Names the moved module ``module`` (old dotted name) defines at its top level; None when unreadable."""
        if module not in self.attributes:
            self.attributes[module] = None
            base = module.replace(".", "/")
            for old in (f"{base}.py", f"{base}/__init__.py"):
                if self.root is not None and old in self.files and (self.root / self.files[old]).is_file():
                    self.attributes[module] = _defined_names(self.root / self.files[old])
                    break
        return self.attributes[module]


AMBIGUOUS_FILES = {"cli.py", "utils.py", "config.py", "setup.py", "__init__.py", "conftest.py"}


def _defined_names(file: Path) -> frozenset[str] | None:
    try:
        body = ast.parse(file.read_text(encoding="utf-8-sig")).body
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None
    names: set[str] = set()
    pending = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name.split(".")[0]) for a in node.names if a.name != "*")
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            pending.extend(ast.iter_child_nodes(node))
    return frozenset(names)


@dataclass
class Stats:
    rewrites: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)
    samples: list[tuple[str, int, str, str]] = field(default_factory=list)
    files_changed: int = 0
    files_read: int = 0
    unparsed: list[str] = field(default_factory=list)

    def add(self, other: Stats) -> None:
        self.rewrites.update(other.rewrites)
        self.skips.update(other.skips)
        self.samples.extend(other.samples)
        self.files_changed += other.files_changed
        self.files_read += other.files_read
        self.unparsed += other.unparsed


# ---------------------------------------------------------------- text rewriting


def rewrite_text(text: str, plan: Plan, stats: Stats | None = None) -> str:
    """Rewrite one piece of prose, line by line (tree-diagram lines are left as they are)."""
    stats = stats if stats is not None else Stats()
    out, fenced = [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            out.append(line)
        elif _TREE_LINE.search(line):
            if _mentions(line, plan):
                stats.skips["tree-line"] += 1
            out.append(line)
        else:
            out.append(_rewrite_line(line, plan, stats, fenced or line.lstrip().startswith((">>>", "..."))))
    return "\n".join(out)


def _mentions(line: str, plan: Plan) -> bool:
    return any(old.split("/")[0] in line for old in list(plan.dirs)[:50] + list(plan.unique_root)[:50]) or \
        any(k in line for k in plan.renames)


def _rewrite_line(line: str, plan: Plan, stats: Stats, fenced: bool = False) -> str:
    edits: list[tuple[int, int, str]] = []
    imported = _import_spans(line, plan, stats, edits, fenced)
    for match in _TOKEN.finditer(line):
        start, token = match.start(), match.group()
        if any(s <= start < e for s, e in imported):
            continue
        blob = _BLOB.match(token)
        if blob:
            start, token = start + blob.end(), token[blob.end():]
        prev = line[:start]
        if blob:
            kind = "blob-url"
        elif prev.endswith("/"):
            if _first_segment_moved(token, plan):
                stats.skips["prefixed-path"] += 1
            continue
        else:
            kind = None
        found = _path(token, plan, stats) if "/" in token else _bare(token, plan, stats, prev)
        if found:
            length, new, rule = found
            edits.append((start, start + length, new))
            stats.rewrites[kind or rule] += 1
    return _apply(line, edits)


def _first_segment_moved(token: str, plan: Plan) -> bool:
    head = token.split("/")[0]
    return head in plan.dirs or any(f.startswith(head + "/") for f in plan.files)


def _apply(line: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, new in sorted(edits, reverse=True):
        line = line[:start] + new + line[end:]
    return line


def _path(token: str, plan: Plan, stats: Stats):
    """(length of the old text, its replacement, rule) for a token with a ``/``."""
    clean = token
    # A longest-first walk over the cut points of the token.
    cuts = [m.start() for m in re.finditer(r"[./]", clean)] + [len(clean)]
    for cut in sorted(set(cuts), reverse=True):
        head, rest = clean[:cut], clean[cut:]
        if not head or "/" not in head:
            continue
        if head in plan.files and _tail_ok(rest):
            return len(head), plan.files[head], "path-file"
        stem = head + ".py"
        if stem in plan.files and head not in plan.dirs and _tail_ok(rest) and rest != "/" and not rest.startswith("/"):
            return len(head), plan.files[stem][:-3], "path-stem"
        base = head.rstrip("/")
        if base in plan.dirs and not rest.startswith("/") and _tail_ok(rest) and (head.endswith("/") or "/" in base):
            if base in AMBIGUOUS_ROOTS:
                stats.skips["bare-dir"] += 1
                continue
            return len(base), plan.dirs[base], "path-dir"
    if "*" in token:
        head, _, pattern = token.rpartition("/")
        if head in AMBIGUOUS_ROOTS and not pattern.strip("*"):
            stats.skips["bare-dir"] += 1  # ``cron/*`` takes the whole folder, like a bare ``cron/``
            return None
        if head in plan.dirs or any(PurePosixPath(f).parent.as_posix() == head for f in plan.files):
            hits = [f for f in plan.files if PurePosixPath(f).parent.as_posix() == head
                    and fnmatch.fnmatch(PurePosixPath(f).name, pattern)]
            if hits:
                return len(head), PurePosixPath(plan.files[hits[0]]).parent.as_posix(), "path-glob"
    return None


def _tail_ok(rest: str) -> bool:
    """What may follow a matched path: nothing, sentence punctuation, or a ``.symbol`` suffix."""
    if rest == "" or re.fullmatch(r"\.+", rest):
        return True
    symbol = re.fullmatch(r"\.([A-Za-z_]\w*)(?:\.[A-Za-z_]\w*)*\.*", rest)
    return bool(symbol) and symbol.group(1).lower() not in FILE_EXT


def _bare(token: str, plan: Plan, stats: Stats, before: str = ""):
    """A token without ``/``: a root file name, or a dotted module name."""
    core = _DOTTED.match(token)
    if not core:
        return None
    name = core.group()
    rest = token[len(name):]
    if rest.startswith("-"):
        return None
    last = name.rsplit(".", 1)[-1].lower()
    if "." in name and last in FILE_EXT:
        file = name
        if file in plan.files:
            if file in plan.unique_root:
                return len(file), plan.files[file], "path-file"
            stats.skips["ambiguous-root-file"] += 1
        return None
    return _dotted(name, plan, stats, importing=False, before=before)


def _dotted(name: str, plan: Plan, stats: Stats, importing: bool, before: str = ""):
    if "." not in name and not importing:
        return None
    new = current_name(name, plan.renames)
    if new is None or new == name:
        return None
    if not importing:
        if CONFIG_DENY.match(name) or _config_key(name, plan):
            stats.skips["config-key"] += 1
            return None
        if name in plan.i18n_keys or _I18N_CALL.search(before):
            stats.skips["i18n-key"] += 1
            return None
        segments = name.split(".")
        if segments[0] in AMBIGUOUS_ROOTS and (len(segments) < 2 or ".".join(segments[:2]) not in plan.packages):
            stats.skips["ambiguous-dotted"] += 1
            return None
        if not _resolves(name, plan):
            stats.skips["unresolved-dotted"] += 1
            return None
    return len(name), new, "import" if importing else "dotted"


def _resolves(name: str, plan: Plan) -> bool:
    """``name`` is a moved module or package, or a path into one.

    ``module.attr`` resolves when the module defines ``attr``. A tail that names nothing the module defines
    (``gateway.run.X``, a placeholder or an attribute that moved to a sibling) still counts after a module two
    or more segments deep, whose name cannot be a config key; after a bare package that is also a config
    section (``plugins.enabled``, ``agent.reasoning_effort``) it does not.
    """
    parts = name.split(".")
    if name in plan.packages:
        return True
    for size in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:size])
        if prefix in plan.packages:
            attributes = plan.module_attributes(prefix) if prefix in plan.modules else None
            return (attributes is not None and parts[size] in attributes) or size > 1 or parts[0] not in AMBIGUOUS_ROOTS
    return False


def _config_key(name: str, plan: Plan) -> bool:
    parts = name.split(".")
    return any(".".join(parts[:k]) in plan.config_keys for k in range(2, len(parts) + 1))


def _code_spans(line: str) -> list[tuple[int, int]]:
    """Inner ranges of the backtick spans on a line (the backticks themselves excluded)."""
    return [(m.start() + len(m.group(1)), m.end() - len(m.group(1))) for m in _CODE_SPAN.finditer(line)]


def _import_spans(line: str, plan: Plan, stats: Stats, edits: list, fenced: bool = False) -> list[tuple[int, int]]:
    """``from M import ...``, ``import M``, ``-m M``: the code places a bare top-level word is rewritten."""
    spans, code = [], _code_spans(line)
    for match in _IMPORT.finditer(line):
        keyword, name = match.group(1).strip(), match.group(2).rstrip(".")
        site = next((span for span in code if span[0] <= match.start() < span[1]), None)
        if keyword == "import" and re.search(r"\bfrom\s+[\w.]+\s+$", line[:match.start()]):
            continue
        found = _dotted(name, plan, stats, importing=True)
        if not found:
            continue
        if not _is_code(line, match, keyword, name, site, fenced):
            stats.skips["prose-import"] += 1
            continue
        length, new, _ = found
        begin, start = match.start(2), match.start(1)
        if keyword == "import" and "." not in name and new.endswith("." + name):
            start, new = match.start(1), f"from {new.rpartition('.')[0]} import {name}"
        else:
            start = begin
        edits.append((start, begin + length, new))
        spans.append((begin, begin + len(match.group(2))))
        stats.rewrites["import"] += 1
    return spans


def _is_code(line: str, match: re.Match, keyword: str, name: str, site: tuple[int, int] | None, fenced: bool) -> bool:
    """Whether the match is code: in a code span or fence, or a Python import statement that starts the line."""
    in_code = fenced or site is not None
    if keyword.startswith("-m"):
        return in_code
    head = _STATEMENT.match(line, match.start())
    if head is None or not _parses(head.group()):
        return False
    rest = line[head.end():site[1] if site else len(line)].strip()
    if in_code:
        return not rest or not (rest[0].isalnum() or rest[0] == "_")
    if not _STATEMENT_START.fullmatch(line[:match.start()]):
        return False
    return rest.rstrip(".,;:") == "" or rest[:1] in (";", "#")


def _parses(statement: str) -> bool:
    try:
        ast.parse(statement.replace("...", "_") + "_)" if statement.endswith("(") else statement.replace("...", "_"))
    except SyntaxError:
        return False
    return True


# ---------------------------------------------------------------- per file kind


def rewrite_python(source: str, plan: Plan, stats: Stats) -> str:
    """Rewrite only comment tokens and docstring tokens."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        stats.unparsed.append("tokenize")
        return source
    lines = source.splitlines(keepends=True)
    starts = [0]
    for text in lines:
        starts.append(starts[-1] + len(text))
    docs = _docstring_ranges(source, lines)
    edits = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            body = tok.string[1:]
            if tok.start[0] == 1 and body.startswith("!"):
                continue
            new = rewrite_text(body, plan, stats)
            if new != body:
                begin = starts[tok.start[0] - 1] + tok.start[1] + 1
                edits.append((begin, begin + len(body), new))
        elif tok.type == tokenize.STRING and any(a <= tok.start and tok.end <= b for a, b in docs):
            if re.match(r"(?i)[bf]+['\"]", tok.string):
                continue
            new = rewrite_text(tok.string, plan, stats)
            if new != tok.string:
                begin = starts[tok.start[0] - 1] + tok.start[1]
                end = starts[tok.end[0] - 1] + tok.end[1]
                edits.append((begin, end, new))
    for begin, end, new in sorted(edits, reverse=True):
        source = source[:begin] + new + source[end:]
    return source


def _docstring_ranges(source: str, lines: list[str]):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                end_row, end_col = first.end_lineno or first.lineno, first.end_col_offset or 0
                ranges.append(((first.lineno, _char(lines, first.lineno, first.col_offset)),
                               (end_row, _char(lines, end_row, end_col))))
    return ranges


def _char(lines: list[str], row: int, byte_col: int) -> int:
    return len(lines[row - 1].encode()[:byte_col].decode(errors="ignore"))


def rewrite_yaml_comments(source: str, plan: Plan, stats: Stats) -> str:
    out = []
    for line in source.split("\n"):
        at = _comment_start(line)
        out.append(line if at < 0 else line[:at + 1] + rewrite_text(line[at + 1:], plan, stats))
    return "\n".join(out)


def _comment_start(line: str) -> int:
    if line.lstrip().startswith("#"):
        return line.index("#")
    for match in re.finditer(r"\s#", line):
        before = line[:match.start()]
        if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
            return match.start() + 1
    return -1


def excluded(path: str) -> bool:
    pure = PurePosixPath(path)
    return any(pure.full_match(pattern) if hasattr(pure, "full_match") else _glob(path, pattern)
               for pattern in EXCLUDED)


def _glob(path: str, pattern: str) -> bool:
    regex = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, path) is not None


@functools.cache
def _generated() -> frozenset[str]:
    """Repo-relative paths of the shims and byte copies at old module paths (all .py files)."""
    from generate_shims import generated

    return frozenset(generated())


def rewrite_file(path: str, source: str, plan: Plan, stats: Stats) -> str:
    if excluded(path):
        stats.skips["excluded-file"] += 1
        return source
    if path.endswith(".py") and path in _generated():
        stats.skips["generated-shim"] += 1
        return source
    if path.endswith(".py"):
        return rewrite_python(source, plan, stats)
    if path.endswith(TEXT_SUFFIXES):
        return rewrite_text(source, plan, stats)
    if path.endswith(YAML_SUFFIXES):
        return rewrite_yaml_comments(source, plan, stats)
    return source


# ---------------------------------------------------------------- tree driver


def locale_keys(root: Path) -> frozenset[str]:
    """Leaf keys of ``locales/en.yaml``, joined by dots (the file is nested maps of scalars)."""
    file = root / "locales" / "en.yaml"
    if not file.is_file():
        return frozenset()
    keys: set[str] = set()
    stack: list[tuple[int, str]] = []
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"( *)([\w.\-]+):(?:\s+(\S.*))?", line)
        if not match:
            continue
        indent, key, value = len(match.group(1)), match.group(2), match.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [key])
        if value is None:
            stack.append((indent, key))
        else:
            keys.add(path)
    return frozenset(keys)


def synthetic_plan(root: Path, moves: list[str]) -> Plan:
    """A rename table for a move that has not happened: each path goes under ``hermes_agent/``."""
    tree = git(root, "ls-files").splitlines()
    paths, renames = [], {}
    for move in moves:
        move = move.rstrip("/")
        hit = [p for p in tree if p == move or p.startswith(move + "/")]
        paths += [(p, f"hermes_agent/{p}") for p in hit]
        module = move.removesuffix(".py")
        renames[module.replace("/", ".")] = "hermes_agent." + module.replace("/", ".")
    return Plan(renames, paths, tree, config_keys(root), locale_keys(root), root)


def commit_plan(root: Path, commit: str) -> Plan:
    tree = git(root, "ls-tree", "-r", "--name-only", f"{commit}^").splitlines()
    return Plan(module_renames(root, commit), renamed_paths(root, commit), tree, config_keys(root),
                locale_keys(root), root)


def run(root: Path, plan: Plan, only: list[str], write: bool) -> Stats:
    total = Stats()
    files = git(root, "ls-files").splitlines()
    if only:
        files = [f for f in files if any(f == p or f.startswith(p.rstrip("/") + "/") for p in only)]
    for path in files:
        if not path.endswith((".py",) + TEXT_SUFFIXES + YAML_SUFFIXES) or excluded(path):
            continue
        file = root / path
        try:
            source = file.read_text(encoding="utf-8-sig")
            with file.open("rb") as fh:
                bom = fh.read(3) == codecs.BOM_UTF8
        except (UnicodeDecodeError, OSError):
            continue
        one = Stats()
        new = rewrite_file(path, source, plan, one)
        one.files_read = 1
        if new != source:
            one.files_changed = 1
            for number, (a, b) in enumerate(zip(source.split("\n"), new.split("\n")), 1):
                if a != b:
                    one.samples.append((path, number, a, b))
            if write:
                file.write_bytes((codecs.BOM_UTF8 if bom else b"") + new.encode("utf-8"))
        total.add(one)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--from-commit", help="the move commit (git mv only)")
    parser.add_argument("--synthetic-move", nargs="*", default=[], help="rehearse moving these paths under hermes_agent/")
    parser.add_argument("--paths", nargs="*", default=[], help="limit to these files or directories")
    parser.add_argument("--check", action="store_true", help="write nothing; fail when anything would change")
    parser.add_argument("--samples", type=int, default=0, help="print this many evenly spaced changed lines")
    parser.add_argument("--stats-json", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if bool(args.from_commit) == bool(args.synthetic_move):
        parser.error("give exactly one of --from-commit or --synthetic-move")
    plan = commit_plan(root, args.from_commit) if args.from_commit else synthetic_plan(root, args.synthetic_move)
    stats = run(root, plan, args.paths, write=not args.check)
    summary = {"files_read": stats.files_read, "files_changed": stats.files_changed,
               "rewrites": dict(stats.rewrites), "total_rewrites": sum(stats.rewrites.values()),
               "skips": dict(stats.skips), "changed_lines": len(stats.samples), "unparsed": len(stats.unparsed)}
    print(json.dumps(summary, indent=2))
    if args.stats_json:
        args.stats_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.samples and stats.samples:
        step = max(1, len(stats.samples) // args.samples)
        for path, number, before, after in stats.samples[::step][:args.samples]:
            print(f"\n{path}:{number}\n-{before.strip()[:200]}\n+{after.strip()[:200]}")
    return 1 if args.check and stats.files_changed else 0


if __name__ == "__main__":
    sys.exit(main())
