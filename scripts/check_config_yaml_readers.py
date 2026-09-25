#!/usr/bin/env python3
"""Fail when a user ``config.yaml`` is read by anything but the config backend.

Every read of ``<hermes home>/config.yaml`` must go through ``hermes_cli.config_backend``
(``get_config_backend()`` / the path helpers there) so a non-file backend (design §4.1, D10) sees
every reader. A direct ``open`` / ``read_text`` / ``yaml.safe_load`` / ``fast_safe_load`` /
``load_yaml_file_readonly`` / ``stat`` of a config path silently keeps reading the local file —
the reader twin of the #92554 writer class, which regrew each time a new caller picked the raw
primitive again.

Flags, in the scanned trees:

* ``open(p)``, ``p.open()``, ``p.read_text()``, ``p.read_bytes()``, ``p.stat()``,
  ``p.exists()``, ``p.is_file()``, ``os.stat(p)``, ``os.path.exists(p)``, ``os.path.isfile(p)``,
  ``os.path.getmtime(p)``, ``os.path.getsize(p)``, ``load_yaml_file_readonly(p)``,
  ``fast_safe_load(p)``, ``yaml.load(p)`` / ``safe_load`` / ``full_load`` / ``unsafe_load`` /
  ``compose`` (and their ``*_all`` forms) and any ``*_load_yaml*`` / ``*_read_yaml*`` helper
  call, where ``p`` is a config path: an expression naming ``config.yaml``,
  any ``*config_path()`` call (``get_config_path()``, ``_active_config_path()``), or a local name / parameter bound to one
  (``cfg_path = home / "config.yaml"`` then ``open(cfg_path)``);
* ``atomic_roundtrip_yaml_update`` / ``atomic_roundtrip_yaml_save`` / ``atomic_write_text`` of a
  config path (the bypass writers; route them through ``write_config_document`` /
  ``write_config_key``).

Existence checks count: for a backend whose user layer is not a local file, "no local
config.yaml" must never mean "use defaults" (design §4.2) — ``config_exists()`` asks the backend.

Suppress a true false positive with ``# config-reader: ok — <why>`` on the call's line; a
marker without a reason after the dash does not suppress.

Usage: python3 scripts/check_config_yaml_readers.py [paths...]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREES = (
    "hermes_cli", "agent", "gateway", "tui_gateway", "cron", "plugins", "tools", "acp_adapter",
    "cli.py", "utils.py", "hermes_constants.py", "hermes_logging.py", "hermes_time.py", "run_agent.py",
    "model_tools.py", "batch_runner.py")
# The backend module itself, and the on-disk primitives it wraps.
ALLOWED_FILES = {ROOT / "hermes_cli" / "config_backend.py", ROOT / "utils.py"}
# Modules whose ``config_path`` / ``path`` names are their OWN config file, never a hermes
# config.yaml (the name heuristic would flag every line of them).
FOREIGN_CONFIG_FILES = {
    "plugins/memory/honcho": "Honcho's own honcho.json",
    "plugins/memory/openviking": "OpenViking's own ovcli config",
    "agent/proxy_sources/iron_proxy.py": "iron-proxy's proxy.yaml",
}
SUPPRESS = "# config-reader: ok"
# The escape must carry its reason: ``# config-reader: ok — <why>`` (``-`` / ``--`` accepted).
SUPPRESS_RE = re.compile(r"# config-reader: ok\s*(?:—|--?)\s*\S")

# An expression that evaluates to a user config.yaml path.
CONFIG_PATH_EXPR_RE = re.compile(r"""["']config\.yaml["']|\w*config_path\(\)""")
# Parameters that conventionally carry the user config path.
CONFIG_PARAM_NAMES = {"config_path", "cfg_path", "config_yaml_path"}
PATH_METHOD_READS = {"open", "read_text", "read_bytes", "stat", "exists", "is_file"}
FUNC_READS = {
    "exists", "isfile", "open", "getmtime", "getsize", "stat", "load_yaml_file_readonly", "fast_safe_load",
    # Every PyYAML entry point that parses a stream, not just safe_load.
    "safe_load", "safe_load_all", "full_load", "full_load_all", "unsafe_load", "unsafe_load_all",
    "load_all", "compose", "compose_all"}
BYPASS_WRITERS = {"atomic_roundtrip_yaml_update", "atomic_roundtrip_yaml_save", "atomic_write_text"}
YAML_HELPER_RE = re.compile(r"(load|read)_yaml|yaml_(load|read)")


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _scope_nodes(scope: ast.AST):
    """Nodes of *scope* without descending into nested function/class bodies."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


class _Scanner:
    def __init__(self, path: Path):
        self.path = path
        self.src = path.read_text(encoding="utf-8-sig")
        self.lines = re.split(r"\r\n|\r|\n", self.src)  # the tokenizer's line breaks, not splitlines()'s
        self.blines = [line.encode("utf-8") for line in self.lines]
        self.rel = path.relative_to(ROOT).as_posix()
        self.problems: list[str] = []

    def seg(self, node: ast.AST) -> str:
        # ast.get_source_segment re-splits the whole file per call (quadratic on config.py).
        start, end = getattr(node, "lineno", None), getattr(node, "end_lineno", None)
        if start is None or end is None:
            return ""
        # col offsets are UTF-8 byte offsets.
        c0, c1 = getattr(node, "col_offset"), getattr(node, "end_col_offset")
        b = self.blines
        if start == end:
            return b[start - 1][c0:c1].decode("utf-8", "replace")
        return b"\n".join([b[start - 1][c0:], *b[start:end - 1], b[end - 1][:c1]]).decode("utf-8", "replace")

    def is_config_path(self, node: ast.AST, names: set[str]) -> bool:
        if isinstance(node, ast.Name) and node.id in names:
            return True
        if isinstance(node, ast.Call) and _func_name(node.func) in {"Path", "str", "expanduser", "resolve"}:
            inner = node.args[0] if node.args else (node.func.value if isinstance(node.func, ast.Attribute) else None)
            if inner is not None and self.is_config_path(inner, names):
                return True
        if isinstance(node, ast.BoolOp):
            return any(self.is_config_path(v, names) for v in node.values)
        return bool(CONFIG_PATH_EXPR_RE.search(self.seg(node)))

    def config_names(self, scope: ast.AST, inherited: set[str]) -> set[str]:
        names = set(inherited)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs):
                if a.arg in CONFIG_PARAM_NAMES:
                    names.add(a.arg)
        # Two passes so `a = home / "config.yaml"; b = a` binds b too.
        for _ in range(2):
            for node in _scope_nodes(scope):
                targets: list = []
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                    targets, value = [node.target], node.value
                else:
                    continue
                if self.is_config_path(value, names):
                    names.update(t.id for t in targets if isinstance(t, ast.Name))
        return names

    def flag(self, node: ast.Call, why: str) -> None:
        line = self.lines[node.lineno - 1]
        if SUPPRESS_RE.search(line):
            return
        if SUPPRESS in line:
            why = f"{why}; the escape needs a reason (`{SUPPRESS} — <why>`)"
        self.problems.append(f"{self.rel}:{node.lineno}: {why} — go through hermes_cli.config_backend")

    def check_call(self, node: ast.Call, names: set[str]) -> None:
        name = _func_name(node.func)
        if name is None:
            return
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        first = node.args[0] if node.args else None
        if name in BYPASS_WRITERS and first is not None and self.is_config_path(first, names):
            self.flag(node, f"bypass write of a config path ({self.seg(first)})")
            return
        if name in PATH_METHOD_READS and receiver is not None and self.is_config_path(receiver, names):
            self.flag(node, f"direct {name}() of a config path ({self.seg(receiver)})")
            return
        is_reader = name in FUNC_READS or YAML_HELPER_RE.search(name) is not None
        if name == "load" and receiver is not None and "yaml" in self.seg(receiver).lower():
            is_reader = True
        if is_reader and first is not None and (receiver is None or name != "open") and self.is_config_path(first, names):
            self.flag(node, f"direct {name}() of a config path ({self.seg(first)[:60]})")

    def scan(self) -> list[str]:
        try:
            tree = ast.parse(self.src)
        except SyntaxError:
            return []

        def walk(scope: ast.AST, inherited: set[str]) -> None:
            names = self.config_names(scope, inherited)
            for node in _scope_nodes(scope):
                if isinstance(node, ast.Call):
                    self.check_call(node, names)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    walk(node, names)

        walk(tree, set())
        return self.problems


def scan_file(path: Path) -> list[str]:
    return _Scanner(path).scan()


def main(argv: list[str]) -> int:
    targets = [ROOT / a for a in argv] or [ROOT / t for t in DEFAULT_TREES]
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files.extend(p for p in t.rglob("*.py") if "node_modules" not in p.parts and "tests" not in p.parts)
        elif t.suffix == ".py" and t.exists():
            files.append(t)
    problems: list[str] = []
    for f in sorted(set(files)):
        if f.resolve() in ALLOWED_FILES:
            continue
        rel = f.resolve().relative_to(ROOT).as_posix() if f.resolve().is_relative_to(ROOT) else ""
        if any(rel == p or rel.startswith(p + "/") for p in FOREIGN_CONFIG_FILES):
            continue
        problems.extend(scan_file(f))
    if problems:
        print("config.yaml must only be read through hermes_cli.config_backend:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"check_config_yaml_readers: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
