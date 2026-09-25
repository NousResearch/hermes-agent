#!/usr/bin/env python3
"""Fail when a ``config.yaml`` is written by anything but the comment-preserving writer.

Every writer of ``~/.hermes/config.yaml`` (and ``profiles/*/config.yaml``) must go through
``hermes_cli.config.atomic_config_write`` → ``utils.atomic_roundtrip_yaml_save`` (ruamel
round-trip). A PyYAML dump (``yaml.dump`` / ``yaml.safe_dump`` / ``utils.atomic_yaml_write``)
of a config path re-serialises the parsed dict and destroys every user comment — the #92554
class, which regressed several times because each new writer picked the plain dumper again.

Flags, in the scanned trees:

* a call to ``atomic_yaml_write`` / ``yaml.dump`` / ``yaml.safe_dump`` / ``safe_dump`` whose
  first argument names a config path (``config_path``, ``cfg_path``, ``config.yaml``,
  ``get_config_path()``, ``_active_config_path()``, ``live_path``);
* ``<config path>.write_text(... dump(...) ...)``;
* any ``yaml.dump`` / ``yaml.safe_dump`` / ``atomic_yaml_write`` call inside ``hermes_cli/config.py``
  or ``hermes_cli/config_*.py`` (the config system has exactly one writer).

Suppress a true false positive with ``# config-writer: ok — <why>`` on the call's line.

Usage: python3 scripts/check_config_yaml_writers.py [paths...]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREES = ("hermes_cli", "agent", "gateway", "tui_gateway", "cron", "plugins", "tools", "cli.py", "utils.py")
# The writer module itself and the on-disk primitive it wraps.
ALLOWED_FILES = {ROOT / "utils.py"}
DUMPERS = {"atomic_yaml_write", "safe_dump", "dump"}
CONFIG_PATH_RE = re.compile(
    r"config_path|cfg_path|config\.yaml|get_config_path\(\)|_active_config_path\(\)|\blive_path\b")
CONFIG_MODULE_RE = re.compile(r"^hermes_cli/config(_[a-z_]+)?\.py$")
SUPPRESS = "# config-writer: ok"


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_yaml_dump(call: ast.Call, src: str) -> bool:
    name = _func_name(call.func)
    if name == "atomic_yaml_write":
        return True
    if name in ("safe_dump", "dump"):
        # ``yaml.dump`` / ``yaml.safe_dump`` / ``yaml_rt.dump``; plain ``json.dump`` is not YAML.
        receiver = ast.get_source_segment(src, call.func.value) if isinstance(call.func, ast.Attribute) else ""
        return "yaml" in (receiver or "").lower() or name == "safe_dump"
    return False


def scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    rel = path.relative_to(ROOT).as_posix()
    in_config_module = bool(CONFIG_MODULE_RE.match(rel))
    problems: list[str] = []

    def flag(node: ast.Call, why: str) -> None:
        line = lines[node.lineno - 1]
        if SUPPRESS in line:
            return
        problems.append(f"{rel}:{node.lineno}: {why} — route it through hermes_cli.config.atomic_config_write")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_yaml_dump(node, src):
            if in_config_module:
                flag(node, "PyYAML dump inside the config system")
                continue
            first = ast.get_source_segment(src, node.args[0]) if node.args else ""
            if first and CONFIG_PATH_RE.search(first):
                flag(node, f"PyYAML dump of a config path ({first})")
        elif _func_name(node.func) == "write_text" and isinstance(node.func, ast.Attribute):
            receiver = ast.get_source_segment(src, node.func.value) or ""
            body = ast.get_source_segment(src, node) or ""
            if CONFIG_PATH_RE.search(receiver) and "dump(" in body:
                flag(node, f"write_text of a YAML dump onto a config path ({receiver})")
    return problems


def main(argv: list[str]) -> int:
    targets = [ROOT / a for a in argv] or [ROOT / t for t in DEFAULT_TREES]
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files.extend(p for p in t.rglob("*.py") if "node_modules" not in p.parts)
        elif t.suffix == ".py" and t.exists():
            files.append(t)
    problems: list[str] = []
    for f in sorted(set(files)):
        if f.resolve() in ALLOWED_FILES:
            continue
        problems.extend(scan_file(f))
    if problems:
        print("config.yaml must only be written through the comment-preserving writer:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"check_config_yaml_writers: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
