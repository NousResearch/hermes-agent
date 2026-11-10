"""Old module names a move leaves in ``pyproject.toml``: entry points and package-data keys change, package-list rows
with no generated old-path file behind them go; the rest is flagged."""
from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

from renames import current_name

_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$")
_ENTRY = re.compile(r"""^\s*(?:"[^"]*"|'[^']*'|[\w.\-]+)\s*=\s*(["'])([\w.]+)(?::[\w.]+)?\1""")
_DATA_KEY = re.compile(r"""^(\s*)(?:"([\w.]+)"|'([\w.]+)'|([\w]+))(\s*=)""")
_STRING = re.compile(r"""(["'])([^"'\n]*)\1""")
_PACKAGE_ROW = re.compile(r"""^\s*(["'])([\w.]+?)(\.\*)?\1\s*,?\s*(?:#.*)?$""")
_ENTRY_TABLES = ("project.scripts", "project.gui-scripts")
_PACKAGE_TABLES = ("tool.setuptools.packages.find",)
_REQUIRED = "hermes_agent/compat/required_names.py"


def rewrite_pyproject(root: Path, plan, write: bool) -> tuple[int, list[str]]:
    """Rewrite ``pyproject.toml`` for ``plan``; return (sites rewritten, flag lines).

    Entry-point values (``module:attr``) and ``[tool.setuptools.package-data]`` keys follow their module. A
    package-list row (``"pkg"``, ``"pkg.*"``) of a moved package stays while a generated old-path file lives under
    it (``required_names.py``) and goes otherwise: ``hermes_agent.*`` already covers the new place. Any other string
    naming a moved module or path is flagged.
    """
    file = root / "pyproject.toml"
    if not file.is_file():
        return 0, []
    bom = "\ufeff" if file.read_bytes().startswith(b"\xef\xbb\xbf") else ""
    lines = file.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    shimmed = _shimmed_packages(root, plan)
    table, count, flags, out = "", 0, [], []
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            out.append(line)
            continue
        if header := _HEADER.match(line):
            table = re.sub(r"""["']""", "", header.group(1))
            out.append(line)
            continue
        new_line, changed = _rewrite_line(line, table, plan, shimmed)
        if changed:
            count += 1
            if new_line is not None:
                out.append(new_line)
            continue
        out.append(line)
        if not (table in _PACKAGE_TABLES and (row := _PACKAGE_ROW.match(line)) and row.group(2) in shimmed):
            flags += _flags(line, number, table, plan)
    if count and write:
        file.write_text(bom + "".join(out), encoding="utf-8")
    return count, flags


def _rewrite_line(line: str, table: str, plan, shimmed: set[str]) -> tuple[str | None, bool]:
    """(new line, or None to drop it; whether it changed)."""
    if (table in _ENTRY_TABLES or table.startswith("project.entry-points.")) and (entry := _ENTRY.match(line)) \
            and (new := current_name(entry.group(2), plan.renames)) is not None:
        return line[:entry.start(2)] + new + line[entry.end(2):], True
    if table == "tool.setuptools.package-data" and (key := _DATA_KEY.match(line)) \
            and (new := current_name(next(filter(None, key.group(2, 3, 4))), plan.renames)) is not None:
        return f'{key.group(1)}"{new}"{key.group(5)}{line[key.end():]}', True
    if table in _PACKAGE_TABLES and (row := _PACKAGE_ROW.match(line)) and plan.is_module(row.group(2)) \
            and row.group(2) not in shimmed:
        return None, True
    return line, False


def _shimmed_packages(root: Path, plan) -> set[str]:
    """Old package names that keep a generated old-path file under them after this move."""
    file = root / _REQUIRED
    if not file.is_file():
        return set()
    node = next((n for n in ast.parse(file.read_text(encoding="utf-8-sig")).body if isinstance(n, ast.AnnAssign)
                 and getattr(n.target, "id", None) == "REQUIRED_NAMES"), None)
    required = ast.literal_eval(node.value) if node is not None else {}
    table = {**plan.moves, **plan.renames}
    out = set()
    for name, kind in required.items():
        old = ".".join(PurePosixPath(name).with_suffix("").parts) if kind in ("by-path", "exists") else name
        old = old.removesuffix(".__init__")
        if current_name(old, table) is None:
            continue
        parts = old.split(".")
        out |= {".".join(parts[:size]) for size in range(1, len(parts) + 1)}
    return out


def _flags(line: str, number: int, table: str, plan) -> list[str]:
    out = []
    for match in _STRING.finditer(line):
        text = match.group(2)
        name = text.removesuffix(".*").partition(":")[0]
        if plan.is_target(name) and text not in plan.config_keys:
            reason = ("package list names a moved package; keep the old name only while a shim lives there"
                      if table.startswith("tool.setuptools") else "TOML value names a moved module; fix by hand")
            out.append(f"pyproject.toml:{number}  toml  {text}  {reason}")
        elif (found := plan.find_path(text)) is not None:
            out.append(f"pyproject.toml:{number}  toml  {found[0]}  TOML value names a moved path (now {found[1]})")
    return out
