"""Old module names a move leaves in ``pyproject.toml``: entry points and package-data keys change; the rest is flagged."""
from __future__ import annotations

import re
from pathlib import Path

from renames import current_name

_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$")
_ENTRY = re.compile(r"""^\s*(?:"[^"]*"|'[^']*'|[\w.\-]+)\s*=\s*(["'])([\w.]+)(?::[\w.]+)?\1""")
_DATA_KEY = re.compile(r"""^(\s*)(?:"([\w.]+)"|'([\w.]+)'|([\w]+))(\s*=)""")
_STRING = re.compile(r"""(["'])([^"'\n]*)\1""")
_ENTRY_TABLES = ("project.scripts", "project.gui-scripts")


def rewrite_pyproject(root: Path, plan, write: bool) -> tuple[int, list[str]]:
    """Rewrite ``pyproject.toml`` for ``plan``; return (sites rewritten, flag lines).

    Entry-point values (``module:attr``) and ``[tool.setuptools.package-data]`` keys follow their module. Any
    other string naming a moved module or path is flagged: package lists stay as they are, since the old
    directory may still hold a compat shim while ``hermes_agent.*`` already covers the new place.
    """
    file = root / "pyproject.toml"
    if not file.is_file():
        return 0, []
    bom = "\ufeff" if file.read_bytes().startswith(b"\xef\xbb\xbf") else ""
    lines = file.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    table, count, flags = "", 0, []
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        if header := _HEADER.match(line):
            table = re.sub(r"""["']""", "", header.group(1))
            continue
        if (table in _ENTRY_TABLES or table.startswith("project.entry-points.")) and (entry := _ENTRY.match(line)) \
                and (new := current_name(entry.group(2), plan.renames)) is not None:
            lines[number - 1] = line[:entry.start(2)] + new + line[entry.end(2):]
            count += 1
            continue
        if table == "tool.setuptools.package-data" and (key := _DATA_KEY.match(line)) \
                and (new := current_name(next(filter(None, key.group(2, 3, 4))), plan.renames)) is not None:
            lines[number - 1] = f'{key.group(1)}"{new}"{key.group(5)}{line[key.end():]}'
            count += 1
            continue
        flags += _flags(line, number, table, plan)
    if count and write:
        file.write_text(bom + "".join(lines), encoding="utf-8")
    return count, flags


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
