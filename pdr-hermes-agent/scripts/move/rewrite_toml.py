"""Old module names a move leaves in ``pyproject.toml``: entry points and package-data keys change, package-list rows
with no generated old-path file behind them go, ``[tool.hermes] update-startup-files`` follows its files; the rest is
flagged."""
from __future__ import annotations

import ast
import importlib.util
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
_SHIM_WRITER = "scripts/move/generate_shims.py"
_STARTUP_TABLE, _STARTUP_KEY = "tool.hermes", "update-startup-files"
_STARTUP_START = re.compile(rf"^\s*{_STARTUP_KEY}\s*=\s*\[")
_STARTUP_ENTRY = re.compile(r'"([^"\\\n]+)"')
_STARTUP_WIDTH = 120
"""A startup-list line that grows past this many characters is wrapped; shorter lines keep their layout."""


def rewrite_pyproject(root: Path, plan, write: bool) -> tuple[int, list[str]]:
    """Rewrite ``pyproject.toml`` for ``plan``; return (sites rewritten, flag lines).

    Entry-point values (``module:attr``) and ``[tool.setuptools.package-data]`` keys follow their module. A
    package-list row (``"pkg"``, ``"pkg.*"``) of a moved package stays while a generated old-path file lives under
    it (``required_names.py``) and goes otherwise: ``hermes_agent.*`` already covers the new place. Each
    ``update-startup-files`` entry this move relocated becomes its new path, and the old path stays beside it while a
    file stays there (generated, or hand-written). Any other string naming a moved module or path is flagged.
    """
    file = root / "pyproject.toml"
    if not file.is_file():
        return 0, []
    bom = "\ufeff" if file.read_bytes().startswith(b"\xef\xbb\xbf") else ""
    lines = file.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    shimmed = _shimmed_packages(root, plan)
    startup = _startup_span(lines)
    table, count, flags, out = "", 0, [], []
    if startup is not None:
        first, last = startup
        new_lines, count = _rewrite_startup(lines[first:last + 1], dict(plan.paths), _generated_paths(root))
        lines = lines[:first] + new_lines + lines[last + 1:]
        startup = (first, first + len(new_lines) - 1)
    for number, line in enumerate(lines, 1):
        if startup is not None and startup[0] <= number - 1 <= startup[1]:
            out.append(line)
            continue
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


def _startup_span(lines: list[str]) -> tuple[int, int] | None:
    """First and last index of the ``[tool.hermes] update-startup-files`` array; None when there is none."""
    table, first = "", None
    for index, line in enumerate(lines):
        if first is None:
            if header := _HEADER.match(line):
                table = re.sub(r"""["']""", "", header.group(1))
            elif table == _STARTUP_TABLE and _STARTUP_START.match(line):
                first = index
        if first is None:
            continue
        code = _STARTUP_ENTRY.sub("", line).partition("#")[0]
        if "]" in (code.partition("[")[2] if index == first else code):
            return first, index
    return None


def _rewrite_startup(block: list[str], moved: dict[str, str], generated: set[str]) -> tuple[list[str], int]:
    """The startup-list lines with each entry this move relocated respelled, and how many entries changed.

    The old path is kept, followed by the new one, while a file stays at the old path; an entry whose new
    path the list already names is not added twice, so a second run changes nothing.
    """
    present = set(_STARTUP_ENTRY.findall("".join(block)))
    count, out = 0, []

    def entry(match: re.Match) -> str:
        nonlocal count
        old, separator = match.group(1), match.group(2) or ""
        if old not in moved:
            return match.group(0)
        names = ([old] if old in generated else []) + ([moved[old]] if moved[old] not in present else [])
        present.add(moved[old])
        if names == [old]:
            return match.group(0)
        count += 1
        return ", ".join(f'"{name}"' for name in names) + separator if names else ""

    for line in block:
        new = re.sub(r'"([^"\\\n]+)"(\s*,[ \t]*)?', entry, line)
        if new == line:
            out.append(line)
        elif new.strip():
            out += _wrapped(new)
    return out, count


def _wrapped(line: str) -> list[str]:
    """``line`` split into lines of at most ``_STARTUP_WIDTH`` characters when it is a plain row of entries."""
    body = line.rstrip("\r\n")
    if len(body) <= _STARTUP_WIDTH or not re.fullmatch(r'\s*(?:"[^"\n]*",\s*)*"[^"\n]*",?\s*', body):
        return [line]
    indent, ending = body[:len(body) - len(body.lstrip())], line[len(body):]
    rows, row = [], ""
    for name in _STARTUP_ENTRY.findall(body):
        piece = f'"{name}",'
        if row and len(indent) + len(row) + 1 + len(piece) > _STARTUP_WIDTH:
            rows.append(row)
            row = piece
        else:
            row = f"{row} {piece}" if row else piece
    rows.append(row)
    return [indent + row + ending for row in rows]


def _generated_paths(root: Path) -> set[str]:
    """Repository paths that keep a file once their module has moved: the ones ``generate_shims.py`` writes for
    this tree, and the hand-written ones it requires."""
    script = root / _SHIM_WRITER
    if not script.is_file():
        return set()
    spec = importlib.util.spec_from_file_location("_move_generate_shims", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.generated()) | set(module.hand_written())


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
        old = ".".join(PurePosixPath(name).with_suffix("").parts) if kind in ("by-path", "exists", "copy") else name
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
