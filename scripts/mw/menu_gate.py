#!/usr/bin/env python3
"""Portable menu-close decision gate (mw-menu-gate).

A menu may be called "done" only when a machine verifies every required
checklist item has real evidence under the project tree.  Loads a central
checklist YAML, evaluates each item LIVE, derives pass/fail counts from
evaluation (never hardcoded), and decides:

  exit 0  — all blocking items pass (menu closeable)
  exit 1  — one or more blocking items red
  exit 2  — usage / config / checklist error

Config and paths are portable: no hardcoded repo paths; --root defaults to
cwd; checklist defaults to <root>/.work/menu-checklist.yaml.

stdlib-only core · optional PyYAML · Python 3.9+
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_RED = 1
EXIT_ERR = 2

DEFAULT_CHECKLIST_REL = Path(".work") / "menu-checklist.yaml"
DEFAULT_CMD_TIMEOUT = 120

KNOWN_VERIFY = frozenset(
    {"file_glob", "file_grep", "row_in", "command", "evidence_file"}
)

# Tests may force the fallback YAML loader even when PyYAML is installed.
_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained; no dependency on other scripts)
# ---------------------------------------------------------------------------

def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    # strip trailing unquoted comment
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    # simple inline list: [a, b, "c"]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts: List[str] = []
        buf = ""
        in_q: Optional[str] = None
        for ch in inner:
            if in_q:
                buf += ch
                if ch == in_q:
                    in_q = None
                continue
            if ch in "\"'":
                in_q = ch
                buf += ch
                continue
            if ch == ",":
                parts.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return [_scalar(p) for p in parts]
    low = v.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous / unrepresentable construct."""


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for the checklist schema.

    Handles: nested mappings by 2-space indent, lists of mappings (`- key: val`
    and dash-on-its-own-line with mapping keys on following indented lines),
    quoted/unquoted scalars, booleans, null, inline `[a, b]` lists, `#` comments,
    and blank lines.  Enough for menu-checklist.yaml — not a full YAML 1.1 parser.

    On constructs that cannot be represented unambiguously, raises MiniYamlError
    instead of silently returning a partial/altered structure (fail closed).
    """
    lines = [
        l
        for l in (raw.rstrip() for raw in text.splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    pos = 0

    def indent_of(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def _is_list_item(s: str) -> bool:
        """True for ``- `` prefix or bare ``-`` (dash alone on the line)."""
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
        """Read nested keys under a list-item mapping (indent > base_indent)."""
        nonlocal pos
        while pos < len(lines):
            nline = lines[pos]
            ni = indent_of(nline)
            if ni <= base_indent:
                break
            ns = nline.strip()
            if _is_list_item(ns):
                break
            if ":" not in ns:
                raise MiniYamlError(
                    f"unrepresentable YAML at line content {ns!r}: "
                    "expected key: value under list item"
                )
            nk, _, nv = ns.partition(":")
            pos += 1
            if nv.strip():
                d[nk.strip()] = _scalar(nv)
            elif pos < len(lines) and indent_of(lines[pos]) > ni:
                d[nk.strip()] = parse_block(indent_of(lines[pos]))
            else:
                d[nk.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal pos
        result: Any = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur < indent:
                break
            if cur > indent:
                raise MiniYamlError(
                    f"orphan deeper indent without a parent key near: {line.strip()!r}"
                )
            s = line.strip()
            if _is_list_item(s):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError(
                        f"mixed mapping/list structure near: {s!r}"
                    )
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    # Dash alone (or empty after dash): nested block on next lines
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_indent = indent_of(lines[pos])
                        nested_s = lines[pos].strip()
                        # Mapping keys on following indented lines (no leading dash)
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(nested_indent))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    # mapping entry on the dash line: `- id: foo`
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    # dash-line sole key: with nested block as its value
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    # plain scalar list item
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(
                        f"mixed list/mapping structure near: {s!r}"
                    )
                if ":" not in s:
                    raise MiniYamlError(
                        f"unrepresentable YAML near: {s!r} (expected key: value)"
                    )
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML text: prefer PyYAML unless forced to mini loader.

    MiniYamlError propagates so callers can exit 2 (fail closed) rather than
    accept a silently truncated structure.
    """
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        data = _yaml.safe_load(text)
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# templating / checklist normalize
# ---------------------------------------------------------------------------

def apply_template(value: str, site: str, menu: str) -> str:
    """Replace <SITE>, <site>, <menu> placeholders in a string."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("<SITE>", site)
    s = s.replace("<site>", site)
    s = s.replace("<menu>", menu)
    return s


def under_root(root: Path, rel: str) -> Path:
    """Join a project-relative path under root (no containment check).

    Important: do NOT use str.lstrip('./') — that strips *any* leading
    ``.`` / ``/`` characters and would turn ``.work/foo`` into ``work/foo``.
    """
    if rel is None:
        return root
    s = str(rel).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if s.startswith("/"):
        # still treat as relative to root for portability (no absolute escape)
        s = s.lstrip("/")
    return root / s


def _realpath(p: Path) -> Path:
    """Canonical absolute path, resolving symlinks (stdlib realpath)."""
    return Path(os.path.realpath(str(p)))


def path_escapes_root(root: Path, candidate: Path) -> bool:
    """True if candidate's real path is outside root's real path."""
    root_r = _realpath(root)
    cand_r = _realpath(candidate)
    try:
        cand_r.relative_to(root_r)
        return False
    except ValueError:
        return True


def contained_path(root: Path, rel: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve rel under root; fail closed if the real path escapes --root.

    Returns (path, None) on success, or (None, reason) on escape/error.
    Reason always includes ``path escapes --root: <rel>`` when applicable.
    """
    if rel is None or str(rel).strip() == "":
        return None, "empty path"
    joined = under_root(root, rel)
    # Fast reject for explicit .. components (also covered by realpath)
    norm = str(rel).replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    parts = Path(norm).parts
    if ".." in parts:
        return None, f"path escapes --root: {rel}"
    if path_escapes_root(root, joined):
        return None, f"path escapes --root: {rel}"
    return joined, None


# Safe site/menu id charset: letters, digits, . _ - / ; must not start with -
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._/][A-Za-z0-9._\-/]*$")


def is_safe_id(value: str) -> bool:
    """True if value is a safe site/menu id (no spaces, no shell metacharacters)."""
    if not value or not isinstance(value, str):
        return False
    if value.startswith("-"):
        return False
    return bool(_SAFE_ID_RE.fullmatch(value))


def _as_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


class ChecklistStructureError(ValueError):
    """Raised when checklist sections/items are malformed (fail closed)."""


def normalize_sections(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ordered list of {key, title, items} from preferred or legacy shape.

    Preferred::
        sections:
          - key: M1
            title: ...
            items: [...]

    Legacy list-of-maps already preferred; also accept::
        sections:
          M1: {title, items}

    Or top-level keys (not ``meta``/``sections``) mapping to {title, items}.

    Fail closed (raises ChecklistStructureError): never silently drop a
    malformed section/item and proceed with survivors that could green.
    """
    if not isinstance(data, dict):
        raise ValueError("checklist root must be a mapping")

    raw = data.get("sections")
    out: List[Dict[str, Any]] = []

    def _section_items(entry: Dict[str, Any], sec_label: str) -> Any:
        """Preserve items for structural validation (do not silently coerce)."""
        items = entry.get("items")
        if items is None:
            return []
        return items

    def _append_section(entry: Any, key_hint: Any = None) -> None:
        if not isinstance(entry, dict):
            label = repr(key_hint) if key_hint is not None else repr(entry)
            raise ChecklistStructureError(
                f"section {label}: must be a mapping with items"
            )
        key = entry.get("key") or entry.get("id") or key_hint or ""
        key_s = str(key) if key is not None else ""
        if not key_s.strip() and key_hint is not None:
            key_s = str(key_hint)
        out.append(
            {
                "key": key_s,
                "title": entry.get("title", key_s),
                "items": _section_items(entry, key_s),
            }
        )

    if isinstance(raw, list):
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ChecklistStructureError(
                    f"section at index {i}: must be a mapping with items "
                    f"(got {type(entry).__name__})"
                )
            _append_section(entry)
        return out

    if isinstance(raw, dict):
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                raise ChecklistStructureError(
                    f"section {key!r}: must be a mapping with items "
                    f"(got {type(entry).__name__})"
                )
            _append_section(entry, key_hint=key)
        return out

    # legacy: top-level section keys
    skip = {"meta", "sections"}
    for key, entry in data.items():
        if key in skip:
            continue
        if not isinstance(entry, dict):
            continue
        if "items" not in entry and "title" not in entry:
            continue
        _append_section(entry, key_hint=key)
    return out


def validate_checklist(sections: List[Dict[str, Any]]) -> Optional[str]:
    """Structural fail-closed validation. Returns error message or None if OK.

    Never allows 0/0 closeable=True from a broken checklist: empty items,
    non-mapping items, missing id/verify, unknown verify, empty/whitespace
    file_grep patterns → all return a config error string for exit 2.
    """
    if not sections:
        return "checklist has no sections/items"

    total = 0
    for sec in sections:
        key = sec.get("key", "?")
        items = sec.get("items")
        if not isinstance(items, list) or len(items) == 0:
            return (
                f"section {key!r}: items must be a non-empty list of mappings"
            )
        for raw in items:
            if not isinstance(raw, dict):
                return (
                    f"section {key!r}: item is not a mapping "
                    f"(got {type(raw).__name__})"
                )
            iid = raw.get("id")
            if iid is None or str(iid).strip() == "":
                return f"section {key!r}: item missing id"
            iid_s = str(iid)
            verify_raw = raw.get("verify")
            if verify_raw is None or str(verify_raw).strip() == "":
                return f"item {iid_s!r} missing verify"
            v = str(verify_raw).strip()
            if v not in KNOWN_VERIFY:
                return f"unknown verify type {v!r} on item {iid_s!r}"
            if v == "file_grep":
                pats = raw.get("patterns")
                empty = (
                    pats is None
                    or (isinstance(pats, list) and len(pats) == 0)
                    or (isinstance(pats, str) and not str(pats).strip())
                )
                if empty:
                    return f"file_grep item {iid_s} has no patterns"
                # FIX A2: any empty/whitespace member vacuous-matches every file
                pat_list = [pats] if isinstance(pats, str) else list(pats)
                for p in pat_list:
                    if p is None or not str(p).strip():
                        return f"file_grep item {iid_s} has an empty pattern"
            total += 1

    if total == 0:
        return "checklist has no evaluable items (0 items)"
    return None


# ---------------------------------------------------------------------------
# per-type evaluators (LIVE under root)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    status: str  # "pass" | "fail"
    reason: str


def _normalize_rel_pattern(pattern: str) -> str:
    s = str(pattern).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if s.startswith("/"):
        s = s.lstrip("/")
    return s


def _is_md_table_data_row(line: str) -> bool:
    """True only for a real markdown table data row (not comment/heading/sep/header).

    Requirements:
    - after strip, starts with ``|``
    - not an HTML comment or markdown heading
    - not a separator row (only pipes, dashes, colons, spaces)
    - not a pure header-like separator-adjacent row with only dash cells
    """
    s = line.strip()
    if not s.startswith("|"):
        return False
    if s.startswith("<!--") or s.startswith("#"):
        return False
    # separator: | --- | :---: | --- |
    if re.fullmatch(r"\|?[\s\-:|]+\|?", s):
        return False
    # cells that are exclusively dashes/colons (separator variants)
    cells = [c.strip() for c in s.strip("|").split("|")]
    if cells and all(re.fullmatch(r":?-{1,}:?", c or "") for c in cells):
        return False
    return True


def eval_file_glob(root: Path, pattern: str) -> EvalResult:
    """PASS if >=1 regular FILE matches glob under root (no path-escape).

    Directories that match the glob do NOT count as evidence (FIX F).
    """
    if not pattern:
        return EvalResult("fail", "file_glob: empty glob pattern")
    rel = _normalize_rel_pattern(pattern)
    # Fail closed on .. / symlink escape before any I/O greening
    if ".." in Path(rel).parts:
        return EvalResult("fail", f"path escapes --root: {rel}")
    # Probe the non-glob prefix path for escape (e.g. ../outside.txt)
    probe = under_root(root, rel)
    if path_escapes_root(root, probe):
        return EvalResult("fail", f"path escapes --root: {rel}")

    matches = sorted(globmod.glob(str(root / rel), recursive=True))
    if not matches:
        try:
            matches = [str(p) for p in root.glob(rel) if p.exists()]
        except Exception:
            matches = []

    safe: List[str] = []
    for m in matches:
        mp = Path(m)
        if path_escapes_root(root, mp):
            return EvalResult("fail", f"path escapes --root: {rel}")
        # FIX F: only regular files count (not directories named like the target)
        if not mp.is_file():
            continue
        safe.append(m)

    if safe:
        return EvalResult("pass", f"found {len(safe)} match(es)")
    return EvalResult("fail", f"no files match glob: {rel}")


def eval_evidence_file(root: Path, pattern: str) -> EvalResult:
    """Semantically human-attached proof; same rule as file_glob (files only)."""
    r = eval_file_glob(root, pattern)
    if r.status == "pass":
        return EvalResult("pass", f"evidence present ({r.reason})")
    return EvalResult("fail", f"evidence missing: {r.reason}")


def eval_file_grep(root: Path, file_rel: str, patterns: Sequence[str]) -> EvalResult:
    """PASS if file exists under root and contains every pattern (substring)."""
    if not file_rel:
        return EvalResult("fail", "file_grep: empty file path")
    if not patterns:
        # Defense in depth — config validation should already exit 2
        return EvalResult("fail", "file_grep: no patterns")
    # FIX A2 defense: empty/whitespace pattern is a substring of everything
    for p in patterns:
        if p is None or not str(p).strip():
            return EvalResult("fail", "file_grep: empty pattern")
    path, esc = contained_path(root, file_rel)
    if esc is not None:
        if esc.startswith("path escapes"):
            return EvalResult("fail", esc)
        return EvalResult("fail", f"file_grep: {esc}")
    assert path is not None
    if not path.is_file():
        return EvalResult("fail", f"file missing: {file_rel}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return EvalResult("fail", f"cannot read {file_rel}: {exc}")
    missing = [p for p in patterns if p not in text]
    if missing:
        return EvalResult(
            "fail",
            f"missing pattern(s): {missing!r} in {file_rel}",
        )
    return EvalResult("pass", f"all {len(patterns)} pattern(s) found in {file_rel}")


def _md_table_cells(line: str) -> List[str]:
    """Split a markdown table row into stripped cell strings.

    Leading/trailing empty splits from the outer ``|`` are dropped so that
    ``| a | b |`` yields ``["a", "b"]``.
    """
    s = line.strip()
    if not s.startswith("|"):
        return []
    # split on |; first/last empties from outer pipes are discarded
    parts = s.split("|")
    cells = [c.strip() for c in parts]
    # drop leading empty (before first |) and trailing empty (after last |)
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def eval_row_in(
    root: Path,
    file_rel: str,
    key: str,
    pass_marker: Optional[str],
) -> EvalResult:
    """PASS if a markdown TABLE data row has an EXACT cell equal to key.

    Only lines that start with ``|`` after strip count. Comments (``<!--``),
    headings (``#``), and table separator rows are ignored.

    FIX E2: key must equal some cell exactly (==), never a substring of a
    longer cell (``menuX`` must not match ``menuX-old``). When pass_marker
    is given, some cell in the same row must contain or equal the marker.
    """
    if not file_rel:
        return EvalResult("fail", "row_in: empty file path")
    if not key:
        return EvalResult("fail", "row_in: empty key")
    path, esc = contained_path(root, file_rel)
    if esc is not None:
        if esc.startswith("path escapes"):
            return EvalResult("fail", esc)
        return EvalResult("fail", f"row_in: {esc}")
    assert path is not None
    if not path.is_file():
        return EvalResult("fail", f"file missing: {file_rel}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return EvalResult("fail", f"cannot read {file_rel}: {exc}")
    for line in lines:
        if not _is_md_table_data_row(line):
            continue
        cells = _md_table_cells(line)
        if key not in cells:
            continue
        if pass_marker is not None:
            marker = str(pass_marker)
            # marker may be exact cell or contained in a cell (emoji status)
            if not any(marker == c or marker in c for c in cells):
                continue
        marker_note = f" + marker {pass_marker!r}" if pass_marker else ""
        return EvalResult("pass", f"row found for key {key!r}{marker_note}")
    if pass_marker is not None:
        return EvalResult(
            "fail",
            f"no row with key {key!r} and marker {pass_marker!r} in {file_rel}",
        )
    return EvalResult("fail", f"no row with key {key!r} in {file_rel}")


def eval_command(
    cmd: str,
    timeout: float,
    cwd: Path,
) -> Tuple[EvalResult, Dict[str, Any]]:
    """Run cmd via shlex.split WITHOUT shell=True. PASS iff exit code 0."""
    extra: Dict[str, Any] = {"cmd": cmd, "exit_code": None, "stdout": "", "stderr": ""}
    if not cmd or not str(cmd).strip():
        return EvalResult("fail", "command: empty cmd"), extra
    try:
        argv = shlex.split(str(cmd))
    except ValueError as exc:
        return EvalResult("fail", f"command: shlex error: {exc}"), extra
    if not argv:
        return EvalResult("fail", "command: empty argv after split"), extra
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        extra["stdout"] = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        extra["stderr"] = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        extra["exit_code"] = None
        extra["timeout"] = True
        return (
            EvalResult("fail", f"command timed out after {timeout}s: {cmd}"),
            extra,
        )
    except FileNotFoundError:
        extra["exit_code"] = 127
        return EvalResult("fail", f"command not found: {argv[0]}"), extra
    except OSError as exc:
        return EvalResult("fail", f"command OS error: {exc}"), extra

    extra["exit_code"] = proc.returncode
    extra["stdout"] = proc.stdout or ""
    extra["stderr"] = proc.stderr or ""
    if proc.returncode == 0:
        return EvalResult("pass", f"command exit 0: {cmd}"), extra
    return (
        EvalResult("fail", f"command exit {proc.returncode}: {cmd}"),
        extra,
    )


def evaluate_item(
    item: Dict[str, Any],
    root: Path,
    site: str,
    menu: str,
    cmd_timeout: float,
) -> Tuple[EvalResult, Dict[str, Any]]:
    """Dispatch one checklist item. Returns (EvalResult, command_extra_or_empty)."""
    verify = str(item.get("verify") or "").strip()
    if verify not in KNOWN_VERIFY:
        # config error — caller maps this to exit 2
        raise ValueError(f"unknown verify type: {verify!r} (item {item.get('id')!r})")

    if verify in ("file_glob", "evidence_file"):
        pattern = apply_template(str(item.get("glob") or ""), site, menu)
        if verify == "file_glob":
            return eval_file_glob(root, pattern), {}
        return eval_evidence_file(root, pattern), {}

    if verify == "file_grep":
        file_rel = apply_template(str(item.get("file") or ""), site, menu)
        raw_pats = item.get("patterns")
        if isinstance(raw_pats, str):
            patterns = [apply_template(raw_pats, site, menu)]
        else:
            patterns = [apply_template(str(p), site, menu) for p in _as_list(raw_pats)]
        return eval_file_grep(root, file_rel, patterns), {}

    if verify == "row_in":
        file_rel = apply_template(str(item.get("file") or ""), site, menu)
        key = apply_template(str(item.get("key") or ""), site, menu)
        marker = item.get("pass_marker")
        if marker is not None:
            marker = apply_template(str(marker), site, menu)
        return eval_row_in(root, file_rel, key, marker), {}

    if verify == "command":
        cmd = apply_template(str(item.get("cmd") or ""), site, menu)
        return eval_command(cmd, timeout=cmd_timeout, cwd=root)

    raise ValueError(f"unknown verify type: {verify!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# evaluation orchestration
# ---------------------------------------------------------------------------

@dataclass
class ItemReport:
    id: str
    verify: str
    status: str
    reason: str
    blocking: bool
    scope: str
    check: str = ""
    command_detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "verify": self.verify,
            "status": self.status,
            "reason": self.reason,
        }
        if self.command_detail:
            d["command"] = self.command_detail
        return d


@dataclass
class SectionReport:
    key: str
    title: str
    items: List[ItemReport] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for i in self.items if i.status == "pass")

    @property
    def count(self) -> int:
        return len(self.items)

    def blocking_reds(self) -> List[ItemReport]:
        return [i for i in self.items if i.status == "fail" and i.blocking]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "pass": self.pass_count,
            "count": self.count,
            "items": [i.as_dict() for i in self.items],
        }


def _item_blocking(item: Dict[str, Any]) -> bool:
    b = item.get("blocking", True)
    if b is None:
        return True
    return bool(b)


def _item_scope(item: Dict[str, Any]) -> str:
    s = str(item.get("scope") or "menu").strip().lower()
    return s if s in ("menu", "site") else "menu"


def evaluate_menu(
    checklist: Dict[str, Any],
    root: Path,
    site: str,
    menu: str,
    cmd_timeout: float,
    include_site_scope: bool = True,
    site_only: bool = False,
) -> List[SectionReport]:
    """Evaluate checklist sections for one menu (and optionally site-scope items).

    If site_only=True, only site-scope items are evaluated (used in --all).
    If include_site_scope=False, site-scope items are skipped (menu-only pass).
    """
    sections = normalize_sections(checklist)
    reports: List[SectionReport] = []

    for sec in sections:
        srep = SectionReport(key=sec["key"], title=str(sec.get("title") or sec["key"]))
        for raw in sec.get("items") or []:
            if not isinstance(raw, dict):
                continue
            scope = _item_scope(raw)
            if site_only and scope != "site":
                continue
            if not include_site_scope and scope == "site":
                continue

            iid = str(raw.get("id") or raw.get("key") or "?")
            verify = str(raw.get("verify") or "").strip()
            blocking = _item_blocking(raw)

            try:
                result, cmd_extra = evaluate_item(
                    raw, root, site, menu, cmd_timeout
                )
            except ValueError:
                raise

            srep.items.append(
                ItemReport(
                    id=iid,
                    verify=verify,
                    status=result.status,
                    reason=result.reason,
                    blocking=blocking,
                    scope=scope,
                    check=str(raw.get("check") or ""),
                    command_detail=cmd_extra or {},
                )
            )
        if srep.items:
            reports.append(srep)
    return reports


def summarize(sections: List[SectionReport]) -> Dict[str, Any]:
    """Live totals from evaluation results (never hardcoded)."""
    items = [i for s in sections for i in s.items]
    p = sum(1 for i in items if i.status == "pass")
    f = sum(1 for i in items if i.status == "fail")
    reds = [i.id for i in items if i.status == "fail" and i.blocking]
    return {
        "total": {"pass": p, "fail": f, "count": len(items)},
        "blocking_red": reds,
        "closeable": len(reds) == 0,
    }


def build_report(
    site: str,
    menu: str,
    sections: List[SectionReport],
) -> Dict[str, Any]:
    summary = summarize(sections)
    return {
        "site": site,
        "menu": menu,
        "total": summary["total"],
        "blocking_red": summary["blocking_red"],
        "sections": [s.as_dict() for s in sections],
        "closeable": summary["closeable"],
    }


def format_human(report: Dict[str, Any], sections: List[SectionReport]) -> str:
    lines: List[str] = []
    for srep in sections:
        reds = srep.blocking_reds()
        if reds:
            red_ids = ", ".join(r.id for r in reds)
            lines.append(
                f"{srep.key}: {srep.pass_count}/{srep.count} "
                f"({len(reds)} blocking red: {red_ids})"
            )
        else:
            lines.append(f"{srep.key}: {srep.pass_count}/{srep.count}")
    total = report["total"]
    n_red = len(report["blocking_red"])
    flag = "YES" if report["closeable"] else "NO"
    lines.append(
        f"menu {report['menu']}: {total['pass']}/{total['count']} "
        f"· blocking red {n_red} · CLOSEABLE={flag}"
    )
    # list reds with reasons
    for srep in sections:
        for item in srep.items:
            if item.status == "fail":
                tag = "blocking" if item.blocking else "non-blocking"
                lines.append(f"  RED [{tag}] {item.id}: {item.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# menu discovery for --all
# ---------------------------------------------------------------------------

def discover_menus(checklist: Dict[str, Any], root: Path, site: str) -> List[str]:
    """Menus from meta.menus, else directory names under harvest path template."""
    meta = checklist.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    menus = meta.get("menus")
    if isinstance(menus, list) and menus:
        return [str(m) for m in menus]

    paths = meta.get("paths") or {}
    harvest = ""
    if isinstance(paths, dict):
        harvest = str(paths.get("harvest") or "")
    if not harvest:
        return []

    # Replace site; treat <menu> segment as wildcard directory
    templ = apply_template(harvest, site, "__MENU__")
    # find the __MENU__ component
    parts = Path(templ).parts
    try:
        idx = parts.index("__MENU__")
    except ValueError:
        # no <menu> placeholder — look at the path as a directory of menus
        base = root / templ.rstrip("/").lstrip("./")
        if base.is_dir():
            return sorted(
                p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
            )
        return []

    parent = root.joinpath(*parts[:idx]) if idx > 0 else root
    if not parent.is_dir():
        return []
    return sorted(
        p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def resolve_checklist(root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else (root / explicit)
        if not p.is_file():
            raise FileNotFoundError(f"checklist not found: {p}")
        return p
    default = root / DEFAULT_CHECKLIST_REL
    if not default.is_file():
        raise FileNotFoundError(
            f"checklist not found: pass --checklist or create {DEFAULT_CHECKLIST_REL} under --root"
        )
    return default


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_SENTINEL = "__ALL_MENUS__"


def _preprocess_argv(argv: Optional[Sequence[str]]) -> List[str]:
    """Allow positional ``--all`` and option-like menu ids (e.g. ``-rf``).

    ``--all`` is rewritten to a non-dashed sentinel so argparse accepts it as
    a positional. Tokens that look like short options but are not known flags
    are kept as positionals when they fill the menu slot, so FIX D validation
    can reject them with exit 2 instead of argparse "unrecognized arguments".
    """
    raw = list(argv) if argv is not None else list(sys.argv[1:])
    value_opts = {"--checklist", "--root", "--cmd-timeout"}
    known_flags = {"--json", "--all", "--help", "-h", *value_opts}

    # First pass: rewrite --all; leave structure intact
    pass1: List[str] = []
    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--all":
            pass1.append(_ALL_SENTINEL)
            i += 1
            continue
        pass1.append(a)
        if a in value_opts and i + 1 < len(raw):
            pass1.append(raw[i + 1])
            i += 2
            continue
        i += 1

    # Second pass: if a dashed token is in a positional slot (site/menu) and is
    # not a known flag, prefix with a private marker so argparse won't eat it.
    # run() strips the marker before validation.
    out: List[str] = []
    positionals_seen = 0
    i = 0
    while i < len(pass1):
        a = pass1[i]
        if a in value_opts and i + 1 < len(pass1):
            out.append(a)
            out.append(pass1[i + 1])
            i += 2
            continue
        if a in known_flags or a.startswith("--"):
            out.append(a)
            i += 1
            continue
        # positional candidate
        if a.startswith("-") and a != _ALL_SENTINEL and positionals_seen < 2:
            # option-like menu/site id — shield from argparse option parsing
            out.append(_DASHED_ID_PREFIX + a)
            positionals_seen += 1
            i += 1
            continue
        if not a.startswith("-") or a == _ALL_SENTINEL:
            positionals_seen += 1
        out.append(a)
        i += 1
    return out


_DASHED_ID_PREFIX = "__DASHED_ID__:"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="menu_gate.py",
        description=(
            "Evaluate a portable menu-close checklist LIVE against the project tree. "
            "Exit 0 if all blocking items pass; 1 if any blocking red; 2 on config error."
        ),
    )
    p.add_argument("site", help="Site id (substituted for <SITE>/<site>)")
    p.add_argument(
        "menu",
        help="Menu id, or --all to evaluate every discovered menu",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object to stdout",
    )
    p.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Path to checklist YAML (default: <root>/.work/menu-checklist.yaml)",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root for live checks (default: cwd)",
    )
    p.add_argument(
        "--cmd-timeout",
        type=float,
        default=DEFAULT_CMD_TIMEOUT,
        metavar="N",
        help=f"Timeout seconds for command verify (default {DEFAULT_CMD_TIMEOUT})",
    )
    return p.parse_args(_preprocess_argv(argv))


def run(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. Returns process exit code."""
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        code = e.code
        if code is None:
            return EXIT_OK
        return int(code) if isinstance(code, int) else EXIT_ERR

    root = (args.root or Path.cwd()).resolve()
    site = args.site
    menu_arg = args.menu
    # unwrap dashed-id shield from _preprocess_argv (FIX D)
    if isinstance(site, str) and site.startswith(_DASHED_ID_PREFIX):
        site = site[len(_DASHED_ID_PREFIX) :]
    if isinstance(menu_arg, str) and menu_arg.startswith(_DASHED_ID_PREFIX):
        menu_arg = menu_arg[len(_DASHED_ID_PREFIX) :]
    if menu_arg == _ALL_SENTINEL:
        menu_arg = "--all"
    cmd_timeout = float(args.cmd_timeout)

    # FIX D: validate site / menu ids before any evaluation (argument injection)
    if not is_safe_id(site):
        print(f"error: invalid site/menu id: {site!r}", file=sys.stderr)
        return EXIT_ERR
    if menu_arg != "--all" and not is_safe_id(menu_arg):
        print(f"error: invalid site/menu id: {menu_arg!r}", file=sys.stderr)
        return EXIT_ERR

    try:
        checklist_path = resolve_checklist(root, args.checklist)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR

    try:
        checklist = load_yaml_file(checklist_path)
    except MiniYamlError as exc:
        print(
            f"error: checklist YAML not representable by mini parser "
            f"({checklist_path}): {exc}",
            file=sys.stderr,
        )
        return EXIT_ERR
    except Exception as exc:
        print(f"error: failed to load checklist {checklist_path}: {exc}", file=sys.stderr)
        return EXIT_ERR

    if not isinstance(checklist, dict):
        print("error: checklist root must be a mapping", file=sys.stderr)
        return EXIT_ERR

    try:
        sections_norm = normalize_sections(checklist)
    except ChecklistStructureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR
    # FIX A + B: structural validation (empty patterns, 0 items, missing id/verify, …)
    v_err = validate_checklist(sections_norm)
    if v_err is not None:
        print(f"error: {v_err}", file=sys.stderr)
        return EXIT_ERR

    if menu_arg == "--all":
        raw_menus = discover_menus(checklist, root, site)
        menus: List[str] = []
        skipped: List[str] = []
        for m in raw_menus:
            if is_safe_id(m):
                menus.append(m)
            else:
                skipped.append(m)
                print(
                    f"warning: skipping invalid menu id {m!r} "
                    f"(not a safe site/menu charset)",
                    file=sys.stderr,
                )
        # FIX D2: any skipped discovered menu means incomplete coverage —
        # cannot certify closeable; exit 2 (config/usage).
        if skipped:
            print(
                f"error: cannot certify: {len(skipped)} menu(s) skipped: "
                f"{', '.join(skipped)}",
                file=sys.stderr,
            )
            return EXIT_ERR
        if not menus:
            print(
                "error: --all found no menus "
                "(set meta.menus or create harvest/<menu>/ directories)",
                file=sys.stderr,
            )
            return EXIT_ERR

        # site-scope once
        try:
            site_sections = evaluate_menu(
                checklist,
                root,
                site,
                menu="__site__",
                cmd_timeout=cmd_timeout,
                include_site_scope=True,
                site_only=True,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERR

        menu_reports: List[Dict[str, Any]] = []
        menu_section_map: Dict[str, List[SectionReport]] = {}
        all_sections_for_total: List[SectionReport] = list(site_sections)

        site_report = build_report(site, "__site__", site_sections)

        for m in menus:
            try:
                msec = evaluate_menu(
                    checklist,
                    root,
                    site,
                    menu=m,
                    cmd_timeout=cmd_timeout,
                    include_site_scope=False,
                    site_only=False,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_ERR
            # combine menu + site for per-menu closeable decision
            combined = site_sections + msec
            rep = build_report(site, m, combined)
            menu_reports.append(rep)
            menu_section_map[m] = combined
            all_sections_for_total.extend(msec)

        # aggregate totals: site once + every menu item (no double-count site)
        agg = summarize(all_sections_for_total)
        # Fail closed: never green on zero evaluated items
        if agg["total"]["count"] == 0:
            print("error: checklist evaluation produced 0 items", file=sys.stderr)
            return EXIT_ERR
        blocking_red = list(agg["blocking_red"])
        closeable = len(blocking_red) == 0

        aggregate = {
            "site": site,
            "menu": "--all",
            "menus": menu_reports,
            "site_scope": site_report,
            "total": agg["total"],
            "blocking_red": blocking_red,
            "closeable": closeable,
            "sections": [s.as_dict() for s in all_sections_for_total],
        }

        if args.json:
            print(json.dumps(aggregate, ensure_ascii=False))
        else:
            print(f"site {site} — evaluating {len(menus)} menu(s)")
            if site_sections:
                print("--- site-scope (once) ---")
                print(format_human(site_report, site_sections))
            for mr in menu_reports:
                m = mr["menu"]
                print(f"--- menu {m} ---")
                print(format_human(mr, menu_section_map[m]))
            total = aggregate["total"]
            flag = "YES" if aggregate["closeable"] else "NO"
            print(
                f"TOTAL --all: {total['pass']}/{total['count']} "
                f"· blocking red {len(aggregate['blocking_red'])} · CLOSEABLE={flag}"
            )

        return EXIT_OK if aggregate["closeable"] else EXIT_RED

    # single menu
    menu = menu_arg
    try:
        sections = evaluate_menu(
            checklist,
            root,
            site,
            menu=menu,
            cmd_timeout=cmd_timeout,
            include_site_scope=True,
            site_only=False,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERR

    report = build_report(site, menu, sections)
    # Fail closed: never green on zero evaluated items
    if report["total"]["count"] == 0:
        print("error: checklist evaluation produced 0 items", file=sys.stderr)
        return EXIT_ERR

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(format_human(report, sections))

    return EXIT_OK if report["closeable"] else EXIT_RED


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
