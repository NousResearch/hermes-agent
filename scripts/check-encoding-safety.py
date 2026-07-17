#!/usr/bin/env python3
"""
AST-based checker for encoding-safety footguns on user-writable files.

Flags patterns that silently drop, mangle, or permanently rewrite keys when
a user edits Hermes state with a Windows/Notepad/PowerShell encoding
(UTF-8 BOM, UTF-16 "Unicode", cp1252). Sibling of check-windows-footguns.py.

Usage:
    # Scan staged changes (default when run from a git checkout)
    python scripts/check-encoding-safety.py

    # Scan the full tree (full-repo audit)
    python scripts/check-encoding-safety.py --all

    # Scan a specific file or directory
    python scripts/check-encoding-safety.py path/to/file.py path/to/dir/

    # Scan only modified files vs. main
    python scripts/check-encoding-safety.py --diff main

Exit status:
    0 — no encoding-safety issues found (or all matches suppressed)
    1 — at least one unsuppressed match

Suppress an intentional use with:
    load_dotenv(path, encoding="utf-8")  # encoding-safety: ok — reason

Rules (derived from the env-class regression class):
    R1  open()/read_text()/load_dotenv()/dotenv_values() with encoding="utf-8"
        (not utf-8-sig) on a USER-WRITABLE path — BOM sticks to first key.
    R2  decode fallback chains that re-decode as latin-1 without BOM-stripping
        (sanitize/latin-1 shape) — UTF-8 BOM survives as U+FEFF on the key.
    R3  errors="replace" decodes whose output is WRITTEN BACK to disk
        (destructive-rewrite shape) — UTF-16 BOM becomes U+FFFD + permanent
        rewrite of the user's file.

"User-writable" is an explicit allowlist (not a heuristic): basenames and
path fragments Hermes documents as user-edited state under HERMES_HOME /
managed scope. Expand the allowlist deliberately; false-positive scope is
a config choice.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPRESS_MARKER = re.compile(r"#\s*encoding-safety\s*:\s*ok\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# User-writable allowlist
# ---------------------------------------------------------------------------
# Derived from website/docs/user-guide/configuration.md "Directory Structure"
# plus managed-scope admin files and plugin configs users/admins edit.
# Keep this explicit — expanding it is a deliberate product decision.

USER_WRITABLE_BASENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".op.env",
        "config.yaml",
        "config.yml",
        "auth.json",
        "SOUL.md",
        "MEMORY.md",
        "USER.md",
        "SKILL.md",
        "plugin.yaml",
        "plugin.yml",
        "jobs.json",
    }
)

# Substring matches against the source of a path expression (string literal
# contents, f-string chunks, or variable names). Lower-cased comparison.
USER_WRITABLE_PATH_FRAGMENTS: tuple[str, ...] = (
    ".env",
    "config.yaml",
    "config.yml",
    "auth.json",
    "soul.md",
    "memory.md",
    "user.md",
    "skill.md",
    "plugin.yaml",
    "plugin.yml",
    "jobs.json",
    "/memories",
    "\\memories",
    "/skills",
    "\\skills",
    "/cron",
    "\\cron",
    "memories/",
    "skills/",
    "cron/",
    ".op.env",
)

# Variable / attribute names that almost always hold a user-writable path
# in this codebase. Lower-cased.
USER_WRITABLE_VAR_HINTS: tuple[str, ...] = (
    "env_path",
    "env_file",
    "dotenv_path",
    "user_env",
    "project_env",
    "managed_env",
    "op_env",
    "config_path",
    "config_file",
    "config_yaml_path",
    "yaml_path",  # only when also .yaml/.yml content — still a config hint
    "managed_config",
    "soul_path",
    "auth_path",
    "auth_file",
    "memory_path",
    "memory_file",
    "skill_path",
    "skill_md",
    "skill_file",
    "found_skill_md",
    "cron_path",
    "jobs_path",
    "jobs_file",
    "plugin_yaml",
    "plugin_config",
)

# Functions that load dotenv files — their target is always an env file,
# so R1 applies whenever encoding="utf-8" is used (path allowlist implied).
DOTENV_FUNCS: frozenset[str] = frozenset({"load_dotenv", "dotenv_values"})

# Text-open call names we inspect for encoding=.
TEXT_OPEN_FUNCS: frozenset[str] = frozenset({"open", "read_text"})

# Encoding literals considered "plain utf-8" (BOM-blind). Not utf-8-sig.
_PLAIN_UTF8 = frozenset(
    {
        "utf-8",
        "utf8",
        "UTF-8",
        "UTF8",
        "Utf-8",
    }
)

# Encoding that correctly strips a UTF-8 BOM.
_UTF8_SIG = frozenset(
    {
        "utf-8-sig",
        "utf8-sig",
        "UTF-8-SIG",
        "UTF8-SIG",
    }
)

# latin-1 / iso-8859-1 fallback encodings (never strip a UTF-8 BOM).
_LATIN1 = frozenset(
    {
        "latin-1",
        "latin1",
        "iso-8859-1",
        "iso8859-1",
        "LATIN-1",
        "LATIN1",
        "ISO-8859-1",
    }
)

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    "website/build",
    "optional-skills",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".png",
    ".jpg",
    ".gif",
    ".ico",
    ".svg",
    ".mp4",
    ".mp3",
    ".wav",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".lock",
    ".min.js",
    ".min.css",
}

EXCLUDED_FILES = {
    "scripts/check-encoding-safety.py",
    "scripts/check-windows-footguns.py",
    "CONTRIBUTING.md",
}


@dataclass(frozen=True)
class Finding:
    rule: str
    lineno: int
    col: int
    message: str
    fix: str
    line_text: str = ""


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.Call) -> str | None:
    """Return the trailing attribute/name of a Call (open, read_text, load_dotenv)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kw(node: ast.Call, name: str) -> ast.AST | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _dict_str_entries(node: ast.AST | None) -> dict[str, str]:
    """Extract {str: str} entries from a Dict literal (best-effort)."""
    out: dict[str, str] = {}
    if not isinstance(node, ast.Dict):
        return out
    for k, v in zip(node.keys, node.values):
        ks = _literal_str(k)
        vs = _literal_str(v)
        if ks is not None and vs is not None:
            out[ks] = vs
    return out


def _resolve_kwargs_dict(call: ast.Call, func_node: ast.AST | None) -> dict[str, str]:
    """Resolve open(path, **read_kw) style dicts assigned in the enclosing function.

    Handles the historical sanitize shape::

        read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
        with open(path, **read_kw) as f:
            ...
    """
    merged: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Name) and func_node is not None:
            name = kw.value.id
            # Walk assignments in the enclosing function (simple Name targets).
            for stmt in ast.walk(func_node):
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name) and t.id == name:
                            merged.update(_dict_str_entries(stmt.value))
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == name and stmt.value is not None:
                        merged.update(_dict_str_entries(stmt.value))
        elif kw.arg is not None:
            lit = _literal_str(kw.value)
            if lit is not None:
                merged[kw.arg] = lit
    return merged


def _call_kwargs(call: ast.Call, func_node: ast.AST | None = None) -> dict[str, str]:
    """Keyword string literals on a Call, including resolved **dict splats."""
    out: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            lit = _literal_str(kw.value)
            if lit is not None:
                out[kw.arg] = lit
    # Overlay **dict resolution (does not override explicit keywords).
    resolved = _resolve_kwargs_dict(call, func_node)
    for k, v in resolved.items():
        out.setdefault(k, v)
    return out


def _encoding_literal(node: ast.Call, func_node: ast.AST | None = None) -> str | None:
    return _call_kwargs(node, func_node).get("encoding")


def _errors_literal(node: ast.Call, func_node: ast.AST | None = None) -> str | None:
    return _call_kwargs(node, func_node).get("errors")


def _is_path_open_method(node: ast.Call) -> bool:
    """True for ``path.open(...)`` / ``Path(...).open(...)`` (mode is arg 0).

    Distinct from builtins ``open(path, mode)`` where mode is arg 1.
    """
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"os", "io", "builtins"}
        )
    )


def _open_mode(node: ast.Call) -> str | None:
    """Return the mode string for open()/Path.open(), or None if unspecified."""
    mode_kw = _literal_str(_kw(node, "mode"))
    if mode_kw is not None:
        return mode_kw
    if _is_path_open_method(node):
        # Path.open(mode='r', ...) — mode is first positional if present
        if node.args:
            return _literal_str(node.args[0])
        return None
    # builtins.open(path, mode, ...)
    if len(node.args) >= 2:
        return _literal_str(node.args[1])
    return None


def _is_text_open(node: ast.Call) -> bool:
    """True if this open()/Path.open() is text mode (not binary)."""
    name = _call_name(node)
    if name == "read_text":
        return True
    if name not in {"open"}:
        return False
    mode = _open_mode(node)
    if mode is None:
        return True  # default is text
    return "b" not in mode


def _is_read_open(node: ast.Call) -> bool:
    """True if this is a text *read* (R1 only cares about reads, not writes).

    Writing with encoding='utf-8' is correct and desirable — we must not flag
    ``open(path, "w", encoding="utf-8")``.
    """
    name = _call_name(node)
    if name == "read_text":
        return True
    if name == "open":
        if not _is_text_open(node):
            return False
        mode = _open_mode(node)
        if mode is None:
            return True  # default 'r'
        # Any mode containing 'r' (r, rt, r+, etc.) is a read; pure w/a/x are not.
        if "r" in mode:
            return True
        return False
    return False


def _path_expr_source(node: ast.AST | None, source: str) -> str:
    """Best-effort source text of a path expression for allowlist matching."""
    if node is None:
        return ""
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def _path_looks_user_writable(path_src: str) -> bool:
    """True if path_src mentions an allowlisted user-writable location."""
    if not path_src:
        return False
    lowered = path_src.lower()
    # Strip quotes for basename check on pure string literals.
    bare = path_src.strip().strip("'\"")
    if bare in USER_WRITABLE_BASENAMES or bare.split("/")[-1] in USER_WRITABLE_BASENAMES:
        return True
    if any(frag in lowered for frag in USER_WRITABLE_PATH_FRAGMENTS):
        return True
    # Variable-name hints: env_path, config_path, ...
    # Match as whole identifier-ish tokens inside the expression.
    for hint in USER_WRITABLE_VAR_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", lowered):
            return True
    return False


def _dotenv_path_node(node: ast.Call) -> ast.AST | None:
    """Path argument of load_dotenv / dotenv_values."""
    # dotenv_path= or first positional
    for name in ("dotenv_path", "path"):
        n = _kw(node, name)
        if n is not None:
            return n
    if node.args:
        return node.args[0]
    return None


def _open_path_node(node: ast.Call) -> ast.AST | None:
    name = _call_name(node)
    if name == "read_text":
        # Path.read_text() — receiver is the path
        if isinstance(node.func, ast.Attribute):
            return node.func.value
        return None
    if name == "open":
        # builtins.open(path) or Path.open() — if Attribute, receiver is path
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
        ):
            # path.open(...) form
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {
                "os",
                "io",
                "builtins",
            }:
                # Heuristic: Path.open has no path positional; open(path) does.
                if not node.args or _literal_str(node.args[0]) in {None, "r", "rt", "w", "a", "x"}:
                    # likely path.open(mode=...)
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in {
                        "open",
                    }:
                        pass
                    else:
                        # Prefer first arg if it looks like a path, else receiver
                        if node.args and _literal_str(node.args[0]) not in {
                            "r",
                            "rt",
                            "w",
                            "a",
                            "x",
                            "rb",
                            "wb",
                        }:
                            return node.args[0]
                        return node.func.value
        if node.args:
            return node.args[0]
        file_kw = _kw(node, "file")
        if file_kw is not None:
            return file_kw
    return None


def _line_suppressed(source_lines: list[str], lineno: int) -> bool:
    if lineno < 1 or lineno > len(source_lines):
        return False
    return bool(SUPPRESS_MARKER.search(source_lines[lineno - 1]))


def _has_bom_strip_nearby(text: str) -> bool:
    """True if the fallback block strips a UTF-8 BOM before re-decoding."""
    markers = (
        "BOM_UTF8",
        "utf-8-sig",
        "utf8_sig",
        "\\ufeff",
        "\ufeff",
        "codecs.BOM",
        "lstrip('\\ufeff')",
        'lstrip("\\ufeff")',
        "startswith(codecs.BOM",
        "startswith(b'\\xef\\xbb\\xbf')",
        'startswith(b"\\xef\\xbb\\xbf")',
    )
    return any(m in text for m in markers)


# ---------------------------------------------------------------------------
# Rule scanners (AST)
# ---------------------------------------------------------------------------

# Tight name patterns — deliberately NOT bare "env"/"memory"/"auth"/"plugin"
# (those match pyvenv, memory_limit, oauth_credentials, _discover_plugins and
# create a flood of false positives). Prefer path allowlist + caller analysis;
# function-name hints are a last resort for helpers like _cached_read(path).
_ENVISH_FUNC_RE = re.compile(
    r"(?:"
    r"cached_read|"
    r"load_dotenv|dotenv_values|"
    r"load_hermes_env|_load_hermes_env|load_env|save_env|set_env|write_env|"
    r"sanitize_env|_sanitize_env|"
    r"load_config|save_config|load_managed|apply_managed|"
    r"save_env_value|set_env_value|load_env_value|"
    r"memory_setup|_write_env_vars|_write_env\b|"
    r"quote_env"
    r")",
    re.IGNORECASE,
)


def _func_suggests_user_writable(func: ast.AST) -> bool:
    name = getattr(func, "name", "") or ""
    return bool(_ENVISH_FUNC_RE.search(name))


def _enclosing_function_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map call/handler node id → innermost enclosing FunctionDef."""
    mapping: dict[int, ast.AST] = {}

    def visit(node: ast.AST, func: ast.AST | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
        mapping[id(node)] = func  # type: ignore[assignment]
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(tree, None)
    return mapping


def _function_param_names(func: ast.AST | None) -> set[str]:
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    names: set[str] = set()
    args = func.args
    for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        names.add(a.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _callers_pass_user_writable(
    tree: ast.AST, func_name: str, param_index: int, source: str
) -> bool:
    """True if any same-module call to func_name passes a user-writable path
    at the given positional index (or as the matching keyword).

    Covers helpers like managed_scope._cached_read(path, ...) where the open()
    inside only sees the bare parameter name, but callers pass managed/.env.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != func_name:
            continue
        path_node = None
        if len(node.args) > param_index:
            path_node = node.args[param_index]
        if path_node is None:
            continue
        if _path_looks_user_writable(_path_expr_source(path_node, source)):
            return True
    return False


def _scan_r1(tree: ast.AST, source: str, source_lines: list[str]) -> list[Finding]:
    """R1: plain utf-8 (not utf-8-sig) on user-writable text reads / dotenv loads."""
    findings: list[Finding] = []
    enclosing = _enclosing_function_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        func_node = enclosing.get(id(node))
        enc = _encoding_literal(node, func_node)
        if enc is None or enc not in _PLAIN_UTF8:
            continue
        if name not in DOTENV_FUNCS and not _is_read_open(node):
            continue

        if name in DOTENV_FUNCS:
            # dotenv always targets an env file — R1 applies unconditionally
            # when encoding is plain utf-8.
            path_node = _dotenv_path_node(node)
            path_src = _path_expr_source(path_node, source)
            rule_detail = "load_dotenv/dotenv_values"
        elif name in TEXT_OPEN_FUNCS or (
            isinstance(node.func, ast.Attribute) and node.func.attr in {"open", "read_text"}
        ):
            path_node = _open_path_node(node)
            path_src = _path_expr_source(path_node, source)
            user_writable = _path_looks_user_writable(path_src)
            # Bare parameter forwarded from a caller that passes .env / config.yaml
            if (
                not user_writable
                and isinstance(path_node, ast.Name)
                and isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and path_node.id in _function_param_names(func_node)
            ):
                # Assume first path-like param is index 0 for helpers like
                # _cached_read(path, cache, parse).
                params = [
                    a.arg
                    for a in list(func_node.args.posonlyargs) + list(func_node.args.args)
                ]
                try:
                    idx = params.index(path_node.id)
                except ValueError:
                    idx = 0
                if _callers_pass_user_writable(tree, func_node.name, idx, source):
                    user_writable = True
                    path_src = path_src or f"{func_node.name}({path_node.id}=…user-writable)"
            if not user_writable and func_node is not None and _func_suggests_user_writable(
                func_node
            ):
                # Function name itself is env/config/managed-ish and opens with
                # plain utf-8 — flag even when the path is a bare parameter.
                user_writable = True
            if not user_writable:
                continue
            rule_detail = f"{name}()"
        else:
            continue

        if _line_suppressed(source_lines, node.lineno):
            continue

        findings.append(
            Finding(
                rule="R1",
                lineno=node.lineno,
                col=node.col_offset,
                message=(
                    f"{rule_detail} uses encoding={enc!r} on user-writable path "
                    f"({path_src or 'dotenv target'}). A UTF-8 BOM (Notepad / "
                    f"PowerShell 5.1 Set-Content -Encoding UTF8) sticks to the "
                    f"first key name as U+FEFF and silently drops it from "
                    f"os.environ under its canonical name."
                ),
                fix=(
                    'Use encoding="utf-8-sig" (strips a leading BOM, no-op for '
                    "BOM-less UTF-8)."
                ),
                line_text=source_lines[node.lineno - 1].rstrip()
                if 0 < node.lineno <= len(source_lines)
                else "",
            )
        )
    return findings


def _scan_r2(tree: ast.AST, source: str, source_lines: list[str]) -> list[Finding]:
    """R2: UnicodeDecodeError → latin-1 re-decode without BOM strip."""
    findings: list[Finding] = []
    enclosing = _enclosing_function_map(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Match `except UnicodeDecodeError` (or tuple including it)
        if not _handler_catches_unicode_decode(node):
            continue
        body_src = _handler_body_source(node, source)
        if not body_src:
            continue
        # Look for latin-1 re-decode / load_dotenv(encoding="latin-1")
        has_latin1 = False
        func_node = enclosing.get(id(node))
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                enc = _encoding_literal(child, func_node)
                if enc and enc in _LATIN1:
                    has_latin1 = True
                    break
                # .decode("latin-1")
                if (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "decode"
                    and child.args
                ):
                    lit = _literal_str(child.args[0])
                    if lit and lit in _LATIN1:
                        has_latin1 = True
                        break
        if not has_latin1:
            continue
        if _has_bom_strip_nearby(body_src):
            continue
        if _line_suppressed(source_lines, node.lineno):
            continue
        findings.append(
            Finding(
                rule="R2",
                lineno=node.lineno,
                col=node.col_offset,
                message=(
                    "UnicodeDecodeError fallback re-decodes as latin-1 without "
                    "stripping a leading UTF-8 BOM. The BOM survives as U+FEFF "
                    "on the first key (same silent drop as R1 on the primary path)."
                ),
                fix=(
                    "Before latin-1 decode: if raw.startswith(codecs.BOM_UTF8): "
                    "raw = raw[len(codecs.BOM_UTF8):]  — or prefer encoding="
                    '"utf-8-sig" on the primary path so this fallback is rare.'
                ),
                line_text=source_lines[node.lineno - 1].rstrip()
                if 0 < node.lineno <= len(source_lines)
                else "",
            )
        )
    return findings


def _handler_catches_unicode_decode(node: ast.ExceptHandler) -> bool:
    if node.type is None:
        return False
    if isinstance(node.type, ast.Name):
        return node.type.id == "UnicodeDecodeError"
    if isinstance(node.type, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id == "UnicodeDecodeError"
            for elt in node.type.elts
        )
    return False


def _handler_body_source(node: ast.ExceptHandler, source: str) -> str:
    chunks: list[str] = []
    for stmt in node.body:
        seg = ast.get_source_segment(source, stmt)
        if seg:
            chunks.append(seg)
    return "\n".join(chunks)


def _scan_r3(tree: ast.AST, source: str, source_lines: list[str]) -> list[Finding]:
    """R3: errors='replace' read of user-writable file written back to disk.

    Detects the destructive-rewrite shape: a function reads with
    errors='replace' and later writes to the same path. On UTF-16-BOM
    input, utf-8-sig+replace yields U+FFFD prefixes that get permanently
    committed to the user's .env. Handles both inline kwargs and the
    historical ``open(path, **read_kw)`` dict-splat form.
    """
    findings: list[Finding] = []

    for func in _iter_functions(tree):
        reads: list[tuple[ast.Call, str]] = []  # (call, path_src)
        writes: list[tuple[ast.AST, str]] = []
        envish = _func_suggests_user_writable(func)

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            # Read side: open/read_text with errors="replace"
            if name in {"open", "read_text"} and _is_text_open(node):
                err = _errors_literal(node, func)
                if err == "replace":
                    path_node = _open_path_node(node)
                    path_src = _path_expr_source(path_node, source)
                    if _path_looks_user_writable(path_src) or envish:
                        reads.append((node, path_src))
            # Write side: write_text, open(mode w/a), os.fdopen(... "w"),
            # atomic_replace, writelines
            if name == "write_text":
                path_node = (
                    node.func.value
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                writes.append((node, _path_expr_source(path_node, source)))
            elif name == "open" and _is_write_mode(node):
                path_node = _open_path_node(node)
                writes.append((node, _path_expr_source(path_node, source)))
            elif name == "fdopen":
                mode = None
                if len(node.args) >= 2:
                    mode = _literal_str(node.args[1])
                if mode is None:
                    mode = _literal_str(_kw(node, "mode"))
                # os.fdopen(fd, "w", encoding=...) — mode may be positional
                # with encoding as kw-only after. Treat text writes when mode
                # is w/a/x, or when encoding= is present without binary mode
                # (common mkstemp+fdopen pattern).
                kwargs = _call_kwargs(node, func)
                if (mode and any(c in mode for c in "wax")) or (
                    mode is None and kwargs.get("encoding")
                ):
                    if mode is None or "b" not in (mode or ""):
                        writes.append((node, _path_expr_source(node, source)))
            elif name == "atomic_replace":
                if len(node.args) >= 2:
                    writes.append(
                        (node, _path_expr_source(node.args[1], source))
                    )
            elif name == "writelines":
                writes.append((node, ""))
            elif name == "replace" and isinstance(node.func, ast.Attribute):
                # os.replace(tmp, path)
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and len(node.args) >= 2
                ):
                    writes.append(
                        (node, _path_expr_source(node.args[1], source))
                    )

        if not reads or not writes:
            continue

        for read_call, read_path in reads:
            if _line_suppressed(source_lines, read_call.lineno):
                continue
            # Co-occurrence in same function is enough for the
            # tempfile + atomic_replace dance (write path often a tmp name).
            findings.append(
                Finding(
                    rule="R3",
                    lineno=read_call.lineno,
                    col=read_call.col_offset,
                    message=(
                        f"errors='replace' read of user-writable path "
                        f"({read_path or 'user file'}) whose result is written "
                        f"back to disk in the same function. On UTF-16-BOM "
                        f"input (Notepad 'Unicode'), undecodable bytes become "
                        f"U+FFFD and the rewrite permanently corrupts the "
                        f"first key."
                    ),
                    fix=(
                        "Sniff BOM bytes before decoding (UTF-32 before UTF-16 "
                        "before UTF-8). Refuse-to-mangle on unknown binary; "
                        "decode UTF-16 correctly; never persist U+FFFD-prefixed "
                        "content."
                    ),
                    line_text=source_lines[read_call.lineno - 1].rstrip()
                    if 0 < read_call.lineno <= len(source_lines)
                    else "",
                )
            )
    return findings


def _iter_functions(tree: ast.AST) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _is_write_mode(node: ast.Call) -> bool:
    mode = None
    if len(node.args) >= 2:
        mode = _literal_str(node.args[1])
    if mode is None:
        mode = _literal_str(_kw(node, "mode"))
    if mode is None:
        return False
    return any(c in mode for c in "wax")


# ---------------------------------------------------------------------------
# File I/O + CLI (mirrors check-windows-footguns.py)
# ---------------------------------------------------------------------------


def should_scan_file(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS:
        return False
    for suffix in EXCLUDED_SUFFIXES:
        if str(path).endswith(suffix):
            return False
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    if rel in EXCLUDED_FILES:
        return False
    # Fixture trees for this checker live under tests/fixtures/encoding_safety/
    # and are scanned only when paths are passed explicitly — skip under --all.
    if "tests/fixtures/encoding_safety" in rel.replace("\\", "/"):
        return False
    if path.suffix in {".py", ".pyw", ".pyi"}:
        return True
    return False


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            if should_scan_file(p) or _is_explicit_fixture(p):
                yield p
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
                for fname in files:
                    fpath = Path(root) / fname
                    if should_scan_file(fpath) or _is_explicit_fixture(fpath):
                        yield fpath


def _is_explicit_fixture(path: Path) -> bool:
    """Allow scanning fixture files when the user passed their path explicitly."""
    s = str(path).replace("\\", "/")
    return path.suffix == ".py" and "encoding_safety" in s


def scan_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    findings: list[Finding] = []
    findings.extend(_scan_r1(tree, source, source_lines))
    findings.extend(_scan_r2(tree, source, source_lines))
    findings.extend(_scan_r3(tree, source, source_lines))
    # Stable order
    findings.sort(key=lambda f: (f.lineno, f.col, f.rule))
    return findings


def get_staged_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [REPO_ROOT / f for f in out.splitlines() if f.strip()]


def get_diff_files(ref: str) -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", f"{ref}...HEAD", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [REPO_ROOT / f for f in out.splitlines() if f.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Flag encoding-safety footguns on user-writable Hermes state files."
        )
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific files/dirs to scan (default: staged changes).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Scan the full repository Python packages + scripts.",
    )
    p.add_argument(
        "--diff",
        metavar="REF",
        help="Scan files changed vs. the given git ref (e.g. --diff main).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all known encoding-safety rules and exit.",
    )
    return p.parse_args(argv)


def print_rules() -> None:
    print("Known encoding-safety rules checked by this script:\n")
    rules = [
        (
            "R1",
            'open()/read_text()/load_dotenv()/dotenv_values() with encoding="utf-8" '
            "(not utf-8-sig) on a user-writable path",
            'encoding="utf-8-sig"',
        ),
        (
            "R2",
            "UnicodeDecodeError → latin-1 re-decode without stripping a UTF-8 BOM",
            "Strip codecs.BOM_UTF8 before latin-1; prefer utf-8-sig primary",
        ),
        (
            "R3",
            'errors="replace" decode of a user-writable file written back to disk',
            "BOM-sniff (UTF-32→UTF-16→UTF-8); refuse-to-mangle; never persist U+FFFD",
        ),
    ]
    for name, msg, fix in rules:
        print(f"  {name}: {msg}")
        print(f"      Fix: {fix}")
        print()
    print("User-writable allowlist (explicit; expand deliberately):")
    print(f"  basenames: {', '.join(sorted(USER_WRITABLE_BASENAMES))}")
    print(f"  path fragments: {', '.join(USER_WRITABLE_PATH_FRAGMENTS)}")
    print()
    print("Suppress:  # encoding-safety: ok — reason")


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)

    if args.list:
        print_rules()
        return 0

    if args.all:
        roots = [
            REPO_ROOT / "hermes_cli",
            REPO_ROOT / "gateway",
            REPO_ROOT / "tools",
            REPO_ROOT / "cron",
            REPO_ROOT / "agent",
            REPO_ROOT / "plugins",
            REPO_ROOT / "scripts",
            REPO_ROOT / "acp_adapter",
            REPO_ROOT / "acp_registry",
            REPO_ROOT / "tui_gateway",
        ]
        roots = [r for r in roots if r.exists()]
    elif args.diff:
        roots = get_diff_files(args.diff)
    elif args.paths:
        roots = [p.resolve() for p in args.paths]
    else:
        roots = get_staged_files()
        if not roots:
            print(
                "No staged files to scan. Pass --all for a full-repo scan, "
                "--diff <ref> for a range diff, or paths explicitly.",
                file=sys.stderr,
            )
            return 0

    total_matches = 0
    files_scanned = 0
    for path in iter_files(roots):
        files_scanned += 1
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        for finding in scan_file(path):
            print(f"{rel}:{finding.lineno}: [{finding.rule}]")
            if finding.line_text:
                print(f"    {finding.line_text.strip()}")
            print(f"    — {finding.message}")
            print(f"    Fix: {finding.fix.splitlines()[0]}")
            print()
            total_matches += 1

    if total_matches:
        print(
            f"\n✗ {total_matches} encoding-safety issue(s) found across "
            f"{files_scanned} file(s) scanned.",
            file=sys.stderr,
        )
        print(
            "  If an individual match is intentional, suppress it with "
            "`# encoding-safety: ok — reason` on the same line.\n"
            "  Run with --list to see all rules and the user-writable allowlist.",
            file=sys.stderr,
        )
        return 1

    print(
        f"✓ No encoding-safety issues found ({files_scanned} file(s) scanned)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
