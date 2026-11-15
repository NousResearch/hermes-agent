#!/usr/bin/env python3
"""Fail when new code reaches a profile's ``.env`` or ``auth.json`` file directly.

Every read and write of a profile's secrets goes through the selected secret store
(``hermes_secret_routing``: ``read_profile_env``, ``read_env_text``, ``route_env_write``,
``read_auth_json`` and the ``hermes_cli.auth`` store load/save, ``save_env_value`` /
``remove_env_value``). A store that keeps secrets off disk only sees what passes through there: a
new ``open(home / ".env")`` or ``json.loads((home / "auth.json").read_text())`` silently keeps
reading the local file, which then holds the store's bootstrap entries only.

Counted, per file, in the scanned trees:

* a path built to one of those files: ``<expr> / ".env"``, ``<expr> / "auth.json"``, and
  ``Path(...)`` / ``os.path.join(...)`` / ``.joinpath(...)`` / ``.with_name(...)`` / ``.file(...)``
  with either name as an argument;
* a call of a helper that returns one: ``get_env_path()``, ``_auth_file_path()``,
  ``_global_auth_file_path()``, ``auth_json_path()``, ``_auth_json_path()``.

A file name in a list of names (a deny list, a marker set) is not a path and is not counted.

What legitimately stays is ratcheted in ``scripts/secret_file_access_allowlist.json``: per file,
the number of such sites and why they may stay (the local backend's own file code; a path shown
to the user; a marker or deny-list check; file tooling gated on the secret store's
``supports_file_tooling``; process-global names that never leave the local file; another tool's
own file). A file with more sites than allowed fails; a file with fewer fails too, asking for the
allowlist to be lowered, so a moved site cannot be re-added unnoticed. Suppress one true false
positive with ``# secret-file: ok — <why>`` on its line.

Usage: python3 scripts/check_secret_file_access.py [--write-allowlist]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "scripts" / "secret_file_access_allowlist.json"
TREES = ("hermes_cli", "agent", "gateway", "tui_gateway", "cron", "plugins", "tools", "acp_adapter", "pm")
SKIP_PARTS = {"tests", "__pycache__", "node_modules", "vendor"}
# The router itself and the local backend: the one place each path is built on purpose.
EXEMPT = {"hermes_secret_routing.py", "hermes_secret_port_local.py"}
SECRET_FILES = {".env", "auth.json"}
PATH_HELPERS = {"get_env_path", "_auth_file_path", "_global_auth_file_path", "auth_json_path", "_auth_json_path"}
PATH_BUILDERS = {"Path", "PurePath", "join", "joinpath", "with_name", "file"}
# The routed readers and writers: a path handed straight to one of them goes through the store.
ROUTED = {"read_profile_env", "read_env_text", "env_file_exists", "route_env_write", "routed_bootstrap_env",
          "read_auth_json", "auth_json_exists", "_load_auth_store", "_save_auth_store"}
SUPPRESS_RE = re.compile(r"# secret-file: ok\s*(?:—|--?)\s*\S")


def _is_secret_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in SECRET_FILES


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def sites(path: Path) -> list[int]:
    """Line numbers of the counted sites in *path* (suppressed lines excluded)."""
    src = path.read_text(encoding="utf-8-sig")
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    found: list[int] = []
    routed = {id(arg) for node in ast.walk(tree)
              if isinstance(node, ast.Call) and _func_name(node.func) in ROUTED for arg in node.args}
    for node in ast.walk(tree):
        if id(node) in routed:
            continue
        hit = False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and _is_secret_name(node.right):
            hit = True
        elif isinstance(node, ast.Call):
            name = _func_name(node.func)
            hit = name in PATH_HELPERS or (name in PATH_BUILDERS and any(_is_secret_name(a) for a in node.args))
        lineno = getattr(node, "lineno", 0)
        if hit and not SUPPRESS_RE.search(lines[lineno - 1] if 0 < lineno <= len(lines) else ""):
            found.append(lineno)
    return sorted(found)


def scanned_files() -> list[Path]:
    files = sorted(p for p in ROOT.glob("*.py"))
    for tree in TREES:
        files.extend(sorted(p for p in (ROOT / tree).rglob("*.py") if not SKIP_PARTS & set(p.parts)))
    return [p for p in files if p.name not in EXEMPT or p.parent != ROOT]


def counts() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for path in scanned_files():
        found = sites(path)
        if found:
            out[path.relative_to(ROOT).as_posix()] = found
    return out


def main(argv: list[str]) -> int:
    current = counts()
    allow = json.loads(ALLOWLIST.read_text(encoding="utf-8-sig")) if ALLOWLIST.exists() else {}
    if "--write-allowlist" in argv:
        out = {rel: {"count": len(lines), "why": (allow.get(rel) or {}).get("why", "TODO")}
               for rel, lines in sorted(current.items())}
        ALLOWLIST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    problems: list[str] = []
    for rel, lines in sorted(current.items()):
        allowed = (allow.get(rel) or {}).get("count", 0)
        if len(lines) > allowed:
            problems.append(
                f"{rel}: {len(lines)} direct .env/auth.json path site(s) (lines {', '.join(map(str, lines))}), "
                f"{allowed} allowed — read and write secrets through hermes_secret_routing (read_profile_env, "
                "read_auth_json, save_env_value, the auth store), or mark a true false positive "
                "`# secret-file: ok — <why>`")
        elif len(lines) < allowed:
            problems.append(f"{rel}: {len(lines)} site(s), {allowed} allowed — lower its count in "
                            f"{ALLOWLIST.relative_to(ROOT).as_posix()} (the allowlist only ratchets down)")
    for rel, entry in sorted(allow.items()):
        # A file this tree does not have cannot regress (a partial checkout, a deleted file).
        if rel not in current and entry.get("count", 0) and (ROOT / rel).exists():
            problems.append(f"{rel}: no sites left — remove it from {ALLOWLIST.relative_to(ROOT).as_posix()}")
        if not str(entry.get("why") or "").strip() or entry.get("why") == "TODO":
            problems.append(f"{rel}: the allowlist entry needs a reason")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
