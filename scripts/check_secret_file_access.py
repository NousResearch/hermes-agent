#!/usr/bin/env python3
"""Fail when new code names a profile's ``.env`` or ``auth.json`` file outside the secret store.

Every read and write of a profile's secrets goes through the selected secret store
(``hermes_secret_routing``: ``read_profile_env``, ``read_env_text``, ``route_env_write``,
``read_auth_json`` and the ``hermes_cli.auth`` store load/save, ``save_env_value`` /
``remove_env_value``). A store that keeps secrets off disk only sees what passes through there: a
new ``open(home / ".env")`` or ``name = "auth.json"; (home / name).read_text()`` silently keeps
reading the local file, which then holds the store's bootstrap entries only.

A site, in the scanned trees (everything but the router itself and the local backend, the one
place each path is built on purpose):

* any string literal that IS one of those file names, or ends in ``/<name>``, wherever it flows
  next (a path, a variable, a list of names, an argument): this checker does not follow it;
* a call of a helper that returns one of those paths: ``get_env_path()``, ``_auth_file_path()``,
  ``_global_auth_file_path()``, ``auth_json_path()``, ``_auth_json_path()``.

Not a site: anything inside the arguments of a routed reader or writer the file IMPORTS from
``hermes_secret_routing`` (or ``hermes_cli.auth``'s store load/save, imported by name or reached
through the imported module), unless the file defines or assigns that name itself: a local helper
of the same name is not the router. Docstrings and bare string statements are not sites.

Every remaining site must be listed, per line, with its reason, in
``scripts/secret_file_access_allowlist.json`` (``{file: [{"line": <stripped source line>,
"why": <reason>}, ...]}``). A site whose line is not listed fails, so new behaviour in an
allowlisted file fails as it would anywhere else; a listed line that is gone fails too, asking for
it to be removed. Counts cannot exceed the allowlist. An addition is permitted only through an
explicit allowlist edit with a per-line reason in the same change, visible to review.

This is a ratchet, not a proof. It reads source text: a path built from computed strings, a file
name that arrives from another module, or a reader reached through an alias it cannot see is not
caught. Routed-call argument subtrees are exempt, including side effects during argument
evaluation; import rebinding can also impersonate a routed name. These are known bypasses, not
security boundaries. It keeps known sites explicit and makes additions reviewed allowlist lines.

Usage: python3 scripts/check_secret_file_access.py [--write-allowlist] [--base REV] [--head REV]
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "scripts" / "secret_file_access_allowlist.json"
ALLOWLIST_REL = "scripts/secret_file_access_allowlist.json"
TREES = ("hermes_cli", "agent", "gateway", "tui_gateway", "cron", "plugins", "tools", "acp_adapter", "pm")
SKIP_PARTS = {"tests", "__pycache__", "node_modules", "vendor"}
# The router itself and the local backend: the one place each path is built on purpose.
EXEMPT = {"hermes_secret_routing.py", "hermes_secret_port_local.py"}
SECRET_FILES = (".env", "auth.json")
PATH_HELPERS = {"get_env_path", "_auth_file_path", "_global_auth_file_path", "auth_json_path", "_auth_json_path"}
ROUTING_MODULE = "hermes_secret_routing"
AUTH_MODULE = "hermes_cli.auth"
# The routed readers and writers, by the module that provides them.
ROUTED = {
    ROUTING_MODULE: frozenset({
        "read_profile_env", "read_env_text", "env_file_exists", "route_env_write", "routed_bootstrap_env",
        "routed_profile_env", "read_auth_json", "auth_json_exists", "auth_view_version"}),
    AUTH_MODULE: frozenset({"_load_auth_store", "_save_auth_store"}),
}


def _is_secret_literal(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    return any(node.value == name or node.value.endswith(("/" + name, "\\" + name)) for name in SECRET_FILES)


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _imported_routers(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    """The routed callables this file imported by name (their local names), and the routing
    modules it imported (``{local alias: module}``: ``import hermes_secret_routing as r``,
    ``from hermes_cli import auth``)."""
    names: set[str] = set()
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name in ROUTED.get(node.module, ()):
                    names.add(alias.asname or alias.name)
                if f"{node.module}.{alias.name}" in ROUTED:
                    modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ROUTED and (alias.asname or "." not in alias.name):
                    modules[alias.asname or alias.name] = alias.name
    return names, modules


def _locally_bound(tree: ast.AST) -> set[str]:
    """Names this file defines or assigns itself: a local helper is never trusted as the router."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def _is_routed_call(node: ast.Call, names: set[str], modules: dict[str, str], bound: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in names and func.id not in bound
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id not in bound:
        module = modules.get(func.value.id)
        return module is not None and func.attr in ROUTED[module]
    return False


def sites(path: Path) -> list[tuple[int, str]]:
    """``(line number, stripped source line)`` of each counted site in *path*, one per site."""
    src = path.read_text(encoding="utf-8-sig")
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    names, modules = _imported_routers(tree)
    bound = _locally_bound(tree)
    # Docstrings and bare string statements are never a path.
    skipped = {id(node.value) for node in ast.walk(tree)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_routed_call(node, names, modules, bound):
            for arg in (*node.args, *(kw.value for kw in node.keywords)):
                skipped.update(id(inner) for inner in ast.walk(arg))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        if _is_secret_literal(node) or (isinstance(node, ast.Call) and _func_name(node.func) in PATH_HELPERS):
            lineno = getattr(node, "lineno", 0)
            found.append((lineno, lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""))
    return sorted(found)


def scanned_files() -> list[Path]:
    files = sorted(p for p in ROOT.glob("*.py"))
    for tree in TREES:
        files.extend(sorted(p for p in (ROOT / tree).rglob("*.py") if not SKIP_PARTS & set(p.parts)))
    return [p for p in files if p.name not in EXEMPT or p.parent != ROOT]


def counts() -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for path in scanned_files():
        found = sites(path)
        if found:
            out[path.relative_to(ROOT).as_posix()] = found
    return out


def _entry_count(entry: object) -> int:
    """A file's allowed-site count in either allowlist shape (a list of lines, or ``{"count": n}``)."""
    if isinstance(entry, list):
        return len(entry)
    return int(entry.get("count", 0)) if isinstance(entry, dict) else 0


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              stdin=subprocess.DEVNULL, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def base_allowlist(base: str | None, head: str | None = None) -> dict | None:
    """The allowlist at *base* (default: the merge-base of *head*, else HEAD, with ``origin/main``,
    else ``main``); None when there is no base or it had no allowlist (nothing to ratchet on yet)."""
    if base is None:
        for ref in ("origin/main", "main"):
            out = _git("merge-base", head or "HEAD", ref)
            if out and out.strip():
                base = out.strip()
                break
    text = None if base is None else _git("show", f"{base}:{ALLOWLIST_REL}")
    try:
        data = json.loads(text) if text is not None else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _write_allowlist(current: dict[str, list[tuple[int, str]]], allow: dict) -> None:
    """Rewrite the allowlist to the current sites, keeping each listed line's reason (a file's
    old single reason for its new lines; ``TODO``, which fails, for a file never listed)."""
    out: dict[str, list[dict[str, str]]] = {}
    for rel, found in sorted(current.items()):
        old = allow.get(rel)
        reasons: dict[str, list[str]] = {}
        default = (old.get("why") or "TODO") if isinstance(old, dict) else "TODO"
        for item in old if isinstance(old, list) else ():
            reasons.setdefault(str(item.get("line", "")), []).append(str(item.get("why") or "TODO"))
        entries = []
        for _lineno, text in found:
            pool = reasons.get(text)
            entries.append({"line": text, "why": pool.pop(0) if pool else default})
        out[rel] = entries
    ALLOWLIST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def problems_in(current: dict[str, list[tuple[int, str]]], allow: dict, base_allow: dict | None) -> list[str]:
    problems: list[str] = []
    for rel in sorted(set(current) | set(allow)):
        entry = allow.get(rel)
        if entry is not None and not isinstance(entry, list):
            problems.append(f"{rel}: the allowlist entry must list each allowed line with its reason")
            continue
        listed = [item for item in entry or () if isinstance(item, dict)]
        found = current.get(rel, [])
        allowed = Counter(str(item.get("line", "")) for item in listed)
        seen: Counter = Counter()
        for lineno, text in found:
            seen[text] += 1
            if seen[text] > allowed[text]:
                problems.append(
                    f"{rel}:{lineno}: a direct .env/auth.json site that is not allowlisted — read and write "
                    "secrets through hermes_secret_routing (read_profile_env, read_auth_json, save_env_value, "
                    f"the auth store), or list this line with its reason in {ALLOWLIST_REL}")
        # A file this tree does not have cannot regress (a partial checkout, a deleted file).
        if found or (ROOT / rel).exists():
            for text, n in (allowed - Counter(text for _lineno, text in found)).items():
                problems.append(f"{rel}: {n} allowlisted line(s) no longer present ({text!r}): remove them "
                                f"from {ALLOWLIST_REL} (the allowlist only falls)")
        if any(not str(item.get("why") or "").strip() or item.get("why") == "TODO" for item in listed):
            problems.append(f"{rel}: every allowlisted line needs a reason")
    for rel, entry in sorted(allow.items()) if base_allow is not None else ():
        previous = base_allow.get(rel)
        if previous == entry:
            continue
        # An explicit allowlist edit is the review surface for additions. Site matching above
        # still bounds each file by its listed lines; missing reasons always fail.
        if not isinstance(entry, list) or any(not str(item.get("why") or "").strip()
                                              or item.get("why") == "TODO" for item in entry):
            problems.append(f"{rel}: an allowlist addition needs a per-line reason in this change")
    return problems


def main(argv: list[str]) -> int:
    current = counts()
    allow = json.loads(ALLOWLIST.read_text(encoding="utf-8-sig")) if ALLOWLIST.exists() else {}
    if "--write-allowlist" in argv:
        _write_allowlist(current, allow)
        return 0
    problems = problems_in(current, allow, base_allowlist(_option(argv, "--base"), _option(argv, "--head")))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
