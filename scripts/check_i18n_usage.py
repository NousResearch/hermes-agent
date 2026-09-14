#!/usr/bin/env python3
"""Fail when a ``t()`` call in first-party Python does not match ``locales/en.yaml``.

Two rules, both silent at runtime and therefore invisible without a static check:

  1. The key must exist in the English catalog. ``t()`` returns the bare key when
     English misses it, so the message ships and users read the dotted key path.
     That is how ``gateway.verbose.mode_log`` reached users and had to be
     backfilled into every catalog by hand.
  2. A literal call must pass every ``{placeholder}`` its English value uses. A
     missing format kwarg does not raise: ``t()`` logs and returns the
     unformatted template, so users read a literal ``{count}``.

The catalog parity tests in ``tests/agent/test_i18n.py`` compare catalogs to each
other and cannot see either failure, because a key missing from *every* catalog
is perfectly consistent.

Call resolution is lexically scoped: a name is looked up in the innermost scope
that binds it, class bodies are skipped for nested functions, and any name bound
to something other than the i18n import is treated as shadowed. So a
function-local import never classifies calls elsewhere in the file, and a local
``t`` parameter or assignment is never mistaken for the catalog lookup.

Recognized import forms::

    from agent.i18n import t            # t("k")
    from agent.i18n import t as _t      # _t("k")
    from agent import i18n              # i18n.t("k")
    import agent.i18n as i18n_module    # i18n_module.t("k")
    import agent.i18n                   # agent.i18n.t("k")

Keys built at runtime (variables, f-strings) cannot be checked statically and are
skipped, as are calls that forward ``**kwargs``.

Exit 1 with a file:line list on any hit. Run: python scripts/check_i18n_usage.py
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCALES_DIR = ROOT / "locales"

# Directories whose ``t()`` calls are not first-party production strings: test
# fixtures invent keys on purpose, and the JS/docs trees have no Python callers.
SKIP_DIRS = {
    ".git", "__pycache__", ".worktrees", "node_modules", "build",
    "tests", "web", "website", "apps", "ui-tui", "docs", "assets",
}

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# ``t()``'s own parameter, not a format placeholder: passing lang= does not
# satisfy a ``{lang}`` token in the catalog value.
_T_PARAMETERS = frozenset({"lang"})

# What a local name is bound to, for the scope resolver below.
T_FUNC = "t-func"          # the t() catalog lookup itself
I18N_MODULE = "i18n-module"  # the agent.i18n module
AGENT_PACKAGE = "agent-package"  # the top package, with agent.i18n imported
SHADOWED = "shadowed"      # bound to something else entirely


@dataclass(frozen=True)
class TCall:
    """One statically analyzable ``t("dotted.key", ...)`` call site."""

    path: str
    lineno: int
    key: str
    kwargs: frozenset[str]
    star_kwargs: bool


class _Scope:
    """One lexical scope and the names it binds."""

    __slots__ = ("kind", "names")

    def __init__(self, kind: str) -> None:
        self.kind = kind  # "module" | "function" | "class"
        self.names: dict[str, str] = {}

    def bind(self, name: str, kind: str) -> None:
        previous = self.names.get(name)
        if previous is not None and previous != kind:
            # Two different bindings for one name in one scope: which one wins
            # depends on execution order, so claim nothing.
            self.names[name] = SHADOWED
        else:
            self.names[name] = kind


def _resolve(name: str, stack: list[_Scope]) -> str | None:
    """Resolve ``name`` against the enclosing scopes, innermost first.

    Class bodies are visible only to code written directly inside them, never to
    functions nested within them, which is how Python itself resolves names.
    """
    for depth in range(len(stack) - 1, -1, -1):
        scope = stack[depth]
        if scope.kind == "class" and depth != len(stack) - 1:
            continue
        kind = scope.names.get(name)
        if kind is not None:
            return kind
    return None


def _default_exprs(args: ast.arguments) -> list[ast.expr]:
    """Default values, which are evaluated in the *enclosing* scope."""
    return [d for d in [*args.defaults, *args.kw_defaults] if d is not None]


def _bind_params(scope: _Scope, args: ast.arguments) -> None:
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        scope.bind(arg.arg, SHADOWED)
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            scope.bind(extra.arg, SHADOWED)


def _bind_import(scope: _Scope, alias: ast.alias) -> None:
    if alias.asname:
        kind = I18N_MODULE if alias.name == "agent.i18n" else SHADOWED
        scope.bind(alias.asname, kind)
        return
    if alias.name == "agent.i18n":
        # ``import agent.i18n`` binds the top-level name and makes the submodule
        # reachable through it.
        scope.bind("agent", AGENT_PACKAGE)
        return
    top = alias.name.partition(".")[0]
    if top != "agent":
        scope.bind(top, SHADOWED)
    # ``import agent`` / ``import agent.tools`` bind the same package object
    # without making agent.i18n reachable, and must not cancel an
    # ``import agent.i18n`` elsewhere in the scope, so leave the name alone.


def _bind_import_from(scope: _Scope, node: ast.ImportFrom) -> None:
    for alias in node.names:
        if alias.name == "*":
            continue
        kind = SHADOWED
        if node.level == 0:
            if node.module == "agent.i18n" and alias.name == "t":
                kind = T_FUNC
            elif node.module == "agent" and alias.name == "i18n":
                kind = I18N_MODULE
        scope.bind(alias.asname or alias.name, kind)


def _bind_node(scope: _Scope, node: ast.AST) -> None:
    """Record every name ``node`` binds in ``scope``.

    Nested scopes are not entered: their bodies bind in their own scope. What
    they do bind here is their own name, plus the decorators, bases and default
    values that are evaluated where they are written.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        scope.bind(node.name, SHADOWED)
        for expr in [*node.decorator_list, *_default_exprs(node.args)]:
            _bind_node(scope, expr)
        return
    if isinstance(node, ast.ClassDef):
        scope.bind(node.name, SHADOWED)
        for expr in [*node.decorator_list, *node.bases]:
            _bind_node(scope, expr)
        for keyword in node.keywords:
            _bind_node(scope, keyword.value)
        return
    if isinstance(node, ast.Lambda):
        for expr in _default_exprs(node.args):
            _bind_node(scope, expr)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return
    if isinstance(node, ast.Import):
        for alias in node.names:
            _bind_import(scope, alias)
        return
    if isinstance(node, ast.ImportFrom):
        _bind_import_from(scope, node)
        return
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        for name in node.names:
            scope.bind(name, SHADOWED)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        scope.bind(node.id, SHADOWED)
        return
    if isinstance(node, ast.ExceptHandler) and node.name:
        scope.bind(node.name, SHADOWED)
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        scope.bind(node.name, SHADOWED)
    if isinstance(node, ast.MatchMapping) and node.rest:
        scope.bind(node.rest, SHADOWED)
    for child in ast.iter_child_nodes(node):
        _bind_node(scope, child)


def _open_scope(kind: str, body: list[ast.AST]) -> _Scope:
    scope = _Scope(kind)
    for node in body:
        _bind_node(scope, node)
    return scope


def _is_t_call(node: ast.Call, stack: list[_Scope]) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return _resolve(func.id, stack) == T_FUNC
    if isinstance(func, ast.Attribute) and func.attr == "t":
        value = func.value
        if isinstance(value, ast.Name):
            return _resolve(value.id, stack) == I18N_MODULE
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "i18n"
            and isinstance(value.value, ast.Name)
        ):
            return _resolve(value.value.id, stack) == AGENT_PACKAGE
    return False


def _record_call(node: ast.Call, path: str, calls: list[TCall]) -> None:
    if not node.args:
        return
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return  # key built at runtime: nothing to check statically
    names = {kw.arg for kw in node.keywords}
    calls.append(
        TCall(
            path=path,
            lineno=node.lineno,
            key=first.value,
            kwargs=frozenset(name for name in names if name is not None),
            star_kwargs=None in names,
        )
    )


def _walk(node: ast.AST, stack: list[_Scope], path: str, calls: list[TCall]) -> None:
    """Visit ``node``, opening a new scope for every construct that makes one."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for expr in [*node.decorator_list, *_default_exprs(node.args)]:
            _walk(expr, stack, path, calls)
        inner = _open_scope("function", node.body)
        _bind_params(inner, node.args)
        for stmt in node.body:
            _walk(stmt, [*stack, inner], path, calls)
        return
    if isinstance(node, ast.ClassDef):
        for expr in [*node.decorator_list, *node.bases]:
            _walk(expr, stack, path, calls)
        for keyword in node.keywords:
            _walk(keyword.value, stack, path, calls)
        inner = _open_scope("class", node.body)
        for stmt in node.body:
            _walk(stmt, [*stack, inner], path, calls)
        return
    if isinstance(node, ast.Lambda):
        for expr in _default_exprs(node.args):
            _walk(expr, stack, path, calls)
        inner = _Scope("function")
        _bind_params(inner, node.args)
        _bind_node(inner, node.body)
        _walk(node.body, [*stack, inner], path, calls)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        inner = _Scope("function")
        for generator in node.generators:
            _bind_node(inner, generator.target)
        for child in ast.iter_child_nodes(node):
            _walk(child, [*stack, inner], path, calls)
        return
    if isinstance(node, ast.Call) and _is_t_call(node, stack):
        _record_call(node, path, calls)
    for child in ast.iter_child_nodes(node):
        _walk(child, stack, path, calls)


def scan_source(src: str, path: str = "<string>") -> list[TCall]:
    """Return every statically analyzable ``t()`` call in one Python source."""
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return []
    module = _open_scope("module", tree.body)
    calls: list[TCall] = []
    for stmt in tree.body:
        _walk(stmt, [module], path, calls)
    return calls


def flatten(catalog: dict | None, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in (catalog or {}).items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def load_english_catalog() -> dict[str, object]:
    with (LOCALES_DIR / "en.yaml").open("r", encoding="utf-8") as handle:
        return flatten(yaml.safe_load(handle))


def missing_keys(calls: list[TCall], english: dict[str, object]) -> list[str]:
    """Call sites naming a key the English catalog does not define."""
    return sorted(
        f"{call.path}:{call.lineno}: t({call.key!r})"
        for call in calls
        if call.key not in english
    )


def missing_placeholders(calls: list[TCall], english: dict[str, object]) -> list[str]:
    """Call sites omitting a ``{placeholder}`` their English value interpolates."""
    problems = []
    for call in calls:
        if call.star_kwargs:
            continue  # forwarded kwargs are not statically knowable
        value = english.get(call.key)
        if not isinstance(value, str):
            continue
        gap = set(PLACEHOLDER_RE.findall(value)) - (call.kwargs - _T_PARAMETERS)
        if gap:
            problems.append(
                f"{call.path}:{call.lineno}: t({call.key!r}) does not supply {sorted(gap)}"
            )
    return sorted(problems)


def iter_python_files():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield path, rel


def collect_calls() -> list[TCall]:
    calls: list[TCall] = []
    for path, rel in iter_python_files():
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "i18n" not in src:
            continue  # cheap pre-filter before paying for a parse
        calls.extend(scan_source(src, str(rel)))
    return calls


def main() -> int:
    english = load_english_catalog()
    calls = collect_calls()
    missing = missing_keys(calls, english)
    gaps = missing_placeholders(calls, english)
    if missing:
        print("❌ t() keys used in code but absent from locales/en.yaml")
        print("   (users would read the raw dotted key path):")
        for line in missing:
            print("  " + line)
    if gaps:
        if missing:
            print()
        print("❌ t() calls omitting a format kwarg their catalog entry uses")
        print("   (users would read a literal {placeholder}):")
        for line in gaps:
            print("  " + line)
    if missing or gaps:
        print(
            f"\n{len(missing) + len(gaps)} site(s). Add the key to locales/en.yaml "
            "and every other catalog, or pass the missing kwarg."
        )
        return 1
    print(f"✅ {len(calls)} static t() call(s) agree with locales/en.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
