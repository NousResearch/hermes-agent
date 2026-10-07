"""Guard against plugin-hook wire-up drift.

Every hook name in ``VALID_HOOKS`` must have a live production dispatch
site, and every literal hook name passed to ``invoke_hook``/``has_hook``
must be declared in ``VALID_HOOKS``.

Motivation (research spike #64180, PR #65658): OpenCode's ``permission.ask``
hook sat typed-but-dead for 6+ months after a subsystem rewrite deleted its
dispatch site while the published types kept compiling — plugin authors got
compile-time success and runtime no-ops. Hermes dispatches several hooks
through wrapper functions that take the hook name as a parameter (the
kanban lifecycle events, the approval observers, ``_notify_session_boundary``),
which is exactly the structure where a refactor can silently orphan a hook.

The liveness check is deliberately loose — a hook name counts as dispatched
if its string literal appears anywhere in production Python outside the
declaration/tooling modules — so wrapper indirection doesn't false-positive.
When a rewrite removes a dispatch path entirely, the literal disappears with
it and this test fails the same day.
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories that contain no production dispatch sites.
_SKIP_DIRS = {
    ".git",
    ".claude",
    "__pycache__",
    "node_modules",
    "apps",
    "docs",
    "tests",
    "tests-js",
    "ui-tui",
    "web",
    "website",
}

# Modules where a hook-name literal does not prove a live dispatch path:
# plugins.py declares VALID_HOOKS itself; hooks.py holds synthetic
# `hermes hooks test` payloads keyed by hook name.
_DECLARATION_MODULES = {
    REPO_ROOT / "hermes_cli" / "plugins.py",
    REPO_ROOT / "hermes_cli" / "hooks.py",
}


def _production_py_files():
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.relative_to(REPO_ROOT).parts)
        if parts & _SKIP_DIRS:
            continue
        yield path


def _parse_valid_hooks():
    tree = ast.parse((REPO_ROOT / "hermes_cli" / "plugins.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == "VALID_HOOKS" and isinstance(value, ast.Set):
            return {e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    raise AssertionError("VALID_HOOKS set literal not found in hermes_cli/plugins.py")


def test_every_valid_hook_has_a_live_dispatch_reference():
    valid_hooks = _parse_valid_hooks()
    assert valid_hooks, "VALID_HOOKS parsed empty — parser broke, not the catalog"

    missing = set(valid_hooks)
    patterns = {hook: re.compile(r"""["']%s["']""" % re.escape(hook)) for hook in valid_hooks}
    for path in _production_py_files():
        if path in _DECLARATION_MODULES:
            continue
        if not missing:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for hook in list(missing):
            if patterns[hook].search(text):
                missing.discard(hook)

    assert not missing, (
        "Hooks declared in VALID_HOOKS but with no production dispatch "
        "reference (typed-but-dead — plugins registering these silently "
        f"never fire): {sorted(missing)}. If a hook is intentionally "
        "reserved before its dispatch lands, add it to an explicit "
        "allowlist here with a comment pointing at the tracking issue."
    )


def test_every_literal_hook_dispatch_is_declared():
    valid_hooks = _parse_valid_hooks()
    undeclared = {}
    for path in _production_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name not in ("invoke_hook", "has_hook"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            # Dynamic names (variables, f-strings) are legitimate wrapper
            # patterns — only literal names can be checked for typos.
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value not in valid_hooks:
                    site = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                    undeclared.setdefault(first.value, []).append(site)

    assert not undeclared, (
        f"Hook names dispatched but not declared in VALID_HOOKS (typo or "
        f"missing declaration): {undeclared}"
    )
