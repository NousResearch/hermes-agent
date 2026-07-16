"""Static guard: the todo-store modules must stay LLM/agent-client free.

All three modules' docstrings promise "No LLM/agent-client imports: stdlib
+ sqlite3 + hermes_cli.sqlite_util only." — these are meant to be safe to
call from the scheduled morning-run path, Discord button handlers, and a
one-off seed script without ever touching a model provider.

The real "this module invokes an LLM" import shape elsewhere in the repo
(e.g. run_agent.py) is ``from agent.<submodule> import ...`` — see
run_agent.py's ``from agent.process_bootstrap import ...``,
``from agent.anthropic_adapter import ...``-style imports (agent/ is the
package that wraps model-provider calls), and plugins/model-providers/*
which import from ``providers``. This test parses each module's AST (rather
than grepping for the substring "agent", which would false-positive on
unrelated names like "agent_runtime_helpers" as a *local* variable, or on
this repo's own doc comments) and asserts no import statement's module path
starts with ``agent`` or ``providers`` as a top-level package.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MODULES = [
    _REPO_ROOT / "plugins" / "life_ops" / "todo_store.py",
    _REPO_ROOT / "plugins" / "life_ops" / "away_mode.py",
    _REPO_ROOT / "plugins" / "life_ops" / "todo_store_seed.py",
]

_FORBIDDEN_TOP_LEVEL_PACKAGES = {"agent", "providers"}


def _imported_top_level_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                packages.add(node.module.split(".")[0])
    return packages


@pytest.mark.parametrize("module_path", _MODULES, ids=lambda p: p.name)
def test_module_exists(module_path):
    assert module_path.is_file(), f"expected {module_path} to exist"


@pytest.mark.parametrize("module_path", _MODULES, ids=lambda p: p.name)
def test_no_agent_or_providers_import(module_path):
    packages = _imported_top_level_packages(module_path)
    forbidden_hit = packages & _FORBIDDEN_TOP_LEVEL_PACKAGES
    assert not forbidden_hit, (
        f"{module_path.name} imports from {sorted(forbidden_hit)} — "
        "this module must stay LLM/agent-client free per its module docstring"
    )


@pytest.mark.parametrize("module_path", _MODULES, ids=lambda p: p.name)
def test_only_expected_import_surface(module_path):
    """Belt-and-braces: every top-level import package is stdlib, sqlite3,
    hermes_cli, or plugins.life_ops (self-references between the three
    modules, e.g. away_mode importing todo_store's todos_db_path)."""
    allowed = {
        "__future__",
        "argparse",
        "datetime",
        "json",
        "logging",
        "os",
        "re",
        "sqlite3",
        "hermes_cli",
        "plugins",
        "pathlib",
        "typing",
    }
    packages = _imported_top_level_packages(module_path)
    unexpected = packages - allowed
    assert not unexpected, f"{module_path.name} has unexpected imports: {sorted(unexpected)}"
