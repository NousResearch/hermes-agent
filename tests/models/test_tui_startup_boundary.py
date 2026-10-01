"""Phase 5.8.6.1: freeze TUI startup ownership and exact credential debt."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _tree(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path):
    found = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_tui_startup_uses_canonical_model_domains():
    paths = (ROOT / "tui_gateway" / "agent_factory.py",
             ROOT / "tui_gateway" / "model_startup_route.py")
    forbidden = ("hermes_cli.model_switch", "hermes_cli.model_selection",
                 "hermes_cli.models", "hermes_cli.model_catalog")
    offenders = [
        f"{path.relative_to(ROOT)} -> {module}"
        for path in paths for module in _imports(path)
        if any(module == name or module.startswith(name + ".") for name in forbidden)
    ]
    assert offenders == []

    for name in ("application_model_aliases.py", "application_model_facts.py"):
        path = ROOT / name
        assert path.exists()
        assert not any(
            module == "gateway" or module.startswith("gateway.")
            for module in _imports(path)
        )
    assert "resolve_startup_seed" in {
        node.name for node in _tree(paths[1]).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_tui_startup_runtime_provider_exception_is_exact():
    path = ROOT / "tui_gateway" / "agent_factory.py"
    found = set()

    def collect(node, scope="<module>"):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "hermes_cli.runtime_provider"
        ):
            found.update((scope, node.module, alias.name) for alias in node.names)
        for child in ast.iter_child_nodes(node):
            collect(child, scope)

    collect(_tree(path))
    assert found == {
        ("_resolve_runtime_with_fallback", "hermes_cli.runtime_provider",
         "resolve_runtime_provider")
    }
