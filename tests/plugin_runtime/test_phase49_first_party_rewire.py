"""Permanent first-party plugin ownership checks retained from Phase 4.9."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_FACADE = ROOT / "hermes_cli" / "plugins.py"

SOURCE_ROOTS = (
    "acp_adapter",
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "plugin_runtime",
    "plugins",
    "providers",
    "tools",
    "tui_gateway",
)
ROOT_SOURCES = (
    "cli.py",
    "hermes_state.py",
    "run_agent.py",
)

# The Phase 4.9 migration is complete; these exact-zero assertions remain as a
# permanent regression check for first-party consumers.
PHASE49_REWIRE_DEBT_IMPORTS = 0
PHASE49_REWIRE_DEBT_FILES: set[str] = set()

GATEWAY_REWIRE_DEBT_IMPORTS = 0
GATEWAY_REWIRE_DEBT_FILES: set[str] = set()

PLUGIN_RUNTIME_REWIRE_DEBT_IMPORTS = 0
PLUGIN_RUNTIME_REWIRE_DEBT_FILES: set[str] = set()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@cache
def _production_sources() -> tuple[Path, ...]:
    sources = [ROOT / name for name in ROOT_SOURCES if (ROOT / name).exists()]
    for root_name in SOURCE_ROOTS:
        root = ROOT / root_name
        if root.exists():
            sources.extend(root.rglob("*.py"))
    return tuple(sources)


def _imports_cli_plugins(path: Path) -> list[tuple[str, int, str]]:
    source = path.read_text(encoding="utf-8")
    if "hermes_cli.plugins" not in source and not (
        "from hermes_cli import" in source and "plugins" in source
    ):
        return []

    relative = path.relative_to(ROOT).as_posix()
    violations: list[tuple[str, int, str]] = []

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "hermes_cli.plugins":
                    violations.append((relative, node.lineno, "import hermes_cli.plugins"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "hermes_cli.plugins":
                violations.append((relative, node.lineno, "from hermes_cli.plugins import"))
            elif node.module == "hermes_cli" and any(
                alias.name == "plugins" for alias in node.names
            ):
                violations.append((relative, node.lineno, "from hermes_cli import plugins"))

    return violations


@cache
def _first_party_cli_plugin_imports() -> tuple[tuple[str, int, str], ...]:
    violations: list[tuple[str, int, str]] = []
    for path in _production_sources():
        if path == LEGACY_FACADE:
            continue
        violations.extend(_imports_cli_plugins(path))
    return tuple(sorted(violations))


def _scoped_cli_plugin_imports(prefix: str) -> list[tuple[str, int, str]]:
    return [
        violation
        for violation in _first_party_cli_plugin_imports()
        if violation[0].startswith(prefix)
    ]


def _imports_cli_plugin_compat(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if "hermes_cli.plugin_compat" not in source and not (
        "from hermes_cli import" in source and "plugin_compat" in source
    ):
        return False

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            if any(alias.name == "hermes_cli.plugin_compat" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "hermes_cli.plugin_compat":
                return True
            if node.module == "hermes_cli" and any(
                alias.name == "plugin_compat" for alias in node.names
            ):
                return True
    return False


def test_phase49_first_party_rewire_debt_is_exact() -> None:
    violations = _first_party_cli_plugin_imports()

    assert len(violations) == PHASE49_REWIRE_DEBT_IMPORTS
    assert {path for path, _, _ in violations} == PHASE49_REWIRE_DEBT_FILES


def test_phase49_gateway_rewire_debt_is_exact() -> None:
    violations = _scoped_cli_plugin_imports("gateway/")

    assert len(violations) == GATEWAY_REWIRE_DEBT_IMPORTS
    assert {path for path, _, _ in violations} == GATEWAY_REWIRE_DEBT_FILES


def test_phase49_plugin_runtime_backedge_debt_is_exact() -> None:
    violations = _scoped_cli_plugin_imports("plugin_runtime/")

    assert len(violations) == PLUGIN_RUNTIME_REWIRE_DEBT_IMPORTS
    assert {path for path, _, _ in violations} == PLUGIN_RUNTIME_REWIRE_DEBT_FILES


def test_phase49_gateway_has_no_cli_plugin_compat_dependency() -> None:
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "gateway").rglob("*.py")
        if _imports_cli_plugin_compat(path)
    ]

    assert violations == []
