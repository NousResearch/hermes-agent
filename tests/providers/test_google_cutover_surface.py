"""AST guard: core modules never import the Google model-provider plugin or its moved adapter modules."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_ROOTS = (
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "providers",
    "tools",
    "tui_gateway",
)
_CORE_FILES = ("run_agent.py", "model_tools.py")

_FORBIDDEN_PREFIXES = (
    "agent.gemini_native_adapter",
    "agent.gemini_schema",
    "agent.gemini_long_decode",
    "plugins.model_providers.google",
    "plugins.model_providers.gemini",
    "plugins.model_providers.solstice",
)


def _core_python_files() -> list[Path]:
    files = [REPO_ROOT / name for name in _CORE_FILES if (REPO_ROOT / name).is_file()]
    for root in _CORE_ROOTS:
        base = REPO_ROOT / root
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_FORBIDDEN_PREFIXES):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(_FORBIDDEN_PREFIXES):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} from {node.module}")
    return hits


def test_core_has_no_imports_of_moved_google_adapter_or_plugin():
    violations = [hit for path in _core_python_files() for hit in _forbidden_imports(path)]
    assert violations == [], "Core must talk to the Google provider through ProviderProfile seams:\n" + "\n".join(violations)


def test_google_plugin_has_no_private_core_imports():
    from hermes_cli.plugin_validate_private_imports import private_import_findings

    plugin_dir = REPO_ROOT / "plugins" / "model-providers" / "google"
    if not plugin_dir.is_dir():
        return
    assert private_import_findings(plugin_dir) == []
