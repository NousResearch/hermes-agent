"""Architecture guards for the completed process-identity extraction."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "plugins",
    "profiles",
    "providers",
    "runtime",
    "storage",
    "auth",
    "kanban",
    "automation",
    "tools",
)


def _python_sources():
    for dirname in PRODUCTION_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def test_retired_cli_process_identity_owner_stays_deleted():
    assert not (ROOT / "hermes_cli" / "process_identity.py").exists()


def test_production_never_imports_retired_cli_process_identity():
    violations = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "hermes_cli.process_identity" for alias in node.names):
                    violations.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom):
                if node.module == "hermes_cli.process_identity":
                    violations.append(str(path.relative_to(ROOT)))
                elif node.module == "hermes_cli" and any(
                    alias.name == "process_identity" for alias in node.names
                ):
                    violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_runtime_package_has_no_cli_imports():
    violations = []
    for path in (ROOT / "runtime").rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "hermes_cli" or alias.name.startswith("hermes_cli.") for alias in node.names):
                    violations.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "hermes_cli" or node.module.startswith("hermes_cli.")):
                    violations.append(str(path.relative_to(ROOT)))
    assert violations == []