"""Ownership guards for Gateway service-manager backends."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import gateway.s6_manager as s6_manager
import gateway.service_manager as service_manager


@pytest.mark.parametrize("module", [service_manager, s6_manager])
def test_service_manager_owners_do_not_import_hermes_cli(module) -> None:
    """Gateway service ownership must not route back through the CLI namespace."""
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not [name for name in imports if name == "hermes_cli" or name.startswith("hermes_cli.")]


def test_service_manager_factory_owns_s6_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generic factory resolves s6 directly from Gateway ownership."""
    monkeypatch.setattr(service_manager, "detect_service_manager", lambda: "s6")
    manager = service_manager.get_service_manager()
    assert isinstance(manager, s6_manager.S6ServiceManager)
