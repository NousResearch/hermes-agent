"""Phase 2 ownership guards for the Gateway / CLI boundary."""
from __future__ import annotations

import ast
from pathlib import Path

import gateway.migration as migration


def _import_refs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    refs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            refs.append(node.module)
            refs.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return refs


def test_gateway_python_never_imports_phase2_cli_owners() -> None:
    """Gateway control/topology/lifecycle must never route back through CLI owners."""
    gateway_root = Path(migration.__file__).parent
    forbidden_prefixes = ("hermes_cli.gateway", "hermes_cli.service_manager")
    profile_lifecycle = {
        "hermes_cli.profiles._check_gateway_running",
        "hermes_cli.profiles._maybe_register_gateway_service",
        "hermes_cli.profiles._maybe_unregister_gateway_service",
        "hermes_cli.profiles._cleanup_gateway_service",
        "hermes_cli.profiles._stop_profile_backends",
        "hermes_cli.profiles._stop_gateway_process",
    }
    offenders: list[tuple[str, str]] = []
    for path in gateway_root.rglob("*.py"):
        for ref in _import_refs(path):
            if ref.startswith(forbidden_prefixes) or ref == "hermes_cli.profiles" or ref in profile_lifecycle:
                offenders.append((str(path.relative_to(gateway_root)), ref))

    assert offenders == []


def test_migration_domain_does_not_render_terminal_output() -> None:
    path = Path(migration.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert print_calls == []
