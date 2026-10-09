"""Phase 2 ownership guards for the Gateway / CLI boundary."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETIRED_GATEWAY_CLIENT = "hermes_cli.gateway_client"


def _retired_gateway_client_refs(path: Path) -> list[int]:
    """Return source lines that still name the deleted Phase 2 client owner."""
    return [
        lineno
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if RETIRED_GATEWAY_CLIENT in line
    ]


def test_embedded_python_imports_cannot_hide_retired_gateway_client(tmp_path: Path) -> None:
    probe = tmp_path / "probe.ts"
    probe.write_text(
        "const SCRIPT = `\n"
        "from hermes_cli.gateway_client import _session_ticket\n"
        "`\n",
        encoding="utf-8",
    )
    assert _retired_gateway_client_refs(probe) == [2]


def test_tracked_source_never_names_retired_gateway_client() -> None:
    tracked = subprocess.run(
        ["git", "grep", "-l", RETIRED_GATEWAY_CLIENT, "--", "*.py", "*.ts", "*.tsx"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    this_file = Path(__file__).resolve()
    offenders: list[tuple[str, int]] = []
    for relative_text in tracked:
        path = (PROJECT_ROOT / relative_text).resolve()
        if path == this_file:
            continue
        for lineno in _retired_gateway_client_refs(path):
            offenders.append((relative_text, lineno))

    assert offenders == []


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
