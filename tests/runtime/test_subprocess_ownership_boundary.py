"""Architecture guards for the completed subprocess/runtime extraction."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
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
    "tui_gateway",
)
RUNTIME_OWNERS = (
    ROOT / "runtime" / "subprocess_compat.py",
    ROOT / "runtime" / "git_subprocess.py",
    ROOT / "runtime" / "processes.py",
    ROOT / "runtime" / "process_identity.py",
)


def _python_sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def test_subprocess_runtime_owners_exist_and_cli_owner_stays_deleted():
    for path in RUNTIME_OWNERS:
        assert path.exists(), path.relative_to(ROOT)
    assert (ROOT / "gateway" / "windows_launch.py").exists()
    assert not (ROOT / "hermes_cli" / "_subprocess_compat.py").exists()


def test_no_source_references_retired_cli_subprocess_module():
    violations = []
    needle = "hermes_cli" + "._subprocess_compat"
    for path in _python_sources():
        if needle in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_runtime_subprocess_owners_do_not_import_upper_layers():
    forbidden = ("hermes_cli", "nous_cli", "agent", "gateway")
    violations = []
    for path in RUNTIME_OWNERS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                    violations.append((str(path.relative_to(ROOT)), module))
    assert violations == []


def test_gateway_breakaway_metadata_has_one_owner():
    declaration = '_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"'
    owners = []
    for path in _python_sources():
        if declaration in path.read_text(encoding="utf-8"):
            owners.append(path.relative_to(ROOT).as_posix())
    assert owners == ["gateway/windows_launch.py"]
