"""Architecture guards for the stdio/resource-limit runtime extraction."""

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
    ROOT / "runtime" / "stdio.py",
    ROOT / "runtime" / "resource_limits.py",
)
def _python_sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def test_runtime_owners_exist_and_cli_owners_stay_deleted():
    for path in RUNTIME_OWNERS:
        assert path.exists(), path.relative_to(ROOT)
    assert not (ROOT / "hermes_cli" / "stdio.py").exists()
    assert not (ROOT / "hermes_cli" / "resource_limits.py").exists()


def test_no_source_references_retired_cli_modules():
    violations = []
    needles = (
        "hermes_cli" + ".stdio",
        "hermes_cli" + ".resource_limits",
        "hermes_cli" + "/stdio.py",
        "hermes_cli" + "/resource_limits.py",
    )
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        if any(needle in source for needle in needles):
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_runtime_owners_do_not_import_upper_layers():
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


def test_resource_limits_does_not_load_configuration():
    source = (ROOT / "runtime" / "resource_limits.py").read_text(encoding="utf-8")
    assert "load_config_readonly" not in source
