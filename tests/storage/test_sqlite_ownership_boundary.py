"""Architecture guards for the SQLite runtime/storage extraction."""

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
OWNERS = (
    ROOT / "runtime" / "sqlite_runtime.py",
    ROOT / "storage" / "sqlite_util.py",
    ROOT / "storage" / "sqlite_safe_read.py",
)
COMPAT_FACADE = ROOT / "hermes_cli" / "sqlite_safe_read.py"


def _python_sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def test_sqlite_owners_exist_and_retired_cli_owners_stay_deleted():
    for path in OWNERS:
        assert path.exists(), path.relative_to(ROOT)
    assert not (ROOT / "hermes_cli" / "sqlite_runtime.py").exists()
    assert not (ROOT / "hermes_cli" / "sqlite_util.py").exists()


def test_sqlite_safe_read_cli_facade_stays_compat_only():
    source = COMPAT_FACADE.read_text(encoding="utf-8")
    assert "PLUGIN-COMPAT" in source
    assert "SQLITE_HEADER_MAGIC" in source
    for runtime_name in (
        "connect_tracked",
        "offline_file_access",
        "read_header_bytes_preopen",
        "file_length_matches_header",
        "TrackedConnection",
        "LiveConnectionError",
    ):
        assert runtime_name not in source


def test_no_source_references_retired_sqlite_cli_modules():
    runtime_needle = "hermes_cli" + ".sqlite_runtime"
    util_needle = "hermes_cli" + ".sqlite_util"
    safe_read_needle = "hermes_cli" + ".sqlite_safe_read"
    violations = []
    for path in _python_sources():
        if path == COMPAT_FACADE:
            continue
        source = path.read_text(encoding="utf-8")
        for needle in (runtime_needle, util_needle, safe_read_needle):
            if needle in source:
                violations.append((str(path.relative_to(ROOT)), needle))
    assert violations == []


def test_sqlite_owners_do_not_import_cli_or_gateway_layers():
    forbidden = ("hermes_cli", "nous_cli", "agent", "gateway")
    violations = []
    for path in OWNERS:
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
