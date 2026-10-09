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
SAFE_READ_COMPAT = ROOT / "hermes_cli" / "sqlite_safe_read.py"
RUNTIME_COMPAT = ROOT / "hermes_cli" / "sqlite_runtime.py"


def _python_sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def test_sqlite_owners_exist_and_cli_util_stays_deleted():
    for path in OWNERS:
        assert path.exists(), path.relative_to(ROOT)
    assert SAFE_READ_COMPAT.exists()
    assert RUNTIME_COMPAT.exists()
    assert not (ROOT / "hermes_cli" / "sqlite_util.py").exists()


def test_sqlite_cli_facades_stay_compat_only():
    safe_source = SAFE_READ_COMPAT.read_text(encoding="utf-8")
    assert "PLUGIN-COMPAT" in safe_source
    assert "from storage.sqlite_safe_read import (" in safe_source
    for name in (
        "LiveConnectionError",
        "has_live_connection",
        "offline_file_access",
        "read_header_bytes_preopen",
        "SQLITE_HEADER_MAGIC",
    ):
        assert name in safe_source
    for implementation_name in (
        "connect_tracked",
        "file_length_matches_header",
        "TrackedConnection",
        "UntrackableConnectionError",
    ):
        assert implementation_name not in safe_source

    runtime_source = RUNTIME_COMPAT.read_text(encoding="utf-8")
    assert "from runtime.sqlite_runtime import probe_sqlite_runtime" in runtime_source
    assert "subprocess.run" not in runtime_source
    assert "class SQLiteRuntimeInfo" not in runtime_source


def test_no_source_references_retired_sqlite_cli_modules():
    runtime_needle = "hermes_cli" + ".sqlite_runtime"
    util_needle = "hermes_cli" + ".sqlite_util"
    safe_read_needle = "hermes_cli" + ".sqlite_safe_read"
    violations = []
    compatibility_facades = {SAFE_READ_COMPAT, RUNTIME_COMPAT}
    registry_owner = ROOT / "storage" / "sqlite_safe_read.py"
    for path in _python_sources():
        if path in compatibility_facades:
            continue
        source = path.read_text(encoding="utf-8")
        for needle in (runtime_needle, util_needle, safe_read_needle):
            # The one exception is a cached-module identity lookup needed when
            # an old daemon still has a live SQLite descriptor across the update.
            if path == registry_owner and needle == safe_read_needle:
                continue
            if needle in source:
                violations.append((str(path.relative_to(ROOT)), needle))
    owner_source = registry_owner.read_text(encoding="utf-8")
    assert owner_source.count('sys.modules.get("hermes_cli.sqlite_safe_read")') == 1
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
