"""Phase 4.8 ownership gate for plugin compatibility runtime."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "plugin_runtime"
CANONICAL_OWNER = ROOT / "plugin_runtime" / "compat.py"
WARNING_CONTRACT = ROOT / "plugin_runtime" / "compat_warning.py"
LEGACY_FACADE = ROOT / "hermes_cli" / "plugin_compat.py"
LEGACY_OWNER = "hermes_cli.plugin_compat"

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


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _production_sources() -> list[Path]:
    sources = [ROOT / name for name in ROOT_SOURCES if (ROOT / name).exists()]
    for root_name in SOURCE_ROOTS:
        root = ROOT / root_name
        if root.exists():
            sources.extend(root.rglob("*.py"))
    return sources


def _legacy_imports() -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    for path in _production_sources():
        if path == LEGACY_FACADE:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == LEGACY_OWNER:
                        violations.append(
                            (path.relative_to(ROOT).as_posix(), node.lineno, alias.name)
                        )
            elif isinstance(node, ast.ImportFrom) and node.module == LEGACY_OWNER:
                violations.append(
                    (path.relative_to(ROOT).as_posix(), node.lineno, node.module)
                )
    return violations


def test_phase48_canonical_runtime_owners_exist() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in (CANONICAL_OWNER, WARNING_CONTRACT, LEGACY_FACADE)
        if not path.exists()
    ]

    assert missing == []


def test_phase48_first_party_production_has_no_legacy_compat_imports() -> None:
    assert _legacy_imports() == []


def test_phase48_plugin_compat_is_not_a_scheduled_manifest_facade() -> None:
    manifest = json.loads((ROOT / "compat_manifest.json").read_text(encoding="utf-8"))
    facades = {entry["facade"] for entry in manifest["entries"]}

    assert LEGACY_OWNER not in facades


def test_phase48_documented_warning_category_requires_minimal_old_path_facade() -> None:
    docs = (ROOT / "COMPAT_MANIFEST.md").read_text(encoding="utf-8")

    assert "hermes_cli.plugin_compat.HermesPluginCompatWarning" in docs


def test_phase48_contributor_docs_name_runtime_owner() -> None:
    docs = (ROOT / "plugins" / "AGENTS.md").read_text(encoding="utf-8")

    assert "plugin_runtime.compat.COMPAT_REMOVAL_DATE" in docs
    assert "`plugin_runtime/compat.py` is the single source" in docs
    assert "hermes_cli.plugin_compat.COMPAT_REMOVAL_DATE" not in docs
    assert "`hermes_cli/plugin_compat.py` is the single source" not in docs


def test_phase48_old_path_facade_exposes_only_canonical_warning_category() -> None:
    import hermes_cli.plugin_compat as legacy
    import plugin_runtime.compat as runtime
    import plugin_runtime.compat_warning as warning_contract

    assert legacy.HermesPluginCompatWarning is runtime.HermesPluginCompatWarning
    assert runtime.HermesPluginCompatWarning is warning_contract.HermesPluginCompatWarning
    assert legacy.__all__ == ("HermesPluginCompatWarning",)
    assert not hasattr(legacy, "warn_once")
    assert not hasattr(legacy, "compat_report")
    assert not hasattr(legacy, "scan_plugin")
    assert not hasattr(legacy, "COMPAT_REMOVAL_DATE")


def test_phase48_legacy_facade_contains_no_runtime_implementation() -> None:
    tree = _tree(LEGACY_FACADE)

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
    imports = [
        (node.module, tuple(alias.name for alias in node.names))
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert imports == [
        ("plugin_runtime.compat_warning", ("HermesPluginCompatWarning",))
    ]
