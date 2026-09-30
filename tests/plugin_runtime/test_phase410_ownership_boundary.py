"""Permanent Phase 4.10 ownership ratchet for the plugin runtime boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_RUNTIME_ROOT = ROOT / "plugin_runtime"
PLUGIN_COMPAT_FACADE = ROOT / "hermes_cli" / "plugin_compat.py"
PLUGINS_FACADE = ROOT / "hermes_cli" / "plugins.py"
CONFIG_BRIDGE = PLUGIN_RUNTIME_ROOT / "config_bridge.py"

SOURCE_ROOTS = (
    "acp_adapter",
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "nous_cli",
    "plugin_runtime",
    "plugins",
    "providers",
    "tools",
    "tui_gateway",
)
ROOT_SOURCES = (
    "cli.py",
    "hermes_state.py",
    "model_tools.py",
    "run_agent.py",
)

# These were implementation owners before plugin_runtime became canonical. They are
# intentionally absent: compatibility/process-boundary modules are not retired here.
RETIRED_PLUGIN_RUNTIME_MODULES = (
    "hermes_cli.plugin_capabilities",
    "hermes_cli.plugin_index",
    "hermes_cli.plugins_activation_live",
    "hermes_cli.plugins_discovery",
    "hermes_cli.plugins_dispatch",
    "hermes_cli.plugins_ledger",
    "hermes_cli.plugins_loader",
    "hermes_cli.plugins_manifest",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _production_sources() -> tuple[Path, ...]:
    sources = [ROOT / name for name in ROOT_SOURCES if (ROOT / name).exists()]
    for root_name in SOURCE_ROOTS:
        root = ROOT / root_name
        if root.exists():
            sources.extend(root.rglob("*.py"))
    return tuple(sorted(set(sources)))


def _imports(path: Path) -> tuple[tuple[int, str], ...]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
    return tuple(found)


def _module_matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _violations(
    paths: tuple[Path, ...],
    forbidden: tuple[str, ...],
    *,
    allowed: dict[Path, tuple[str, ...]] | None = None,
) -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    allowed = allowed or {}
    for path in paths:
        permitted = allowed.get(path, ())
        for lineno, module in _imports(path):
            if any(_module_matches(module, prefix) for prefix in forbidden) and not any(
                _module_matches(module, prefix) for prefix in permitted
            ):
                violations.append((path.relative_to(ROOT).as_posix(), lineno, module))
    return sorted(violations)


def test_phase410_plugin_runtime_has_no_application_backedges() -> None:
    violations = _violations(
        tuple(PLUGIN_RUNTIME_ROOT.rglob("*.py")),
        ("gateway", "nous_cli", "hermes_cli"),
        allowed={CONFIG_BRIDGE: ("hermes_cli.config", "hermes_cli.plugins_state")},
    )

    assert violations == []


def test_phase410_gateway_does_not_import_cli_plugin_runtime_surfaces() -> None:
    forbidden = ("hermes_cli.plugins", "hermes_cli.plugin_compat") + RETIRED_PLUGIN_RUNTIME_MODULES
    violations = _violations(tuple((ROOT / "gateway").rglob("*.py")), forbidden)

    assert violations == []


def test_phase410_first_party_does_not_consume_legacy_plugins_api() -> None:
    forbidden = ("hermes_cli.plugins",)
    sources = tuple(
        path
        for path in _production_sources()
        if path not in {PLUGINS_FACADE, PLUGIN_COMPAT_FACADE}
    )
    violations = _violations(sources, forbidden, allowed={CONFIG_BRIDGE: ("hermes_cli.plugins_state",)})

    assert violations == []


def test_phase410_retired_plugin_runtime_paths_remain_absent() -> None:
    missing_enforcement = [
        module.replace(".", "/") + ".py"
        for module in RETIRED_PLUGIN_RUNTIME_MODULES
        if (ROOT / (module.replace(".", "/") + ".py")).exists()
    ]

    assert missing_enforcement == []


def test_phase410_compatibility_facades_are_facade_only() -> None:
    plugin_compat_tree = _tree(PLUGIN_COMPAT_FACADE)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in plugin_compat_tree.body
    )

    plugins_tree = _tree(PLUGINS_FACADE)
    assert not any(isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef)) for node in plugins_tree.body)
    assert [
        node.name
        for node in plugins_tree.body
        if isinstance(node, ast.FunctionDef)
    ] == ["__getattr__"]
