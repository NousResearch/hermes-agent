"""Phase 4.3 ownership gates for plugin registration and unload lifecycle."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_REGISTRATION = ROOT / "plugin_runtime" / "registration.py"
RUNTIME_OWNERSHIP = ROOT / "plugin_runtime" / "ownership.py"
RUNTIME_SCOPE = ROOT / "plugin_runtime" / "scope.py"
RETIRED_LEDGER = ROOT / "hermes_cli" / "plugins_ledger.py"
RETIRED_LIFECYCLE = ROOT / "registration_lifecycle.py"


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_registration_runtime_has_no_cli_back_edges():
    for path in (RUNTIME_REGISTRATION, RUNTIME_OWNERSHIP, RUNTIME_SCOPE):
        imports = _imports(path)
        assert not any(
            module == "hermes_cli" or module.startswith("hermes_cli.")
            for module in imports
        ), path.relative_to(ROOT)


def test_legacy_registration_owners_are_retired():
    assert not RETIRED_LEDGER.exists()
    assert not RETIRED_LIFECYCLE.exists()


def test_first_party_consumers_do_not_use_retired_registration_paths():
    retired = ("hermes_cli.plugins_ledger", "registration_lifecycle")
    violations = []
    for package in ("hermes_cli", "plugin_runtime", "plugins"):
        for path in (ROOT / package).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if any(name in source for name in retired):
                violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_plugin_manager_binds_canonical_runtime_ownership():
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    import plugin_runtime.ownership as ownership
    import plugin_runtime.registration as registration

    assert issubclass(PluginManager, ownership.PluginOwnershipMixin)
    assert plugins.PluginRegistration is registration.PluginRegistration
    assert PluginManager._track_registration is ownership.PluginOwnershipMixin._track_registration
    assert PluginManager._dispose_registrations is ownership.PluginOwnershipMixin._dispose_registrations
    assert PluginManager.unload is ownership.PluginOwnershipMixin.unload
