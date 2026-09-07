"""
Interface compliance tests for all plugin-based gateway platforms.

Discovers platforms dynamically under ``plugins/platforms/`` — no manual
enumeration — and verifies each one implements the required contract.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PLATFORMS_DIR = PROJECT_ROOT / "plugins" / "platforms"


def _discover_platform_plugins() -> list[str]:
    """Return names of all bundled platform plugins."""
    if not PLATFORMS_DIR.is_dir():
        return []
    names = []
    for child in sorted(PLATFORMS_DIR.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            names.append(child.name)
    return names


# Dynamically parametrise over discovered platforms
_PLATFORM_NAMES = _discover_platform_plugins()


@pytest.fixture
def clean_registry():
    """Yield with a clean platform registry, restoring state afterwards."""
    from gateway.platform_registry import platform_registry

    original = dict(platform_registry._entries)
    platform_registry._entries.clear()
    yield platform_registry
    platform_registry._entries.clear()
    platform_registry._entries.update(original)


class _MockPluginContext:
    """Minimal mock of hermes_cli.plugins.PluginContext.

    Only implements register_platform so we can exercise the plugin's
    register() entrypoint without importing the real plugin system.
    """

    def __init__(self):
        self.registered_names: list[str] = []

    def register_platform(
        self,
        *,
        name: str,
        label: str,
        adapter_factory: Any,
        check_fn: Any,
        **kwargs: Any,
    ) -> None:
        from gateway.platform_registry import platform_registry, PlatformEntry

        entry = PlatformEntry(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            **kwargs,
        )
        platform_registry.register(entry)
        self.registered_names.append(name)


def _import_platform_module(name: str) -> ModuleType:
    """Import plugins.platforms.<name> in a test-safe way."""
    # Make sure the project root is on sys.path so relative imports work
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    module = importlib.import_module(f"plugins.platforms.{name}")
    return module


@pytest.mark.parametrize("platform_name", _PLATFORM_NAMES)
def test_plugin_registers_valid_platform_entry(platform_name: str, clean_registry):
    """Calling register() must create a valid PlatformEntry."""
    module = _import_platform_module(platform_name)
    ctx = _MockPluginContext()
    module.register(ctx)

    assert platform_name in ctx.registered_names

    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform_name)
    assert entry is not None, f"{platform_name} did not register an entry"
    assert entry.name == platform_name
    assert entry.label
    assert callable(entry.adapter_factory)
    assert callable(entry.check_fn)


@pytest.mark.parametrize("platform_name", _PLATFORM_NAMES)
def test_platform_entry_has_required_fields(platform_name: str, clean_registry):
    """PlatformEntry must have the mandatory metadata fields."""
    module = _import_platform_module(platform_name)
    ctx = _MockPluginContext()
    module.register(ctx)

    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform_name)
    assert entry is not None

    # Mandatory fields
    assert isinstance(entry.name, str) and entry.name
    assert isinstance(entry.label, str) and entry.label
    assert callable(entry.adapter_factory)
    assert callable(entry.check_fn)

    # Optional but recommended fields
    if entry.validate_config is not None:
        assert callable(entry.validate_config)
    if entry.is_connected is not None:
        assert callable(entry.is_connected)
    if entry.setup_fn is not None:
        assert callable(entry.setup_fn)


