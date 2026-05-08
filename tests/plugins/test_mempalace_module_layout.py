from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "memory" / "mempalace"
MODULE_NAME = "plugins.memory.mempalace"


def _ensure_package(name: str, package_dir: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader and (package_dir / "__init__.py").exists():
        spec.loader.exec_module(module)


def _load_plugin_root():
    _ensure_package("plugins", REPO_ROOT / "plugins")
    _ensure_package("plugins.memory", REPO_ROOT / "plugins" / "memory")
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_publishable_module_layout_exists():
    for filename in [
        "provider.py",
        "tools.py",
        "hooks.py",
        "schemas.py",
    ]:
        assert (PLUGIN_DIR / filename).exists(), filename


def test_provider_is_reexported_from_package_root():
    root = _load_plugin_root()
    from plugins.memory.mempalace.provider import MemPalaceMemoryProvider

    assert root.MemPalaceMemoryProvider is MemPalaceMemoryProvider
    assert root.MemPalaceMemoryProvider.__module__ == "plugins.memory.mempalace.provider"
