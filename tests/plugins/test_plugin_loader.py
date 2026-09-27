"""Regression tests for directory-plugin module loading."""

from __future__ import annotations

import logging
import sys

from plugins.plugin_loader import load_plugin_module


def test_failed_sibling_is_removed_before_init_handles_missing_import(tmp_path):
    """A failed eager sibling import must remain catchable as ModuleNotFoundError."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "broken.py").write_text(
        "from .missing_dependency import value\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "try:\n"
        "    from .broken import value\n"
        "except ModuleNotFoundError:\n"
        "    fallback_used = True\n",
        encoding="utf-8",
    )
    module_name = "test_plugin_loader_package.failed_sibling"

    try:
        module = load_plugin_module(
            module_name,
            plugin_dir,
            parents=(),
            logger=logging.getLogger(__name__),
        )

        assert module is not None
        assert module.fallback_used is True
        assert f"{module_name}.broken" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)


def test_successful_sibling_remains_available_on_loaded_module(tmp_path):
    """Cleaning failed siblings must not alter the eager success path."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "helper.py").write_text("value = 42\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "from .helper import value\n",
        encoding="utf-8",
    )
    module_name = "test_plugin_loader_package.successful_sibling"

    try:
        module = load_plugin_module(
            module_name,
            plugin_dir,
            parents=(),
            logger=logging.getLogger(__name__),
        )

        assert module is not None
        assert module.value == 42
        assert module.helper.value == 42
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)
