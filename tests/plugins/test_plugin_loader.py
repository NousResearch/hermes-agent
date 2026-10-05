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


def test_concurrent_first_loads_all_see_the_initialized_module(tmp_path):
    """The module is reserved in sys.modules before it executes: without serialization a second
    caller got the half-initialized module (no ``register``) and the engine silently fell back."""
    import threading
    import time

    plugin_dir = tmp_path / "slow_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(
        "import time\ntime.sleep(0.2)\ndef register(ctx):\n    pass\n", encoding="utf-8")
    module_name = "test_plugin_loader_package.slow_concurrent"
    barrier = threading.Barrier(8)
    seen = []

    def load():
        barrier.wait()
        mod = load_plugin_module(module_name, plugin_dir, parents=(), logger=logging.getLogger(__name__))
        seen.append(hasattr(mod, "register"))

    try:
        threads = [threading.Thread(target=load) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == [True] * 8
    finally:
        for name in tuple(sys.modules):
            if name.startswith("test_plugin_loader_package"):
                sys.modules.pop(name, None)


def test_unrelated_plugins_load_in_parallel(tmp_path):
    """The load lock is per module name: six independent slow plugins must not serialize."""
    import threading
    import time

    names = []
    for i in range(6):
        plugin_dir = tmp_path / f"par_{i}"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            "import time\ntime.sleep(0.3)\ndef register(ctx):\n    pass\n", encoding="utf-8")
        names.append((f"test_plugin_loader_package.par_{i}", plugin_dir))
    barrier = threading.Barrier(len(names))
    seen = []

    def load(module_name, plugin_dir):
        barrier.wait()
        mod = load_plugin_module(module_name, plugin_dir, parents=(), logger=logging.getLogger(__name__))
        seen.append(hasattr(mod, "register"))

    try:
        threads = [threading.Thread(target=load, args=n) for n in names]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        assert seen == [True] * 6
        assert elapsed < 1.2, f"unrelated plugins serialized ({elapsed:.2f}s for 6 x 0.3s)"
    finally:
        for name in tuple(sys.modules):
            if name.startswith("test_plugin_loader_package"):
                sys.modules.pop(name, None)
