"""Regression tests for directory-plugin module loading."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from types import SimpleNamespace

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


def test_concurrent_load_waits_bounded_for_the_first_load(tmp_path, monkeypatch):
    """A second caller never receives the half-built shell published mid-load, and a hung import
    in the first caller does not block it forever: it refuses after the bounded wait."""
    import plugins.plugin_loader as loader

    monkeypatch.setattr(loader, "_CONCURRENT_LOAD_WAIT_SECS", 0.3)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    gate_name = "_test_plugin_loader_race_gate"
    gate = SimpleNamespace(entered=threading.Event(), release=threading.Event())
    sys.modules[gate_name] = gate
    (plugin_dir / "slow.py").write_text(
        f"import sys\ngate = sys.modules[{gate_name!r}]\n"
        "gate.entered.set()\ngate.release.wait(10)\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    module_name = "test_plugin_loader_package.concurrent_load"
    results = {}

    def load(key):
        results[key] = load_plugin_module(module_name, plugin_dir, parents=(),
                                          logger=logging.getLogger(__name__))

    first = threading.Thread(target=load, args=("first",))
    second = threading.Thread(target=load, args=("second",))
    try:
        first.start()
        assert gate.entered.wait(5)
        second.start()
        second.join(5)
        assert not second.is_alive(), "second caller blocked on the hung first load"
        assert results["second"] is None
        load("stalled")  # later callers refuse at once while the import is still hung
        assert results["stalled"] is None
        gate.release.set()
        first.join(10)
        assert hasattr(results["first"], "register")
        load("after")
        assert results["after"] is results["first"]

        # A cross-thread cycle through importlib's own module lock (thread A imports a helper that
        # loads the plugin; thread B loads the plugin, which imports the helper) must not deadlock.
        monkeypatch.setattr(loader, "_CONCURRENT_LOAD_WAIT_SECS", 60)
        cyc_dir, helper = tmp_path / "cyc", "_test_plugin_loader_cycle_helper"
        cyc_dir.mkdir()
        cyc_name = "test_plugin_loader_package.cycle"
        gate.helper, gate.loader = threading.Event(), threading.Event()
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / f"{helper}.py").write_text(
            f"import sys, logging, pathlib\nfrom plugins.plugin_loader import load_plugin_module\n"
            f"gate = sys.modules[{gate_name!r}]\ngate.helper.set()\ngate.loader.wait(10)\n"
            f"mod = load_plugin_module({cyc_name!r}, pathlib.Path({str(cyc_dir)!r}), parents=(),"
            " logger=logging.getLogger('x'))\n", encoding="utf-8")
        (cyc_dir / "__init__.py").write_text(
            f"import sys\nsys.modules[{gate_name!r}].loader.set()\nimport {helper}\n"
            "def register(ctx):\n    pass\n", encoding="utf-8")
        a = threading.Thread(target=importlib.import_module, args=(helper,), daemon=True)
        b = threading.Thread(target=load_plugin_module, args=(cyc_name, cyc_dir),
                             kwargs={"parents": (), "logger": logging.getLogger(__name__)}, daemon=True)
        a.start()
        assert gate.helper.wait(5)
        b.start()
        a.join(5)
        b.join(5)
        assert not a.is_alive() and not b.is_alive(), "loader lock and import lock deadlocked"
        assert hasattr(sys.modules[cyc_name], "register")
    finally:
        gate.release.set()
        getattr(gate, "loader", threading.Event()).set()
        sys.modules.pop(gate_name, None)
        sys.modules.pop("_test_plugin_loader_cycle_helper", None)
        for name in tuple(sys.modules):
            if name.startswith("test_plugin_loader_package."):
                sys.modules.pop(name, None)
