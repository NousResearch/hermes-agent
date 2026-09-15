"""Regression: ``_evict_modules`` must survive concurrent mutation of ``sys.modules``."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from hermes_cli import plugins_loader

# Enough entries that the filtering pass spans several GIL switches.
_MODULE_COUNT = 5000
_EVICTIONS = 300
# Batched so the mapping's *size* genuinely differs across the eviction's iteration window.
_CHURN_BATCH = 50


@pytest.fixture
def isolated_modules(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Point the loader at a private ``sys.modules`` stand-in.

    A real dict, because the fix rests on ``list(dict)`` being one atomic C-level copy —
    a Python-level mapping stub would not exercise that property.
    """
    modules: dict[str, object] = {
        f"unrelated_{i}": types.ModuleType(f"unrelated_{i}") for i in range(_MODULE_COUNT)
    }
    monkeypatch.setattr(plugins_loader, "sys", types.SimpleNamespace(modules=modules))
    return modules


def test_evict_modules_drops_module_and_submodules(isolated_modules: dict) -> None:
    for name in ("hermes_plugins.aop", "hermes_plugins.aop.client", "hermes_plugins.aop_other"):
        isolated_modules[name] = types.ModuleType(name)

    plugins_loader._evict_modules("hermes_plugins.aop")

    assert "hermes_plugins.aop" not in isolated_modules
    assert "hermes_plugins.aop.client" not in isolated_modules
    # A sibling sharing the prefix without the dot boundary must survive.
    assert "hermes_plugins.aop_other" in isolated_modules


def test_evict_modules_tolerates_concurrent_imports(isolated_modules: dict) -> None:
    """A concurrent import must not abort the plugin load.

    Iterating the live mapping raised ``RuntimeError: dictionary changed size during
    iteration``; ``PluginLoader._load`` catches it and degrades to
    ``Failed to load plugin '<name>': ...``, so none of that plugin's tools ever register
    and every later lookup reports the tool as unknown.
    """
    stop = threading.Event()
    failures: list[BaseException] = []

    def churn() -> None:
        i = 0
        while not stop.is_set():
            batch = [f"late_import_{i}_{j}" for j in range(_CHURN_BATCH)]
            for key in batch:
                isolated_modules[key] = None
            for key in batch:
                isolated_modules.pop(key, None)
            i += 1

    def evict() -> None:
        try:
            for i in range(_EVICTIONS):
                name = f"hermes_plugins.victim_{i}"
                isolated_modules[name] = None
                isolated_modules[f"{name}.sub"] = None
                plugins_loader._evict_modules(name)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion below
            failures.append(exc)

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force frequent handoffs so the race is deterministic
    churner = threading.Thread(target=churn, daemon=True)
    churner.start()
    try:
        evictor = threading.Thread(target=evict)
        evictor.start()
        evictor.join(timeout=60)
        assert not evictor.is_alive(), "eviction thread hung"
    finally:
        stop.set()
        churner.join(timeout=10)
        sys.setswitchinterval(previous_interval)

    assert not failures, f"eviction raised under concurrent import: {failures[0]!r}"
