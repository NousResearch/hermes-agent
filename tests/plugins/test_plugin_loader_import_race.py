"""Behavior contract: the plugin loader must never expose a half-initialized module.

Child agents run concurrently in a shared-process ThreadPoolExecutor
(delegate_task, max_concurrent_children) and therefore share one ``sys.modules``.
The shared loader (``plugins/plugin_loader.py::load_plugin_module``) registers
the module in ``sys.modules`` BEFORE ``exec_module()`` runs it, and reuses any
entry that already has a ``__file__``. Without serialization, a second
concurrent caller can observe the module (it's in ``sys.modules``) while its
top-level code has NOT finished executing — so ``register`` / the engine class
are not defined yet — grab that half-initialized shell, find no engine, and
silently fall back to the built-in compressor.

This reproduces exactly that window deterministically: a coordination Event is
set from *inside* the plugin's module-top-level (i.e. mid-``exec_module``, after
the loader has already populated ``sys.modules``). The racing thread waits on
that Event, guaranteeing it enters the loader in the vulnerable window.

Contract: BOTH threads must receive a valid engine. Pre-fix, the racing thread
gets ``None`` (built-in fallback). Post-fix, the load lock serializes the
critical section so the racing thread waits for the full load and gets the
engine.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

import plugins.context_engine as ce


_COORD_MODULE = "_hermes_test_race_coord"
_PLUGIN_NAME = "raceplug"
_PLUGIN_MODULE = f"plugins.context_engine.{_PLUGIN_NAME}"


@pytest.fixture
def race_plugin(tmp_path, monkeypatch):
    """Create a fake context-engine plugin whose exec is slow enough to race."""
    # Shared coordination object, reachable from the plugin's top-level code
    # via sys.modules (the plugin is exec'd in a fresh namespace).
    coord = types.ModuleType(_COORD_MODULE)
    setattr(coord, "a_in_exec", threading.Event())
    setattr(coord, "exec_window_s", 0.5)
    sys.modules[_COORD_MODULE] = coord

    plugin_dir = tmp_path / _PLUGIN_NAME
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(
        "import sys, time\n"
        f"_c = sys.modules[{_COORD_MODULE!r}]\n"
        # Signal that we are INSIDE exec_module: the loader has already put this
        # module into sys.modules, but register/_RaceEngine are NOT defined yet.
        "_c.a_in_exec.set()\n"
        # Hold the vulnerable window open long enough for the racing thread to
        # observe the half-initialized module.
        "time.sleep(_c.exec_window_s)\n"
        "\n"
        "class _RaceEngine:\n"
        "    name = 'raceplug'\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_context_engine(_RaceEngine())\n",
        encoding="utf-8",
    )

    # Point the loader at our tmp plugin dir.
    monkeypatch.setattr(ce, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path)

    yield plugin_dir

    # Teardown: drop injected modules so other tests get a clean slate.
    sys.modules.pop(_COORD_MODULE, None)
    sys.modules.pop(_PLUGIN_MODULE, None)


def test_concurrent_load_never_returns_half_initialized_module(race_plugin):
    coord = sys.modules[_COORD_MODULE]
    results: dict[str, object] = {}

    def load_a():
        results["a"] = ce.load_context_engine(_PLUGIN_NAME)

    def load_b():
        # Enter ONLY after A is mid-exec (sys.modules populated, module not yet
        # fully executed) — the exact window the race lives in.
        assert coord.a_in_exec.wait(timeout=5.0), "plugin exec never started"
        results["b"] = ce.load_context_engine(_PLUGIN_NAME)

    ta = threading.Thread(target=load_a, name="loader-A")
    tb = threading.Thread(target=load_b, name="loader-B")
    ta.start()
    tb.start()
    ta.join(timeout=10.0)
    tb.join(timeout=10.0)

    assert not ta.is_alive() and not tb.is_alive(), "loader threads did not finish"

    # The contract: neither thread may observe a half-initialized module.
    # Pre-fix, thread B races into the window and gets None (built-in fallback).
    assert results["a"] is not None, "thread A failed to load the engine"
    assert results["b"] is not None, (
        "thread B observed a half-initialized module and fell back "
        "(the partial-import race is not serialized)"
    )
    assert getattr(results["a"], "name", None) == _PLUGIN_NAME
    assert getattr(results["b"], "name", None) == _PLUGIN_NAME


def test_load_lock_exists_and_is_a_lock():
    """The serialization primitive must exist (guards against a silent refactor)."""
    import plugins.plugin_loader as loader
    lock = getattr(loader, "_LOAD_LOCK", None)
    assert lock is not None, "plugin_loader must define _LOAD_LOCK"
    # RLock/Lock expose acquire/release; assert it behaves as a context manager.
    with lock:
        pass
