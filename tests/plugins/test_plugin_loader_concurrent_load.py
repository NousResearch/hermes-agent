"""Concurrent plugin-module load guard — regression test for the startup race.

Background (observed 2026-09-08, dashboard/gateway restart):
  After a restart the process builds agents concurrently (one AIAgent per
  session). Both call ``load_plugin_module`` for the same plugin module at the
  same time. The loader puts the *unevaluated* module object into
  ``sys.modules`` before ``exec_module`` runs (needed so intra-plugin relative
  imports resolve), so the second caller can hit the half-built module in the
  cache: ``__file__`` is already present but ``register`` is not defined yet →
  the subclass scan finds nothing → the loader returns None → the provider/engine
  is silently disabled for that agent (no recall, no memory tools/context
  engine). Observed live: exactly one of two concurrent agent builds logged
  "Memory provider 'X' loaded but no provider instance found" while the sibling
  build 9ms later registered it successfully.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

# Import order stability: loading the loader package first leaves
# ``plugins.memory`` unbound on the ``plugins`` package, which breaks string
# monkeypatch targets ("plugins.memory._get_user_plugins_dir") in unrelated
# tests that assume the attribute exists. Bind it explicitly.
import plugins.memory  # noqa: F401  (side effect: binds plugins.memory)
from plugins import plugin_loader

# Slow plugin: sleeps during module evaluation (exec_module), keeping the
# module in sys.modules as a half-built object long enough for a concurrent
# caller to observe it.
_SLOW_SOURCE = """\
import time
time.sleep(0.2)


def register(ctx):
    ctx.register_memory_provider(object())
"""


@pytest.fixture
def slow_plugin(tmp_path):
    plugin_dir = tmp_path / "slowplugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_SLOW_SOURCE, encoding="utf-8")
    yield plugin_dir
    sys.modules.pop("plugins.memory.slowplugin", None)


def test_concurrent_load_slow_plugin(slow_plugin):
    """Two concurrent loads of the same module must both get the evaluated module.

    Before the fix the second loader can observe the half-built module
    (sys.modules entry exists, exec_module still sleeping) → returns None.
    With the fix the load is serialized per module, so the second caller waits
    and then reuses the fully-evaluated module.
    """
    results: list = []
    results_lock = threading.Lock()
    module_name = "plugins.memory.slowplugin"

    def _load():
        got = plugin_loader.load_plugin_module(
            module_name,
            slow_plugin,
            parents=("plugins", "plugins.memory"),
            logger=logging.getLogger("test"),
            synthetic_namespace="_hermes_user_memory",
        )
        # Record the register state IMMEDIATELY at return time: a half-built
        # module is the SAME object the first loader finishes later, so a
        # deferred check (after join) would see register defined and miss the
        # race window entirely.
        with results_lock:
            results.append((got is not None, hasattr(got, "register") if got is not None else None))

    t1 = threading.Thread(target=_load)
    t2 = threading.Thread(target=_load)
    t1.start()
    # Deterministically wait until the half-built module is observable
    # (sys.modules entry present, register not yet defined), THEN start t2.
    deadline = time.time() + 5
    while time.time() < deadline:
        mod = sys.modules.get(module_name)
        if mod is not None and not hasattr(mod, "register"):
            break
        time.sleep(0.005)
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(results) == 2, f"loaders did not both finish: {results!r}"
    assert all(ok and has_register for ok, has_register in results), (
        "Bug: concurrent load returned a half-built module for one caller — "
        "the module was in sys.modules but exec_module had not finished (no "
        "register defined), and the caller reused it instead of waiting. Live "
        "effect: the provider/engine is silently disabled for that agent "
        "(observed 2026-09-08: 'Memory provider ... loaded but no provider "
        "instance found' on one of two concurrent agent builds)."
    )
