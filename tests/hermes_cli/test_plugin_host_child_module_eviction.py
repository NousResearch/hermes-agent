"""Regression: the plugin host child must survive concurrent mutation of ``sys.modules``.

``plugins_loader._evict_modules`` was fixed for this (1f6e0ddb52, hardened in 952d55d2e7), but the
plugin host child (``plugins.isolation: host``) kept two copies of the live-mapping comprehension:
``HostRuntime.op_unload`` and the failed-import cleanup in ``_import_plugin``. The host runs plugin
async code on its ``plugin-host-loop`` thread and plugin-started threads, so a lazy import there
races either eviction and raises ``RuntimeError: dictionary changed size during iteration``. On
unload the RPC fails and the plugin's modules stay cached; on a failed import the plugin's own
exception is replaced by that RuntimeError.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from hermes_cli import plugin_host_child

# Same sizing as tests/hermes_cli/test_plugin_loader_module_eviction.py: enough entries that the
# filtering pass spans several GIL switches, and a batched churn so the size genuinely changes.
_MODULE_COUNT = 5000
_ROUNDS = 300
_CHURN_BATCH = 50


@pytest.fixture
def isolated_modules(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Point the host child at a private ``sys.modules`` stand-in (a real dict: the fix rests on
    ``list(dict)`` being one atomic C-level copy). Only the eviction paths may run under it."""
    modules: dict[str, object] = {
        f"unrelated_{i}": types.ModuleType(f"unrelated_{i}") for i in range(_MODULE_COUNT)
    }
    monkeypatch.setattr(plugin_host_child, "sys", types.SimpleNamespace(modules=modules))
    return modules


@pytest.fixture
def fast_thread_switching() -> Iterator[None]:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _runtime() -> plugin_host_child.HostRuntime:
    """The tables ``op_unload`` reads, without starting the host's loop thread or channel."""
    runtime = object.__new__(plugin_host_child.HostRuntime)
    runtime.refs, runtime.owners, runtime.loads = {}, {}, {}
    runtime.unload_callbacks, runtime.modules = {}, {}
    runtime._lock = threading.Lock()
    return runtime


def _race(isolated_modules: dict, work) -> tuple[list[BaseException], int]:
    """Run ``work`` on a thread while another thread imports and drops modules."""
    stop = threading.Event()
    failures: list[BaseException] = []
    churn_rounds = 0

    def churn() -> None:
        nonlocal churn_rounds
        while not stop.is_set():
            batch = [f"late_import_{churn_rounds}_{j}" for j in range(_CHURN_BATCH)]
            for key in batch:
                isolated_modules[key] = None
            for key in batch:
                isolated_modules.pop(key, None)
            churn_rounds += 1

    def run() -> None:
        try:
            work()
        except BaseException as exc:  # recorded for the caller's assertion
            failures.append(exc)

    churner = threading.Thread(target=churn, daemon=True)
    churner.start()
    try:
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=60)
        assert not worker.is_alive(), "eviction thread hung"
    finally:
        stop.set()
        churner.join(timeout=10)
    return failures, churn_rounds


def test_unload_evicts_plugin_modules_under_concurrent_imports(
    isolated_modules: dict, fast_thread_switching: None,
) -> None:
    runtime = _runtime()

    def unload_many() -> None:
        for i in range(_ROUNDS):
            name = f"hermes_plugins.victim_{i}"
            isolated_modules[name] = None
            isolated_modules[f"{name}.sub"] = None
            runtime.modules[f"victim_{i}"] = name
            assert runtime.op_unload({"plugin_key": f"victim_{i}"}) == {"errors": []}

    failures, churn_rounds = _race(isolated_modules, unload_many)

    assert not failures, f"op_unload raised under concurrent import: {failures[0]!r}"
    assert churn_rounds > 0, "churn thread never ran — the race window was never opened"
    leftover = [name for name in isolated_modules if name.startswith("hermes_plugins.victim_")]
    assert not leftover, f"unload left entries behind: {leftover[:5]}"


def test_failed_import_keeps_its_own_error_under_concurrent_imports(
    isolated_modules: dict, fast_thread_switching: None, tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "broken_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("raise ValueError('plugin failed to import')\n")
    params = {"path": str(plugin_dir), "module_name": "hermes_plugins.broken_plugin"}
    raised: list[BaseException] = []

    def import_many() -> None:
        for _ in range(_ROUNDS):
            isolated_modules["hermes_plugins.broken_plugin.helper"] = None
            try:
                plugin_host_child._import_plugin(params)
            except BaseException as exc:  # the plugin's own error is the expected outcome
                raised.append(exc)

    failures, churn_rounds = _race(isolated_modules, import_many)

    assert not failures
    assert churn_rounds > 0, "churn thread never ran — the race window was never opened"
    assert len(raised) == _ROUNDS
    wrong = [exc for exc in raised if not isinstance(exc, ValueError)]
    assert not wrong, f"the plugin's ValueError was replaced: {wrong[0]!r}"
    leftover = [name for name in isolated_modules if name.startswith("hermes_plugins.broken_plugin")]
    assert not leftover, f"failed import left entries behind: {leftover}"
