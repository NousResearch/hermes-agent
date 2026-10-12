"""Tests for the parallel deferred-platform load in ``PlatformRegistry._resolve_all()``.

Loaders are independent module imports, so ``all_entries()`` runs them through a thread pool
(~1.0s instead of ~2.0s for the 22 bundled platforms, where telegram alone is ~800ms). What must
hold regardless of the pool: every pending loader runs exactly once, and a pool that cannot start
degrades to the original sequential pass rather than dropping platforms.
"""

import threading

import pytest

from gateway.platform_registry import PlatformEntry, PlatformRegistry


def _entry(name: str) -> PlatformEntry:
    return PlatformEntry(name=name, label=name.title(), adapter_factory=lambda _config: object(),
                         check_fn=lambda: True)


@pytest.fixture
def registry_with_deferred():
    """Registry holding 6 deferred platforms, plus the per-name call counter."""
    registry = PlatformRegistry()
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def make_loader(name: str):
        def loader() -> None:
            with lock:
                calls[name] = calls.get(name, 0) + 1
            registry.register(_entry(name))
        return loader

    names = [f"plat{i}" for i in range(6)]
    for name in names:
        registry.register_deferred(name, make_loader(name))
    return registry, calls, names


def test_all_entries_resolves_every_deferred_loader(registry_with_deferred):
    registry, calls, names = registry_with_deferred

    entries = registry.all_entries()

    assert sorted(e.name for e in entries) == sorted(names)
    assert calls == {name: 1 for name in names}, "each loader must run exactly once"


def test_large_sets_use_the_pool(registry_with_deferred, monkeypatch):
    """The whole point: more than a couple of loaders run concurrently, not one after another."""
    import concurrent.futures as futures

    registry, _calls, names = registry_with_deferred
    real_pool = futures.ThreadPoolExecutor
    used = {"pool": False}

    class RecordingPool(real_pool):
        def __init__(self, *args, **kwargs):
            used["pool"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(futures, "ThreadPoolExecutor", RecordingPool)

    entries = registry.all_entries()

    assert used["pool"] is True
    assert sorted(e.name for e in entries) == sorted(names)


def test_pool_failure_falls_back_to_sequential(registry_with_deferred, monkeypatch):
    """A thread pool that cannot start must not cost us platforms."""
    registry, calls, names = registry_with_deferred

    class UnusablePool:
        def __init__(self, *a, **kw):
            raise RuntimeError("can't start a thread here")

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", UnusablePool)

    entries = registry.all_entries()

    assert sorted(e.name for e in entries) == sorted(names)
    assert calls == {name: 1 for name in names}


def test_small_sets_skip_the_pool(monkeypatch):
    """Two platforms are not worth a pool; they resolve inline."""
    registry = PlatformRegistry()
    used = {"pool": False}

    class MarkingPool:
        def __init__(self, *a, **kw):
            used["pool"] = True
            raise RuntimeError("should not be reached")

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", MarkingPool)
    for name in ("solo", "duo"):
        registry.register_deferred(name, (lambda n: lambda: registry.register(_entry(n)))(name))

    assert sorted(e.name for e in registry.all_entries()) == ["duo", "solo"]
    assert used["pool"] is False


def test_loader_exception_does_not_block_the_others(monkeypatch):
    """One broken adapter must not take the rest of the listing down with it."""
    registry = PlatformRegistry()

    def boom() -> None:
        raise ImportError("adapter SDK missing")

    registry.register_deferred("broken", boom)
    for name in ("alpha", "beta", "gamma"):
        registry.register_deferred(name, (lambda n: lambda: registry.register(_entry(n)))(name))

    assert sorted(e.name for e in registry.all_entries()) == ["alpha", "beta", "gamma"]
