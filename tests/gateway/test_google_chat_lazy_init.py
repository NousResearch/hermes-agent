"""Concurrent passive dependency probes must not publish a half-loaded adapter."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from plugins.platforms.google_chat import adapter


@pytest.fixture
def fresh_loader(monkeypatch):
    monkeypatch.setattr(adapter, "_google_modules_loaded", False)
    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)
    for name, _, _ in adapter._GOOGLE_IMPORTS:
        monkeypatch.setattr(adapter, name, None)
    modules = {}
    expected = {}
    for name, module, attr in adapter._GOOGLE_IMPORTS:
        value = object()
        modules[module] = SimpleNamespace(**{attr: value}) if attr else value
        expected[name] = value
    monkeypatch.setattr(adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__))
    return modules, expected


def test_concurrent_passive_probes_wait_for_complete_imports(fresh_loader, monkeypatch):
    modules, expected = fresh_loader
    importing = threading.Event()
    release = threading.Event()
    second_reached_boundary = threading.Event()
    real_lock = threading.Lock()
    first_thread = []

    class ObservedLock:
        def __enter__(self):
            if first_thread and threading.get_ident() != first_thread[0]:
                second_reached_boundary.set()
            real_lock.acquire()

        def __exit__(self, *args):
            real_lock.release()

    monkeypatch.setattr(adapter, "_google_modules_lock", ObservedLock(), raising=False)

    def load(module):
        if not first_thread:
            first_thread.append(threading.get_ident())
            importing.set()
            assert release.wait(10), "test did not release the paused import"
        return modules[module]

    monkeypatch.setattr(adapter.importlib, "import_module", load)

    def second_probe():
        try:
            result = adapter.check_google_chat_requirements()
            return result, {name: getattr(adapter, name) for name in expected}
        finally:
            second_reached_boundary.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(adapter.check_google_chat_requirements)
        try:
            assert importing.wait(10)
            second = pool.submit(second_probe)
            assert second_reached_boundary.wait(10)
        finally:
            release.set()
        assert first.result(timeout=10) is True
        result, observed = second.result(timeout=10)
    assert result is True, "concurrent probe saw unavailable while imports were still running"
    assert observed == expected


def test_success_is_cached(fresh_loader, monkeypatch):
    _, expected = fresh_loader
    assert adapter.check_google_chat_requirements() is True
    assert {name: getattr(adapter, name) for name in expected} == expected
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: pytest.fail("reimported"))
    assert adapter.check_google_chat_requirements() is True


def test_missing_optional_dependency_is_cached_without_partial_globals(fresh_loader, monkeypatch):
    modules, expected = fresh_loader
    missing = adapter._GOOGLE_IMPORTS[-1][1]
    calls = []

    def load(module):
        calls.append(module)
        if module == missing:
            raise ImportError("optional dependency absent")
        return modules[module]

    monkeypatch.setattr(adapter.importlib, "import_module", load)
    assert adapter.check_google_chat_requirements() is False
    previous = list(calls)
    assert adapter.check_google_chat_requirements() is False
    assert calls == previous
    assert all(getattr(adapter, name) is None for name in expected)


def test_unexpected_import_error_does_not_poison_retry(fresh_loader, monkeypatch):
    modules, _ = fresh_loader

    def broken(_):
        raise RuntimeError("transient loader failure")

    monkeypatch.setattr(adapter.importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="transient loader failure"):
        adapter.check_google_chat_requirements()
    monkeypatch.setattr(adapter.importlib, "import_module", modules.__getitem__)
    assert adapter.check_google_chat_requirements() is True
