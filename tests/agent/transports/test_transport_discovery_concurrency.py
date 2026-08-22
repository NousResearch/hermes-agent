"""Concurrency regressions for transport discovery."""

import builtins
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import agent.transports as transports


class _ObservedLock:
    """Expose when the second caller reaches lock acquisition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._attempts = 0
        self.second_attempted = threading.Event()

    def __enter__(self) -> "_ObservedLock":
        with self._state_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.second_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def test_discovery_is_published_after_one_serialized_import_sweep(monkeypatch) -> None:
    target_imports = (
        "agent.transports.anthropic",
        "agent.transports.codex",
        "agent.transports.chat_completions",
        "agent.transports.bedrock",
    )
    observed_lock = _ObservedLock()
    first_import_started = threading.Event()
    release_first_import = threading.Event()
    imported: list[str] = []
    original_import = builtins.__import__

    def controlled_import(name: str, *args: object, **kwargs: object) -> object:
        if name in target_imports:
            imported.append(name)
            if name == target_imports[0]:
                first_import_started.set()
                assert release_first_import.wait(timeout=5)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(transports, "_discovered", False)
    monkeypatch.setattr(transports, "_discover_lock", observed_lock)
    monkeypatch.setattr(builtins, "__import__", controlled_import)

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        first = pool.submit(transports._discover_transports)
        assert first_import_started.wait(timeout=5)
        assert transports._discovered is False

        second = pool.submit(transports._discover_transports)
        assert observed_lock.second_attempted.wait(timeout=5)
        assert second.done() is False
        assert imported == [target_imports[0]]

        release_first_import.set()
        first.result(timeout=5)
        second.result(timeout=5)
    finally:
        release_first_import.set()
        pool.shutdown(wait=True)

    assert transports._discovered is True
    assert imported == list(target_imports)


def test_failed_discovery_is_not_published(monkeypatch) -> None:
    original_import = builtins.__import__

    def fail_first_transport(name: str, *args: object, **kwargs: object) -> object:
        if name == "agent.transports.anthropic":
            raise RuntimeError("transport import failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(transports, "_discovered", False)
    monkeypatch.setattr(builtins, "__import__", fail_first_transport)

    with pytest.raises(RuntimeError, match="transport import failed"):
        transports._discover_transports()

    assert transports._discovered is False
