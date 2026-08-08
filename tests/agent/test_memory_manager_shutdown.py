"""Regression tests for synchronized MemoryManager shutdown."""

import threading

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _ShutdownProvider(MemoryProvider):
    @property
    def name(self):
        return "shutdown-test"

    def __init__(self, entered=None, release=None):
        self.shutdown_calls = 0
        self.entered = entered
        self.release = release

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        self.shutdown_calls += 1
        if self.entered:
            self.entered.set()
        if self.release:
            assert self.release.wait(timeout=2)


def test_shutdown_all_is_idempotent():
    manager = MemoryManager()
    provider = _ShutdownProvider()
    manager.add_provider(provider)

    manager.shutdown_all()
    manager.shutdown_all()

    assert provider.shutdown_calls == 1


def test_shutdown_all_serializes_concurrent_callers():
    entered = threading.Event()
    release = threading.Event()
    manager = MemoryManager()
    provider = _ShutdownProvider(entered, release)
    manager.add_provider(provider)

    first = threading.Thread(target=manager.shutdown_all)
    second = threading.Thread(target=manager.shutdown_all)
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    # The second caller must wait for the first, rather than invoking the
    # provider concurrently while its resources are being torn down.
    assert second.is_alive()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert provider.shutdown_calls == 1
