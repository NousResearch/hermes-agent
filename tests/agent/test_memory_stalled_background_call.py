"""A background provider call that never returns pins its mem-sync worker thread forever (Python cannot
stop a thread). MemoryManagers are rebuilt per agent, so without a process-wide view every fresh manager
started one more call against the hung backend and pinned one more thread. Calls for a provider are now
skipped while an earlier call for it is stuck past the stall threshold."""
import threading

import pytest

import agent.memory_manager as mm
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from tests.waiting import wait_until


class _HangingProvider(MemoryProvider):
    def __init__(self, name: str = "hangs"):
        self._name = name
        self.release = threading.Event()
        self.started = threading.Event()
        self.sync_calls = 0
        self.prefetch_calls = 0
        self.hang_next = True

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def get_tool_schemas(self):
        return []

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self.prefetch_calls += 1

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "") -> None:
        self.sync_calls += 1
        if self.hang_next:
            self.hang_next = False
            self.started.set()
            self.release.wait(10)


@pytest.fixture(autouse=True)
def _short_stall(monkeypatch):
    monkeypatch.setattr(mm, "_BACKGROUND_CALL_STALL_S", 0.0)
    MemoryManager._BACKGROUND_CALLS.clear()
    MemoryManager._BACKGROUND_SKIPS.clear()
    yield
    MemoryManager._BACKGROUND_CALLS.clear()
    MemoryManager._BACKGROUND_SKIPS.clear()


def _manager(provider):
    mgr = MemoryManager()
    mgr.add_provider(provider)
    return mgr


def _mem_sync_threads():
    return [t for t in threading.enumerate() if t.name.startswith("mem-sync")]


def test_stuck_call_stops_fresh_managers_from_pinning_more_threads():
    provider = _HangingProvider()
    first = _manager(provider)
    first.sync_all("hello", "hi")
    assert provider.started.wait(5)
    before = len(_mem_sync_threads())

    for _ in range(5):
        fresh = _manager(provider)
        fresh.sync_all("again", "reply")
        fresh.queue_prefetch_all("query")
        assert fresh.flush_pending(timeout=5)  # returns: nothing blocked behind the hung call
        fresh.shutdown_all()

    assert provider.sync_calls == 1 and provider.prefetch_calls == 0
    assert MemoryManager._BACKGROUND_SKIPS[(first._home_key, provider.name)] == 10
    # Idle workers of shut-down executors exit asynchronously.
    wait_until(lambda: len(_mem_sync_threads()) <= before, timeout=5, what="idle mem-sync workers to exit")

    provider.release.set()
    assert first.flush_pending(timeout=5)
    assert (first._home_key, provider.name) not in MemoryManager._BACKGROUND_CALLS
    first.sync_all("next", "turn")  # calls resume once the stuck one returns
    assert first.flush_pending(timeout=5)
    assert provider.sync_calls == 2
    first.shutdown_all()


def test_slow_call_below_threshold_is_not_skipped(monkeypatch):
    monkeypatch.setattr(mm, "_BACKGROUND_CALL_STALL_S", 60.0)
    provider = _HangingProvider()
    first = _manager(provider)
    first.sync_all("hello", "hi")
    assert provider.started.wait(5)

    second = _manager(provider)
    second.sync_all("again", "reply")
    assert second.flush_pending(timeout=5)
    assert provider.sync_calls == 2

    provider.release.set()
    for m in (first, second):
        m.shutdown_all()
