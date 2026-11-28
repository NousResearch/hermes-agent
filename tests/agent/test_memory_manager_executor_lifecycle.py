"""A manager's ``mem-sync`` worker must not outlive its queued work.

Every ``AIAgent`` builds its own ``MemoryManager``. The first background write lazily starts a
single-worker ``mem-sync`` ``ThreadPoolExecutor`` that then parked forever: the gateway ran 175
idle ``mem-sync_0`` threads (0.0 % CPU, ``futex_wait``) after two days, one per session since boot,
because retiring a manager's worker depended on an explicit release that not every drop path
performs. The worker is retired when its queue drains, so the count tracks pending writes, not
managers or sessions.
"""
import threading
import time

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager

_WORKER_PREFIX = "mem-sync"


class _RecordingProvider(MemoryProvider):
    """Records every write it is handed, so the test can read the probe write back."""

    _name = "probe"

    def __init__(self):
        self.writes: list[tuple] = []
        self.shutdowns = 0

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

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.writes.append((user_content, assistant_content))

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def shutdown(self) -> None:
        self.shutdowns += 1


def _live_workers() -> int:
    return sum(1 for t in threading.enumerate()
               if t.name.startswith(_WORKER_PREFIX) and t.is_alive())


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    """Bounded wait for a predicate — the worker retires from its own done callback, off-thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _manager_with_provider() -> tuple:
    manager = MemoryManager()
    provider = _RecordingProvider()
    manager.add_provider(provider)
    return manager, provider


def test_idle_worker_retires_after_its_write_drains():
    """The worker must not survive its queue: N managers leave no N parked threads behind."""
    baseline = _live_workers()
    managers = []
    for i in range(8):
        manager, provider = _manager_with_provider()
        manager.sync_all(f"user-{i}", f"assistant-{i}", session_id=f"s{i}")
        assert manager.flush_pending(timeout=10) is True
        assert provider.writes == [(f"user-{i}", f"assistant-{i}")]  # the write landed
        managers.append(manager)

    for manager in managers:
        assert _wait_until(lambda m=manager: m._sync_executor is None), \
            "the mem-sync executor outlived its queued work"
    # The retired workers stop on their None sentinel; the count must fall back to the baseline
    # however many managers were built, instead of parking one thread per manager.
    assert _wait_until(lambda: _live_workers() == baseline), \
        f"mem-sync worker count grew with the number of managers: {_live_workers()} live"


def test_next_write_rebuilds_the_worker_and_keeps_order():
    """Retiring the worker must not drop or reorder the next turn's write."""
    manager, provider = _manager_with_provider()
    for i in range(3):
        manager.sync_all(f"turn-{i}", "ok", session_id="s")
        assert manager.flush_pending(timeout=10) is True

    assert provider.writes == [(f"turn-{i}", "ok") for i in range(3)]
    assert _wait_until(lambda: manager._sync_executor is None)


def test_release_soft_still_runs_once_per_manager():
    """Eviction semantics unchanged: one provider shutdown per manager, however often released."""
    managers = []
    for i in range(4):
        manager, provider = _manager_with_provider()
        manager.sync_all("u", "a", session_id=f"s{i}")
        managers.append((manager, provider))

    for manager, provider in managers:
        manager.release_soft()
        manager.release_soft()  # idempotent: the teardown claim already taken
        assert provider.shutdowns == 1
        assert manager._sync_executor is None


def test_a_wedged_provider_never_blocks_another_managers_write():
    """The per-manager worker is what keeps one stuck backend from starving every other session."""
    stuck = threading.Event()

    class _WedgedProvider(_RecordingProvider):
        _name = "wedged"

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            stuck.wait(10)
            super().sync_turn(user_content, assistant_content, session_id=session_id)

    wedged_manager = MemoryManager()
    wedged_manager.add_provider(_WedgedProvider())
    other_manager, other_provider = _manager_with_provider()
    try:
        wedged_manager.sync_all("u", "a", session_id="s1")  # occupies the wedged manager's worker
        other_manager.sync_all("u2", "a2", session_id="s2")
        assert other_manager.flush_pending(timeout=5) is True
        assert other_provider.writes == [("u2", "a2")]
    finally:
        stuck.set()
    assert wedged_manager.flush_pending(timeout=5) is True
