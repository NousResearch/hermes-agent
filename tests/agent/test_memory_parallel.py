import threading

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _BlockingProvider(MemoryProvider):
    """Provider that blocks on an Event until released.

    Uses deterministic synchronization instead of time.sleep so tests are
    fast, never flaky, and never block the interpreter longer than needed.
    """

    def __init__(self, name: str):
        self._name = name
        self.sync_done = False
        self.prefetch_done = False
        self._block = threading.Event()

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
        self._block.wait(timeout=5)
        return f"result_from_{self._name}"

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self._block.wait(timeout=5)
        self.prefetch_done = True

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self._block.wait(timeout=5)
        self.sync_done = True

    def release(self):
        self._block.set()

    def get_tool_schemas(self):
        return []


class _SignalingProvider(MemoryProvider):
    """Provider that sets an Event when its work completes."""

    def __init__(self, name: str):
        self._name = name
        self.sync_done = False
        self.prefetch_done = False
        self.sync_event = threading.Event()
        self.prefetch_event = threading.Event()

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
        return f"result_from_{self._name}"

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self.prefetch_done = True
        self.prefetch_event.set()

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.sync_done = True
        self.sync_event.set()

    def get_tool_schemas(self):
        return []


def test_prefetch_all_preserves_registration_order():
    mgr = MemoryManager()
    mgr.add_provider(_SignalingProvider("builtin"))
    mgr.add_provider(_SignalingProvider("external"))

    result = mgr.prefetch_all("hello")

    assert result == "result_from_builtin\n\nresult_from_external"


def test_background_sync_is_isolated_per_registered_provider():
    """A blocked provider must not prevent another provider from syncing."""
    mgr = MemoryManager()
    stuck = _BlockingProvider("builtin")
    fast = _SignalingProvider("external")
    mgr.add_provider(stuck)
    mgr.add_provider(fast)

    mgr.sync_all("hello", "response")

    # The fast provider should complete quickly even though the stuck
    # provider is still blocked on its Event. Deterministic wait, no sleep.
    assert fast.sync_event.wait(timeout=5)
    assert fast.sync_done is True
    assert stuck.sync_done is False  # stuck provider hasn't been released

    # Release the stuck provider so flush_pending can complete
    stuck.release()
    assert mgr.flush_pending(timeout=5) is True
    assert stuck.sync_done is True


def test_background_prefetch_is_isolated_per_registered_provider():
    """A blocked provider must not prevent another provider from prefetching."""
    mgr = MemoryManager()
    stuck = _BlockingProvider("builtin")
    fast = _SignalingProvider("external")
    mgr.add_provider(stuck)
    mgr.add_provider(fast)

    mgr.queue_prefetch_all("hello")

    # The fast provider should complete quickly even though the stuck
    # provider is still blocked on its Event. Deterministic wait, no sleep.
    assert fast.prefetch_event.wait(timeout=5)
    assert fast.prefetch_done is True
    assert stuck.prefetch_done is False  # stuck provider hasn't been released

    # Release the stuck provider so flush_pending can complete
    stuck.release()
    assert mgr.flush_pending(timeout=5) is True
    assert stuck.prefetch_done is True
