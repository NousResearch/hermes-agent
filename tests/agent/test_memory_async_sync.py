"""Regression guard: end-of-turn memory sync must not block the turn.

Before this fix, ``MemoryManager.sync_all`` / ``queue_prefetch_all`` looped
``provider.sync_turn`` / ``provider.queue_prefetch`` INLINE on the
turn-completion path. A provider making a blocking network/daemon call (a
misconfigured Hindsight daemon was observed blocking ~298s before failing)
held ``run_conversation`` open long after the user saw their response, so
every interface (CLI, TUI, gateway) kept the agent marked "running" for
minutes and any follow-up message triggered an aggressive interrupt that
dropped the message.

The fix dispatches provider work to per-provider background executors.
``sync_all`` / ``queue_prefetch_all`` return immediately; the work completes
(or fails, logged) in the background. ``flush_pending`` provides a barrier
for session boundaries and deterministic tests. ``shutdown_all`` drains the
executors with a bounded timeout so a wedged provider can't hang teardown.
"""
import threading
import time

import pytest

from agent import memory_manager as memory_manager_module
from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


class _SlowProvider(MemoryProvider):
    """Provider whose sync/prefetch block, simulating a slow backend."""

    _name = "slow"

    def __init__(self, delay: float = 1.0):
        self._delay = delay
        self.sync_done = False
        self.prefetch_done = False

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
        time.sleep(self._delay)
        self.prefetch_done = True

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        time.sleep(self._delay)
        self.sync_done = True

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


class _ContendedProvider(MemoryProvider):
    """Provider that keeps sync busy so the test can observe prefetch start."""

    _name = "contended"

    def __init__(self):
        self.sync_started = threading.Event()
        self.allow_sync_finish = threading.Event()
        self.prefetch_started = threading.Event()

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
        self.prefetch_started.set()

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.sync_started.set()
        assert self.allow_sync_finish.wait(timeout=5), "test failed to release sync_turn"

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


class _ShutdownBlockedProvider(MemoryProvider):
    """Provider whose sync and prefetch lanes both stay busy during shutdown."""

    _name = "shutdown-blocked"

    def __init__(self):
        self.sync_started = threading.Event()
        self.prefetch_started = threading.Event()
        self.allow_sync_finish = threading.Event()
        self.allow_prefetch_finish = threading.Event()

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
        self.prefetch_started.set()
        assert self.allow_prefetch_finish.wait(timeout=5), "test failed to release prefetch"

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.sync_started.set()
        assert self.allow_sync_finish.wait(timeout=5), "test failed to release sync_turn"

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


def test_sync_all_does_not_block_on_slow_provider():
    """The crux of the fix: a slow provider must NOT stall the caller."""
    mgr = MemoryManager()
    mgr.add_provider(_SlowProvider(delay=2.0))

    t0 = time.time()
    mgr.sync_all("hi", "hey", session_id="s1")
    mgr.queue_prefetch_all("hi", session_id="s1")
    elapsed = time.time() - t0

    # Provider blocks 2s per call inline; off-thread dispatch returns ~instantly.
    assert elapsed < 0.5, f"turn-completion path blocked {elapsed:.2f}s"


def test_background_work_still_completes():
    """Dispatching off-thread must not silently drop the write."""
    mgr = MemoryManager()
    p = _SlowProvider(delay=0.1)
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1")
    mgr.queue_prefetch_all("hi", session_id="s1")

    assert mgr.flush_pending(timeout=10) is True
    assert p.sync_done is True
    assert p.prefetch_done is True


def test_queue_prefetch_can_start_while_same_provider_sync_is_still_running():
    """Prefetch must not wait for the same provider's slow sync lane to clear."""
    mgr = MemoryManager()
    p = _ContendedProvider()
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1")
    assert p.sync_started.wait(timeout=1), "sync_turn never started"

    mgr.queue_prefetch_all("hi", session_id="s1")
    prefetch_started = p.prefetch_started.wait(timeout=1)
    p.allow_sync_finish.set()

    assert prefetch_started, (
        "queue_prefetch_all() did not start until sync_turn finished; "
        "same-provider contention still exists"
    )
    assert mgr.flush_pending(timeout=5) is True


def test_flush_pending_no_executor_is_true():
    """flush_pending must be a no-op (return True) before any sync ran."""
    mgr = MemoryManager()
    assert mgr.flush_pending(timeout=1) is True


def test_no_providers_does_not_create_executor():
    """Builtin-only / no-provider sessions must not spawn an executor."""
    mgr = MemoryManager()
    mgr.sync_all("hi", "hey")
    mgr.queue_prefetch_all("hi")
    assert not mgr._sync_executors
    assert not mgr._prefetch_executors


def test_shutdown_all_is_bounded_with_wedged_provider():
    """A provider that never returns must not hang teardown."""
    mgr = MemoryManager()
    mgr.add_provider(_SlowProvider(delay=30.0))
    mgr.sync_all("hi", "hey")

    t0 = time.time()
    mgr.shutdown_all()
    elapsed = time.time() - t0

    # Bounded by the background drain timeout (5s) plus a little slack.
    assert elapsed < 8.0, f"shutdown blocked {elapsed:.1f}s on wedged provider"


def test_shutdown_all_drains_sync_and_prefetch_with_one_timeout(monkeypatch):
    """Sync and prefetch teardown should share one shutdown drain timeout."""
    monkeypatch.setattr(memory_manager_module, "_BACKGROUND_DRAIN_TIMEOUT_S", 0.5)
    mgr = MemoryManager()
    p = _ShutdownBlockedProvider()
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey")
    assert p.sync_started.wait(timeout=1), "sync_turn never started"
    mgr.queue_prefetch_all("hi")
    assert p.prefetch_started.wait(timeout=1), "queue_prefetch never started"

    t0 = time.time()
    try:
        mgr.shutdown_all()
        elapsed = time.time() - t0
    finally:
        p.allow_sync_finish.set()
        p.allow_prefetch_finish.set()

    assert elapsed < 0.85, f"shutdown used serial lane timeouts: {elapsed:.2f}s"


def test_writes_are_serialized_in_order():
    """Single-worker executor must preserve turn ordering (N before N+1)."""
    order = []

    class _OrderProvider(_SlowProvider):
        _name = "order"

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            order.append(user_content)

    mgr = MemoryManager()
    mgr.add_provider(_OrderProvider(delay=0.0))
    for i in range(5):
        mgr.sync_all(f"turn-{i}", "resp", session_id="s1")
    assert mgr.flush_pending(timeout=10) is True
    assert order == [f"turn-{i}" for i in range(5)]
