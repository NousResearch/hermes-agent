"""Regression guard: end-of-turn memory sync must not block the turn.

Before this fix, ``MemoryManager.sync_all`` / ``queue_prefetch_all`` looped
``provider.sync_turn`` / ``provider.queue_prefetch`` INLINE on the
turn-completion path. A provider making a blocking network/daemon call (a
misconfigured Hindsight daemon was observed blocking ~298s before failing)
held ``run_conversation`` open long after the user saw their response, so
every interface (CLI, TUI, gateway) kept the agent marked "running" for
minutes and any follow-up message triggered an aggressive interrupt that
dropped the message.

The fix dispatches provider work to a single-worker background executor.
``sync_all`` / ``queue_prefetch_all`` return immediately; the work completes
(or fails, logged) in the background. ``flush_pending`` provides a barrier
for session boundaries and deterministic tests. ``shutdown_all`` drains the
executor with a bounded timeout so a wedged provider can't hang teardown.
"""
import time

import pytest

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


def test_flush_pending_no_executor_is_true():
    """flush_pending must be a no-op (return True) before any sync ran."""
    mgr = MemoryManager()
    assert mgr.flush_pending(timeout=1) is True


def test_no_providers_does_not_create_executor():
    """Builtin-only / no-provider sessions must not spawn an executor."""
    mgr = MemoryManager()
    mgr.sync_all("hi", "hey")
    mgr.queue_prefetch_all("hi")
    assert mgr._sync_executor is None


def test_shutdown_all_is_bounded_with_wedged_provider():
    """A provider that never returns must not hang teardown."""
    mgr = MemoryManager()
    mgr.add_provider(_SlowProvider(delay=30.0))
    mgr.sync_all("hi", "hey")

    t0 = time.time()
    mgr.shutdown_all()
    elapsed = time.time() - t0

    # Bounded by _SYNC_DRAIN_TIMEOUT_S (5s) plus a little slack.
    assert elapsed < 8.0, f"shutdown blocked {elapsed:.1f}s on wedged provider"


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


def test_on_session_end_flushes_pending_sync_before_provider_hook():
    """Session-end hooks must see all turns queued before the boundary.

    Providers such as Supermemory buffer turn data in ``sync_turn`` and write
    the full conversation from ``on_session_end``. If the manager calls
    ``on_session_end`` before draining the background executor, a one-shot CLI
    process can exit with queued turns still missing from the provider's
    session-end write.
    """
    import threading

    first_sync_started = threading.Event()
    allow_first_sync_to_finish = threading.Event()

    class _SessionProvider(_SlowProvider):
        _name = "session-provider"

        def __init__(self):
            super().__init__(delay=0.0)
            self.buffered_turns = []
            self.session_end_seen = None

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            if user_content == "turn-1":
                first_sync_started.set()
                assert allow_first_sync_to_finish.wait(timeout=5), "test did not release first sync_turn"
            self.buffered_turns.append((user_content, assistant_content, session_id))

        def on_session_end(self, messages):
            self.session_end_seen = list(self.buffered_turns)

    mgr = MemoryManager()
    provider = _SessionProvider()
    mgr.add_provider(provider)

    mgr.sync_all("turn-1", "stored-1", session_id="cli-one-shot")
    assert first_sync_started.wait(timeout=5), "first background sync did not start"
    mgr.sync_all("turn-2", "stored-2", session_id="cli-one-shot")

    release_timer = threading.Timer(0.2, allow_first_sync_to_finish.set)
    release_timer.start()
    t0 = time.monotonic()
    try:
        mgr.on_session_end([{"role": "user", "content": "turn-2"}])
    finally:
        release_timer.cancel()
        allow_first_sync_to_finish.set()
    elapsed = time.monotonic() - t0

    # Without a drain in on_session_end, the hook returns immediately after
    # seeing an incomplete buffer because turn-2 is still queued behind the
    # blocked first sync. A correct implementation waits for the queued work.
    assert elapsed >= 0.15, "on_session_end returned before queued sync drained"

    assert provider.session_end_seen == [
        ("turn-1", "stored-1", "cli-one-shot"),
        ("turn-2", "stored-2", "cli-one-shot"),
    ]
