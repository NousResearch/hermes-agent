"""Regression tests for #73297: memory rollback after /reset.

The gateway's ``_cleanup_agent_resources`` (the cleanup chokepoint that
``/reset`` and every other gateway session rotation invokes to tear down the
cached agent) used to call ``shutdown_memory_provider()`` WITHOUT first
draining the memory manager's serialized background write worker.
``shutdown_memory_provider`` -> ``shutdown_all`` only gives that worker a
bounded (~5s) drain and abandons whatever is still queued past it, so a
/reset could silently drop writes the session had already handed off — the
next session then loaded stale memory.

The CLI exit path already drains via ``MemoryManager.flush_pending`` before
shutdown (cli.py); these tests pin the same contract on the gateway path.

The fix: in ``_cleanup_agent_resources``, call
``agent._memory_manager.flush_pending(timeout=10)`` BEFORE
``shutdown_memory_provider``.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
import agent.memory_manager as _mm_module
from gateway.run import GatewayRunner


# How long the "gate" write occupies the single background worker. Chosen so a
# queued write reliably stays PENDING behind it for the duration of the test
# (the worker is FIFO, single-threaded). Kept small for test speed.
_GATE_DELAY_S = 0.6


class _RecordingProvider(MemoryProvider):
    """Provider that records completed writes to ``self.recorded`` (the "disk").

    A write whose assistant payload is the sentinel ``__GATE__`` sleeps for
    ``gate_delay`` first — it occupies the manager's single background worker
    so a subsequent real write sits PENDING behind it until the gate clears.
    """

    _name = "recording"

    def __init__(self, gate_delay: float = _GATE_DELAY_S):
        self._gate_delay = gate_delay
        self.recorded: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        if assistant_content == "__GATE__":
            time.sleep(self._gate_delay)
            return
        self.recorded.append(assistant_content)


def _make_agent(mgr: MemoryManager) -> SimpleNamespace:
    """Build a minimal agent wired to a real MemoryManager.

    ``shutdown_memory_provider`` mirrors ``AIAgent.shutdown_memory_provider``
    (run_agent.py): end-of-session notification then ``shutdown_all``. The
    memory manager is the live one carrying the queued writes, so the gateway
    cleanup path exercises the real flush -> shutdown ordering.
    """

    def _shutdown_memory_provider(messages=None):
        mgr.on_session_end(messages or [])
        mgr.shutdown_all()

    return SimpleNamespace(
        _memory_manager=mgr,
        _session_messages=[{"role": "user", "content": "earlier turn"}],
        shutdown_memory_provider=_shutdown_memory_provider,
    )


def test_cleanup_flushes_pending_writes_before_shutdown(monkeypatch):
    """#73297 invariant: ``disk_state_after_reset >= last_known_state_before_reset``.

    A real write ("fact-1") is queued on the memory manager's background
    worker behind a slow gate write, so it is PENDING when /reset cleanup
    runs. ``shutdown_all``'s drain is shortened (via ``_SYNC_DRAIN_TIMEOUT_S``)
    to model the production condition where the bounded drain abandons queued
    work — the condition the issue reports. Without the pre-shutdown
    ``flush_pending``, "fact-1" is abandoned and never reaches the provider
    (the on-disk store). With the fix, ``flush_pending`` drains the queue
    BEFORE the unreliable shutdown drain, so the write lands.
    """
    # Shorten shutdown_all's drain so it abandons the still-pending write,
    # modelling the bounded-drain abandonment at the heart of #73297. The
    # fix's flush_pending uses its OWN (longer) barrier, independent of this
    # timeout, so it still drains the queue.
    monkeypatch.setattr(_mm_module, "_SYNC_DRAIN_TIMEOUT_S", 0.1)

    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    mgr.initialize_all("sess-old")

    # Occupy the single worker with a slow gate write, then queue the real
    # write behind it. "fact-1" is now PENDING (FIFO, worker busy with gate).
    mgr.sync_all("user msg a", "__GATE__", session_id="sess-old")
    mgr.sync_all("user msg b", "fact-1", session_id="sess-old")

    last_known_state_before_reset = ["fact-1"]
    assert provider.recorded == [], "precondition: write must still be pending"

    agent = _make_agent(mgr)

    # Run the gateway cleanup path the /reset handler runs. No explicit flush
    # here — the contract is that cleanup itself flushes before shutdown.
    GatewayRunner._cleanup_agent_resources(object(), agent)

    disk_state_after_reset = provider.recorded
    assert set(last_known_state_before_reset) <= set(disk_state_after_reset), (
        f"#73297 regression: pending memory write lost across reset. "
        f"expected >= {last_known_state_before_reset}, got {disk_state_after_reset}"
    )


def test_cleanup_calls_flush_pending_before_shutdown_memory_provider():
    """Structural contract: the gateway cleanup drains the memory manager
    BEFORE shutting the provider down (ordering matters — shutdown's drain is
    the unreliable one the fix works around)."""
    agent = MagicMock()
    agent._session_messages = [{"role": "user", "content": "hi"}]
    agent._memory_manager = MagicMock()

    # Track call order across both objects via a parent mock.
    parent = MagicMock()
    parent.attach_mock(agent._memory_manager.flush_pending, "flush_pending")
    parent.attach_mock(agent.shutdown_memory_provider, "shutdown_memory_provider")

    GatewayRunner._cleanup_agent_resources(object(), agent)

    agent._memory_manager.flush_pending.assert_called_once_with(timeout=10)
    parent.assert_has_calls(
        [call.flush_pending(timeout=10), call.shutdown_memory_provider([{"role": "user", "content": "hi"}])],
        any_order=False,
    )
