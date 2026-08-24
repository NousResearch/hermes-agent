"""Tests for MemoryManager.commit_session_boundary_async.

The /new session boundary must deliver on_session_end (old-session
extraction) strictly BEFORE on_session_switch (provider rebinding to the
new session), without blocking the caller. Both hooks run as one task on
the manager's single serialized background worker.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records hook invocations with thread identity."""

    def __init__(self, end_delay: float = 0.0):
        self.calls: List[tuple] = []
        self._end_delay = end_delay
        self._caller_thread_ids: List[int] = []

    # Required ABC surface (minimal no-ops)
    @property
    def name(self) -> str:
        return "recorder"

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def initialize(self, agent: Any = None, **kwargs) -> bool:  # type: ignore[override]
        return True

    def build_system_prompt(self) -> str:  # type: ignore[override]
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:  # type: ignore[override]
        self.calls.append(("sync_turn", kwargs.get("session_id", "")))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._end_delay:
            time.sleep(self._end_delay)
        self._caller_thread_ids.append(threading.get_ident())
        self.calls.append(("end", list(messages)))

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self.calls.append(("switch", new_session_id, kwargs.get("reset")))


def _make_manager(provider: _RecordingProvider) -> MemoryManager:
    mm = MemoryManager()
    mm._providers.append(provider)  # bypass add_provider validation for the stub
    return mm


def test_boundary_commit_delivers_end_strictly_before_switch():
    """Even with a slow (LLM-like) extraction, switch waits for end."""
    provider = _RecordingProvider(end_delay=0.15)
    mm = _make_manager(provider)

    msgs = [{"role": "user", "content": "old turn"}]
    mm.commit_session_boundary_async(
        msgs, new_session_id="new-sid", parent_session_id="old-sid"
    )
    # DETERMINISTIC non-blocking witness — replaces `assert elapsed < 0.1`.
    #
    # The old form timed `commit_session_boundary_async` and required it under
    # 100ms, which makes the scheduler part of the assertion: thread startup
    # alone can exceed that on a loaded box, flipping the inequality with
    # nothing wrong in the code under test.
    #
    # The real contract is that the caller returns WITHOUT waiting for the slow
    # extraction. Assert it directly: the background `on_session_end` sleeps
    # 0.15s before recording anything, so if the caller had blocked on it, the
    # provider would already have recorded the "end" call by the time we get
    # here. An empty call list is a positive witness that /new was not gated.
    assert provider.calls == [], (
        "commit_session_boundary_async blocked on the slow extraction: "
        f"provider already recorded {provider.calls} before the caller returned"
    )

    assert mm.flush_pending(timeout=30)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch"], f"ordering violated: {provider.calls}"
    assert provider.calls[0] == ("end", msgs)
    assert provider.calls[1] == ("switch", "new-sid", True)
    # And it genuinely ran off the caller's thread.
    assert provider._caller_thread_ids[0] != threading.get_ident()




def test_boundary_commit_switch_still_fires_when_end_raises():
    """A failing provider extraction must not strand providers on the old sid."""

    class _ExplodingEndProvider(_RecordingProvider):
        def on_session_end(self, messages):  # type: ignore[override]
            raise RuntimeError("provider extraction blew up")

    provider = _ExplodingEndProvider()
    mm = _make_manager(provider)

    mm.commit_session_boundary_async([{"role": "user", "content": "x"}], new_session_id="new-sid")
    assert mm.flush_pending(timeout=5)

    assert ("switch", "new-sid", True) in provider.calls


