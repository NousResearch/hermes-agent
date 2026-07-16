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
    t0 = time.monotonic()
    mm.commit_session_boundary_async(
        msgs, new_session_id="new-sid", parent_session_id="old-sid"
    )
    # Caller returns immediately — the slow extraction must not block /new.
    assert time.monotonic() - t0 < 0.1

    assert mm.flush_pending(timeout=5)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch"], f"ordering violated: {provider.calls}"
    assert provider.calls[0] == ("end", msgs)
    assert provider.calls[1] == ("switch", "new-sid", True)
    # And it genuinely ran off the caller's thread.
    assert provider._caller_thread_ids[0] != threading.get_ident()


def test_boundary_commit_serializes_against_turn_syncs():
    """The boundary task shares the single worker with sync_all — FIFO order
    means a queued boundary can't interleave into a later turn's sync."""
    provider = _RecordingProvider(end_delay=0.05)
    mm = _make_manager(provider)

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "old"}],
        new_session_id="new-sid",
    )
    mm.sync_all("next-session user msg", "assistant reply", session_id="new-sid")

    assert mm.flush_pending(timeout=5)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch", "sync_turn"], f"unexpected order: {provider.calls}"


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


def test_boundary_commit_noop_without_providers():
    mm = MemoryManager()
    # Must not create the executor or raise.
    mm.commit_session_boundary_async([{"role": "user", "content": "x"}], new_session_id="s")
    assert mm._sync_executor is None


class _ScopedRecordingProvider(_RecordingProvider):
    def __init__(self, end_delay: float = 0.0):
        super().__init__(end_delay=end_delay)
        self.scope_switches = []
        self.active_scope_key = "old-key"

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        super().on_session_switch(new_session_id, **kwargs)
        self.scope_switches.append((new_session_id, kwargs.get("memory_scope_key")))
        if "memory_scope_key" in kwargs:
            self.active_scope_key = kwargs["memory_scope_key"]

    def prefetch(self, query: str, **kwargs) -> str:
        return str(self.active_scope_key)


def test_queued_boundaries_consume_scope_key_for_exact_session():
    provider = _ScopedRecordingProvider(end_delay=0.05)
    mm = _make_manager(provider)
    mm.bind_session_scope("two", "key-two")
    mm.bind_session_scope("three", "key-three")
    mm.commit_session_boundary_async([], new_session_id="two")
    mm.commit_session_boundary_async([], new_session_id="three")
    assert mm.flush_pending(timeout=5)
    assert provider.scope_switches == [("two", "key-two"), ("three", "key-three")]


def test_prefetch_waits_for_pending_scope_rotation():
    provider = _ScopedRecordingProvider(end_delay=0.1)
    mm = _make_manager(provider)
    mm.bind_session_scope("new", "new-key")
    mm.commit_session_boundary_async([], new_session_id="new")
    assert mm.prefetch_all("first turn", session_id="new") == "new-key"
