"""Tests for MemoryManager.commit_session_boundary_async.

The /new session boundary must deliver on_session_end (old-session
extraction) strictly BEFORE on_session_switch (provider rebinding to the
new session), without blocking the caller. Both hooks run as one task on
the manager's single serialized background worker.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from agent.memory_manager import MemoryManager, memory_session_context
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records hook invocations with thread identity."""

    def __init__(self, end_delay: float = 0.0):
        self.calls: List[tuple] = []
        self._end_delay = end_delay
        self._caller_thread_ids: List[int] = []
        self.switch_kwargs: List[Dict[str, Any]] = []

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
        self.switch_kwargs.append(dict(kwargs))
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


def test_boundary_commit_captures_target_context_before_queued_switch():
    provider = _RecordingProvider(end_delay=0.15)
    mm = _make_manager(provider)
    stored = {"title": "Target title", "source": "user"}
    agent = SimpleNamespace(
        _session_db=SimpleNamespace(
            get_session_title=lambda _sid: stored["title"],
            get_session_title_source=lambda _sid: stored["source"],
        ),
        _session_title_hint=None,
        _session_title_source=None,
    )
    context = memory_session_context(agent, "new-sid", cwd="/target/project")

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "old"}],
        new_session_id="new-sid",
        **context,
    )
    stored.update(title="Later title", source="llm")
    context.update(cwd="/later/project", session_title="Later title", session_title_source="llm")
    assert mm.flush_pending(timeout=30)

    assert provider.switch_kwargs == [{
        "parent_session_id": "",
        "reset": True,
        "reason": "new_session",
        "cwd": "/target/project",
        "session_title": "Target title",
        "session_title_source": "user",
    }]


def test_replacement_context_clears_stale_title_and_cwd():
    provider = _RecordingProvider()
    mm = _make_manager(provider)
    session_db = SimpleNamespace(
        get_session_title=lambda _sid: None,
        get_session_title_source=lambda _sid: None,
    )
    agent = SimpleNamespace(
        _session_db=session_db,
        _session_title_hint="Old title",
        _session_title_source="user",
    )

    context = memory_session_context(agent, "untitled-session", cwd=None)
    mm.on_session_switch("untitled-session", reset=True, **context)

    assert context == {"cwd": None, "session_title": None, "session_title_source": None}
    assert agent._session_title_hint is None
    assert agent._session_title_source is None
    assert {
        key: provider.switch_kwargs[0][key]
        for key in ("cwd", "session_title", "session_title_source")
    } == context


def test_session_context_preserves_nonempty_cwd_verbatim():
    agent = SimpleNamespace(_session_db=None)

    assert memory_session_context(agent, "sid", cwd=" /project with spaces/ ")["cwd"] == " /project with spaces/ "
    assert memory_session_context(agent, "sid", cwd="")["cwd"] is None


def test_narrow_legacy_switch_signature_remains_compatible():
    class _LegacyProvider(_RecordingProvider):
        def on_session_switch(self, new_session_id: str) -> None:
            self.calls.append(("legacy-switch", new_session_id))

    provider = _LegacyProvider()
    mm = _make_manager(provider)

    mm.on_session_switch(
        "new-sid",
        cwd="/project",
        session_title="Title",
        session_title_source="user",
        reason="resume",
    )

    assert provider.calls == [("legacy-switch", "new-sid")]
