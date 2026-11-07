"""One session boundary reaches a provider's ``on_session_end`` exactly once.

Regression (Desktop/TUI session close): ``tui_gateway.session_lifecycle._finalize_session``
commits the boundary with ``agent.commit_memory_session()`` and ``_teardown_session`` then calls
``agent.close()``  → ``shutdown_memory_provider()``.  Both routes fan ``on_session_end`` out to
the providers, so a provider that writes a session separator on that hook (agent-memory's
``<!-- agent-memory:session_end ... -->``) wrote two per close: 59 pairs across the 51 raw
session records that carry a marker, always milliseconds apart and never the same timestamp —
two live calls, not a retry.

The claim belongs to the host.  A provider cannot tell a boundary commit from a teardown, and a
provider-side dedupe keyed on the session id would also swallow the real end of a session the
user resumed.  A new turn or a session-id rotation re-arms it, because that is a new boundary to
extract — in-place compression commits a boundary without rotating the id.
"""

from __future__ import annotations

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from run_agent import AIAgent

MARKER = "<!-- agent-memory:session_end {n} -->"


class _MarkerProvider(MemoryProvider):
    """Writes one session-end marker per ``on_session_end``, like agent-memory's raw capture."""

    def __init__(self, name: str = "builtin") -> None:
        self._name = name
        self.markers: list[str] = []
        self.shutdown_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:  # pragma: no cover - unused
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, turn_author=None):
        pass

    def queue_prefetch(self, query, *, session_id=""):
        pass

    def on_session_end(self, messages):
        self.markers.append(MARKER.format(n=len(self.markers)))

    def shutdown(self):
        self.shutdown_calls += 1


def _agent(manager: MemoryManager) -> AIAgent:
    """An agent with just the surface the two host routes touch."""
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "local-tui-close"
    return agent


def test_boundary_commit_then_teardown_writes_one_session_end():
    """``_finalize_session`` then ``close()`` is ONE boundary, not two."""
    mgr = MemoryManager()
    provider = _MarkerProvider()
    mgr.add_provider(provider)
    agent = _agent(mgr)
    messages = [{"role": "user", "content": "hi"}]

    agent.commit_memory_session(messages)  # _finalize_session: boundary commit
    agent.shutdown_memory_provider(messages)  # _teardown_session -> close(): teardown

    assert len(provider.markers) == 1

    # Teardown itself still happens: only the duplicate extraction is suppressed.
    assert provider.shutdown_calls == 1


def test_a_new_turn_rearms_the_session_end():
    """In-place compression commits a boundary without rotating the id; the next turn is live again."""
    mgr = MemoryManager()
    provider = _MarkerProvider()
    mgr.add_provider(provider)

    mgr.on_session_end([])
    mgr.on_turn_start(1, "another question")
    mgr.on_session_end([])

    assert len(provider.markers) == 2


def test_a_session_switch_rearms_the_session_end():
    """A rotated session id writes into a new record and gets its own end."""
    mgr = MemoryManager()
    provider = _MarkerProvider()
    mgr.add_provider(provider)

    mgr.on_session_end([])  # /new boundary commit
    mgr.on_session_switch("local-next")
    mgr.on_session_end([])

    assert len(provider.markers) == 2
