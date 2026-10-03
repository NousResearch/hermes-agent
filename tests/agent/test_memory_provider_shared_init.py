"""A non-primary (cron/subagent) re-init must not deactivate a provider instance
already live for a primary session in the same process. See #119180.

Root cause: singleton-style providers (mnemosyne's process-global
``_get_or_create_provider()``) are shared by every agent in a long-lived
backend, because ``load_memory_provider`` returns the same object per agent.
The subagent/cron agent's ``initialize(agent_context=...)`` then tore the live
primary backend down (mnemosyne ``_initialize_locked`` clears ``_beam`` first,
then early-returns on skip contexts), so every later in-session tool call
reported ``Mnemosyne not initialized`` while session-start injection (already
rendered) and the CLI (separate process) stayed healthy.
"""
from agent.memory_manager import MemoryManager


class SingletonBeamProvider:
    """Mimics the pinned mnemosyne provider's init teardown semantics."""

    name = "singleton-beam"

    def __init__(self):
        self.beam = None
        self.agent_context = None
        self.init_calls = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        # Mirror _initialize_locked: teardown first, skip-context early return.
        self.beam = None
        self.agent_context = kwargs.get("agent_context", "primary")
        self.init_calls.append((session_id, self.agent_context))
        if self.agent_context in {"cron", "flush", "subagent", "background", "skill_loop"}:
            return
        self.beam = object()  # live backend

    def get_tool_schemas(self):
        return []

    def tool_ready(self):
        return self.beam is not None


def _manager_with(provider, **kwargs):
    mm = MemoryManager()
    mm.add_provider(provider)
    mm.initialize_all(session_id=kwargs.pop("session_id", "sess"), hermes_home="/tmp/hh", **kwargs)
    return mm


def test_skip_context_reinit_does_not_kill_primary_beam():
    """Primary session is live; a subagent sharing the instance must not kill it."""
    provider = SingletonBeamProvider()
    _manager_with(provider, session_id="primary-sess", agent_context="primary", platform="desktop")
    assert provider.tool_ready()

    _manager_with(provider, session_id="sub-sess", agent_context="subagent", platform="subagent")

    assert provider.tool_ready(), "subagent re-init deactivated the primary session's backend"
    assert provider.agent_context == "primary"
    assert [c for _, c in provider.init_calls] == ["primary"]


def test_fresh_skip_context_init_still_runs():
    """A first-time init under a skip context must still reach the provider."""
    provider = SingletonBeamProvider()
    _manager_with(provider, session_id="cron-sess", agent_context="cron", platform="cron")

    assert [c for _, c in provider.init_calls] == ["cron"]


def test_primary_reinit_still_runs():
    """Session switch (primary -> primary) must still re-initialize."""
    provider = SingletonBeamProvider()
    _manager_with(provider, session_id="sess-a", agent_context="primary", platform="desktop")
    first_beam = provider.beam
    _manager_with(provider, session_id="sess-b", agent_context="primary", platform="desktop")

    assert [s for s, _ in provider.init_calls] == ["sess-a", "sess-b"]
    assert provider.tool_ready()
    assert provider.beam is not first_beam
