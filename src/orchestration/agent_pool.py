"""Agent Context Pool — the TCB pool equivalent.

Following the IBM CICS model, the pool manages the lifecycle of every
AgentContext (TCB) inside the single Gateway process.  It provides:

* lazy loading from disk (CICS TCB creation)
* pseudo-conversational snapshot / restore
* dynamic register / unregister (CICS Resource Definition Online)

CICS analogue
-------------
* CICS TCB pool       →  AgentContextPool
* CICS RDO (dynamic)  →  register() / unregister()
* CICS pseudo-conversational → snapshot() / get_or_create() restore
"""

from __future__ import annotations

from pathlib import Path

from hermes_state import get_hermes_home
from src.orchestration.agent_context import AgentContext


class AgentContextPool:
    """Manages the lifecycle of all agent contexts within one Gateway."""

    def __init__(self, agents_dir: Path | None = None):
        self._agents: dict[str, AgentContext] = {}
        self._agents_dir = agents_dir or (get_hermes_home() / "agents")

    # ── core ─────────────────────────────────────────────────────────

    def get_or_create(self, agent_id: str) -> AgentContext:
        """Return the context for *agent_id*, loading from disk if needed.

        This is the primary entry point.  The first call for a given
        agent id triggers :meth:`AgentContext.from_disk` — subsequent
        calls return the in-memory instance (pseudo-conversational: the
        agent's state accumulates across turns).
        """
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentContext.from_disk(agent_id)
        return self._agents[agent_id]

    # ── persistence ──────────────────────────────────────────────────

    def snapshot(self, agent_id: str) -> None:
        """Persist the agent's state to disk (pseudo-conversational save).

        Safe to call at turn boundaries — writes state.json only, does
        not touch memory or skills.
        """
        ctx = self._agents.get(agent_id)
        if ctx is not None:
            ctx.save_state()

    def snapshot_all(self) -> None:
        """Snapshot every agent in the pool."""
        for ctx in self._agents.values():
            ctx.save_state()

    # ── lifecycle (RDO) ──────────────────────────────────────────────

    def register(self, ctx: AgentContext) -> None:
        """Register a pre-built context (CICS RDO — dynamic add)."""
        self._agents[ctx.id] = ctx

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the pool and snapshot first."""
        self.snapshot(agent_id)
        self._agents.pop(agent_id, None)

    # ── inspection ───────────────────────────────────────────────────

    @property
    def agent_ids(self) -> list[str]:
        return sorted(self._agents.keys())

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def __len__(self) -> int:
        return len(self._agents)
