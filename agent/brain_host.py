"""Phase 3 Brain Host seam (central-brain-openclaw.md §11 "3c/3d").

This is the SEAM ONLY — v1 constructs identically to direct AIAgent() calls so
routing through it is a no-op.  The host is the single future home for:
  * credential-pool / API-key sharing across the ~20 AIAgent construction sites
  * tool-schema caching
  * memory-session sharing
  * per-intent model/policy selection

Richer fields on AgentSpec (toolsets, model_policy, memory_key …) will be added
as later construction sites migrate.  Until then, ``kwargs`` carries the exact
constructor kwargs that the call site already builds.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """Descriptor passed from a construction site to BrainHost.

    intent: short human-readable tag for the construction site
        (e.g. ``"tui_gateway"``).  Used for logging and future routing.
    kwargs: the exact keyword arguments to forward to AIAgent.__init__.
        This thin representation is intentional: additional fields
        (toolsets, model_policy, memory_key) will be added here as
        individual sites migrate; for now the host acts as a transparent
        proxy so parity with direct construction is provably trivially true.
    """

    intent: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class BrainHost:
    """Process-singleton that owns AIAgent construction for migrated call sites.

    Phase 3 "Brain host" seam — one place that will later own
    credential-pool / tool-schema / memory-session sharing across the ~20
    construction sites; v1 deliberately constructs identically to direct
    calls (parity-tested) so routing through it is a no-op.

    Usage::

        from agent.brain_host import AgentSpec, BrainHost
        agent = BrainHost.get().build_agent(AgentSpec(intent="tui_gateway", kwargs=kw))
    """

    _instance: BrainHost | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> "BrainHost":
        """Return (or create) the process-level BrainHost singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def build_agent(self, spec: AgentSpec):
        """Construct and return an AIAgent for *spec*.

        v1: identical to ``AIAgent(**spec.kwargs)`` — no caching, no
        credential-pool lookup, no behaviour change.  Later phases will
        intercept here.
        """
        from run_agent import AIAgent  # lazy: heavy import, only when used

        return AIAgent(**spec.kwargs)
