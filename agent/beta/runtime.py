"""Runtime facade shared by CLI, gateway, and tests.

Surfaces call this class instead of invoking individual Beta modules. It keeps
one stable profile, registry, approval gate, and executor per agent session.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.beta.chief_profile import ChiefProfile, ChiefProfileStore
from agent.beta.orchestrator import BetaRun, orchestrate_request
from agent.beta.risk import ApprovalGate, ApprovalReceipt, Operation
from agent.beta.specialists import SpecialistRegistry, default_specialist_registry


class BetaRuntime:
    def __init__(
        self,
        parent_agent: Any,
        *,
        delegate: Callable[..., str] | None = None,
        executor: Callable[[Operation], str] | None = None,
        registry: SpecialistRegistry | None = None,
        approval_gate: ApprovalGate | None = None,
        profile_store: ChiefProfileStore | None = None,
    ):
        self.parent_agent = parent_agent
        self.delegate = delegate
        self.executor = executor
        self.registry = registry or default_specialist_registry()
        self.approval_gate = approval_gate or ApprovalGate()
        self.profile_store = profile_store or ChiefProfileStore()
        self.profile = self.profile_store.load(
            getattr(parent_agent, "_user_id", None),
            display_name=getattr(parent_agent, "_user_name", None) or "Chief",
        )

    def refresh_profile(self) -> ChiefProfile:
        self.profile = self.profile_store.load(
            getattr(self.parent_agent, "_user_id", None),
            display_name=getattr(self.parent_agent, "_user_name", None) or "Chief",
        )
        return self.profile

    def handle(self, request: str, *, receipts: dict[str, ApprovalReceipt] | None = None) -> BetaRun:
        return orchestrate_request(
            request,
            self.parent_agent,
            delegate=self.delegate,
            executor=self.executor,
            registry=self.registry,
            approval_gate=self.approval_gate,
            approval_receipts=receipts,
            chief_profile=self.profile,
        )


def get_beta_runtime(agent: Any) -> BetaRuntime:
    """Return one session-stable runtime without affecting Hermes mode."""
    runtime = getattr(agent, "_beta_runtime", None)
    if runtime is None:
        runtime = BetaRuntime(agent)
        agent._beta_runtime = runtime
    return runtime
