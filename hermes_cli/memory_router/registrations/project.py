"""Self-registration of the L2-project state capability (Invariant A)."""

from __future__ import annotations

from ..registry import CapabilityRegistry
from ...memory_api.project import ProjectProvider
from ..intents import Intent
from . import _registrar


@_registrar
def register(registry: CapabilityRegistry) -> None:
    # Exactly ONE ProjectProvider instance owns L2 storage resolution.
    project_provider = ProjectProvider()

    def project_handle(method: str, **kw):
        if method == "get":
            return project_provider.get(kw.get("project", ""))
        if method == "set":
            # Human-gated write path only. Hermes never calls this.
            state = kw.get("state")
            if state is None:
                raise RuntimeError("project set requires a state argument")
            return project_provider.set(state, updated_by=kw.get("updated_by"))
        if method == "propose":
            # Hermes-autonomous SUGGESTION. Writes NOTHING (authority B).
            return project_provider.propose_update(kw.get("project", ""), **kw.get("kwargs", {}))
        raise RuntimeError(f"unknown project method {method!r}")

    registry.register(
        "L2-project",
        [Intent.PROJECT_STATE],
        True,
        project_handle,
        provider=project_provider,
        # Opted in by a reviewed product decision: the single active project
        # feeds the "project" slot of the context bundle.
        contributes_to_context=True,
        context_category="project",
    )
