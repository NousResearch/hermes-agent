"""Turn admission middleware for identity, continuity, and scoped context."""

from __future__ import annotations

from dataclasses import dataclass

from .config import HermesTagConfig
from .continuity import ContinuityRecord, ContinuityStore
from .errors import IncompleteScope
from .identity import IdentityStore
from .ledger import HermesTagLedger
from .model import (
    ContextBundle,
    ContinuityMode,
    ExternalIdentity,
    ScopeRef,
    SurfaceRef,
    TurnAdmission,
    new_id,
)
from .omniscience import FactStore
from .scopes import scope_from_surface


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admission: TurnAdmission
    continuity: ContinuityRecord


class TurnAdmissionMiddleware:
    """Create one canonical turn admission after transport authorization."""

    def __init__(
        self,
        ledger: HermesTagLedger,
        config: HermesTagConfig,
        *,
        identities: IdentityStore | None = None,
        continuities: ContinuityStore | None = None,
        facts: FactStore | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config
        self.identities = identities or IdentityStore(ledger)
        self.continuities = continuities or ContinuityStore(ledger)
        self.facts = facts or FactStore(ledger)

    def admit(
        self,
        identity: ExternalIdentity,
        surface: SurfaceRef,
        *,
        event_id: str | None = None,
        project_id: str | None = None,
        continuity_mode: ContinuityMode | None = None,
        explicit_continuity_id: str | None = None,
    ) -> AdmissionResult:
        if identity.platform != surface.platform:
            raise IncompleteScope("actor and surface platforms differ")
        if identity.profile != surface.profile:
            raise IncompleteScope("actor and surface profiles differ")
        if identity.scope_id != surface.scope_id:
            raise IncompleteScope("actor and surface tenant scopes differ")

        if event_id is not None:
            self.ledger.reserve_turn_event(event_id)
        try:
            principal = self.identities.resolve_or_guest(
                identity, allow_guest=self.config.allow_guests
            )
            mode = continuity_mode or self.config.continuity.mode
            if not self.config.continuity.enabled:
                mode = ContinuityMode.ISOLATED
            continuity = self.continuities.resolve_or_create(
                principal,
                surface,
                mode=mode,
                project_id=project_id,
                explicit_id=explicit_continuity_id,
            )
            scope = scope_from_surface(
                surface,
                principal_id=principal.principal_id,
                project_id=project_id or continuity.project_id,
                continuity_id=continuity.continuity_id,
            )
            context = self._context(scope)
            admission = TurnAdmission(
                admission_id=new_id("admission"),
                principal=principal,
                surface=surface,
                scope=scope,
                continuity_id=continuity.continuity_id,
                context=context,
                shadow=self.config.shadow,
            )
            receipt_event_id = event_id or new_id("event")
            self.ledger.append_receipt(
                event_id=receipt_event_id,
                kind="turn.admitted",
                payload={
                    "admission_id": admission.admission_id,
                    "principal_id": principal.principal_id,
                    "guest": principal.guest,
                    "surface_key": surface.key,
                    "scope_digest": scope.digest,
                    "continuity_id": continuity.continuity_id,
                    "context_fact_ids": tuple(fact.fact_id for fact in context.facts),
                    "context_conflicts": context.conflicts,
                    "context_omitted": context.omitted_count,
                    "shadow": admission.shadow,
                },
            )
            if event_id is not None:
                self.ledger.complete_turn_event(
                    event_id,
                    admission_id=admission.admission_id,
                    principal_id=principal.principal_id,
                    surface_key=surface.key,
                    continuity_id=continuity.continuity_id,
                    scope_digest=scope.digest,
                )
            return AdmissionResult(admission=admission, continuity=continuity)
        except Exception:
            if event_id is not None:
                self.ledger.release_turn_event(event_id)
            raise

    def _context(self, scope: ScopeRef) -> ContextBundle:
        if not self.config.context.enabled:
            return ContextBundle(facts=())
        return self.facts.query(
            scope,
            sensitivity_ceiling=self.config.context.sensitivity_ceiling,
            max_facts=self.config.context.max_facts,
            max_chars=self.config.context.max_chars,
        )
