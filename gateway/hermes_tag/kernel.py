"""Composed Hermes Tag governance kernel."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from .capability import CapabilityRegistry
from .config import HermesTagConfig
from .continuity import ContinuityStore
from .enforcement import LeaseAuthority
from .errors import LeaseError, LeaseTampered, PolicyDenied
from .identity import IdentityStore
from .ledger import BudgetLimits, HermesTagLedger, ReceiptRecord
from .middleware import AdmissionResult, TurnAdmissionMiddleware
from .model import (
    ActionIntent,
    ApprovalGrant,
    CapabilityDefinition,
    CapabilityLease,
    ContinuityMode,
    DecisionOutcome,
    ExternalIdentity,
    Fact,
    PolicyDecision,
    Principal,
    RiskLevel,
    SurfaceRef,
    TurnAdmission,
    arguments_digest,
    new_id,
    utc_now,
    utc_text,
)
from .obligations import ObligationPhase, ObligationRegistry
from .omniscience import FactStore
from .policy import ApprovalStore, PolicyEngine, PolicyEvaluation, PolicyRule
from .runtime import RuntimeAuthority

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    intent: ActionIntent
    decision: PolicyDecision
    lease: CapabilityLease | None = None
    token: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionResult:
    lease: CapabilityLease
    receipt: ReceiptRecord
    budget_state: str | None


class _ApprovalAuthorityView:
    """Read-only approval authority exposed by the public kernel facade."""

    __slots__ = ("__store",)

    def __init__(self, store: ApprovalStore) -> None:
        self.__store = store

    def has_exact(
        self,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        now: datetime | None = None,
    ) -> bool:
        return self.__store.has_exact(
            principal_id=principal_id,
            intent_digest=intent_digest,
            scope_digest=scope_digest,
            now=now,
        )

    def get(self, approval_id: str) -> ApprovalGrant:
        return self.__store.get(approval_id)


class _PolicyAuthorityView:
    """Policy evaluation without exposing raw approval-minting storage."""

    __slots__ = ("__engine", "approvals")

    def __init__(self, engine: PolicyEngine) -> None:
        self.__engine = engine
        self.approvals = _ApprovalAuthorityView(engine.approvals)

    @property
    def rules(self) -> tuple[PolicyRule, ...]:
        return self.__engine.rules

    @property
    def budget_limits(self) -> BudgetLimits:
        return self.__engine.budget_limits

    @property
    def approval_risk_floor(self) -> RiskLevel:
        return self.__engine.approval_risk_floor

    def evaluate(
        self,
        principal: Principal,
        intent: ActionIntent,
        definition: CapabilityDefinition,
        *,
        projected_tokens: int | None = 0,
        projected_cost_usd: float | None = 0.0,
        now: datetime | None = None,
    ) -> PolicyEvaluation:
        return self.__engine.evaluate(
            principal,
            intent,
            definition,
            projected_tokens=projected_tokens,
            projected_cost_usd=projected_cost_usd,
            now=now,
        )

    def require_allow(self, evaluation: PolicyEvaluation) -> PolicyDecision:
        return self.__engine.require_allow(evaluation)


class HermesTagKernel:
    """Single process facade over identity, continuity, policy, and audit."""

    def __init__(
        self,
        ledger: HermesTagLedger,
        config: HermesTagConfig,
        *,
        rules: Iterable[PolicyRule] = (),
        lease_authority: LeaseAuthority | None = None,
        capabilities: CapabilityRegistry | None = None,
        obligations: ObligationRegistry | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config
        self.identities = IdentityStore(ledger)
        self.continuities = ContinuityStore(ledger)
        self.facts = FactStore(ledger)
        self.capabilities = capabilities or CapabilityRegistry()
        self.obligations = obligations or ObligationRegistry()
        self.leases = lease_authority
        budget_limits = BudgetLimits(
            hourly_tokens=config.budgets.hourly_tokens,
            daily_tokens=config.budgets.daily_tokens,
            hourly_cost_usd=config.budgets.hourly_cost_usd,
            daily_cost_usd=config.budgets.daily_cost_usd,
        )
        self._policy = PolicyEngine(
            ledger,
            rules=rules,
            budget_limits=budget_limits,
        )
        # Approval issuance is itself a governed HIGH-risk effect. It cannot
        # recursively require an approval merely because it is HIGH risk, so
        # this dedicated engine raises the implicit floor to CRITICAL while
        # retaining default deny, deny precedence, and explicit approval rules.
        self._approval_policy = PolicyEngine(
            ledger,
            rules=rules,
            budget_limits=budget_limits,
            approval_risk_floor=RiskLevel.CRITICAL,
        )
        self.policy = _PolicyAuthorityView(self._policy)
        self.middleware = TurnAdmissionMiddleware(
            ledger,
            config,
            identities=self.identities,
            continuities=self.continuities,
            facts=self.facts,
        )

    def _rollback_failed_authorization(
        self,
        evaluation: PolicyEvaluation,
        *,
        now: datetime | None,
    ) -> None:
        reservation_id = evaluation.decision.budget_reservation_id
        if reservation_id is not None:
            try:
                self.ledger.release_budget(reservation_id, now=now)
            except Exception as exc:  # preserve the root failure
                logger.error(
                    "Hermes Tag failed to release undelivered budget authority: %s",
                    type(exc).__name__,
                )
        if evaluation.approval is not None:
            try:
                self._policy.approvals.restore_consumed(
                    evaluation.approval, now=now
                )
            except Exception as exc:  # preserve the root failure
                logger.error(
                    "Hermes Tag failed to restore undelivered approval authority: %s",
                    type(exc).__name__,
                )

    def _authorize_normalized(
        self,
        policy: PolicyEngine,
        principal: Principal,
        intent: ActionIntent,
        definition: CapabilityDefinition,
        *,
        projected_tokens: int | None,
        projected_cost_usd: float | None,
        now: datetime | None,
    ) -> AuthorizationResult:
        evaluation = policy.evaluate(
            principal,
            intent,
            definition,
            projected_tokens=projected_tokens,
            projected_cost_usd=projected_cost_usd,
            now=now,
        )
        if evaluation.decision.outcome is not DecisionOutcome.ALLOW:
            return AuthorizationResult(intent=intent, decision=evaluation.decision)
        if self.leases is None:
            self._rollback_failed_authorization(evaluation, now=now)
            raise LeaseError(
                "policy allowed the action but no lease signing authority is configured"
            )
        try:
            lease, token = self.leases.issue(
                principal, intent, evaluation.decision, now=now
            )
            self.ledger.append_receipt(
                event_id=f"lease-issued:{lease.lease_id}",
                kind="lease.issued",
                payload={
                    "lease_id": lease.lease_id,
                    "principal_id": lease.principal_id,
                    "continuity_id": lease.continuity_id,
                    "capability": lease.capability,
                    "intent_digest": lease.intent_digest,
                    "scope_digest": lease.scope_digest,
                    "decision_id": lease.decision_id,
                    "approval_id": lease.approval_id,
                    "budget_reservation_id": lease.budget_reservation_id,
                    "obligations": lease.obligations,
                    "expires_at": lease.expires_at,
                },
            )
        except Exception:
            self._rollback_failed_authorization(evaluation, now=now)
            raise
        return AuthorizationResult(
            intent=intent,
            decision=evaluation.decision,
            lease=lease,
            token=token,
        )

    def admit_turn(
        self,
        identity: ExternalIdentity,
        surface: SurfaceRef,
        *,
        event_id: str | None = None,
        project_id: str | None = None,
        continuity_mode: ContinuityMode | None = None,
        explicit_continuity_id: str | None = None,
    ) -> AdmissionResult:
        return self.middleware.admit(
            identity,
            surface,
            event_id=event_id,
            project_id=project_id,
            continuity_mode=continuity_mode,
            explicit_continuity_id=explicit_continuity_id,
        )

    def authorize(
        self,
        principal: Principal,
        *,
        capability: str,
        action: str,
        resource: str,
        arguments: Mapping[str, Any] | tuple[Any, ...] | list[Any] | None,
        scope: Any,
        risk: RiskLevel | str | int = RiskLevel.LOW,
        external_effect: bool = False,
        network_egress: bool = False,
        state_write: bool = False,
        metadata: Mapping[str, Any] | None = None,
        projected_tokens: int | None = 0,
        projected_cost_usd: float | None = 0.0,
        now: datetime | None = None,
    ) -> AuthorizationResult:
        intent, definition = self.capabilities.normalize_intent(
            capability=capability,
            action=action,
            resource=resource,
            arguments_digest=arguments_digest(arguments),
            scope=scope,
            risk=risk,
            external_effect=external_effect,
            network_egress=network_egress,
            state_write=state_write,
            metadata=metadata,
        )
        return self._authorize_normalized(
            self._policy,
            principal,
            intent,
            definition,
            projected_tokens=projected_tokens,
            projected_cost_usd=projected_cost_usd,
            now=now,
        )

    def _require_authenticated_admission(
        self,
        admission: TurnAdmission,
    ) -> Principal:
        registered = self.identities.get_principal(
            admission.principal.principal_id
        )
        if registered != admission.principal:
            raise LeaseTampered(
                "turn admission principal does not match durable identity authority"
            )
        if registered.guest:
            raise PolicyDenied("guest admission cannot grant approval authority")
        return registered

    def _approval_grant_intent(
        self,
        admission: TurnAdmission,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        ttl_seconds: int,
    ) -> tuple[ActionIntent, CapabilityDefinition]:
        self._require_authenticated_admission(admission)
        self.identities.get_principal(principal_id)
        if not re.fullmatch(r"[0-9a-f]{64}", intent_digest):
            raise ValueError("approval target intent_digest must be lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", scope_digest):
            raise ValueError("approval target scope_digest must be lowercase SHA-256")
        if (
            not isinstance(ttl_seconds, int)
            or isinstance(ttl_seconds, bool)
            or not 1 <= ttl_seconds <= 86400
        ):
            raise ValueError(
                "approval ttl_seconds must be an integer between 1 and 86400"
            )
        arguments = {
            "principal_id": principal_id,
            "approver_id": admission.principal.principal_id,
            "intent_digest": intent_digest,
            "scope_digest": scope_digest,
            "ttl_seconds": ttl_seconds,
        }
        return self.capabilities.normalize_intent(
            capability="approval.grant",
            action="grant",
            resource=f"approval:{principal_id}",
            arguments_digest=arguments_digest(arguments),
            scope=admission.scope,
            risk=RiskLevel.HIGH,
            state_write=True,
            metadata={"target_principal_id": principal_id},
        )

    def authorize_approval_grant(
        self,
        admission: TurnAdmission,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> AuthorizationResult:
        """Authorize one exact approval grant under explicit approver policy."""
        intent, definition = self._approval_grant_intent(
            admission,
            principal_id=principal_id,
            intent_digest=intent_digest,
            scope_digest=scope_digest,
            ttl_seconds=ttl_seconds,
        )
        return self._authorize_normalized(
            self._approval_policy,
            admission.principal,
            intent,
            definition,
            projected_tokens=0,
            projected_cost_usd=0.0,
            now=now,
        )

    def verify_effect(
        self,
        token: str,
        intent: ActionIntent,
        *,
        evidence: Mapping[str, Any],
        principal_id: str | None = None,
        decision_id: str | None = None,
        now: datetime | None = None,
    ) -> CapabilityLease:
        if self.leases is None:
            raise LeaseError("no lease signing authority is configured")
        lease = self.leases.verify(
            token,
            intent,
            principal_id=principal_id,
            decision_id=decision_id,
            now=now,
        )
        self.obligations.verify(
            lease.obligations,
            evidence,
            phase=ObligationPhase.PRE_EFFECT,
        )
        self.ledger.reserve_lease_use(lease, now=now)
        return lease

    def complete_effect(
        self,
        token: str,
        intent: ActionIntent,
        *,
        success: bool,
        evidence: Mapping[str, Any],
        actual_tokens: int = 0,
        actual_cost_usd: float = 0.0,
        decision: PolicyDecision | None = None,
        now: datetime | None = None,
    ) -> CompletionResult:
        if self.leases is None:
            raise LeaseError("no lease signing authority is configured")
        lease = self.leases.verify(
            token,
            intent,
            decision_id=decision.decision_id if decision else None,
            now=now,
        )
        if decision is not None:
            if (
                decision.outcome is not DecisionOutcome.ALLOW
                or decision.capability != lease.capability
                or decision.intent_digest != lease.intent_digest
                or decision.scope_digest != lease.scope_digest
                or decision.approval_id != lease.approval_id
                or decision.budget_reservation_id != lease.budget_reservation_id
            ):
                raise LeaseTampered(
                    "completion decision does not match signed lease authority"
                )
        self.ledger.require_reserved_lease_use(lease)
        internally_completed = {"receipt.append", "budget.settle"}
        post_without_receipt = tuple(
            item for item in lease.obligations if item not in internally_completed
        )
        self.obligations.verify(
            post_without_receipt,
            evidence,
            phase=ObligationPhase.POST_EFFECT,
        )

        budget_state: str | None = None
        reservation_id = lease.budget_reservation_id
        if "budget.settle" in lease.obligations and reservation_id is None:
            raise LeaseTampered(
                "signed lease requires budget settlement without a reservation"
            )
        if reservation_id:
            if actual_tokens or actual_cost_usd:
                budget_state = self.ledger.settle_budget(
                    reservation_id,
                    actual_tokens=actual_tokens,
                    actual_cost_usd=actual_cost_usd,
                    now=now,
                ).state
            else:
                budget_state = self.ledger.release_budget(
                    reservation_id, now=now
                ).state

        receipt = self.ledger.append_receipt(
            event_id=f"lease-completion:{lease.lease_id}",
            kind="action.completed" if success else "action.failed",
            payload={
                "lease_id": lease.lease_id,
                "decision_id": lease.decision_id,
                "principal_id": lease.principal_id,
                "continuity_id": lease.continuity_id,
                "capability": lease.capability,
                "intent_digest": lease.intent_digest,
                "scope_digest": lease.scope_digest,
                "success": success,
                "actual_tokens": actual_tokens,
                "actual_cost_usd": actual_cost_usd,
                "budget_state": budget_state,
                "evidence_keys": tuple(sorted(evidence)),
            },
        )
        final_obligations = tuple(
            item
            for item in ("budget.settle", "receipt.append")
            if item in lease.obligations
        )
        if final_obligations:
            self.obligations.verify(
                final_obligations,
                {
                    "budget_settled": budget_state in {"settled", "released"},
                    "receipt_hash": receipt.receipt_hash,
                },
                phase=ObligationPhase.POST_EFFECT,
            )
        self.ledger.complete_lease_use(
            lease,
            success=success,
            receipt_hash=receipt.receipt_hash,
            now=now,
        )
        return CompletionResult(
            lease=lease, receipt=receipt, budget_state=budget_state
        )

    def _revoke_incomplete_approval(
        self,
        grant: ApprovalGrant,
        *,
        now: datetime,
    ) -> None:
        """Remove a grant if its governing effect could not be completed."""
        try:
            with self.ledger.transaction() as connection:
                changed = connection.execute(
                    """
                    DELETE FROM hermes_tag_approvals
                    WHERE approval_id=? AND used_at IS NULL
                    """,
                    (grant.approval_id,),
                ).rowcount
            if changed == 1:
                self.ledger.append_receipt(
                    event_id=new_id("event"),
                    kind="approval.revoked",
                    payload={
                        "approval_id": grant.approval_id,
                        "principal_id": grant.principal_id,
                        "approver_id": grant.approver_id,
                        "reason": "governing effect completion failed",
                        "revoked_at": utc_text(now),
                    },
                )
        except Exception as exc:
            logger.error(
                "Hermes Tag failed to revoke incomplete approval authority: %s",
                type(exc).__name__,
            )

    def grant_approval(
        self,
        admission: TurnAdmission,
        authorization: AuthorizationResult,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> ApprovalGrant:
        """Mint an exact approval only through authenticated, leased authority."""
        decision = authorization.decision
        lease = authorization.lease
        token = authorization.token
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise LeaseError("approval grant was not allowed by policy")
        if lease is None or token is None:
            raise LeaseError("approval grant requires a signed capability lease")

        runtime_authority = RuntimeAuthority(
            admission=admission,
            decision=decision,
            lease=lease,
        )
        expected_intent, _ = self._approval_grant_intent(
            admission,
            principal_id=principal_id,
            intent_digest=intent_digest,
            scope_digest=scope_digest,
            ttl_seconds=ttl_seconds,
        )
        if authorization.intent.digest != expected_intent.digest:
            raise LeaseTampered(
                "approval grant authority does not match the requested grant"
            )

        current = now or utc_now()
        expires_at = utc_text(current + timedelta(seconds=ttl_seconds))
        self.verify_effect(
            token,
            expected_intent,
            evidence={
                "identity_authenticated": True,
                "intent_digest": intent_digest,
                "expires_at": expires_at,
            },
            principal_id=runtime_authority.admission.principal.principal_id,
            decision_id=decision.decision_id,
            now=current,
        )

        try:
            grant = self._policy.approvals.grant(
                principal_id=principal_id,
                approver_id=runtime_authority.admission.principal.principal_id,
                intent_digest=intent_digest,
                scope_digest=scope_digest,
                ttl_seconds=ttl_seconds,
                now=current,
            )
        except Exception:
            try:
                self.complete_effect(
                    token,
                    expected_intent,
                    success=False,
                    evidence={},
                    decision=decision,
                    now=current,
                )
            except Exception as cleanup_exc:
                logger.error(
                    "Hermes Tag failed to close rejected approval effect: %s",
                    type(cleanup_exc).__name__,
                )
            raise

        try:
            self.complete_effect(
                token,
                expected_intent,
                success=True,
                evidence={},
                decision=decision,
                now=current,
            )
        except Exception:
            self._revoke_incomplete_approval(grant, now=current)
            raise
        return grant

    def observe_fact(self, fact: Fact) -> Fact:
        return self.facts.observe(fact)
