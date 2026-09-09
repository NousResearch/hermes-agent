"""Deterministic deny-overrides policy, approvals, and atomic budgets."""

from __future__ import annotations

import fnmatch
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .errors import ApprovalRequired, BudgetExceeded, PolicyDenied, StorageError
from .ledger import BudgetLimits, BudgetReservation, HermesTagLedger
from .model import (
    ActionIntent,
    ApprovalGrant,
    CapabilityDefinition,
    DecisionOutcome,
    PolicyDecision,
    Principal,
    RiskLevel,
    new_id,
    utc_now,
    utc_text,
)


def _patterns(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item.strip() for item in values if item.strip()))
    return result


def _match_any(value: str | None, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    if value is None:
        return False
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One declarative policy rule.

    Every non-empty selector is conjunctive with the others. Values inside one
    selector are alternatives. Glob syntax is supported for capability, action,
    and resource selectors; identity and scope selectors are exact unless the
    operator deliberately supplies a glob.
    """

    rule_id: str
    effect: DecisionOutcome
    capabilities: tuple[str, ...] = ("*",)
    actions: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    principal_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()
    chat_ids: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()
    minimum_risk: RiskLevel | None = None
    require_external_effect: bool | None = None
    require_network_egress: bool | None = None
    require_state_write: bool | None = None
    obligations: tuple[str, ...] = ()
    reason: str = "policy rule matched"
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id must be non-empty")
        object.__setattr__(self, "effect", DecisionOutcome(self.effect))
        for name in (
            "capabilities",
            "actions",
            "resources",
            "principal_ids",
            "roles",
            "platforms",
            "profiles",
            "scope_ids",
            "chat_ids",
            "project_ids",
            "obligations",
        ):
            object.__setattr__(self, name, _patterns(getattr(self, name)))
        if self.minimum_risk is not None:
            object.__setattr__(self, "minimum_risk", RiskLevel.coerce(self.minimum_risk))
        if not self.reason.strip():
            raise ValueError("rule reason must be non-empty")

    def matches(self, principal: Principal, intent: ActionIntent) -> bool:
        scope = intent.scope
        if not _match_any(intent.capability, self.capabilities):
            return False
        if not _match_any(intent.action, self.actions):
            return False
        if not _match_any(intent.resource, self.resources):
            return False
        if not _match_any(principal.principal_id, self.principal_ids):
            return False
        if self.roles and not any(
            _match_any(role, self.roles) for role in principal.roles
        ):
            return False
        for value, patterns in (
            (scope.platform, self.platforms),
            (scope.profile, self.profiles),
            (scope.scope_id, self.scope_ids),
            (scope.chat_id, self.chat_ids),
            (scope.project_id, self.project_ids),
        ):
            if not _match_any(value, patterns):
                return False
        if self.minimum_risk is not None and intent.risk < self.minimum_risk:
            return False
        for actual, expected in (
            (intent.external_effect, self.require_external_effect),
            (intent.network_egress, self.require_network_egress),
            (intent.state_write, self.require_state_write),
        ):
            if expected is not None and actual is not expected:
                return False
        return True


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    decision: PolicyDecision
    approval: ApprovalGrant | None = None


class ApprovalStore:
    """Exact-intent, exact-scope, one-time approval authority."""

    def __init__(self, ledger: HermesTagLedger) -> None:
        self.ledger = ledger

    def grant(
        self,
        *,
        principal_id: str,
        approver_id: str,
        intent_digest: str,
        scope_digest: str,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> ApprovalGrant:
        if ttl_seconds < 1 or ttl_seconds > 86400:
            raise ValueError("approval ttl_seconds must be between 1 and 86400")
        current = now or utc_now()
        grant = ApprovalGrant(
            approval_id=new_id("approval"),
            principal_id=principal_id,
            approver_id=approver_id,
            intent_digest=intent_digest,
            scope_digest=scope_digest,
            issued_at=utc_text(current),
            expires_at=utc_text(current + timedelta(seconds=ttl_seconds)),
        )
        try:
            with self.ledger.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO hermes_tag_approvals(
                        approval_id, principal_id, approver_id, intent_digest,
                        scope_digest, issued_at, expires_at, used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        grant.approval_id,
                        grant.principal_id,
                        grant.approver_id,
                        grant.intent_digest,
                        grant.scope_digest,
                        grant.issued_at,
                        grant.expires_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(
                "approval principal and approver must already exist"
            ) from exc
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="approval.granted",
            payload={
                "approval_id": grant.approval_id,
                "principal_id": grant.principal_id,
                "approver_id": grant.approver_id,
                "intent_digest": grant.intent_digest,
                "scope_digest": grant.scope_digest,
                "expires_at": grant.expires_at,
            },
        )
        return grant

    @staticmethod
    def _grant(row: sqlite3.Row) -> ApprovalGrant:
        return ApprovalGrant(
            approval_id=row["approval_id"],
            principal_id=row["principal_id"],
            approver_id=row["approver_id"],
            intent_digest=row["intent_digest"],
            scope_digest=row["scope_digest"],
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            used_at=row["used_at"],
        )

    def has_exact(
        self,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        now: datetime | None = None,
    ) -> bool:
        """Check current exact authority without consuming it."""
        current_text = utc_text(now or utc_now())
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                """
                SELECT 1 FROM hermes_tag_approvals
                WHERE principal_id=?
                  AND intent_digest=?
                  AND scope_digest=?
                  AND used_at IS NULL
                  AND expires_at>?
                LIMIT 1
                """,
                (principal_id, intent_digest, scope_digest, current_text),
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def consume_exact(
        self,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        now: datetime | None = None,
    ) -> ApprovalGrant | None:
        """Atomically consume the oldest current matching approval."""
        current_text = utc_text(now or utc_now())
        with self.ledger.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM hermes_tag_approvals
                WHERE principal_id=?
                  AND intent_digest=?
                  AND scope_digest=?
                  AND used_at IS NULL
                  AND expires_at>?
                ORDER BY issued_at, approval_id
                LIMIT 1
                """,
                (principal_id, intent_digest, scope_digest, current_text),
            ).fetchone()
            if row is None:
                return None
            changed = connection.execute(
                """
                UPDATE hermes_tag_approvals
                SET used_at=?
                WHERE approval_id=? AND used_at IS NULL AND expires_at>?
                """,
                (current_text, row["approval_id"], current_text),
            ).rowcount
            if changed != 1:
                return None
            payload = dict(row)
            payload["used_at"] = current_text
        grant = self._grant(_DictRow(payload))
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="approval.consumed",
            payload={
                "approval_id": grant.approval_id,
                "principal_id": grant.principal_id,
                "intent_digest": grant.intent_digest,
                "scope_digest": grant.scope_digest,
            },
        )
        return grant

    def consume_exact_with_budget(
        self,
        *,
        principal_id: str,
        intent_digest: str,
        scope_digest: str,
        tokens: int,
        cost_usd: float | None,
        limits: BudgetLimits,
        now: datetime | None = None,
    ) -> tuple[ApprovalGrant | None, BudgetReservation | None]:
        """Consume approval and reserve budget in one SQLite transaction."""
        current = now or utc_now()
        current_text = utc_text(current)
        with self.ledger.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM hermes_tag_approvals
                WHERE principal_id=?
                  AND intent_digest=?
                  AND scope_digest=?
                  AND used_at IS NULL
                  AND expires_at>?
                ORDER BY issued_at, approval_id
                LIMIT 1
                """,
                (principal_id, intent_digest, scope_digest, current_text),
            ).fetchone()
            if row is None:
                return None, None
            reservation = self.ledger._reserve_budget_in_connection(
                connection,
                scope_digest=scope_digest,
                tokens=tokens,
                cost_usd=cost_usd,
                limits=limits,
                now=current,
            )
            changed = connection.execute(
                """
                UPDATE hermes_tag_approvals
                SET used_at=?
                WHERE approval_id=? AND used_at IS NULL AND expires_at>?
                """,
                (current_text, row["approval_id"], current_text),
            ).rowcount
            if changed != 1:
                raise StorageError("approval consumption race was not serializable")
            payload = dict(row)
            payload["used_at"] = current_text
        grant = self._grant(_DictRow(payload))
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="approval.consumed",
            payload={
                "approval_id": grant.approval_id,
                "principal_id": grant.principal_id,
                "intent_digest": grant.intent_digest,
                "scope_digest": grant.scope_digest,
                "budget_reservation_id": reservation.reservation_id,
            },
        )
        return grant, reservation

    def restore_consumed(
        self,
        grant: ApprovalGrant,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Restore a consumed approval when no lease was delivered."""
        if grant.used_at is None:
            return False
        current_text = utc_text(now or utc_now())
        with self.ledger.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE hermes_tag_approvals
                SET used_at=NULL
                WHERE approval_id=? AND used_at=? AND expires_at>?
                """,
                (grant.approval_id, grant.used_at, current_text),
            ).rowcount
        if changed != 1:
            return False
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="approval.restored",
            payload={
                "approval_id": grant.approval_id,
                "principal_id": grant.principal_id,
                "intent_digest": grant.intent_digest,
                "scope_digest": grant.scope_digest,
            },
        )
        return True

    def get(self, approval_id: str) -> ApprovalGrant:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                "SELECT * FROM hermes_tag_approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise StorageError("unknown approval")
        return self._grant(row)


class _DictRow(dict[str, object]):
    """Small mapping adapter for the sqlite-row conversion helper."""


class PolicyEngine:
    """Evaluate one normalized intent under deterministic policy authority."""

    def __init__(
        self,
        ledger: HermesTagLedger,
        *,
        rules: Iterable[PolicyRule] = (),
        budget_limits: BudgetLimits | None = None,
        approval_risk_floor: RiskLevel = RiskLevel.HIGH,
    ) -> None:
        self.ledger = ledger
        self.rules = tuple(sorted(rules, key=lambda item: (-item.priority, item.rule_id)))
        self.budget_limits = budget_limits or BudgetLimits()
        self.approval_risk_floor = RiskLevel.coerce(approval_risk_floor)
        self.approvals = ApprovalStore(ledger)

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
        current = now or utc_now()
        matched = tuple(rule for rule in self.rules if rule.matches(principal, intent))
        matched_ids = tuple(rule.rule_id for rule in matched)
        obligations = set(definition.obligations)
        for rule in matched:
            obligations.update(rule.obligations)

        if principal.principal_id != intent.scope.principal_id:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                ("principal does not own the action scope",),
                obligations,
                matched_ids,
                now=current,
            )

        if principal.guest and not definition.guest_eligible:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                ("unresolved guest is not eligible for this capability",),
                obligations,
                matched_ids,
                now=current,
            )

        deny_rules = tuple(rule for rule in matched if rule.effect is DecisionOutcome.DENY)
        if deny_rules:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                tuple(rule.reason for rule in deny_rules),
                obligations,
                matched_ids,
                now=current,
            )

        approval_rules = tuple(
            rule for rule in matched if rule.effect is DecisionOutcome.REQUIRE_APPROVAL
        )
        allow_rules = tuple(rule for rule in matched if rule.effect is DecisionOutcome.ALLOW)
        if not approval_rules and not allow_rules:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                ("no allow rule matched; default deny",),
                obligations,
                matched_ids,
                now=current,
            )

        needs_approval = bool(approval_rules) or intent.risk >= self.approval_risk_floor
        approval_reasons = tuple(rule.reason for rule in approval_rules) or (
            f"{intent.risk.name.lower()} risk requires exact approval",
        )

        limits = self.budget_limits
        has_token_limit = (
            limits.hourly_tokens is not None or limits.daily_tokens is not None
        )
        has_cost_limit = (
            limits.hourly_cost_usd is not None
            or limits.daily_cost_usd is not None
        )
        if has_token_limit and projected_tokens is None:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                ("token budget is configured but projected token usage is unknown",),
                obligations,
                matched_ids,
                now=current,
            )
        if has_cost_limit and projected_cost_usd is None:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                ("cost budget is configured but projected cost is unknown",),
                obligations,
                matched_ids,
                now=current,
            )
        reserve_tokens = int(projected_tokens or 0)
        reserve_cost = float(projected_cost_usd or 0.0)
        if reserve_tokens < 0 or reserve_cost < 0:
            raise ValueError("projected usage cannot be negative")
        should_reserve = (has_token_limit or has_cost_limit) and bool(
            reserve_tokens or reserve_cost
        )

        approval: ApprovalGrant | None = None
        reservation: BudgetReservation | None = None
        try:
            if needs_approval and should_reserve:
                approval, reservation = self.approvals.consume_exact_with_budget(
                    principal_id=principal.principal_id,
                    intent_digest=intent.digest,
                    scope_digest=intent.scope.digest,
                    tokens=reserve_tokens,
                    cost_usd=reserve_cost,
                    limits=limits,
                    now=current,
                )
            elif needs_approval:
                approval = self.approvals.consume_exact(
                    principal_id=principal.principal_id,
                    intent_digest=intent.digest,
                    scope_digest=intent.scope.digest,
                    now=current,
                )
            elif should_reserve:
                reservation = self.ledger.reserve_budget(
                    scope_digest=intent.scope.digest,
                    tokens=reserve_tokens,
                    cost_usd=reserve_cost,
                    limits=limits,
                    now=current,
                )
        except BudgetExceeded as exc:
            return self._decision(
                intent,
                DecisionOutcome.DENY,
                (str(exc),),
                obligations,
                matched_ids,
                now=current,
            )

        if needs_approval and approval is None:
            return self._decision(
                intent,
                DecisionOutcome.REQUIRE_APPROVAL,
                approval_reasons,
                obligations | {"approval.exact"},
                matched_ids,
                now=current,
            )

        reservation_id = reservation.reservation_id if reservation else None
        if reservation_id is not None:
            obligations.add("budget.settle")

        reasons = tuple(rule.reason for rule in allow_rules) or (
            "approval rule satisfied by exact approval",
        )
        return self._decision(
            intent,
            DecisionOutcome.ALLOW,
            reasons,
            obligations,
            matched_ids,
            approval=approval,
            reservation_id=reservation_id,
            now=current,
        )

    def require_allow(self, evaluation: PolicyEvaluation) -> PolicyDecision:
        decision = evaluation.decision
        if decision.outcome is DecisionOutcome.REQUIRE_APPROVAL:
            raise ApprovalRequired("exact approval is required")
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise PolicyDenied("; ".join(decision.reasons))
        return decision

    def _decision(
        self,
        intent: ActionIntent,
        outcome: DecisionOutcome,
        reasons: tuple[str, ...],
        obligations: Iterable[str],
        matched_rules: tuple[str, ...],
        *,
        approval: ApprovalGrant | None = None,
        reservation_id: str | None = None,
        now: datetime,
    ) -> PolicyEvaluation:
        decision = PolicyDecision(
            decision_id=new_id("decision"),
            outcome=outcome,
            capability=intent.capability,
            intent_digest=intent.digest,
            scope_digest=intent.scope.digest,
            reasons=tuple(reasons),
            obligations=tuple(sorted(set(obligations))),
            matched_rules=matched_rules,
            budget_reservation_id=reservation_id,
            approval_id=approval.approval_id if approval else None,
            decided_at=utc_text(now),
        )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="policy.decision",
            payload={
                "decision_id": decision.decision_id,
                "outcome": decision.outcome.value,
                "capability": decision.capability,
                "intent_digest": decision.intent_digest,
                "scope_digest": decision.scope_digest,
                "matched_rules": decision.matched_rules,
                "obligations": decision.obligations,
                "approval_id": decision.approval_id,
                "budget_reservation_id": decision.budget_reservation_id,
                "reasons": decision.reasons,
            },
        )
        return PolicyEvaluation(decision=decision, approval=approval)
