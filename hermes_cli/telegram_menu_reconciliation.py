"""Revision-aware Telegram native-menu reconciliation and exact settlement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from hermes_cli.telegram_command_projection import (
    TelegramCommandProjection,
    TelegramNativeCommand,
)


class TelegramMenuReconciliationAction(str, Enum):
    """Required native-registration action for one Telegram scope."""

    NOOP = "noop"
    ADOPT = "adopt"
    SET = "set"


class TelegramMenuVerificationStatus(str, Enum):
    """Terminal result of a reconciliation read-back."""

    SETTLED = "settled"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class TelegramMenuSettlement:
    """Last exact native-menu object proved for one Telegram scope."""

    scope: str
    catalog_revision: str
    projection_fingerprint: str


@dataclass(frozen=True, slots=True)
class TelegramMenuReconciliationPlan:
    """Deterministic action required to align one Telegram menu scope."""

    scope: str
    action: TelegramMenuReconciliationAction
    reason: str
    catalog_revision: str
    projection_fingerprint: str
    desired_commands: tuple[tuple[str, str], ...]
    observed_commands: tuple[tuple[str, str], ...]
    prior_settlement: TelegramMenuSettlement | None

    @property
    def requires_set(self) -> bool:
        return self.action is TelegramMenuReconciliationAction.SET

    @property
    def requires_read_back(self) -> bool:
        return self.action is TelegramMenuReconciliationAction.SET


@dataclass(frozen=True, slots=True)
class TelegramMenuVerification:
    """Exact post-reconciliation proof; mismatches never advance state."""

    status: TelegramMenuVerificationStatus
    expected_commands: tuple[tuple[str, str], ...]
    observed_commands: tuple[tuple[str, str], ...]
    settlement: TelegramMenuSettlement | None


def _nonblank_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _native_command_pair(command: object) -> tuple[str, str]:
    if isinstance(command, TelegramNativeCommand):
        return command.command, command.description
    if isinstance(command, Mapping):
        name = command.get("command", command.get("name"))
        description = command.get("description")
    elif isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        if len(command) < 2:
            raise ValueError("Telegram native command tuple requires name and description")
        name, description = command[0], command[1]
    else:
        name = getattr(command, "command", getattr(command, "name", None))
        description = getattr(command, "description", None)

    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("Telegram native command requires string name and description")
    return name, description


def normalize_telegram_native_commands(
    commands: Iterable[object],
) -> tuple[tuple[str, str], ...]:
    """Normalize Bot API objects, mappings, or tuples for exact comparison."""

    return tuple(_native_command_pair(command) for command in commands)


def plan_telegram_menu_reconciliation(
    projection: TelegramCommandProjection,
    observed_commands: Iterable[object],
    *,
    scope: str,
    prior_settlement: TelegramMenuSettlement | None = None,
) -> TelegramMenuReconciliationPlan:
    """Plan exact revision-aware reconciliation for one Telegram menu scope."""

    normalized_scope = _nonblank_text(scope)
    if normalized_scope is None:
        raise ValueError("Telegram menu reconciliation scope must not be blank")

    desired = projection.native_payload
    observed = normalize_telegram_native_commands(observed_commands)
    prior_matches_scope = (
        prior_settlement is not None and prior_settlement.scope == normalized_scope
    )
    prior_is_current = (
        prior_matches_scope
        and prior_settlement is not None
        and prior_settlement.catalog_revision == projection.catalog_revision
        and prior_settlement.projection_fingerprint
        == projection.projection_fingerprint
    )

    if observed == desired:
        action = (
            TelegramMenuReconciliationAction.NOOP
            if prior_is_current
            else TelegramMenuReconciliationAction.ADOPT
        )
        reason = "in_sync" if prior_is_current else "observed_current_projection"
    else:
        action = TelegramMenuReconciliationAction.SET
        if prior_is_current:
            reason = "remote_drift"
        elif prior_matches_scope:
            reason = "revision_changed"
        else:
            reason = "unsettled"

    return TelegramMenuReconciliationPlan(
        scope=normalized_scope,
        action=action,
        reason=reason,
        catalog_revision=projection.catalog_revision,
        projection_fingerprint=projection.projection_fingerprint,
        desired_commands=desired,
        observed_commands=observed,
        prior_settlement=prior_settlement,
    )


def verify_telegram_menu_reconciliation(
    plan: TelegramMenuReconciliationPlan,
    read_back_commands: Iterable[object] | None = None,
) -> TelegramMenuVerification:
    """Settle a plan only after exact payload read-back.

    ``SET`` plans require a post-write read-back. ``ADOPT`` and ``NOOP`` plans
    may settle from the exact preflight observation already captured in the
    plan. Any mismatch returns a typed terminal result with no settlement.
    """

    if read_back_commands is None:
        if plan.action is TelegramMenuReconciliationAction.SET:
            raise ValueError("SET reconciliation requires post-write read-back")
        observed = plan.observed_commands
    else:
        observed = normalize_telegram_native_commands(read_back_commands)

    if observed != plan.desired_commands:
        return TelegramMenuVerification(
            status=TelegramMenuVerificationStatus.MISMATCH,
            expected_commands=plan.desired_commands,
            observed_commands=observed,
            settlement=None,
        )

    settlement = TelegramMenuSettlement(
        scope=plan.scope,
        catalog_revision=plan.catalog_revision,
        projection_fingerprint=plan.projection_fingerprint,
    )
    return TelegramMenuVerification(
        status=TelegramMenuVerificationStatus.SETTLED,
        expected_commands=plan.desired_commands,
        observed_commands=observed,
        settlement=settlement,
    )
