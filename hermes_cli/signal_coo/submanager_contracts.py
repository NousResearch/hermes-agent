"""Torben hidden-submanager capability contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ScopeName = Literal["ea", "gtm", "finance"]


@dataclass(frozen=True)
class SubmanagerContract:
    scope: ScopeName
    owner: str
    hidden_backend: bool
    handle_prefix: str
    purpose: str
    provider: str
    source_health_artifacts: list[str] = field(default_factory=list)
    allowed_mutation_classes: list[str] = field(default_factory=list)
    blocked_until_approval: list[str] = field(default_factory=list)
    user_visible_surface: str = "torben_signal"
    direct_signal_delivery_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def torben_submanager_contracts() -> dict[str, SubmanagerContract]:
    """Return the production Torben capability registry."""

    return {
        "ea": SubmanagerContract(
            scope="ea",
            owner="torben",
            hidden_backend=True,
            handle_prefix="EA",
            purpose="Personal operations: Gmail, calendar, meeting prep, relationship memory, morning brief, and hygiene recommendations.",
            provider="torben_signal_coo",
            source_health_artifacts=[
                "torben-gmail-pubsub-pull-latest.json",
                "torben-gmail-watch-state.json",
                "torben-live-profile-verify-latest.json",
            ],
            allowed_mutation_classes=[
                "private_busy_block_create_when_policy_allows",
                "draft_reply",
                "stage_email_hygiene_recommendation",
            ],
            blocked_until_approval=[
                "send_email",
                "trash_archive_delete_gmail",
                "calendar_attendee_mutation",
                "calendar_event_edit_or_delete",
            ],
        ),
        "gtm": SubmanagerContract(
            scope="gtm",
            owner="torben",
            hidden_backend=True,
            handle_prefix="GTM",
            purpose="Magnus-backed intelligence, article opportunities, post drafts, X/LinkedIn reply candidates, and performance feedback.",
            provider="magnus_xai_grok",
            source_health_artifacts=[
                "torben-gtm-radar-latest.json",
                "torben-gtm-engagement-radar-latest.json",
                "/Users/ericfreeman/magnus/state/gtm-intelligence-radar/latest.json",
            ],
            allowed_mutation_classes=[
                "draft_social_content",
                "stage_reply_candidate",
                "stage_visual_plan",
            ],
            blocked_until_approval=[
                "post_to_x",
                "reply_on_x",
                "post_to_linkedin",
                "schedule_public_content",
            ],
        ),
        "finance": SubmanagerContract(
            scope="finance",
            owner="torben",
            hidden_backend=True,
            handle_prefix="FIN",
            purpose="Ratatosk-backed Robinhood research, staged trade candidates, risk context, performance learning, and future live execution after mandate.",
            provider="ratatosk_robinhood_v01",
            source_health_artifacts=[
                "torben-finance-radar-latest.json",
                "/Users/ericfreeman/ratatosk/state/robinhood-v01/latest-run.json",
                "/Users/ericfreeman/ratatosk/state/trading-halt.json",
                "/Users/ericfreeman/ratatosk/state/circuit-breaker.json",
            ],
            allowed_mutation_classes=[
                "stage_trade_candidate",
                "stage_monarch_review",
                "performance_report",
            ],
            blocked_until_approval=[
                "broker_order_place_cancel_modify",
                "margin_or_options_execution",
                "monarch_finance_mutation",
                "live_capital_risk_change",
            ],
        ),
    }


def validate_torben_submanager_contracts(
    contracts: dict[str, SubmanagerContract] | None = None,
) -> dict[str, Any]:
    """Validate the single-Signal-operator contract."""

    registry = contracts or torben_submanager_contracts()
    errors: list[str] = []
    warnings: list[str] = []
    required = {"ea": "EA", "gtm": "GTM", "finance": "FIN"}
    for scope, prefix in required.items():
        contract = registry.get(scope)
        if contract is None:
            errors.append(f"missing submanager contract: {scope}")
            continue
        if contract.owner != "torben":
            errors.append(f"{scope}: owner must be torben")
        if not contract.hidden_backend:
            errors.append(f"{scope}: hidden_backend must be true")
        if contract.user_visible_surface != "torben_signal":
            errors.append(f"{scope}: user_visible_surface must be torben_signal")
        if contract.direct_signal_delivery_allowed:
            errors.append(f"{scope}: direct Signal delivery must be blocked")
        if contract.handle_prefix != prefix:
            errors.append(f"{scope}: handle prefix must be {prefix}")
        if not contract.blocked_until_approval:
            warnings.append(f"{scope}: no blocked mutation classes documented")
        if not contract.source_health_artifacts:
            warnings.append(f"{scope}: no source health artifacts documented")

    return {
        "status": "pass" if not errors else "fail",
        "contract_count": len(registry),
        "contracts": {scope: contract.to_dict() for scope, contract in registry.items()},
        "errors": errors,
        "warnings": warnings,
    }
