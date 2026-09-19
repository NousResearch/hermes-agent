"""Revision-aware Telegram native-menu reconciliation characterization."""

from types import SimpleNamespace

import pytest

from hermes_cli.telegram_command_projection import (
    TELEGRAM_BOT_API_MAX_COMMANDS,
    build_telegram_command_projection,
)
from hermes_cli.telegram_menu_reconciliation import (
    TelegramMenuReconciliationAction,
    TelegramMenuSettlement,
    TelegramMenuVerificationStatus,
    normalize_telegram_native_commands,
    plan_telegram_menu_reconciliation,
    verify_telegram_menu_reconciliation,
)


def _command(name: str, description: str | None = None, **overrides):
    values = {
        "name": name,
        "description": description or f"Run {name}",
        "aliases": (),
        "command_id": None,
        "visibility": None,
        "hidden": False,
        "debug": False,
        "available": True,
        "unsupported_surfaces": (),
        "supported_surfaces": (),
        "cli_only": False,
        "gateway_only": False,
        "presentation_overrides": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)

def test_native_command_normalization_accepts_bot_api_shapes():
    bot_command = SimpleNamespace(command="status", description="Show status")

    assert normalize_telegram_native_commands(
        [
            bot_command,
            {"command": "new", "description": "New session"},
            ("stop", "Stop work"),
        ]
    ) == (
        ("status", "Show status"),
        ("new", "New session"),
        ("stop", "Stop work"),
    )


def test_exact_observed_projection_is_adopted_then_becomes_noop():
    projection = build_telegram_command_projection(
        [_command("status")], catalog_revision="rev-1"
    )
    observed = [SimpleNamespace(command="status", description="Run status")]

    adopt = plan_telegram_menu_reconciliation(
        projection, observed, scope="default"
    )
    assert adopt.action is TelegramMenuReconciliationAction.ADOPT
    assert adopt.reason == "observed_current_projection"
    adopted = verify_telegram_menu_reconciliation(adopt)
    assert adopted.status is TelegramMenuVerificationStatus.SETTLED
    assert adopted.settlement is not None

    noop = plan_telegram_menu_reconciliation(
        projection,
        observed,
        scope="default",
        prior_settlement=adopted.settlement,
    )
    assert noop.action is TelegramMenuReconciliationAction.NOOP
    assert noop.reason == "in_sync"
    assert noop.requires_set is False
    assert noop.requires_read_back is False


def test_remote_drift_requires_set_and_exact_read_back():
    projection = build_telegram_command_projection(
        [_command("status")], catalog_revision="rev-1"
    )
    settlement = TelegramMenuSettlement(
        scope="default",
        catalog_revision=projection.catalog_revision,
        projection_fingerprint=projection.projection_fingerprint,
    )

    plan = plan_telegram_menu_reconciliation(
        projection,
        [("old", "Stale")],
        scope="default",
        prior_settlement=settlement,
    )

    assert plan.action is TelegramMenuReconciliationAction.SET
    assert plan.reason == "remote_drift"
    assert plan.requires_set is True
    assert plan.requires_read_back is True

    mismatch = verify_telegram_menu_reconciliation(plan, [("status", "Wrong")])
    assert mismatch.status is TelegramMenuVerificationStatus.MISMATCH
    assert mismatch.settlement is None

    settled = verify_telegram_menu_reconciliation(plan, projection.native_payload)
    assert settled.status is TelegramMenuVerificationStatus.SETTLED
    assert settled.settlement == settlement


def test_revision_change_reconciles_and_advances_only_after_read_back():
    old_projection = build_telegram_command_projection(
        [_command("status", description="Old")], catalog_revision="rev-1"
    )
    old_settlement = TelegramMenuSettlement(
        scope="default",
        catalog_revision=old_projection.catalog_revision,
        projection_fingerprint=old_projection.projection_fingerprint,
    )
    new_projection = build_telegram_command_projection(
        [_command("status", description="New")], catalog_revision="rev-2"
    )

    plan = plan_telegram_menu_reconciliation(
        new_projection,
        old_projection.native_payload,
        scope="default",
        prior_settlement=old_settlement,
    )

    assert plan.action is TelegramMenuReconciliationAction.SET
    assert plan.reason == "revision_changed"
    with pytest.raises(ValueError, match="requires post-write read-back"):
        verify_telegram_menu_reconciliation(plan)

    verification = verify_telegram_menu_reconciliation(
        plan, new_projection.native_payload
    )
    assert verification.status is TelegramMenuVerificationStatus.SETTLED
    assert verification.settlement is not None
    assert verification.settlement.catalog_revision == "rev-2"
    assert verification.settlement != old_settlement


def test_same_payload_new_revision_can_be_adopted_without_rewrite():
    old = build_telegram_command_projection(
        [_command("status")], catalog_revision="rev-1"
    )
    current = build_telegram_command_projection(
        [_command("status")], catalog_revision="rev-2"
    )
    old_settlement = TelegramMenuSettlement(
        scope="default",
        catalog_revision=old.catalog_revision,
        projection_fingerprint=old.projection_fingerprint,
    )

    plan = plan_telegram_menu_reconciliation(
        current,
        current.native_payload,
        scope="default",
        prior_settlement=old_settlement,
    )

    assert plan.action is TelegramMenuReconciliationAction.ADOPT
    verification = verify_telegram_menu_reconciliation(plan)
    assert verification.settlement is not None
    assert verification.settlement.catalog_revision == "rev-2"


def test_settlement_from_another_scope_never_authorizes_noop():
    projection = build_telegram_command_projection([_command("status")])
    other_scope = TelegramMenuSettlement(
        scope="all_private_chats",
        catalog_revision=projection.catalog_revision,
        projection_fingerprint=projection.projection_fingerprint,
    )

    plan = plan_telegram_menu_reconciliation(
        projection,
        [("stale", "Stale")],
        scope="default",
        prior_settlement=other_scope,
    )

    assert plan.action is TelegramMenuReconciliationAction.SET
    assert plan.reason == "unsettled"


