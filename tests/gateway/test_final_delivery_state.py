"""Tests for FinalDeliveryState enum, SendResult.delivery_state field,
and effective_delivery_state() helper in gateway/platforms/base.py."""
import pytest
from gateway.platforms.base import (
    SendResult,
    FinalDeliveryState,
    effective_delivery_state,
)


def test_legacy_success_infers_fully_delivered():
    result = SendResult(success=True)
    assert effective_delivery_state(result) is FinalDeliveryState.FULLY_DELIVERED


def test_legacy_failure_infers_not_handled():
    result = SendResult(success=False, error="x")
    assert effective_delivery_state(result) is FinalDeliveryState.NOT_HANDLED


def test_explicit_partial_overrides_success_flag():
    result = SendResult(
        success=False,
        delivery_state=FinalDeliveryState.PARTIALLY_DELIVERED,
    )
    assert effective_delivery_state(result) is FinalDeliveryState.PARTIALLY_DELIVERED


def test_partial_cannot_be_reported_as_success():
    with pytest.raises(ValueError, match="PARTIALLY_DELIVERED"):
        SendResult(
            success=True,
            delivery_state=FinalDeliveryState.PARTIALLY_DELIVERED,
        )


def test_explicit_fully_delivered_with_success():
    result = SendResult(
        success=True,
        delivery_state=FinalDeliveryState.FULLY_DELIVERED,
    )
    assert effective_delivery_state(result) is FinalDeliveryState.FULLY_DELIVERED


def test_explicit_not_handled_with_failure():
    result = SendResult(
        success=False,
        delivery_state=FinalDeliveryState.NOT_HANDLED,
    )
    assert effective_delivery_state(result) is FinalDeliveryState.NOT_HANDLED


def test_delivery_state_is_optional_and_defaults_none():
    result = SendResult(success=True)
    assert result.delivery_state is None
