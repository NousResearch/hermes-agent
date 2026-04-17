"""Tests for the brokerage service state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore


class FakeBroker(BrokerAdapter):
    def __init__(self, result: BrokerSubmissionResult | None = None, *, order_statuses: dict[str, dict] | None = None, positions: list[dict] | None = None):
        self.result = result or BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-123",
            broker_status="Submitted",
        )
        self.order_statuses = order_statuses or {}
        self.submitted: list[TradeIntent] = []
        self._positions = positions or []
        self.last_positions_query: dict[str, str | None] | None = None

    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        self.submitted.append(intent)
        return self.result

    def get_order_status(
        self,
        order_id: str,
        *,
        account_mode: str | None = None,
        expected_quantity: int | None = None,
    ):
        return self.order_statuses.get(order_id)

    def cancel_order(self, order_id: str):
        return None

    def get_positions(self, *, account_mode: str | None = None, account: str | None = None) -> list[dict]:
        self.last_positions_query = {"account_mode": account_mode, "account": account}
        if account is None:
            return self._positions
        return [p for p in self._positions if p.get("account") == account]


def _make_service(tmp_path, broker: BrokerAdapter | None = None) -> BrokerageService:
    settings = BrokerageSettings()
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    return BrokerageService(settings, store, policy, broker or FakeBroker())


def test_create_intent_returns_confirmation_code_and_pending_status(tmp_path):
    service = _make_service(tmp_path)

    result = service.create_intent(
        account_mode="paper",
        symbol="aapl",
        side="buy",
        quantity=10,
        order_type="market",
        asset_class="stock",
        raw_request_text="buy 10 aapl market paper",
        session_id="session-1",
    )

    assert result["status"] == "pending_confirmation"
    assert result["intent_id"].startswith("ti_")
    assert result["confirmation_code"].startswith("T-")
    assert result["preview"]["symbol"] == "AAPL"


def test_confirm_intent_with_wrong_token_fails(tmp_path):
    service = _make_service(tmp_path)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    with pytest.raises(ValueError, match="confirmation"):
        service.confirm_intent(created["intent_id"], "CONFIRM T-WRONG")


def test_confirm_intent_after_expiry_fails(tmp_path):
    service = _make_service(tmp_path)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    with pytest.raises(ValueError, match="expired"):
        service.confirm_intent(
            created["intent_id"],
            f"CONFIRM {created['confirmation_code']}",
            now=datetime.now(timezone.utc) + timedelta(minutes=5),
        )


def test_confirm_intent_with_valid_token_submits_order(tmp_path):
    broker = FakeBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    result = service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    assert result["status"] == "submitted"
    assert result["broker_order_id"] == "ib-123"
    assert len(broker.submitted) == 1


def test_paper_stop_market_order_survives_create_and_confirm(tmp_path):
    broker = FakeBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="SELL",
        quantity=5,
        order_type="STOP",
        stop_price=180.25,
        asset_class="stock",
    )

    assert created["preview"]["stop_price"] == 180.25

    result = service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    assert result["status"] == "submitted"
    assert broker.submitted[0].order_type == "STOP"
    assert broker.submitted[0].stop_price == 180.25


def test_live_trade_defaults_to_configured_live_account_and_freezes_it_in_preview(tmp_path):
    settings = BrokerageSettings(live_enabled=True, default_live_account="U3510752")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = FakeBroker()
    service = BrokerageService(settings, store, policy, broker)

    created = service.create_intent(
        account_mode="live",
        symbol="AAPL",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        asset_class="stock",
    )

    assert created["preview"]["broker_account"] == "U3510752"

    result = service.confirm_intent(
        created["intent_id"],
        f"CONFIRM LIVE BUY 1 AAPL {created['confirmation_code']}",
    )

    assert result["status"] == "submitted"
    assert broker.submitted[0].broker_account == "U3510752"


def test_explicit_live_account_overrides_default_live_account(tmp_path):
    settings = BrokerageSettings(live_enabled=True, default_live_account="U3510752")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = FakeBroker()
    service = BrokerageService(settings, store, policy, broker)

    created = service.create_intent(
        account_mode="live",
        symbol="AAPL",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        asset_class="stock",
        broker_account="U3053904",
    )

    assert created["preview"]["broker_account"] == "U3053904"

    service.confirm_intent(
        created["intent_id"],
        f"CONFIRM LIVE BUY 1 AAPL {created['confirmation_code']}",
    )

    assert broker.submitted[0].broker_account == "U3053904"


def test_live_stop_market_order_uses_default_live_account(tmp_path):
    settings = BrokerageSettings(live_enabled=True, default_live_account="U3510752")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = FakeBroker()
    service = BrokerageService(settings, store, policy, broker)

    created = service.create_intent(
        account_mode="live",
        symbol="AAPL",
        side="SELL",
        quantity=1,
        order_type="STOP",
        stop_price=180.25,
        asset_class="stock",
    )

    assert created["preview"]["broker_account"] == "U3510752"
    assert created["preview"]["stop_price"] == 180.25

    service.confirm_intent(
        created["intent_id"],
        f"CONFIRM LIVE SELL 1 AAPL {created['confirmation_code']}",
    )

    assert broker.submitted[0].broker_account == "U3510752"
    assert broker.submitted[0].order_type == "STOP"
    assert broker.submitted[0].stop_price == 180.25


def test_get_intent_reconciles_submitted_trade_to_filled_status(tmp_path):
    broker = FakeBroker(order_statuses={"ib-123": {"broker_status": "Filled"}})
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")
    result = service.get_intent(created["intent_id"])

    assert result["status"] == "filled"
    assert result["broker_status"] == "Filled"
    events = service.store.list_events(created["intent_id"])
    assert events[-1]["event_type"] == "filled"


def test_get_intent_reconciliation_is_idempotent_for_terminal_status(tmp_path):
    broker = FakeBroker(order_statuses={"ib-123": {"broker_status": "Filled"}})
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")
    first = service.get_intent(created["intent_id"])
    second = service.get_intent(created["intent_id"])

    assert first["status"] == "filled"
    assert second["status"] == "filled"
    events = service.store.list_events(created["intent_id"])
    assert [event["event_type"] for event in events].count("filled") == 1


def test_get_intent_keeps_submitted_status_for_pending_cancel_broker_state(tmp_path):
    broker = FakeBroker(order_statuses={"ib-123": {"broker_status": "PendingCancel"}})
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")
    result = service.get_intent(created["intent_id"])

    assert result["status"] == "submitted"
    assert result["broker_status"] == "PendingCancel"
    events = service.store.list_events(created["intent_id"])
    assert [event["event_type"] for event in events].count("cancelled") == 0


def test_broker_rejection_moves_intent_to_rejected(tmp_path):
    broker = FakeBroker(
        BrokerSubmissionResult(
            accepted=False,
            broker_order_id=None,
            broker_status="Rejected",
            detail="insufficient buying power",
        )
    )
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    result = service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    assert result["status"] == "rejected"
    assert result["detail"] == "insufficient buying power"


def test_cancel_pending_intent_moves_to_cancelled(tmp_path):
    service = _make_service(tmp_path)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    result = service.cancel_intent(created["intent_id"])

    assert result["status"] == "cancelled"


def test_confirmation_code_is_one_time_use(tmp_path):
    broker = FakeBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )
    confirmation = f"CONFIRM {created['confirmation_code']}"

    first = service.confirm_intent(created["intent_id"], confirmation)
    assert first["status"] == "submitted"

    with pytest.raises(ValueError, match="pending_confirmation"):
        service.confirm_intent(created["intent_id"], confirmation)


def test_confirm_intent_consumes_confirmation_code(tmp_path):
    broker = FakeBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )
    code = created["confirmation_code"]
    assert service.store.get_intent(created["intent_id"])["confirmation_code"] == code

    service.confirm_intent(created["intent_id"], f"CONFIRM {code}")

    # Code should be nullified after use
    assert service.store.get_intent(created["intent_id"])["confirmation_code"] is None


def test_transition_graph_prevents_backward_transition(tmp_path):
    """Terminal states cannot transition to any other state."""
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    intent = TradeIntent(
        request_id="req-t1",
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        asset_class="stock",
    )
    store.create_intent(intent, confirmation_code="T-AAAA")
    store.update_status("req-t1", "confirmed")
    store.update_status("req-t1", "submitted")
    store.transition_status("req-t1", "submitted", "filled")

    with pytest.raises(ValueError, match="Illegal state transition"):
        store.update_status("req-t1", "pending_confirmation")

    with pytest.raises(ValueError, match="Illegal state transition"):
        store.transition_status("req-t1", "filled", "submitted")


def test_transition_graph_prevents_skip(tmp_path):
    """Cannot skip the confirmed state."""
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    intent = TradeIntent(
        request_id="req-t2",
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        asset_class="stock",
    )
    store.create_intent(intent, confirmation_code="T-BBBB")

    with pytest.raises(ValueError, match="Illegal state transition"):
        store.transition_status("req-t2", "pending_confirmation", "submitted")


def test_cas_prevents_double_confirm(tmp_path):
    """Two concurrent confirm calls cannot both succeed — CAS wins."""
    broker = FakeBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )
    code = created["confirmation_code"]

    # First confirm succeeds
    service.confirm_intent(created["intent_id"], f"CONFIRM {code}")

    # Second confirm fails — status is no longer pending_confirmation
    with pytest.raises(ValueError, match="pending_confirmation"):
        service.confirm_intent(created["intent_id"], f"CONFIRM {code}")


# --- Broker error handling ---


class CrashingBroker(BrokerAdapter):
    """A broker that always raises an exception on submit."""

    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        raise ConnectionError("TWS connection refused")

    def get_order_status(
        self,
        order_id: str,
        *,
        account_mode: str | None = None,
        expected_quantity: int | None = None,
    ):
        return None

    def cancel_order(self, order_id: str):
        return None


def test_confirm_intent_handles_broker_exception_gracefully(tmp_path):
    broker = CrashingBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    result = service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    assert result["status"] == "submission_error"
    assert "TWS connection refused" in result["detail"]
    assert result["broker_order_id"] is None

    # Verify persisted status
    stored = service.get_intent(created["intent_id"])
    assert stored["status"] == "submission_error"


def test_confirm_intent_logs_submission_error_event(tmp_path):
    broker = CrashingBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    events = service.store.list_events(created["intent_id"])
    event_types = [e["event_type"] for e in events]
    assert "submission_error" in event_types


def test_cancel_rejected_intent_fails(tmp_path):
    broker = FakeBroker(
        BrokerSubmissionResult(
            accepted=False,
            broker_order_id=None,
            broker_status="Rejected",
            detail="insufficient buying power",
        )
    )
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    with pytest.raises(ValueError, match="pending_confirmation"):
        service.cancel_intent(created["intent_id"])


def test_cancel_submission_error_intent_fails(tmp_path):
    broker = CrashingBroker()
    service = _make_service(tmp_path, broker=broker)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.confirm_intent(created["intent_id"], f"CONFIRM {created['confirmation_code']}")

    with pytest.raises(ValueError, match="pending_confirmation"):
        service.cancel_intent(created["intent_id"])


def test_cancel_cancelled_intent_fails(tmp_path):
    service = _make_service(tmp_path)
    created = service.create_intent(
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
    )

    service.cancel_intent(created["intent_id"])

    with pytest.raises(ValueError, match="pending_confirmation"):
        service.cancel_intent(created["intent_id"])


def test_get_positions_delegates_to_broker(tmp_path):
    positions = [
        {"account": "DUQ218494", "symbol": "AAPL", "position": 5.0, "avg_cost": 264.98, "account_mode": "paper"},
    ]
    broker = FakeBroker(positions=positions)
    service = _make_service(tmp_path, broker=broker)

    result = service.get_positions(account_mode="paper")

    assert result == positions
    assert broker.last_positions_query == {"account_mode": "paper", "account": None}


def test_get_positions_delegates_account_filter_to_broker(tmp_path):
    positions = [
        {"account": "DUQ218494", "symbol": "AAPL", "position": 5.0, "avg_cost": 264.98, "account_mode": "paper"},
        {"account": "U3510752", "symbol": "NFLX", "position": 10.0, "avg_cost": 900.5, "account_mode": "live"},
    ]
    broker = FakeBroker(positions=positions)
    service = _make_service(tmp_path, broker=broker)

    result = service.get_positions(account_mode="live", account="U3510752")

    assert result == [positions[1]]
    assert broker.last_positions_query == {"account_mode": "live", "account": "U3510752"}


def test_get_positions_returns_empty_list_when_no_positions(tmp_path):
    service = _make_service(tmp_path)

    result = service.get_positions()

    assert result == []
