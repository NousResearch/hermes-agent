"""End-to-end coverage for the paper-trading confirmation flow."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from brokerage.app import create_app
from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore
import tools.brokerage_tool as brokerage_tool


class FakeBroker(BrokerAdapter):
    def __init__(self, result: BrokerSubmissionResult, *, order_statuses: dict[str, dict] | None = None):
        self.result = result
        self.order_statuses = order_statuses or {}
        self.submitted: list[TradeIntent] = []

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


def _build_stack(
    tmp_path: Path,
    broker_result: BrokerSubmissionResult,
    *,
    order_statuses: dict[str, dict] | None = None,
) -> tuple[TestClient, FakeBroker]:
    settings = BrokerageSettings(enabled=True, service_token="test-token")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = FakeBroker(broker_result, order_statuses=order_statuses)
    service = BrokerageService(settings, store, policy, broker)
    app = create_app(service=service, auth_token="test-token")
    return TestClient(app), broker


def _patch_tool_client(monkeypatch, client: TestClient) -> None:
    class _BridgeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, *, json=None, headers=None):
            response = client.request(method, url, json=json, headers=headers)
            request = httpx.Request(method, url)
            return httpx.Response(
                status_code=response.status_code,
                json=response.json(),
                request=request,
            )

    monkeypatch.setattr(brokerage_tool.httpx, "Client", _BridgeClient)
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {
            "enabled": True,
            "service_url": "http://testserver",
            "service_token": "test-token",
        },
    )


def test_end_to_end_tool_driven_paper_trade_reaches_submitted_status(tmp_path, monkeypatch):
    client, broker = _build_stack(
        tmp_path,
        BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-accepted-1",
            broker_status="Submitted",
        ),
    )
    _patch_tool_client(monkeypatch, client)

    created = json.loads(
        brokerage_tool.create_trade_intent_tool(
            {
                "account_mode": "paper",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "asset_class": "stock",
                "raw_user_text": "buy 10 shares of aapl at market in paper",
            }
        )
    )
    confirmed = json.loads(
        brokerage_tool.confirm_trade_intent_tool(
            {
                "intent_id": created["intent_id"],
                "confirmation_text": f"CONFIRM {created['confirmation_code']}",
            }
        )
    )
    status = json.loads(
        brokerage_tool.get_trade_intent_status_tool({"intent_id": created["intent_id"]})
    )

    assert created["status"] == "pending_confirmation"
    assert confirmed == {
        "intent_id": created["intent_id"],
        "status": "submitted",
        "broker_order_id": "ib-accepted-1",
        "broker_status": "Submitted",
        "detail": None,
    }
    assert status["status"] == "submitted"
    assert status["ibkr_order_id"] == "ib-accepted-1"
    assert [intent.symbol for intent in broker.submitted] == ["AAPL"]


def test_end_to_end_tool_status_check_reconciles_filled_trade(tmp_path, monkeypatch):
    client, broker = _build_stack(
        tmp_path,
        BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-accepted-1",
            broker_status="Submitted",
        ),
        order_statuses={"ib-accepted-1": {"broker_status": "Filled"}},
    )
    _patch_tool_client(monkeypatch, client)

    created = json.loads(
        brokerage_tool.create_trade_intent_tool(
            {
                "account_mode": "paper",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "asset_class": "stock",
            }
        )
    )
    brokerage_tool.confirm_trade_intent_tool(
        {
            "intent_id": created["intent_id"],
            "confirmation_text": f"CONFIRM {created['confirmation_code']}",
        }
    )

    status = json.loads(
        brokerage_tool.get_trade_intent_status_tool({"intent_id": created["intent_id"]})
    )

    assert status["status"] == "filled"
    assert status["broker_status"] == "Filled"
    assert [intent.request_id for intent in broker.submitted] == [created["intent_id"]]


def test_end_to_end_tool_driven_paper_trade_records_broker_rejection(tmp_path, monkeypatch):
    client, broker = _build_stack(
        tmp_path,
        BrokerSubmissionResult(
            accepted=False,
            broker_order_id=None,
            broker_status="Rejected",
            detail="insufficient buying power",
        ),
    )
    _patch_tool_client(monkeypatch, client)

    created = json.loads(
        brokerage_tool.create_trade_intent_tool(
            {
                "account_mode": "paper",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "asset_class": "stock",
            }
        )
    )
    confirmed = json.loads(
        brokerage_tool.confirm_trade_intent_tool(
            {
                "intent_id": created["intent_id"],
                "confirmation_text": f"CONFIRM {created['confirmation_code']}",
            }
        )
    )
    status = json.loads(
        brokerage_tool.get_trade_intent_status_tool({"intent_id": created["intent_id"]})
    )

    assert confirmed == {
        "intent_id": created["intent_id"],
        "status": "rejected",
        "broker_order_id": None,
        "broker_status": "Rejected",
        "detail": "insufficient buying power",
    }
    assert status["status"] == "rejected"
    assert status["ibkr_order_id"] is None
    assert [intent.request_id for intent in broker.submitted] == [created["intent_id"]]
