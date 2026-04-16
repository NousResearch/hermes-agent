"""Tests for the FastAPI brokerage service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brokerage.app import create_app
from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore


class FakeBroker(BrokerAdapter):
    def __init__(self, result: BrokerSubmissionResult | None = None):
        self.result = result or BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-123",
            broker_status="Submitted",
        )

    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        return self.result

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


def _make_client(tmp_path: Path, *, token: str = "test-token") -> TestClient:
    settings = BrokerageSettings(service_token=token)
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    service = BrokerageService(settings, store, policy, FakeBroker())
    app = create_app(service=service, auth_token=token)
    return TestClient(app)


def _auth_headers(token: str = "test-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_healthz_returns_ok(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "connected" in data
    assert "mode" in data


def test_create_trade_intent_requires_bearer_auth(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/trade-intents",
        json={
            "account_mode": "paper",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "asset_class": "stock",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid authorization header"


def test_create_trade_intent_returns_pending_preview_and_confirmation_code(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "paper",
            "symbol": "aapl",
            "side": "buy",
            "quantity": 10,
            "order_type": "market",
            "asset_class": "stock",
            "raw_request_text": "buy 10 shares of aapl at market in paper",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["intent_id"].startswith("ti_")
    assert body["status"] == "pending_confirmation"
    assert body["confirmation_code"].startswith("T-")
    assert body["preview"] == {
        "account_mode": "paper",
        "side": "BUY",
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": "MARKET",
        "asset_class": "stock",
    }


def test_confirm_trade_intent_submits_order(tmp_path):
    client = _make_client(tmp_path)
    created = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "paper",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "asset_class": "stock",
        },
    ).json()

    response = client.post(
        f"/trade-intents/{created['intent_id']}/confirm",
        headers=_auth_headers(),
        json={"confirmation_text": f"CONFIRM {created['confirmation_code']}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent_id": created["intent_id"],
        "status": "submitted",
        "broker_order_id": "ib-123",
        "broker_status": "Submitted",
        "detail": None,
    }


def test_cancel_trade_intent_returns_cancelled_status(tmp_path):
    client = _make_client(tmp_path)
    created = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "paper",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "asset_class": "stock",
        },
    ).json()

    response = client.post(
        f"/trade-intents/{created['intent_id']}/cancel",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["intent_id"] == created["intent_id"]
    assert response.json()["status"] == "cancelled"


def test_get_trade_intent_returns_current_status(tmp_path):
    client = _make_client(tmp_path)
    created = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "paper",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "asset_class": "stock",
        },
    ).json()

    response = client.get(
        f"/trade-intents/{created['intent_id']}",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent_id"] == created["intent_id"]
    assert body["status"] == "pending_confirmation"
    assert body["confirmation_code"] == created["confirmation_code"]
