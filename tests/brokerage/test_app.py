"""Tests for the FastAPI brokerage service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brokerage.app import build_service, create_app
from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore


class FakeBroker(BrokerAdapter):
    def __init__(self, result: BrokerSubmissionResult | None = None, positions: list[dict] | None = None):
        self.result = result or BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-123",
            broker_status="Submitted",
        )
        self._positions = positions or []
        self.last_positions_query: dict[str, str | None] | None = None

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

    def get_positions(self, *, account_mode: str | None = None, account: str | None = None) -> list[dict]:
        self.last_positions_query = {"account_mode": account_mode, "account": account}
        if account is None:
            return self._positions
        return [p for p in self._positions if p.get("account") == account]


def _make_client(tmp_path: Path, *, token: str = "test-token", positions: list[dict] | None = None) -> TestClient:
    settings = BrokerageSettings(service_token=token)
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    service = BrokerageService(settings, store, policy, FakeBroker(positions=positions))
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


def test_build_service_loads_brokerage_settings_from_hermes_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "brokerage:\n"
        "  live_enabled: true\n"
        "  default_live_account: U3510752\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    service = build_service()

    assert service.settings.live_enabled is True
    assert service.settings.default_live_account == "U3510752"


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


def test_create_trade_intent_accepts_paper_stop_market_order(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "paper",
            "symbol": "AAPL",
            "side": "SELL",
            "quantity": 5,
            "order_type": "STOP",
            "stop_price": 180.25,
            "asset_class": "stock",
        },
    )

    assert response.status_code == 201
    assert response.json()["preview"] == {
        "account_mode": "paper",
        "side": "SELL",
        "symbol": "AAPL",
        "quantity": 5,
        "order_type": "STOP",
        "asset_class": "stock",
        "stop_price": 180.25,
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


def test_create_trade_intent_accepts_explicit_broker_account(tmp_path):
    settings = BrokerageSettings(service_token="test-token", live_enabled=True)
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    service = BrokerageService(settings, store, policy, FakeBroker())
    app = create_app(service=service, auth_token="test-token")
    client = TestClient(app)

    response = client.post(
        "/trade-intents",
        headers=_auth_headers(),
        json={
            "account_mode": "live",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 1,
            "order_type": "MARKET",
            "asset_class": "stock",
            "broker_account": "u3510752",
        },
    )

    assert response.status_code == 201
    assert response.json()["preview"]["broker_account"] == "U3510752"


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


def test_get_positions_returns_empty_when_no_positions(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/positions", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["positions"] == []
    assert body["count"] == 0


def test_get_positions_returns_broker_positions(tmp_path):
    positions = [
        {"account": "DUQ218494", "symbol": "AAPL", "position": 5.0, "avg_cost": 264.98, "account_mode": "paper"},
        {"account": "DUQ218494", "symbol": "MSFT", "position": 3.0, "avg_cost": 420.50, "account_mode": "paper"},
    ]
    client = _make_client(tmp_path, positions=positions)

    response = client.get("/positions", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["positions"] == positions


def test_get_positions_passes_account_mode_and_account_query_params(tmp_path):
    positions = [
        {"account": "DUQ218494", "symbol": "AAPL", "position": 5.0, "avg_cost": 264.98, "account_mode": "paper"},
        {"account": "U3510752", "symbol": "NFLX", "position": 10.0, "avg_cost": 900.5, "account_mode": "live"},
    ]
    settings = BrokerageSettings(service_token="test-token")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = FakeBroker(positions=positions)
    service = BrokerageService(settings, store, policy, broker)
    app = create_app(service=service, auth_token="test-token")
    client = TestClient(app)

    response = client.get("/positions?account_mode=live&account=U3510752", headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["positions"] == [positions[1]]
    assert broker.last_positions_query == {"account_mode": "live", "account": "U3510752"}


def test_get_positions_requires_auth(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/positions")

    assert response.status_code == 401
