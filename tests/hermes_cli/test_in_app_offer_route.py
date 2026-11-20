"""``/api/in-app-offer``: the dashboard's read of the per-profile in-app offer memo."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import in_app_offer
from hermes_cli.web_routers import in_app_offer as route

OFFER = {
    "id": "offer_1",
    "title": "Welcome back",
    "body": "Here is an offer.",
    "cta_label": "See the offer",
    "cta_url": "https://portal.nousresearch.com/billing",
}


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def test_plain_read_answers_from_the_memo_without_fetching(client, monkeypatch):
    monkeypatch.setattr(in_app_offer, "pending_offer", lambda: dict(OFFER))

    def no_fetch():
        raise AssertionError("a plain read must not fetch the account")

    monkeypatch.setattr(in_app_offer, "refresh_from_portal", no_fetch)

    res = client.get("/api/in-app-offer")

    assert res.status_code == 200
    assert res.json() == {"in_app_offer": OFFER}


def test_refresh_fetches_the_account_first(client, monkeypatch):
    calls = []
    monkeypatch.setattr(in_app_offer, "refresh_from_portal", lambda: calls.append(1) or dict(OFFER))

    res = client.get("/api/in-app-offer?refresh=1")

    assert calls == [1]
    assert res.json() == {"in_app_offer": OFFER}


def test_no_offer_is_null_not_an_error(client, monkeypatch):
    monkeypatch.setattr(in_app_offer, "pending_offer", lambda: None)

    res = client.get("/api/in-app-offer")

    assert res.status_code == 200
    assert res.json() == {"in_app_offer": None}


def test_a_failing_read_answers_null(client, monkeypatch):
    def boom():
        raise RuntimeError("memo exploded")

    monkeypatch.setattr(in_app_offer, "refresh_from_portal", boom)

    res = client.get("/api/in-app-offer?refresh=1")

    assert res.status_code == 200
    assert res.json() == {"in_app_offer": None}


def test_the_route_is_mounted_on_the_dashboard_app():
    from hermes_cli.web_server import app

    assert any(getattr(r, "path", None) == "/api/in-app-offer" for r in app.routes)


def test_event_route_forwards_to_report_event(client, monkeypatch):
    calls = []
    monkeypatch.setattr(in_app_offer, "report_event", lambda offer_id, event: calls.append((offer_id, event)) or True)

    res = client.post("/api/in-app-offer/event", json={"offer_id": "offer_1", "event": "clicked"})

    assert res.status_code == 200
    assert res.json() == {"reported": True}
    assert calls == [("offer_1", "clicked")]


def test_event_route_never_errors(client, monkeypatch):
    def boom(offer_id, event):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(in_app_offer, "report_event", boom)

    res = client.post("/api/in-app-offer/event", json={"offer_id": "offer_1", "event": "shown"})

    assert res.status_code == 200
    assert res.json() == {"reported": False}
