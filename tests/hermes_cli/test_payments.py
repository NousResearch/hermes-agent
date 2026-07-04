"""Tests for the dashboard Payments storage and API surface."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
    try:
        import hermes_cli.payments as payments

        monkeypatch.setattr(payments, "get_hermes_home", lambda: home)
        monkeypatch.setattr(
            payments,
            "load_env",
            lambda: {},
        )
    except Exception:
        pass
    return home


def _write_requests(home: Path, requests: list[dict]) -> None:
    payments_dir = home / "payments"
    payments_dir.mkdir(parents=True, exist_ok=True)
    (payments_dir / "requests.json").write_text(
        json.dumps({"requests": requests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class TestPaymentsStore:
    def test_list_requests_reports_sources_and_normalizes_records(self, isolated_home):
        (isolated_home / "google_token.json").write_text('{"token": "x"}', encoding="utf-8")
        (isolated_home / ".env").write_text("EMAIL_ADDRESS=bills@example.test\n", encoding="utf-8")
        _write_requests(
            isolated_home,
            [
                {
                    "id": "pay-1",
                    "source": "gmail",
                    "source_label": "Gmail",
                    "status": "new",
                    "confidence": "high",
                    "received_at": "2026-07-01T10:00:00+00:00",
                    "vendor": "Example Vendor",
                    "title": "July invoice",
                    "amount": {"value": 42, "currency": "GBP", "display": "GBP 42.00"},
                }
            ],
        )

        import hermes_cli.payments as payments

        importlib.reload(payments)
        result = payments.list_payment_requests()

        gmail = next(source for source in result["sources"] if source["id"] == "gmail")
        email = next(source for source in result["sources"] if source["id"] == "email")
        uploads = next(source for source in result["sources"] if source["id"] == "uploads")

        assert gmail["connected"] is True
        assert email["connected"] is True
        assert uploads["connected"] is True
        assert result["requests"][0]["vendor"] == "Example Vendor"
        assert result["requests"][0]["amount"]["display"] == "GBP 42.00"

    def test_update_status_persists_change(self, isolated_home):
        _write_requests(
            isolated_home,
            [
                {
                    "id": "pay-1",
                    "status": "new",
                    "vendor": "Example Vendor",
                }
            ],
        )

        import hermes_cli.payments as payments

        importlib.reload(payments)
        updated = payments.update_payment_status("pay-1", "paid")

        assert updated["status"] == "paid"

        saved = json.loads((isolated_home / "payments" / "requests.json").read_text(encoding="utf-8"))
        assert saved["requests"][0]["status"] == "paid"
        assert saved["requests"][0]["updated_at"]

    def test_sync_gmail_payment_requests_extracts_fields(self, isolated_home, monkeypatch):
        import hermes_cli.payments as payments

        importlib.reload(payments)

        def fake_run_google_api(args):
            if args[:2] == ["gmail", "search"]:
                return [
                    {
                        "id": "msg-1",
                        "threadId": "thread-1",
                        "from": "Acme Billing <billing@acme.test>",
                        "subject": "Invoice INV-1001",
                        "date": "Fri, 01 Jul 2026 12:00:00 +0000",
                        "snippet": "Amount due GBP 42.00 by 2026-07-05",
                        "labels": ["INBOX"],
                    }
                ]
            if args[:2] == ["gmail", "get"]:
                return {
                    "id": "msg-1",
                    "threadId": "thread-1",
                    "from": "Acme Billing <billing@acme.test>",
                    "subject": "Invoice INV-1001",
                    "date": "Fri, 01 Jul 2026 12:00:00 +0000",
                    "body": (
                        "Invoice number: INV-1001\n"
                        "Amount due: GBP 42.00\n"
                        "Due date: 2026-07-05\n"
                        "Sort code: 12-34-56\n"
                        "Account number: 12345678\n"
                        "Payment reference: ACME-1001\n"
                    ),
                }
            raise AssertionError(args)

        monkeypatch.setattr(payments, "_run_google_api", fake_run_google_api)

        result = payments.sync_gmail_payment_requests(query="invoice", max_results=5)

        assert result["imported"] == 1
        saved = payments.list_payment_requests()["requests"][0]
        assert saved["id"] == "gmail:msg-1"
        assert saved["vendor"] == "Acme Billing"
        assert saved["amount"]["display"] == "GBP 42.00"
        assert saved["sort_code"] == "12-34-56"
        assert saved["account_number"] == "12345678"
        assert saved["payment_reference"] == "ACME-1001"
        assert saved["invoice_number"] == "INV-1001"
        assert saved["confidence"] in {"medium", "high"}


class TestPaymentsApi:
    @pytest.fixture(autouse=True)
    def _setup_client(self, isolated_home, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", isolated_home / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_list_and_update_payments(self, isolated_home):
        _write_requests(
            isolated_home,
            [
                {
                    "id": "pay-1",
                    "source": "gmail",
                    "source_label": "Gmail",
                    "status": "needs_review",
                    "vendor": "Example Vendor",
                    "title": "July invoice",
                }
            ],
        )

        listed = self.client.get("/api/payments")
        assert listed.status_code == 200
        payload = listed.json()
        assert payload["requests"][0]["id"] == "pay-1"

        updated = self.client.post("/api/payments/pay-1/status", json={"status": "ready_to_pay"})
        assert updated.status_code == 200
        assert updated.json()["status"] == "ready_to_pay"

    def test_update_rejects_unknown_status(self):
        updated = self.client.post("/api/payments/missing/status", json={"status": "bogus"})
        assert updated.status_code == 400

    def test_sync_payments_endpoint(self, monkeypatch):
        import hermes_cli.payments as payments

        def fake_sync(query, max_results):
            assert query == payments.DEFAULT_GMAIL_QUERY
            assert max_results == payments.DEFAULT_GMAIL_MAX_RESULTS
            return {
                "source": "gmail",
                "query": query,
                "fetched": 1,
                "imported": 1,
                "updated": 0,
                "requests": [],
            }

        monkeypatch.setattr(payments, "sync_gmail_payment_requests", fake_sync)
        resp = self.client.post("/api/payments/sync", json={"source": "gmail"})
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1
