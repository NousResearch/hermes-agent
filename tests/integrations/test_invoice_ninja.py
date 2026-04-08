from __future__ import annotations

import json

import httpx

from integrations.invoice_ninja import InvoiceNinjaClient, InvoiceNinjaConfig


def _client(handler):
    transport = httpx.MockTransport(handler)
    return InvoiceNinjaClient(
        InvoiceNinjaConfig(base_url="https://invoicing.co", api_token="test-token"),
        client=httpx.Client(transport=transport),
    )


def test_build_url_normalizes_api_path():
    client = InvoiceNinjaClient(
        InvoiceNinjaConfig(base_url="https://invoicing.co/", api_token="test-token"),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))),
    )

    assert client._build_url("invoices") == "https://invoicing.co/api/v1/invoices"
    assert client._build_url("/api/v1/clients") == "https://invoicing.co/api/v1/clients"


def test_request_sets_auth_header_and_returns_json():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["token"] = request.headers.get("X-API-TOKEN")
        captured["accept"] = request.headers.get("Accept")
        return httpx.Response(200, json={"data": [1, 2, 3]})

    client = _client(handler)
    result = client.request("GET", "/invoices", params={"filter": "acme"})

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["data"] == {"data": [1, 2, 3]}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://invoicing.co/api/v1/invoices?filter=acme"
    assert captured["token"] == "test-token"
    assert captured["accept"] == "application/json"


def test_append_invoice_line_item_updates_existing_invoice():
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = json.loads(request.content.decode())
        requests.append((request.method, request.url.path, payload))

        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": "inv-123",
                    "client_id": "client-1",
                    "date": "2026-04-08",
                    "due_date": "2026-04-15",
                    "line_items": [
                        {"notes": "first item", "quantity": 1, "cost": 10},
                    ],
                },
            )

        assert request.method == "PUT"
        assert payload is not None
        assert payload["client_id"] == "client-1"
        assert len(payload["line_items"]) == 2
        assert payload["line_items"][1]["notes"] == "second item"
        assert "id" not in payload
        return httpx.Response(
            200,
            json={
                "id": "inv-123",
                "client_id": "client-1",
                "line_items": payload["line_items"],
            },
        )

    client = _client(handler)
    result = client.append_invoice_line_item("inv-123", {"notes": "second item", "quantity": 2, "cost": 15})

    assert result["ok"] is True
    assert result["data"]["id"] == "inv-123"
    assert len(result["data"]["line_items"]) == 2
    assert requests[0][0] == "GET"
    assert requests[1][0] == "PUT"


def test_record_payment_infers_client_id_from_invoice():
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = json.loads(request.content.decode())
        requests.append((request.method, request.url.path, payload))

        if request.method == "GET":
            return httpx.Response(200, json={"id": "inv-123", "client_id": "client-9"})

        assert request.method == "POST"
        assert payload is not None
        assert payload["client_id"] == "client-9"
        assert payload["amount"] == 42.5
        assert payload["invoices"] == [{"invoice_id": "inv-123", "amount": "42.5"}]
        assert payload["type_id"] == "2"
        return httpx.Response(200, json={"id": "pay-1", **payload})

    client = _client(handler)
    result = client.record_payment("inv-123", 42.5, payment_type_id="2", transaction_reference="check-99")

    assert result["ok"] is True
    assert result["data"]["id"] == "pay-1"
    assert requests[0][0] == "GET"
    assert requests[1][0] == "POST"


def test_expense_tools_use_expected_paths():
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = json.loads(request.content.decode())
        calls.append((request.method, request.url.path, payload))

        if request.method == "POST" and request.url.path == "/api/v1/expenses":
            if len([c for c in calls if c[0] == "POST" and c[1] == "/api/v1/expenses"]) == 1:
                assert payload == {"client_id": "client-9", "amount": 18.75, "private_notes": "meal"}
            else:
                assert payload == {
                    "client_id": "client-9",
                    "amount": 18.75,
                    "transaction_reference": "R-123",
                    "private_notes": "meal",
                }
            return httpx.Response(200, json={"id": "exp-1", **payload})

        if request.method == "GET" and request.url.path == "/api/v1/expenses":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "keep-1", "client_id": "client-9", "private_notes": "meal", "payment_date": None},
                        {"id": "drop-1", "client_id": "client-9", "invoice_id": "inv-1", "payment_date": "2026-04-01"},
                    ]
                },
            )

        if request.method == "GET" and request.url.path == "/api/v1/expenses/exp-1":
            return httpx.Response(200, json={"id": "exp-1", "client_id": "client-9"})

        if request.method == "PUT" and request.url.path == "/api/v1/expenses/exp-1":
            assert payload == {"id": "exp-1", "client_id": "client-9", "project_id": "proj-1", "category_id": "cat-1"}
            return httpx.Response(200, json={"id": "exp-1", **payload})

        if request.method == "POST" and request.url.path == "/api/v1/reports/expenses":
            assert payload == {"date_range": "this_month"}
            return httpx.Response(200, json={"ok": True, "report": []})

        raise AssertionError(f"unexpected call: {request.method} {request.url.path}")

    client = _client(handler)

    create_result = client.create_expense({"client_id": "client-9", "amount": 18.75, "private_notes": "meal"})
    receipt_result = client.create_expense_from_receipt(
        {"client_id": "client-9", "amount": 18.75, "notes": "meal", "receipt_number": "R-123"}
    )
    attached_result = client.attach_expense("exp-1", {"project_id": "proj-1", "category_id": "cat-1"})
    list_result = client.list_expenses()
    filtered_result = client.list_unpaid_or_unbilled_expenses(client_id="client-9", query="meal")
    report_result = client.run_expense_report({"date_range": "this_month"})

    assert create_result["ok"] is True
    assert create_result["data"]["id"] == "exp-1"
    assert receipt_result["ok"] is True
    assert receipt_result["data"]["id"] == "exp-1"
    assert attached_result["ok"] is True
    assert attached_result["data"]["project_id"] == "proj-1"
    assert list_result["ok"] is True
    assert list_result["data"] == {"data": [
        {"id": "keep-1", "client_id": "client-9", "private_notes": "meal", "payment_date": None},
        {"id": "drop-1", "client_id": "client-9", "invoice_id": "inv-1", "payment_date": "2026-04-01"},
    ]}
    assert filtered_result["ok"] is True
    assert filtered_result["count"] == 1
    assert filtered_result["data"][0]["id"] == "keep-1"
    assert report_result["ok"] is True
    assert report_result["data"] == {"ok": True, "report": []}
    assert [call[0] for call in calls] == ["POST", "POST", "GET", "PUT", "GET", "GET", "POST"]
