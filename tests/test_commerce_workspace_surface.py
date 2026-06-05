from fastapi import FastAPI
from fastapi.testclient import TestClient

import hashlib
import hmac
import json
import time

from hermes_cli import commerce_workspace_surface as surface


def test_workspace_page_renders_quote_and_records_open(monkeypatch):
    statements = []

    def fake_one(query, **_kwargs):
        if "FROM sales.customer_workspaces" in query:
            return {
                "workspace_id": "workspace-quote-1",
                "document_type": "quote",
                "document_id": "quote-1",
                "customer_name": "Client",
                "status": "pending",
            }
        if "FROM sales.quotes" in query:
            return {
                "quote_id": "quote-1",
                "title": "Website build",
                "currency": "USD",
                "subtotal": 1000,
                "discount_amount": 0,
                "tax_amount": 160,
                "total": 1160,
                "status": "sent",
            }
        return None

    def fake_rows(query, **_kwargs):
        if "FROM sales.quote_items" in query:
            return [{"description": "Design", "quantity": 1, "unit_price": 1000, "line_total": 1160}]
        if "FROM sales.customer_workspace_events" in query:
            return []
        return []

    monkeypatch.setattr(surface.sql, "one", fake_one)
    monkeypatch.setattr(surface.sql, "rows", fake_rows)
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {})

    html = surface.render_workspace_html("token-1")

    assert "Website build" in html
    assert "Design" in html
    assert "Aprobar" in html
    assert any("opened" in statement for statement in statements)


def test_approving_quote_converts_to_order_invoice_and_records_event(monkeypatch):
    statements = []
    calls = []

    monkeypatch.setattr(surface, "_get_workspace", lambda token: {
        "workspace_id": "workspace-quote-1",
        "document_type": "quote",
        "document_id": "quote-1",
        "customer_email": "client@example.com",
        "customer_name": "Client",
        "status": "viewed",
    })
    monkeypatch.setattr(surface, "_workspace_events", lambda workspace_id: [])
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {})
    monkeypatch.setattr(surface, "_notify_agent_workspace_interaction", lambda *args, **kwargs: {"ok": True})

    def fake_order(args):
        calls.append(("order", args))
        return '{"ok": true, "order": {"order_id": "order-1"}, "items": []}'

    def fake_invoice(args):
        calls.append(("invoice", args))
        return '{"ok": true, "invoice": {"invoice_id": "invoice-1"}, "items": []}'

    monkeypatch.setattr(surface.sales_tool, "_handle_order_create", fake_order)
    monkeypatch.setattr(surface.sales_tool, "_handle_invoice_create", fake_invoice)

    result = surface.approve_workspace("token-1", actor_ref="client@example.com", signature="Client")

    assert result["ok"] is True
    assert result["order_id"] == "order-1"
    assert result["invoice_id"] == "invoice-1"
    assert calls[0] == ("order", {"quote_id": "quote-1", "metadata": {"source": "customer_workspace", "workspace_id": "workspace-quote-1"}})
    assert calls[1][0] == "invoice"
    assert any("status='approved'" in statement for statement in statements)
    assert any("approved" in statement and "client@example.com" in statement for statement in statements)


def test_invalid_or_missing_token_renders_generic_agent_placeholder(monkeypatch):
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)

    def fake_one(query, **_kwargs):
        if "FROM sales.customer_workspaces" in query:
            return None
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(surface.sql, "one", fake_one)

    invalid_response = client.get("/w/not-a-real-token")
    missing_response = client.get("/w")

    assert invalid_response.status_code == 200
    assert missing_response.status_code == 200
    assert "Espacio para agentes personalizados" in invalid_response.text
    assert "Personalized agent space" in invalid_response.text
    assert "ear.app" in invalid_response.text
    assert "sitiouno.us" in invalid_response.text
    assert "?lang=en" in invalid_response.text
    assert "?lang=es" in invalid_response.text
    assert "placeholder" not in invalid_response.text.lower()
    assert "—" not in invalid_response.text


def test_public_routes_support_comment_and_reject(monkeypatch):
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)
    actions = []

    monkeypatch.setattr(surface, "render_workspace_html", lambda token, banner=None: f"<html>{token}:{banner or ''}</html>")
    monkeypatch.setattr(surface, "comment_workspace", lambda token, comment, actor_ref=None: actions.append(("comment", token, comment, actor_ref)) or {"ok": True})
    monkeypatch.setattr(surface, "reject_workspace", lambda token, comment=None, actor_ref=None: actions.append(("reject", token, comment, actor_ref)) or {"ok": True})

    comment_response = client.post("/w/token-1/comment", data={"comment": "Tengo una pregunta", "actor_ref": "client@example.com"})
    reject_response = client.post("/w/token-1/reject", data={"comment": "No gracias"})

    assert comment_response.status_code == 200
    assert reject_response.status_code == 200
    assert ("comment", "token-1", "Tengo una pregunta", "client@example.com") in actions
    assert ("reject", "token-1", "No gracias", None) in actions


def test_workspace_actions_use_stored_customer_identity_and_no_email_fields(monkeypatch):
    statements = []

    def fake_one(query, **_kwargs):
        if "FROM sales.customer_workspaces" in query:
            return {
                "workspace_id": "workspace-quote-1",
                "document_type": "quote",
                "document_id": "quote-1",
                "customer_name": "Client Name",
                "customer_email": "client@example.com",
                "status": "pending",
            }
        if "FROM sales.quotes" in query:
            return {
                "quote_id": "quote-1",
                "title": "Website build",
                "currency": "USD",
                "subtotal": 1000,
                "discount_amount": 0,
                "tax_amount": 0,
                "total": 1000,
                "status": "sent",
            }
        return None

    def fake_rows(query, **_kwargs):
        if "FROM sales.quote_items" in query:
            return [{"description": "Design", "quantity": 1, "unit_price": 1000, "line_total": 1000}]
        if "FROM sales.customer_workspace_events" in query:
            return []
        return []

    monkeypatch.setattr(surface.sql, "one", fake_one)
    monkeypatch.setattr(surface.sql, "rows", fake_rows)
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {})

    html = surface.render_workspace_html("token-1")

    assert "name=\"actor_ref\"" not in html
    assert "Email o teléfono" not in html
    assert "Tu email o teléfono" not in html
    assert "client@example.com" in html
    assert "Aprobar cotización" in html
    assert "Rechazar" in html
    assert "button approve" in html
    assert "button reject" in html


def test_invoice_workspace_renders_stripe_payment_button(monkeypatch):
    statements = []

    def fake_one(query, **_kwargs):
        if "FROM sales.customer_workspaces" in query:
            return {
                "workspace_id": "workspace-invoice-1",
                "document_type": "invoice",
                "document_id": "invoice-1",
                "customer_name": "Client Name",
                "customer_email": "client@example.com",
                "status": "viewed",
            }
        if "FROM sales.invoices" in query:
            return {
                "invoice_id": "invoice-1",
                "title": "Invoice build",
                "currency": "USD",
                "total": 104.4,
                "status": "sent",
            }
        if "FROM sales.payment_requests" in query:
            return {
                "payment_request_id": "pay-1",
                "invoice_id": "invoice-1",
                "payment_url": "https://checkout.stripe.com/c/pay/cs_test_123",
                "status": "pending",
            }
        return None

    def fake_rows(query, **_kwargs):
        if "FROM sales.invoice_items" in query:
            return [{"description": "Design", "quantity": 1, "unit_price": 104.4, "line_total": 104.4}]
        if "FROM sales.customer_workspace_events" in query:
            return []
        return []

    monkeypatch.setattr(surface.sql, "one", fake_one)
    monkeypatch.setattr(surface.sql, "rows", fake_rows)
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {})

    html = surface.render_workspace_html("token-1")

    assert "Pagar factura con Stripe" in html
    assert "https://checkout.stripe.com/c/pay/cs_test_123" in html
    assert "Pagar ahora" in html
    assert any("opened" in statement for statement in statements)


def test_stripe_success_and_cancel_routes_render():
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)

    success = client.get("/payments/stripe/success?session_id=cs_test_123")
    cancel = client.get("/payments/stripe/cancel")

    assert success.status_code == 200
    assert "Pago recibido" in success.text
    assert "cs_test_123" in success.text
    assert cancel.status_code == 200
    assert "Pago pendiente" in cancel.text


def test_stripe_webhook_reconciles_checkout_completed(monkeypatch):
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)
    statements = []
    rows_queries = []
    events = []

    monkeypatch.setattr(surface.sql, "runtime_env", lambda: {"STRIPE_WEBHOOK_SECRET": "whsec_test"})

    def fake_statement_one(statement, **_kwargs):
        statements.append(statement)
        if "UPDATE sales.payment_requests" in statement:
            return {"payment_request_id": "pay-1", "invoice_id": "invoice-1", "status": "paid"}
        if "UPDATE sales.invoices" in statement:
            return {"invoice_id": "invoice-1", "status": "paid"}
        if "INSERT INTO sales.customer_workspace_events" in statement:
            events.append(statement)
            return {"workspace_event_id": 1}
        return {}

    def fake_rows(query, **_kwargs):
        rows_queries.append(query)
        if "UPDATE sales.customer_workspaces" in query:
            return [{"workspace_id": "workspace-invoice-1", "document_id": "invoice-1"}]
        return []

    monkeypatch.setattr(surface.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(surface.sql, "rows", fake_rows)

    payload = {
        "id": "evt_test_123",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_123",
                "object": "checkout.session",
                "client_reference_id": "pay-1",
                "metadata": {"payment_request_id": "pay-1", "invoice_id": "invoice-1"},
                "payment_status": "paid",
            }
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    timestamp = int(time.time())
    signature = hmac.new(b"whsec_test", str(timestamp).encode() + b"." + raw, hashlib.sha256).hexdigest()

    response = client.post(
        "/api/payments/stripe/webhook",
        content=raw,
        headers={"stripe-signature": f"t={timestamp},v1={signature}", "content-type": "application/json"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "paid"
    assert body["invoice_status"] == "paid"
    assert body["workspace_count"] == 1
    assert any("UPDATE sales.payment_requests" in statement and "paid" in statement for statement in statements)
    assert any("UPDATE sales.invoices" in statement and "paid" in statement for statement in statements)
    assert any("UPDATE sales.customer_workspaces" in query and "paid" in query for query in rows_queries)
    assert any("paid" in statement and "evt_test_123" in statement for statement in events)


def test_stripe_webhook_rejects_invalid_signature(monkeypatch):
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)

    monkeypatch.setattr(surface.sql, "runtime_env", lambda: {"STRIPE_WEBHOOK_SECRET": "whsec_test"})

    response = client.post(
        "/api/payments/stripe/webhook",
        content=b'{"id":"evt_bad","type":"checkout.session.completed"}',
        headers={"stripe-signature": "t=123,v1=bad"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_stripe_signature"


def test_comment_reject_and_approve_use_workspace_customer_and_create_agent_followups(monkeypatch):
    statements = []
    notifications = []

    workspace = {
        "workspace_id": "workspace-quote-1",
        "document_type": "quote",
        "document_id": "quote-1",
        "customer_email": "client@example.com",
        "customer_name": "Client Name",
        "status": "viewed",
    }

    monkeypatch.setattr(surface, "_get_workspace", lambda token: workspace)
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {})
    monkeypatch.setattr(surface, "_notify_agent_workspace_interaction", lambda workspace_arg, event_type, comment=None, metadata=None: notifications.append((workspace_arg["workspace_id"], event_type, comment, metadata)) or {"ok": True})
    monkeypatch.setattr(surface.sales_tool, "_handle_order_create", lambda args: '{"ok": true, "order": {"order_id": "order-1"}, "items": []}')
    monkeypatch.setattr(surface.sales_tool, "_handle_invoice_create", lambda args: '{"ok": true, "invoice": {"invoice_id": "invoice-1"}, "items": []}')

    surface.comment_workspace("token-1", "Tengo una pregunta")
    surface.reject_workspace("token-1", "No gracias")
    surface.approve_workspace("token-1")

    assert any("client@example.com" in statement and "commented" in statement for statement in statements)
    assert any("client@example.com" in statement and "rejected" in statement for statement in statements)
    assert any("client@example.com" in statement and "approved" in statement for statement in statements)
    assert notifications[0][:3] == ("workspace-quote-1", "commented", "Tengo una pregunta")
    assert notifications[1][:3] == ("workspace-quote-1", "rejected", "No gracias")
    assert notifications[2][0:2] == ("workspace-quote-1", "approved")


def test_customer_workspace_create_sends_email_and_records_sent_event(monkeypatch):
    statements = []
    sent_payloads = []

    monkeypatch.setattr(surface.sales_tool.sql, "statement_one", lambda statement, **_kwargs: statements.append(statement) or {
        "workspace_id": "workspace-quote-1",
        "document_type": "quote",
        "document_id": "quote-1",
        "customer_email": "client@example.com",
        "customer_name": "Client Name",
        "public_url": "https://zeus.kidu.app/w/token-1",
    })
    monkeypatch.setattr(surface.sales_tool, "_token", lambda: "token-1")

    from tools import notification_tool

    monkeypatch.setattr(notification_tool, "_email_adapter_send", lambda payload: sent_payloads.append(payload) or {"ok": True, "adapter": "sendgrid", "status": 202})

    result = surface.sales_tool._handle_customer_workspace_create({
        "workspace_id": "workspace-quote-1",
        "document_type": "quote",
        "document_id": "quote-1",
        "customer_email": "client@example.com",
        "customer_name": "Client Name",
        "send_email": True,
    })

    assert '"ok": true' in result
    assert sent_payloads
    assert sent_payloads[0]["to_email"] == "client@example.com"
    assert "https://zeus.kidu.app/w/token-1" in sent_payloads[0]["text"]
    assert any("sent" in statement and "workspace-quote-1" in statement for statement in statements)
