import json

import tools.customer_intent_tool as customer_intent_tool


def test_customer_intent_raise_inserts_pending_intent(monkeypatch):
    captured = {}

    def fake_statement_one(statement, *, user=None):
        captured["statement"] = statement
        captured["user"] = user
        return {
            "intent_id": "intent-msg-123",
            "status": "pending",
            "intent_type": "formal_quote",
            "summary": "Cliente pidió cotización",
        }

    monkeypatch.setattr(customer_intent_tool.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(customer_intent_tool.sql, "runtime_env", lambda: {"CRM_DB_RUNTIME_USER": "crm_runtime"})

    result = json.loads(customer_intent_tool._handle_customer_intent_raise({
        "intent_id": "intent-msg-123",
        "channel": "whatsapp",
        "source_ref": "wamid.abc",
        "conversation_ref": "584120000000@s.whatsapp.net",
        "intent_type": "formal_quote",
        "customer_request_raw": "envíame una cotización",
        "summary": "Cliente pidió cotización de agentes SitioUno",
        "required_action": "Crear y enviar cotización formal",
        "metadata": {"business_id": "sitiouno"},
    }))

    assert result["ok"] is True
    assert result["intent"]["status"] == "pending"
    assert result["acknowledgement"].startswith("Solicitud registrada")
    assert captured["user"] == "crm_runtime"
    assert "INSERT INTO crm.customer_intents" in captured["statement"]
    assert "'pending'" in captured["statement"]
    assert "sophie_acknowledgement_required" in captured["statement"]


def test_customer_intent_raise_requires_raw_request_and_summary(monkeypatch):
    monkeypatch.setattr(customer_intent_tool.sql, "runtime_env", lambda: {"CRM_DB_RUNTIME_USER": "crm_runtime"})

    result = json.loads(customer_intent_tool._handle_customer_intent_raise({"summary": "sin raw"}))

    assert result["error"]
    assert "customer_request_raw is required" in result["error"]


def test_customer_intent_list_defaults_to_pending(monkeypatch):
    captured = {}

    def fake_rows(statement, *, user=None):
        captured["statement"] = statement
        captured["user"] = user
        return [{"intent_id": "intent-1", "status": "pending"}]

    monkeypatch.setattr(customer_intent_tool.sql, "rows", fake_rows)
    monkeypatch.setattr(customer_intent_tool.sql, "runtime_env", lambda: {"CRM_DB_RUNTIME_USER": "crm_runtime"})

    result = json.loads(customer_intent_tool._handle_customer_intent_list({"limit": 5}))

    assert result["ok"] is True
    assert result["count"] == 1
    assert "ci.status = 'pending'" in captured["statement"]
    assert "LIMIT 5" in captured["statement"]
    assert captured["user"] == "crm_runtime"


def test_customer_intent_update_marks_terminal_status(monkeypatch):
    captured = {}

    def fake_statement_one(statement, *, user=None):
        captured["statement"] = statement
        captured["user"] = user
        return {"intent_id": "intent-1", "status": "completed", "processed_at": "now"}

    monkeypatch.setattr(customer_intent_tool.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(customer_intent_tool.sql, "runtime_env", lambda: {"CRM_DB_RUNTIME_USER": "crm_runtime"})

    result = json.loads(customer_intent_tool._handle_customer_intent_update({
        "intent_id": "intent-1",
        "status": "completed",
        "result_summary": "Cotización enviada y registrada en CRM",
    }))

    assert result["ok"] is True
    assert result["intent"]["status"] == "completed"
    assert "processed_at=COALESCE(processed_at, now())" in captured["statement"]
    assert "result_summary='Cotización enviada y registrada en CRM'" in captured["statement"]
