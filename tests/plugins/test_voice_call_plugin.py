"""Tests for the bundled voice_call plugin."""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_voice_call_available_requires_service_url(monkeypatch):
    cfg = importlib.import_module("plugins.voice_call.config")
    monkeypatch.delenv("VOICE_CALL_SERVICE_URL", raising=False)
    assert cfg.voice_call_available() is False
    monkeypatch.setenv("VOICE_CALL_SERVICE_URL", "http://127.0.0.1:8765")
    assert cfg.voice_call_available() is True


def test_voice_call_blocks_empty_purpose(monkeypatch):
    tools = importlib.import_module("plugins.voice_call.tools")
    monkeypatch.setenv("VOICE_CALL_SERVICE_URL", "http://voice.local")
    result = tools.voice_call({"action": "call", "to": "+13105551212", "context": "ask about appointment"})
    assert "purpose is required" in result


def test_voice_call_validates_prefix_and_redacts(monkeypatch):
    tools = importlib.import_module("plugins.voice_call.tools")
    monkeypatch.setenv("VOICE_CALL_SERVICE_URL", "http://voice.local")
    monkeypatch.setenv("VOICE_CALL_ALLOWED_PREFIXES", "+1")
    result = tools.voice_call({
        "action": "call",
        "to": "+442071838750",
        "purpose": "confirm booking",
        "context": "ask whether the booking is still active",
    })
    assert "not allowed" in result
    assert "+442071838750" not in result
    assert "*8750" in result


def test_voice_call_call_posts_to_service(monkeypatch):
    tools = importlib.import_module("plugins.voice_call.tools")
    monkeypatch.setenv("VOICE_CALL_SERVICE_URL", "http://voice.local")
    monkeypatch.delenv("VOICE_CALL_ALLOWED_PREFIXES", raising=False)
    captured = {}

    def fake_request(method, path, payload=None):
        captured.update({"method": method, "path": path, "payload": payload})
        return {"success": True, "call": {"call_id": "vc_test", "to": payload["to"]}}

    monkeypatch.setattr(tools, "_request", fake_request)
    result = tools.voice_call({
        "action": "call",
        "to": "310-555-1212",
        "purpose": "confirm booking",
        "context": "ask whether the booking is still active",
        "escalation_policy": "take_message",
    })
    assert captured["method"] == "POST"
    assert captured["path"] == "/twilio/voice/outbound"
    assert captured["payload"]["to"] == "+13105551212"
    assert "3105551212" not in result
    assert "*1212" in result


def test_service_outbound_dry_run_and_transcript(monkeypatch):
    service = importlib.import_module("plugins.voice_call.service")
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_FROM_NUMBER", raising=False)
    client = TestClient(service.app)

    response = client.post("/twilio/voice/outbound", json={
        "to": "+13105551212",
        "purpose": "confirm booking",
        "context": "ask whether the booking is still active",
        "caller_profile": {"disclosure": "This is Hermes calling for Jason."},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dry_run"] is True
    call_id = data["call"]["call_id"]
    assert data["call"]["to"] == "[redacted-phone:*1212]"

    twiml = client.get(f"/twilio/voice/inbound?call_id={call_id}")
    assert twiml.status_code == 200
    assert twiml.headers["content-type"].startswith("application/xml")
    assert "This is Hermes calling for Jason." in twiml.text
    assert "confirm booking" in twiml.text

    transcript = client.get(f"/twilio/voice/{call_id}/transcript")
    assert transcript.status_code == 200
    assert transcript.json()["success"] is True
    assert transcript.json()["transcript"][0]["event"] == "disclosure"


def test_service_status_and_transfer_without_target(monkeypatch):
    service = importlib.import_module("plugins.voice_call.service")
    monkeypatch.delenv("VOICE_CALL_TRANSFER_NUMBER", raising=False)
    client = TestClient(service.app)
    created = client.post("/twilio/voice/outbound", json={
        "to": "+13105551212",
        "purpose": "confirm booking",
        "context": "ask whether the booking is still active",
    }).json()
    call_id = created["call"]["call_id"]
    status = client.post("/twilio/status", data={"call_id": call_id, "CallStatus": "completed"})
    assert status.status_code == 200
    assert status.json()["call"]["status"] == "completed"
    transfer = client.post(f"/twilio/voice/{call_id}/transfer")
    assert transfer.status_code == 400
    assert "transfer target" in transfer.text
