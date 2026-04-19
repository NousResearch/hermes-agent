from starlette.testclient import TestClient

import gateway.meta_router_server as server
from gateway.meta_router_runtime import RouteDecision


def test_classify_returns_request_metadata(monkeypatch):
    monkeypatch.setattr(
        server,
        "make_route_decision",
        lambda text, source="api", surface="http", session_id=None: RouteDecision(
            request_id="rid-123",
            type="production",
            mode="review",
            directive="[META-ROUTER | production | review]",
            confidence=0.91,
            primary="som",
            secondary="eop-adv-pass",
            budget_multiplier=1.5,
            routing_artifact_version="candidate-0002",
            bypassed=False,
            bypass_reason="",
        ),
    )
    client = TestClient(server.app)

    resp = client.post(
        "/classify",
        json={
            "text": "review the production rollout checklist",
            "source": "openclaw-plugin",
            "surface": "openclaw",
            "session_id": "sess-789",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "rid-123"
    assert data["primary"] == "som"
    assert data["secondary"] == "eop-adv-pass"
    assert data["routing_artifact_version"] == "candidate-0002"
    assert data["bypassed"] is False
    assert data["text_with_directive"].startswith("[META-ROUTER | production | review]")


def test_classify_returns_original_text_when_bypassed(monkeypatch):
    monkeypatch.setattr(
        server,
        "make_route_decision",
        lambda text, source="api", surface="http", session_id=None: RouteDecision(
            request_id="rid-bypass",
            type="code",
            mode="execute",
            directive="",
            confidence=0.0,
            primary="som",
            secondary=None,
            budget_multiplier=1.0,
            routing_artifact_version="candidate-0002",
            bypassed=True,
            bypass_reason="short-ack",
        ),
    )
    client = TestClient(server.app)

    resp = client.post("/classify", json={"text": "ok"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["bypassed"] is True
    assert data["bypass_reason"] == "short-ack"
    assert data["directive"] == ""
    assert data["text_with_directive"] == "ok"

def test_outcome_endpoint_logs_terminal_outcome(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        server,
        "run_outcome_only",
        lambda **kwargs: captured.update(kwargs),
    )
    client = TestClient(server.app)

    resp = client.post(
        "/outcome",
        json={
            "request_id": "rid-openclaw-fail",
            "task_type": "config",
            "session_id": "sess-openclaw",
            "source": "openclaw-plugin",
            "surface": "openclaw",
            "routing_artifact_version": "candidate-0010",
            "success": False,
            "error": "pairing required",
            "duration_ms": 3250,
            "notes": ["phase=plugin-agent-end"],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert captured["request_id"] == "rid-openclaw-fail"
    assert captured["task_type"] == "config"
    assert captured["session_id"] == "sess-openclaw"
    assert captured["source"] == "openclaw-plugin"
    assert captured["surface"] == "openclaw"
    assert captured["routing_artifact_version"] == "candidate-0010"
    assert captured["error"] == "pairing required"
    assert captured["notes_extra"] == ["phase=plugin-agent-end", "success=false"]

