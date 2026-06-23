import pytest


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_dashboard_route_advisory_endpoint(monkeypatch, _isolate_hermes_home):
    def fake_classify(prompt, *, surface, log):
        assert prompt == "Prepare BMI ASCAP radio promotion plan"
        assert surface == "dashboard:composer"
        assert log is False
        return {
            "route_id": "business-growth",
            "profile": "business-growth",
            "owner": "business-growth",
            "confidence": 4.0,
            "advisory_mode": True,
            "auto_execute": False,
        }

    monkeypatch.setattr("hermes_cli.web_server.classify_route_advisory", fake_classify)

    client = _client()
    response = client.post(
        "/api/route",
        json={
            "prompt": "Prepare BMI ASCAP radio promotion plan",
            "surface": "dashboard:composer",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_id"] == "business-growth"
    assert body["profile"] == "business-growth"
    assert body["auto_execute"] is False


def test_dashboard_route_advisory_honors_explicit_log_true(monkeypatch, _isolate_hermes_home):
    def fake_classify(prompt, *, surface, log):
        assert prompt == "Prepare radio route audit"
        assert surface == "dashboard:composer"
        assert log is True
        return {
            "route_id": "business-growth",
            "profile": "business-growth",
            "advisory_mode": True,
            "auto_execute": False,
        }

    monkeypatch.setattr("hermes_cli.web_server.classify_route_advisory", fake_classify)

    client = _client()
    response = client.post(
        "/api/route",
        json={
            "prompt": "Prepare radio route audit",
            "surface": "dashboard:composer",
            "log": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_id"] == "business-growth"
    assert body["auto_execute"] is False


def test_dashboard_route_advisory_requires_prompt(_isolate_hermes_home):
    client = _client()
    response = client.post("/api/route", json={"prompt": "   "})

    assert response.status_code == 400
