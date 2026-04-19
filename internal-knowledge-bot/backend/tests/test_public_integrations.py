import json
import urllib.request


def _register_and_login(client, email: str):
    password = "TestPass123!"
    reg = client.post(
        "/api/auth/register",
        json={
            "company_name": "Acme Oy",
            "name": "Admin",
            "email": email,
            "password": password,
        },
    )
    assert reg.status_code in (200, 409), reg.text

    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_public_provider_registry_lists_seeded_providers(client):
    headers = _register_and_login(client, "public-providers@example.com")

    r = client.get("/api/integrations/public/providers", headers=headers)
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["success"] is True
    assert isinstance(body["providers"], list)
    assert len(body["providers"]) >= 10

    names = {p["name"] for p in body["providers"]}
    assert "open_meteo" in names
    assert "frankfurter" in names


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return 200

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_public_fetch_normalizes_and_logs_run(client, monkeypatch):
    headers = _register_and_login(client, "public-fetch@example.com")

    def fake_urlopen(request, timeout=5):
        return _FakeHTTPResponse({"results": [{"headline": "hello"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    r = client.post(
        "/api/integrations/public/fetch",
        headers=headers,
        json={
            "provider": "open_meteo",
            "path": "/forecast",
            "query": {"latitude": "60.98", "longitude": "25.66", "current": "temperature_2m"},
            "idempotency_key": "public-fetch-1",
        },
    )
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["success"] is True
    assert body["provider"] == "open_meteo"
    assert body["status_code"] == 200
    assert body["item_count"] == 1
    assert body["items"][0]["headline"] == "hello"
    assert body["run_id"] is not None
    assert body["run_status"] == "completed"

    runs = client.get("/api/analytics/runs", headers=headers)
    assert runs.status_code == 200, runs.text
    assert any(row["endpoint"] == "/api/integrations/public/fetch" for row in runs.json())
