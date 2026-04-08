import pytest
from fastapi.testclient import TestClient
from dashboard.app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Hermes Agent Dashboard" in response.text

def test_get_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "theme" in data
    assert "language" in data

def test_update_config():
    response = client.post("/api/config", json={"theme": "light", "language": "fr"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # Verify it updated
    response = client.get("/api/config")
    data = response.json()
    assert data["theme"] == "light"
    assert data["language"] == "fr"

def test_list_sessions():
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
