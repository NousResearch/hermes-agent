"""Preview policy is independent of the app, without a new private-content endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    from hermes_cli import web_server as server

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    (tmp_path / "assets").mkdir()
    monkeypatch.setattr(server.app.state, "web_dist", tmp_path, raising=False)
    return TestClient(server.app)


def test_preview_assets_expose_only_a_sandboxed_empty_shell(client):
    from hermes_cli import web_server as server

    response = client.get("/assets/webapp-preview.html", params={"html": "private-content", "profile": "other"})
    assert response.status_code == 200
    assert "sandbox allow-scripts" in response.headers["content-security-policy"]
    assert "allow-same-origin" not in response.headers["content-security-policy"]
    assert "frame-ancestors 'self'" in response.headers["content-security-policy"]
    # Widgets keep main's reach: plain-http images and APIs, workers and embeds.
    policy = dict(part.strip().split(" ", 1) for part in response.headers["content-security-policy"].split(";"))
    for directive in ("script-src", "img-src", "connect-src", "worker-src", "frame-src"):
        assert "*" in policy[directive].split(), directive
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "private-content" not in response.text
    assert server._SESSION_TOKEN not in response.text
    loader = client.get("/assets/webapp-preview.js")
    assert loader.status_code == 200
    assert loader.headers["content-type"].startswith("text/javascript")
    assert server._SESSION_TOKEN not in loader.text
    assert client.get("/api/fs/read-text", params={"path": "/secret", "profile": "other"}).status_code == 401


@pytest.mark.parametrize("surface", ["gateway", "serve", "dashboard"])
def test_preview_assets_are_webapp_only(client, monkeypatch, surface):
    monkeypatch.setattr(client.app.state, "ui_surface", surface)
    for path in ("/assets/webapp-preview.html", "/assets/webapp-guest.html", "/assets/webapp-preview.js"):
        assert client.get(path).status_code == 404


def _policy(response):
    return dict(part.strip().split(" ", 1) for part in response.headers["content-security-policy"].split(";"))


def test_plugin_guest_shell_caps_at_the_sdk_sandbox_allowlist(client):
    """SandboxedFrame data: HTML runs here, never with same-origin or navigation tokens."""
    from hermes_cli import web_server as server

    response = client.get("/assets/webapp-guest.html", params={"html": "private-content"})
    assert response.status_code == 200
    policy = _policy(response)
    assert set(policy["sandbox"].split()) == {
        "allow-downloads", "allow-forms", "allow-orientation-lock",
        "allow-pointer-lock", "allow-presentation", "allow-scripts",
    }
    assert "'unsafe-inline'" in policy["script-src"].split()
    assert policy["frame-ancestors"] == "'self'"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "private-content" not in response.text
    assert server._SESSION_TOKEN not in response.text
    # File previews keep their own, narrower ceiling.
    assert _policy(client.get("/assets/webapp-preview.html"))["sandbox"] == "allow-scripts"
