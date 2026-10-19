"""A child tab may receive local authority only through a one-use handoff."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def local_webapp(monkeypatch):
    from hermes_cli import web_server as server

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    monkeypatch.setattr("hermes_cli.dashboard_auth.prefix.resolve_public_url", lambda: "")
    return server, TestClient(server.app, base_url="http://127.0.0.1")


@pytest.mark.parametrize("public_url", ["", "http://localhost", "http://127.0.0.1:8080"])
def test_child_handoff_is_private_single_use_and_authorizes_real_api(local_webapp, tmp_path, monkeypatch, public_url):
    monkeypatch.setattr("hermes_cli.dashboard_auth.prefix.resolve_public_url", lambda: public_url)
    server, client = local_webapp
    token = server._SESSION_TOKEN
    headers = {"X-Hermes-Session-Token": token}
    assert client.post("/api/webapp/window-ticket").status_code == 401
    page = client.get("/webapp/window")
    assert page.status_code == 200
    assert token not in page.text and token not in str(page.headers)
    assert page.headers["cache-control"] == "no-store"
    issued = client.post("/api/webapp/window-ticket", headers=headers)
    assert issued.status_code == 200
    ticket = issued.json()["ticket"]
    assert ticket != token
    assert issued.headers["cache-control"] == "no-store"
    path = tmp_path / "handoff.txt"
    path.write_text("child-window-proof", encoding="utf-8")
    assert client.get("/api/fs/read-text", params={"path": str(path)},
                      headers={"X-Hermes-Session-Token": ticket}).status_code == 401
    # An opaque/sibling origin cannot consume even a known ticket.
    for origin in ("", "null", "http://127.0.0.1:9999", "https://127.0.0.1", "http://evil.example"):
        denied = client.post("/webapp/window-session",
                             headers={"Origin": origin, "X-Hermes-Window-Ticket": ticket})
        assert denied.status_code == 403
        assert token not in denied.text
    rebinding = client.post("/webapp/window-session", headers={
        "Host": "evil.example", "Origin": "http://evil.example", "X-Hermes-Window-Ticket": ticket,
    })
    assert rebinding.status_code == 400
    assert client.post("/webapp/window-session", headers=[
        ("Origin", "http://127.0.0.1"), ("Origin", "http://127.0.0.1"),
        ("X-Hermes-Window-Ticket", ticket),
    ]).status_code == 403
    def exchange():
        return client.post("/webapp/window-session",
                           headers={"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": ticket})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: exchange(), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 403]
    granted = next(r for r in responses if r.status_code == 200)
    assert granted.headers["cache-control"] == "no-store"
    assert granted.json()["token"] == token
    read = client.get("/api/fs/read-text", params={"path": str(path)},
                      headers={"X-Hermes-Session-Token": granted.json()["token"]})
    assert read.status_code == 200 and "child-window-proof" in read.text
    # Replay cannot recover the token, nor can it be obtained through a GET.
    assert exchange().status_code == 403
    assert token not in client.get("/webapp/window-session", params={"ticket": ticket}).text


@pytest.mark.parametrize("invalidate", ["expiry", "restart", "gated", "dashboard"])
def test_child_handoff_is_bound_to_local_webapp_start(local_webapp, monkeypatch, invalidate):
    from hermes_cli.web_routers import webapp

    server, client = local_webapp
    issued = client.post("/api/webapp/window-ticket", headers={"X-Hermes-Session-Token": server._SESSION_TOKEN})
    assert issued.status_code == 200
    ticket = issued.json()["ticket"]
    if invalidate == "expiry":
        now = webapp.time.monotonic()
        monkeypatch.setattr(webapp.time, "monotonic", lambda: now + webapp.WINDOW_TICKET_TTL_SECONDS + 1)
    elif invalidate == "restart":
        monkeypatch.setattr(server, "_SESSION_TOKEN", "b" * 43)
    elif invalidate == "gated":
        server.app.state.auth_required = True
    else:
        server.app.state.ui_surface = "dashboard"
    denied = client.post("/webapp/window-session",
                         headers={"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": ticket}, follow_redirects=False)
    assert denied.status_code in (302, 401, 403, 404)
    assert server._SESSION_TOKEN not in denied.text
