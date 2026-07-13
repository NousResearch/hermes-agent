from fastapi.testclient import TestClient


def test_dashboard_plugins_rescan_accepts_get_and_post(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_get_dashboard_plugins", lambda force_rescan=False: [{"name": "example"}])

    client = TestClient(web_server.app)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}

    get_response = client.get("/api/dashboard/plugins/rescan", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json() == {"ok": True, "count": 1}

    post_response = client.post("/api/dashboard/plugins/rescan", headers=headers)
    assert post_response.status_code == 200
    assert post_response.json() == {"ok": True, "count": 1}
