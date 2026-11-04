"""Managed dashboard downloads stream through narrow file tickets (#50172)."""

import pytest

from hermes_cli import web_server, web_server_file_tickets
from tests.hermes_cli import test_web_server_files as managed_files

forced_files_client = managed_files.forced_files_client


@pytest.mark.parametrize("surface", ["dashboard", "webapp"])
def test_managed_download_ticket_keeps_path_route_and_retry_bounds(forced_files_client, monkeypatch, surface):
    client, root = forced_files_client
    monkeypatch.setattr(web_server.app.state, "ui_surface", surface, raising=False)
    path = str(managed_files._seed_file(client, root, name="out/보고서 & 1.txt"))
    secret = root / ".env"
    secret.write_text("SYNTHETIC_TEST_SECRET=fixture", encoding="utf-8")
    assert client.post("/api/files/ticket", json={"route": "managed_download", "path": str(secret)}).status_code == 403
    if surface == "dashboard":
        assert client.post("/api/files/ticket", json={"route": "download", "path": path}).status_code == 404

    minted = client.post("/api/files/ticket", json={"route": "managed_download", "path": path})
    assert minted.status_code == 200, minted.text
    assert minted.headers["cache-control"] == "no-store"
    params = {"path": path, "ticket": minted.json()["ticket"]}
    del client.headers[web_server._SESSION_HEADER_NAME]

    assert client.get("/api/files/download", params={**params, "path": path + ".other"}).status_code == 401
    assert client.get("/api/fs/download", params=params).status_code == 401
    for _ in range(web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS):
        response = client.get("/api/files/download", params=params)
        assert response.status_code == 200, response.text
        assert response.content == b"hello"
        assert response.headers["content-disposition"].startswith("attachment;")
    assert client.get("/api/files/download", params=params).status_code == 401


@pytest.mark.parametrize("scope", [{"profile": "coder"}, {"session_id": "foreign-session"}])
def test_managed_download_ticket_uses_only_the_managed_files_scope(forced_files_client, scope):
    client, root = forced_files_client
    path = str(managed_files._seed_file(client, root))
    response = client.post("/api/files/ticket", json={"route": "managed_download", "path": path, **scope})
    assert response.status_code == 400
