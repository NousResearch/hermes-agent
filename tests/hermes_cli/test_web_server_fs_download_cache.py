"""A download link means "give me the bytes at this path now".

An agent rewrites a delivered file in place and delivers the same path again; the gateway
re-reads and re-checks the file server-side before the second card. Nothing in the reply
told an HTTP cache between a remote client and the gateway that its copy of the first
response was unusable, so the second click could be answered from that copy (#155031).
"""

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import files as file_routes

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def _directives(response):
    return {part.strip().lower() for part in response.headers.get("cache-control", "").split(",")}


def test_local_download_refuses_intermediary_caching(client, tmp_path):
    target = tmp_path / "install.sh"
    target.write_bytes(b"converge build 1c2572cb\n")

    first = client.get("/api/fs/download", params={"path": str(target)})

    assert first.status_code == 200
    assert first.content == b"converge build 1c2572cb\n"
    assert "no-store" in _directives(first), first.headers

    # Rewritten in place at the same path, as the agent's second delivery does it.
    target.write_bytes(b"converge build 8f258731 rebuilt\n")
    second = client.get("/api/fs/download", params={"path": str(target)})

    assert second.content == b"converge build 8f258731 rebuilt\n"
    assert "no-store" in _directives(second), second.headers


def test_local_download_head_refuses_intermediary_caching(client, tmp_path):
    target = tmp_path / "install.sh"
    target.write_bytes(b"converge build 1c2572cb\n")

    head = client.head("/api/fs/download", params={"path": str(target)})

    assert head.status_code == 200
    assert "no-store" in _directives(head), head.headers


def test_ssh_workspace_download_refuses_intermediary_caching(client, monkeypatch):
    class FakeSshFs:
        def inspect_file(self, _path):
            return "/srv/repos/install.sh", 6

        def stream_file(self, _path):
            yield b"123"
            yield b"456"

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: FakeSshFs())
    params = {"path": "/srv/repos/install.sh", "profile": "remote-dev"}

    streamed = client.get("/api/fs/download", params=params)
    head = client.head("/api/fs/download", params=params)

    assert (streamed.status_code, streamed.content) == (200, b"123456")
    assert head.status_code == 200
    for response in (streamed, head):
        assert "no-store" in _directives(response), response.headers


def test_ticketed_browser_download_refuses_intermediary_caching(client, monkeypatch, tmp_path):
    from hermes_cli import web_server_file_tickets

    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    target = root / "install.sh"
    target.write_bytes(b"converge build 1c2572cb\n")

    minted = client.post("/api/files/ticket", json={"route": "download", "path": str(target)})
    assert minted.status_code == 200, minted.text

    response = client.get(
        web_server_file_tickets.ROUTES["download"],
        params={"path": str(target), "ticket": minted.json()["ticket"]},
    )

    assert response.status_code == 200
    assert response.content == b"converge build 1c2572cb\n"
    assert "no-store" in _directives(response), response.headers
