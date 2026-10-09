"""Real HTTP file tickets must never cross named-profile generations."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import secrets
import shutil
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from starlette.responses import FileResponse

from hermes_cli import profile_incarnation, profile_lifecycle, web_server
from hermes_cli.web_routers import files


@pytest.fixture
def file_server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(home))
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", secrets.token_urlsafe(32))
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "serve", raising=False)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(
        web_server.app, lifespan="off", access_log=False, log_level="error",
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    with httpx.Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", trust_env=False, timeout=15) as client:
        try:
            yield client, home
        finally:
            server.should_exit = True
            thread.join(10)
            sock.close()
            assert not thread.is_alive()


def _generation(home, content, *, replace=False):
    """Use the production publication/retirement lease, without service or OS-user operations."""
    target = home / "profiles" / "worker"
    with profile_lifecycle.profile_lifecycle_lease(target):
        if replace:
            old = profile_incarnation.read_profile_incarnation(target)
            profile_lifecycle.begin_profile_retirement(target, old)
            shutil.rmtree(target)

        def initialize(staging):
            staging.mkdir()
            (staging / "SOUL.md").write_text("Synthetic test profile", encoding="utf-8")
            (staging / "clip.mp4").write_bytes(content)
            profile_incarnation.write_fresh_profile_incarnation(staging)

        profile_lifecycle.create_profile_generation("worker", target, target.parent, initialize)
        if replace:
            assert profile_incarnation.read_profile_incarnation(target) != old
    return target


def _ticket(client, route, query):
    minted = client.post("/api/files/ticket", json={"route": route, **query}, headers={
        web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
    })
    assert minted.status_code == 200, minted.text
    assert minted.headers["cache-control"] == "no-store"
    return {**query, "ticket": minted.json()["ticket"]}


def _session_owned_file(home, profile, content):
    """Write the file outside the session owner's home: path-only generation
    binding would miss the owner's retirement."""
    from hermes_state import SessionDB

    (home / "clip.mp4").write_bytes(content)
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("synthetic-session", "desktop", cwd=str(home), profile_name="worker")
    db.close()


def _session_scope(home, profile):
    _session_owned_file(home, profile, b"old-generation")
    return "download", {"path": "clip.mp4", "profile": "worker", "session_id": "synthetic-session"}


# How the ticket names the file: (home, profile) -> (route, query).
_TICKET_SCOPES = {
    "profile": lambda home, profile: ("stream", {"path": str(profile / "clip.mp4"), "profile": "worker"}),
    "implicit": lambda home, profile: ("stream", {"path": str(profile / "clip.mp4")}),
    "managed-relative": lambda home, profile: ("stream", {"path": "profiles/worker/clip.mp4"}),
    "file-uri": lambda home, profile: ("stream", {"path": (profile / "clip.mp4").as_uri()}),
    "session": _session_scope,
}


@pytest.mark.parametrize("scope", list(_TICKET_SCOPES))
def test_http_tickets_bind_profile_and_session_owner_generations(file_server, scope):
    client, home = file_server
    profile = _generation(home, b"old-generation")
    route, query = _TICKET_SCOPES[scope](home, profile)

    params = _ticket(client, route, query)
    unused_download = _ticket(client, "download", query)
    endpoint = f"/api/files/{route}"
    for _ in range(2 if route == "stream" else 1):
        response = client.get(endpoint, params=params, headers={"Range": "bytes=0-2"})
        assert (response.status_code, response.content) == (206, b"old")
        assert response.headers["content-range"] == "bytes 0-2/14"
    if route == "download":
        assert client.get(endpoint, params=params).status_code == 401
        params = _ticket(client, route, query)
    else:
        assert client.head(endpoint, params=params).headers["content-length"] == "14"
        assert client.get(endpoint, params=params, headers={"Range": "bytes=99-100"}).status_code == 416
        multiple = client.get(endpoint, params=params, headers={"Range": "bytes=0-2,4-6"})
        assert multiple.status_code == 206 and b"old" in multiple.content and b"gen" in multiple.content

    _generation(home, b"new-generation", replace=True)
    if scope == "session":
        _session_owned_file(home, profile, b"new-generation")
    stale = client.get(endpoint, params=params, headers={"Range": "bytes=0-2"})
    assert stale.status_code in (401, 403, 404), stale.content
    assert client.get("/api/files/download", params=unused_download).status_code in (401, 403, 404)
    fresh = client.get(endpoint, params=_ticket(client, route, query), headers={"Range": "bytes=0-2"})
    assert (fresh.status_code, fresh.content) == (206, b"new")


@pytest.mark.parametrize("boundary", [
    "before-open",
    pytest.param("after-open", marks=pytest.mark.platforms("posix")),
])
def test_http_ticket_resource_open_is_atomic_with_profile_replacement(file_server, monkeypatch, boundary):
    client, home = file_server
    profile = _generation(home, b"old-generation")
    query = {"path": str(profile / "clip.mp4"), "profile": "worker"}
    params = _ticket(client, "stream", query)
    paused, resume = threading.Event(), threading.Event()

    async def barrier():
        paused.set()
        assert await asyncio.to_thread(resume.wait, 10), "replacement did not release the HTTP request"

    if boundary == "before-open":
        original = files._managed_file_response

        async def before_open(*args, **kwargs):
            await barrier()
            return await original(*args, **kwargs)

        monkeypatch.setattr(files, "_managed_file_response", before_open)
    else:
        original = FileResponse.__call__

        async def before_body(self, scope, receive, send):
            # FileResponse dispatch happens after the guarded response pins its
            # descriptor, but before the unguarded implementation opens a path.
            await barrier()
            return await original(self, scope, receive, send)

        monkeypatch.setattr(FileResponse, "__call__", before_body)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.get, "/api/files/stream", params=params, headers={"Range": "bytes=0-2"})
        try:
            assert paused.wait(10)
            # Completing this before releasing the stream also proves the
            # profile lease is not held across response delivery.
            _generation(home, b"new-generation", replace=True)
        finally:
            resume.set()
        response = pending.result(15)
    if boundary == "before-open":
        assert response.status_code in (401, 403, 404), response.content
    else:
        assert (response.status_code, response.content) == (206, b"old")
    assert client.get("/api/files/stream", params=params).status_code in (401, 403, 404)
