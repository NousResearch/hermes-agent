"""The gateway resolves the Webapp's requests for a ``file:`` URL to the file the URL names.

The Webapp asks in two forms (the renderer half is pinned in
``apps/desktop/src/lib/browser-file-urls.test.ts``). Image and preview reads
(``/api/fs/read-data-url``, ``/api/fs/read-text``) carry the native path the shared ``file:`` URL
parser gives: ``C:\\...`` for a drive letter, ``\\\\host\\share\\...`` for a UNC host. Audio and video
carry the ``file:`` URL itself, ticketed through ``POST /api/files/ticket``, and the gateway decodes it
on its own OS. On Windows both forms reach drive-letter and UNC files.
"""
import base64
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server

PROFILE = "default"


@pytest.fixture
def webapp_client(monkeypatch, tmp_path):
    """The loopback Webapp's API: the browser credential in the session header, no files root."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(home))
    for key, value in (("auth_required", False), ("bound_host", None), ("ui_surface", "webapp")):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield client
    client.close()


@pytest.fixture
def files(tmp_path):
    """A clip, an image and a page, named as the Webapp meets them by their own path (a drive path on
    Windows)."""
    renders = tmp_path / "renders"
    renders.mkdir()
    (renders / "clip one.mp4").write_bytes(b"hello")
    (renders / "shot one.png").write_bytes(b"\x89PNG\r\n\x1a\nimage")
    (renders / "report #1.html").write_text("<p>report</p>", encoding="utf-8")
    return renders


@pytest.fixture
def unc_files(files):
    """The same files through a UNC host: the local administrative share, ``\\\\127.0.0.1\\C$``."""
    drive, rest = os.path.splitdrive(str(files))
    renders = Path(f"\\\\127.0.0.1\\{drive[0]}${rest}")
    if not renders.is_dir():
        pytest.skip("the local administrative share is not reachable")
    return renders


def _plays_through_its_stream_ticket(webapp_client, files):
    url = (files / "clip one.mp4").as_uri()
    minted = webapp_client.post("/api/files/ticket", json={"route": "stream", "path": url, "profile": PROFILE})
    assert minted.status_code == 200, (url, minted.text)

    # The <video> element sends only the ticketed URL, never the session header.
    media_element = TestClient(web_server.app)
    stream = media_element.get(
        "/api/files/stream",
        params={"path": url, "profile": PROFILE, "ticket": minted.json()["ticket"]},
        headers={"Range": "bytes=1-3"},
    )
    media_element.close()
    assert (stream.status_code, stream.content) == (206, b"ell"), (url, stream.text)


def _reads_images_and_previews_by_native_path(webapp_client, files):
    # ``str(path)`` is what the shared parser gives for ``path.as_uri()`` (``apps/shared/src/file-url.ts``).
    shot = files / "shot one.png"
    read = webapp_client.get("/api/fs/read-data-url", params={"path": str(shot), "profile": PROFILE})
    assert read.status_code == 200, (shot, read.text)
    assert read.json()["dataUrl"] == "data:image/png;base64," + base64.b64encode(shot.read_bytes()).decode()

    page = files / "report #1.html"
    preview = webapp_client.get("/api/fs/read-text", params={"path": str(page), "profile": PROFILE})
    assert preview.status_code == 200, (page, preview.text)
    assert preview.json()["text"] == "<p>report</p>"


@pytest.mark.platforms("linux", "windows")
def test_the_webapp_plays_a_file_url_through_its_stream_ticket(webapp_client, files):
    _plays_through_its_stream_ticket(webapp_client, files)


@pytest.mark.platforms("linux", "windows")
def test_the_webapp_reads_images_and_previews_by_native_path(webapp_client, files):
    _reads_images_and_previews_by_native_path(webapp_client, files)


@pytest.mark.platforms("windows")
def test_the_webapp_plays_a_unc_file_url_through_its_stream_ticket(webapp_client, unc_files):
    _plays_through_its_stream_ticket(webapp_client, unc_files)


@pytest.mark.platforms("windows")
def test_the_webapp_reads_unc_images_and_previews_by_native_path(webapp_client, unc_files):
    _reads_images_and_previews_by_native_path(webapp_client, unc_files)
