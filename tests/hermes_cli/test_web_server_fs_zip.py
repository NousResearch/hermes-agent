"""``GET /api/fs/download?format=zip``: a workspace folder saved as one verifiable archive."""

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest

from hermes_cli import web_server
from hermes_cli import web_server_file_tickets as file_tickets
from hermes_cli.web_routers import files_zip

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield test_client


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "managed"
    project = root / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_bytes(b"print('hi')\n")
    (project / "README.md").write_bytes(b"# demo\n")
    (project / ".env").write_bytes(b"SECRET=1\n")
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_bytes(b"ref: refs/heads/main\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_bytes(b"not yours\n")
    if os.name != "nt":
        (project / "escape").symlink_to(outside, target_is_directory=True)
        (project / "src" / "leak.txt").symlink_to(outside / "private.txt")
    return root, project


def test_folder_zip_holds_the_tree_its_header_hash_names_and_stays_in_the_files_root(
    client, tmp_path, monkeypatch,
):
    root, project = _project(tmp_path)
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    response = client.get("/api/fs/download", params={"path": str(project), "format": "zip"})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert "project.zip" in response.headers["content-disposition"]
    assert response.headers["x-content-sha256"] == hashlib.sha256(response.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == ["project/README.md", "project/src/main.py"]
        assert archive.read("project/src/main.py") == b"print('hi')\n"
    assert response.headers["x-hermes-zip-files"] == "2"

    refused = client.get("/api/fs/download", params={"path": str(tmp_path / "outside"), "format": "zip"})
    assert refused.status_code == 403

    monkeypatch.setattr(files_zip, "MAX_BYTES", len(b"# demo\n"))  # README fits, main.py does not
    too_big = client.get("/api/fs/download", params={"path": str(project), "format": "zip"})
    assert too_big.status_code == 413


def test_folder_zip_rides_a_download_ticket(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    _root, project = _project(tmp_path)
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    query = {"path": str(project), "format": "zip"}

    minted = client.post("/api/files/ticket", json={"route": "download", **query})
    assert minted.status_code == 200, minted.text

    anchor = TestClient(web_server.app)  # a download link sends no session header
    try:
        saved = anchor.get(file_tickets.ROUTES["download"], params={**query, "ticket": minted.json()["ticket"]})
        assert saved.status_code == 200, saved.text
        assert saved.headers["x-content-sha256"] == hashlib.sha256(saved.content).hexdigest()
        with zipfile.ZipFile(io.BytesIO(saved.content)) as archive:
            assert "project/src/main.py" in archive.namelist()
        # The ticket is bound to the zip query: the same ticket cannot fetch the folder as a file.
        assert anchor.get(file_tickets.ROUTES["download"],
                          params={"path": str(project), "ticket": minted.json()["ticket"]}).status_code == 401
    finally:
        anchor.close()
