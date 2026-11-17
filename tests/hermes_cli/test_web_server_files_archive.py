"""Whole-directory transfer through the managed files API (``/api/files/archive``)."""

import io
import tarfile

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def archive_client(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    prev = {name: getattr(web_server.app.state, name, None) for name in ("auth_required", "bound_host")}
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client, root
    finally:
        client.close()
        for name, value in prev.items():
            setattr(web_server.app.state, name, value)


def _tar(*members: tuple[str, str, str]) -> bytes:
    """(name, kind, payload) -> an uncompressed tar; payload is the body or the link target."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = payload.encode()
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                info.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE}[kind]
                info.linkname = payload
                tf.addfile(info)
    return buf.getvalue()


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_directory_round_trips_through_download_and_upload(archive_client):
    """A downloaded workspace folder unpacks into another folder with the same files and links;
    existing files the archive names are replaced and others are left alone."""
    client, root = archive_client
    src = root / "project"
    (src / "src" / "pkg").mkdir(parents=True)
    (src / "src" / "pkg" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (src / "README.md").write_text("# project\n", encoding="utf-8")
    (src / "readme-link").symlink_to("README.md")
    (src / ".env").write_text("SECRET=1\n", encoding="utf-8")  # credential file: never leaves the host

    download = client.get("/api/files/archive", params={"path": "project"})
    assert download.status_code == 200, download.text
    assert download.headers["content-type"] == "application/gzip"

    dest = root / "copy"
    dest.mkdir()
    (dest / "README.md").write_text("old\n", encoding="utf-8")
    (dest / "keep.txt").write_text("untouched\n", encoding="utf-8")
    upload = client.put("/api/files/archive", params={"path": "copy"}, content=download.content)
    assert upload.status_code == 200, upload.text
    assert upload.json()["files"] == 2 and upload.json()["symlinks"] == 1

    assert (dest / "src" / "pkg" / "main.py").read_text(encoding="utf-8-sig") == "print('hi')\n"
    assert (dest / "README.md").read_text(encoding="utf-8-sig") == "# project\n"
    assert (dest / "keep.txt").read_text(encoding="utf-8-sig") == "untouched\n"
    assert (dest / "readme-link").is_symlink() and (dest / "readme-link").read_text(encoding="utf-8-sig") == "# project\n"
    assert not (dest / ".env").exists()


@pytest.mark.parametrize("hostile", [
    [("../escape.txt", "file", "pwned")],
    [("/abs/escape.txt", "file", "pwned")],
    [("out", "symlink", "../../outside")],
    [("d", "symlink", "."), ("p", "symlink", "d/..")],  # reads as "." but resolves to the parent
    [("hard", "hardlink", "inside.txt")],
], ids=["dotdot", "absolute", "symlink-escape", "symlink-chain-escape", "hardlink"])
def test_hostile_archive_is_refused_before_anything_is_written(archive_client, hostile):
    """One bad member refuses the whole archive with 400, and none of the good members before it
    land on disk."""
    client, root = archive_client
    (root / "inbox").mkdir(parents=True)
    body = _tar(("inside.txt", "file", "ok"), ("nested/also.txt", "file", "ok"), *hostile)

    response = client.put("/api/files/archive", params={"path": "inbox"}, content=body)

    assert response.status_code == 400, response.text
    assert "Archive rejected" in response.json()["detail"]
    assert _tree(root) == ["inbox"]
    assert not (root / "escape.txt").exists() and not (root.parent / "outside").exists()
