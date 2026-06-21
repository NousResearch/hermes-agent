import base64

import pytest


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as web_server

    root = tmp_path / "files"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    test_client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client, root


def test_preview_markdown_returns_text(client):
    test_client, root = client
    doc = root / "note.md"
    doc.write_text("# Hello\n\nArtifact viewer", encoding="utf-8")

    resp = test_client.get("/api/artifacts/preview", params={"path": str(doc)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "note.md"
    assert data["preview_type"] == "markdown"
    assert data["text"] == "# Hello\n\nArtifact viewer"
    assert data["data_url"] is None if "data_url" in data else True


def test_preview_image_returns_data_url(client):
    test_client, root = client
    img = root / "pixel.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    resp = test_client.get("/api/artifacts/preview", params={"path": str(img)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["preview_type"] == "image"
    assert data["data_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(data["data_url"].split(",", 1)[1]).startswith(b"\x89PNG")


def test_preview_pdf_returns_data_url(client):
    test_client, root = client
    pdf = root / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%test\n")

    resp = test_client.get("/api/artifacts/preview", params={"path": str(pdf)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["preview_type"] == "pdf"
    assert data["data_url"].startswith("data:application/pdf;base64,")


def test_preview_unsupported_returns_metadata_only(client):
    test_client, root = client
    blob = root / "archive.zip"
    blob.write_bytes(b"PK\x03\x04")

    resp = test_client.get("/api/artifacts/preview", params={"path": str(blob)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["preview_type"] == "unsupported"
    assert "text" not in data
    assert "data_url" not in data


def test_preview_rejects_path_outside_locked_root(client, tmp_path):
    test_client, _root = client
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    resp = test_client.get("/api/artifacts/preview", params={"path": str(outside)})

    assert resp.status_code == 403
