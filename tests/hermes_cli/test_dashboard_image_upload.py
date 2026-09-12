"""Real storage/auth contracts reused by the dashboard image picker.

No upload route mocks: browser bytes must become a profile-local readable image.
"""
import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    profile = home / "profiles" / "image-check"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    image = BytesIO()
    Image.new("RGB", (16, 16), "orange").save(image, format="PNG")
    data = image.getvalue()
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    payload = {
        "data_url": "data:image/png;base64," + base64.b64encode(data).decode(),
        "filename": "../../shot\n.png",
    }
    yield client, payload, data, home, profile, _SESSION_HEADER_NAME
    client.close()


def test_upload_preserves_bytes_and_isolates_selected_profile(uploads):
    client, payload, data, home, profile, _ = uploads
    paths = []
    for query, expected_home in (("", home), ("?profile=image-check", profile), ("?profile=image-check", profile)):
        response = client.post("/api/chat/image-upload" + query, json=payload)
        assert response.status_code == 200, response.text
        receipt = response.json()
        path = Path(receipt["path"])
        assert path.parent == expected_home / "images"
        assert path.read_bytes() == data
        assert receipt["bytes"] == len(data)
        assert "\n" not in path.name
        paths.append(path)
    assert len(set(paths)) == len(paths), "same-name uploads must not overwrite earlier images"
    assert len(list((home / "images").iterdir())) == 1
    assert len(list((profile / "images").iterdir())) == 2


def test_unauthorized_and_non_image_uploads_do_not_write_files(uploads):
    client, payload, _, home, profile, header = uploads
    unauthorized = client.post("/api/chat/image-upload", json=payload, headers={header: "invalid-test-token"})
    assert unauthorized.status_code == 401
    for data_url in ("data:image/png;base64,", "data:image/png;base64,bm90LWFuLWltYWdl", "data:text/plain;base64,dGV4dA=="):
        response = client.post("/api/chat/image-upload", json={**payload, "data_url": data_url})
        assert response.status_code == 400, response.text
    assert not (home / "images").exists()
    assert not (profile / "images").exists()
