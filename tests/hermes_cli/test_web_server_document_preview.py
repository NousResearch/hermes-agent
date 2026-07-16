import base64

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_pdf_preview_open_range_and_close(client, tmp_path):
    target = tmp_path / "paper.pdf"
    data = b"%PDF-1.7\n" + bytes(range(128))
    target.write_bytes(data)

    opened = client.post("/api/preview/pdf/open", json={"path": str(target)})
    assert opened.status_code == 200
    descriptor = opened.json()
    assert base64.b64decode(descriptor["initialData"]) == data

    ranged = client.post(
        "/api/preview/pdf/range",
        json={"id": descriptor["id"], "begin": 10, "end": 20, "revision": descriptor["revision"]},
    )
    assert ranged.status_code == 200
    assert base64.b64decode(ranged.json()["data"]) == data[10:20]

    closed = client.post("/api/preview/pdf/close", json={"id": descriptor["id"]})
    assert closed.json() == {"closed": True}


def test_document_preview_routes_require_auth(tmp_path):
    client = TestClient(web_server.app)
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.7\n")

    response = client.post("/api/preview/pdf/open", json={"path": str(target)})

    assert response.status_code == 401
