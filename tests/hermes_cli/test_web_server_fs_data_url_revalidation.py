"""A transcript image is read through /api/fs/read-data-url on every remount. The response carries
a validator so the browser keeps the body in its HTTP cache and revalidates: an unchanged file
answers 304 with no body, a rewritten one its new bytes."""

import base64
import os

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        test_client.close()
        if previous is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


def _data_url(body: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(body).decode("ascii")


def test_an_unchanged_image_revalidates_to_304_and_a_rewritten_one_is_read_again(client, tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"first")
    route, query = "/api/fs/read-data-url", {"path": str(image)}

    first = client.get(route, params=query)
    assert first.status_code == 200
    assert first.json() == {"dataUrl": _data_url(b"first")}
    assert first.headers["cache-control"] == "private, no-cache"
    etag = first.headers["etag"]

    unchanged = client.get(route, params=query, headers={"If-None-Match": etag})
    assert (unchanged.status_code, unchanged.content) == (304, b"")

    assert client.get(route, params=query, headers={"If-None-Match": "*"}).status_code == 304

    # Same length, later mtime: an edit in place that keeps the size is still a new image.
    image.write_bytes(b"FIRST")
    stat = image.stat()
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    rewritten = client.get(route, params=query, headers={"If-None-Match": etag})
    assert rewritten.status_code == 200
    assert rewritten.json() == {"dataUrl": _data_url(b"FIRST")}
    assert rewritten.headers["etag"] != etag
