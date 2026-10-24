"""Dashboard gzip negotiation must preserve finite reads and bypass streams."""

import gzip

import pytest
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient


PAYLOAD = {"messages": [{"content": "transcript ✓ " * 600}]}
PLAIN = JSONResponse(PAYLOAD).body


@pytest.fixture
def dashboard(monkeypatch, tmp_path, _isolate_hermes_home):
    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    download = tmp_path / "export.json"
    download.write_bytes(PLAIN)

    async def endpoint(request):
        kind = request.path_params["kind"]
        if kind == "small":
            return JSONResponse({"ok": True})
        if kind == "stream":
            return StreamingResponse(iter([PLAIN[:100], PLAIN[100:]]), media_type="application/json")
        if kind == "sse":
            return StreamingResponse(iter([b"data: " + PLAIN + b"\n\n"]), media_type="text/event-stream")
        if kind == "file":
            return FileResponse(download, media_type="application/json")
        if kind == "partial":
            return Response(PLAIN, status_code=206, media_type="application/json")
        if kind == "content-range":
            return Response(PLAIN, headers={"Content-Range": "bytes 0-99/100"}, media_type="application/json")
        if kind == "encoded":
            return Response(gzip.compress(PLAIN), headers={"Content-Encoding": "gzip"}, media_type="application/json")
        if kind == "media":
            return Response(PLAIN, media_type="image/png")
        if kind == "text":
            return Response(PLAIN, media_type="text/plain")
        return JSONResponse(PAYLOAD)

    route = Route("/api/_gzip/{kind}", endpoint)
    monkeypatch.setattr(app.router, "routes", [route, *app.router.routes])
    raw = bytearray()

    async def capture(scope, receive, send):
        raw.clear()

        async def capture_send(message):
            if message["type"] == "http.response.body":
                raw.extend(message.get("body", b""))
            await send(message)

        await app(scope, receive, capture_send)

    client = TestClient(capture)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    client.headers.pop("accept-encoding", None)
    yield client, raw
    client.close()


def test_large_json_negotiates_gzip_and_identity(dashboard):
    client, raw = dashboard
    identity = client.get("/api/_gzip/large")
    assert identity.status_code == 200
    assert "content-encoding" not in identity.headers
    assert bytes(raw) == PLAIN == identity.content
    # Both variants say they vary, so a cache never hands one to a client asking for the other.
    assert "Accept-Encoding" in identity.headers["vary"]

    compressed = client.get("/api/_gzip/large", headers={"Accept-Encoding": "gzip"})
    assert compressed.headers["content-encoding"] == "gzip"
    assert "Accept-Encoding" in compressed.headers["vary"]
    assert gzip.decompress(bytes(raw)) == PLAIN
    assert compressed.json() == identity.json() == PAYLOAD


@pytest.mark.parametrize("kind", ["small", "stream", "sse", "file", "partial", "content-range", "media"])
def test_excluded_responses_stay_unencoded(dashboard, kind):
    client, raw = dashboard
    identity = client.get(f"/api/_gzip/{kind}")
    expected = bytes(raw)
    response = client.get(f"/api/_gzip/{kind}", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == identity.status_code
    assert "content-encoding" not in response.headers
    assert "vary" not in identity.headers and "vary" not in response.headers
    assert bytes(raw) == expected


def test_range_request_and_already_encoded_response_are_preserved(dashboard):
    client, raw = dashboard
    response = client.get("/api/_gzip/large", headers={"Accept-Encoding": "gzip", "Range": "bytes=0-99"})
    assert "content-encoding" not in response.headers
    assert bytes(raw) == PLAIN
    response = client.get("/api/_gzip/file", headers={"Accept-Encoding": "gzip", "Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-99/{len(PLAIN)}"
    assert "content-encoding" not in response.headers
    assert bytes(raw) == PLAIN[:100]
    response = client.get("/api/_gzip/encoded", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"
    assert gzip.decompress(bytes(raw)) == PLAIN
    assert response.json() == PAYLOAD


def test_large_text_is_compressed(dashboard):
    client, raw = dashboard
    response = client.get("/api/_gzip/text", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"
    assert gzip.decompress(bytes(raw)) == PLAIN


def test_cors_preflight_preserves_existing_status_and_headers(dashboard):
    client, _raw = dashboard
    response = client.options("/api/_gzip/large", headers={
        "Origin": "http://localhost:5174",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "x-hermes-session-token",
        "Accept-Encoding": "gzip",
    })
    # Starlette's existing preflight is 200 OK (the server comment says 204).
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"
    assert "GET" in response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-headers"] == "x-hermes-session-token"
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize(("accept_encoding", "compressed"), [
    ("gzip", True),
    ("GZip", True),
    ("br, gzip;q=0.5", True),
    ("gzip;q=0", False),
    ("deflate, gzip; q=0.000", False),
    ("gzip;q=junk", False),
    ("x-gzip", False),
    ("br, deflate", False),
])
def test_gzip_follows_accept_encoding_q_values(dashboard, accept_encoding, compressed):
    client, raw = dashboard
    response = client.get("/api/_gzip/large", headers={"Accept-Encoding": accept_encoding})
    assert ("content-encoding" in response.headers) is compressed
    assert "Accept-Encoding" in response.headers["vary"]
    assert (gzip.decompress(bytes(raw)) if compressed else bytes(raw)) == PLAIN
