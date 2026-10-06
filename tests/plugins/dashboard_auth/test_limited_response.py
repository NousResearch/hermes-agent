"""Regression: decoded IdP bodies must not be inflated a second time.

``httpx.stream(...).iter_bytes()`` already applies ``content-encoding``.
Rebuilding ``httpx.Response`` with those headers and the decoded bytes makes
``Response.read()`` inflate again, which is ``incorrect header check`` on a
gzip token response (issue #134128).
"""

from __future__ import annotations

import plugins.dashboard_auth._shared as shared


class _DecodedStream:
    """Stand-in for a consumed ``httpx.stream`` response.

    ``iter_bytes()`` yields the already-decoded body, while the headers still
    advertise the on-wire encoding. That is what ``httpx`` does.
    """

    def __init__(self, method: str, url: str, *, body: bytes, headers: dict[str, str]) -> None:
        import httpx

        self.status_code = 200
        self.headers = headers
        self.request = httpx.Request(method, url)
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_bytes(self, chunk_size: int = 65536):
        _ = chunk_size
        yield self._body


def test_gzip_content_encoding_is_not_decoded_twice(monkeypatch):
    payload = b'{"access_token":"tok","token_type":"bearer"}'

    def fake_stream(method, url, **kwargs):
        _ = kwargs
        return _DecodedStream(
            method,
            url,
            body=payload,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                # Compressed length, not the decoded length. Must not survive.
                "content-length": "11",
            },
        )

    monkeypatch.setattr(shared.httpx, "stream", fake_stream)
    response = shared._request_limited_response("POST", "https://idp.example/token")

    assert response.status_code == 200
    assert response.content == payload
    assert response.json() == {"access_token": "tok", "token_type": "bearer"}
    assert response.headers.get("content-encoding") is None
    assert response.headers.get("content-length") == str(len(payload))


def test_uncompressed_body_still_round_trips(monkeypatch):
    payload = b'{"error":"invalid_request"}'

    def fake_stream(method, url, **kwargs):
        _ = kwargs
        return _DecodedStream(
            method,
            url,
            body=payload,
            headers={"content-type": "application/json", "content-length": str(len(payload))},
        )

    monkeypatch.setattr(shared.httpx, "stream", fake_stream)
    response = shared._request_limited_response("POST", "https://idp.example/token")

    assert response.json()["error"] == "invalid_request"
    assert response.headers.get("content-length") == str(len(payload))
