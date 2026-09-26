"""Gateway enrollment keeps connector failures inside its RuntimeError contract."""

from __future__ import annotations

import http.client
import io

import pytest

from hermes_cli.gateway_enroll import _post_enroll


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def _post(monkeypatch, body: bytes) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(body),
    )


def _enroll() -> dict:
    return {
        "connector_base_url": "https://connector.example",
        "access_token": "token",
        "enrollment_token": "enrollment",
        "gateway_id": "gw-test",
    }


def test_non_json_response_is_a_runtime_error(monkeypatch):
    _post(monkeypatch, b"<html>proxy error</html>")

    with pytest.raises(RuntimeError, match="non-JSON"):
        _post_enroll(**_enroll())


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ConnectionResetError("reset"), id="read-reset"),
        pytest.param(ValueError("unknown url type"), id="bad-url"),
        pytest.param(http.client.RemoteDisconnected("closed"), id="http-framing"),
    ],
)
def test_transport_failures_are_runtime_errors(monkeypatch, failure):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="Connector transport failure"):
        _post_enroll(**_enroll())


def test_http_error_keeps_enrollment_context(monkeypatch):
    import urllib.error

    failure = urllib.error.HTTPError(
        "https://connector.example/relay/enroll",
        503,
        "unavailable",
        {},
        io.BytesIO(b"down"),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        _post_enroll(**_enroll())
