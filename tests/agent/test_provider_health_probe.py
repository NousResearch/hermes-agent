from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from unittest.mock import patch

import httpx
import pytest

from agent.provider_health_probe import ProbeOutcome, probe_provider_endpoint


@contextmanager
def _http_server(*, status: int = 204, headers: dict[str, str] | None = None):
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            requests.append({
                "method": self.command,
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
            })
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("OS/sandbox forbids loopback bind")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_any_http_status_means_endpoint_reachable_without_auth_or_body_read() -> None:
    with _http_server(status=401) as (url, requests):
        result = probe_provider_endpoint(
            base_url=url.replace("http://", "http://user:secret@")
            + "/v1?token=hidden#fragment",
            timeout_seconds=1.0,
        )

    assert result == ProbeOutcome(
        status="reachable",
        http_status=401,
        detail="endpoint returned HTTP 401",
    )
    assert len(requests) == 1
    assert requests[0]["method"] == "HEAD"
    assert requests[0]["path"] == "/v1"
    request_headers = requests[0]["headers"]
    assert isinstance(request_headers, dict)
    assert request_headers["user-agent"] == "hermes-provider-probe"
    assert "authorization" not in request_headers
    assert "cookie" not in request_headers


def test_redirect_is_reachable_and_is_not_followed() -> None:
    with _http_server(
        status=302, headers={"Location": "http://example.invalid/secret"}
    ) as (
        url,
        requests,
    ):
        result = probe_provider_endpoint(
            base_url=url + "/redirect", timeout_seconds=1.0
        )

    assert result.status == "reachable"
    assert result.http_status == 302
    assert len(requests) == 1


def test_connection_failure_means_endpoint_unreachable() -> None:
    with patch("httpx.Client") as client_class:
        client_class.return_value.__enter__.return_value.stream.side_effect = (
            httpx.ConnectError(
                "connection refused",
                request=httpx.Request("HEAD", "http://127.0.0.1:9/v1"),
            )
        )
        result = probe_provider_endpoint(
            base_url="http://127.0.0.1:9/v1",
            timeout_seconds=0.2,
        )

    assert result.status == "unreachable"
    assert result.http_status is None
    assert result.detail == "ConnectError"


def test_invalid_or_unsupported_url_is_not_probed() -> None:
    for url in ("file:///tmp/provider", "not a url", "https:///missing-host"):
        with patch("httpx.Client") as client:
            result = probe_provider_endpoint(base_url=url, timeout_seconds=1.0)

        assert result.status == "unavailable"
        assert result.http_status is None
        client.assert_not_called()


def test_transport_error_detail_is_sanitized_and_low_cardinality() -> None:
    secret_url = "https://alice:password@example.test/v1?api_key=secret#private"
    request = httpx.Request("HEAD", secret_url)

    with patch("httpx.Client") as client_class:
        client_class.return_value.__enter__.return_value.stream.side_effect = (
            httpx.ConnectError(f"could not connect to {secret_url}", request=request)
        )
        result = probe_provider_endpoint(base_url=secret_url, timeout_seconds=0.1)

    assert result == ProbeOutcome(status="unreachable", detail="ConnectError")
    assert "alice" not in result.detail
    assert "password" not in result.detail
    assert "api_key" not in result.detail
    assert "secret" not in result.detail


def test_tls_error_is_not_classified_as_reachable() -> None:
    with patch("httpx.Client") as client_class:
        client_class.return_value.__enter__.return_value.stream.side_effect = (
            httpx.ConnectError(
                "certificate verify failed",
                request=httpx.Request("HEAD", "https://example.test/v1"),
            )
        )
        result = probe_provider_endpoint(
            base_url="https://example.test/v1", timeout_seconds=0.1
        )

    assert result.status == "unreachable"
    assert result.http_status is None


def test_client_is_fresh_auth_free_and_has_independent_bounded_timeouts() -> None:
    with patch("httpx.Client") as client_class:
        response = client_class.return_value.__enter__.return_value.stream.return_value
        response.__enter__.return_value.status_code = 405

        result = probe_provider_endpoint(
            base_url="https://user:password@example.test/v1?key=secret#fragment",
            timeout_seconds=0.75,
        )

    assert result.status == "reachable"
    kwargs = client_class.call_args.kwargs
    timeout = kwargs["timeout"]
    assert timeout.connect == timeout.read == timeout.write == timeout.pool == 0.75
    assert kwargs["follow_redirects"] is False
    assert kwargs["trust_env"] is False
    assert kwargs.get("cookies") in (None, {})
    client_class.return_value.__enter__.return_value.stream.assert_called_once_with(
        "HEAD",
        "https://example.test/v1",
        headers={"User-Agent": "hermes-provider-probe"},
    )


def test_total_wall_clock_timeout_returns_while_owner_eventually_closes_client() -> (
    None
):
    transport_release = threading.Event()
    owner_closed = threading.Event()

    class BlockedClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            owner_closed.set()

        def stream(self, method, url, *, headers):
            assert method == "HEAD"
            assert "secret" not in url
            assert "authorization" not in headers
            transport_release.wait(5)
            raise httpx.ReadTimeout("blocked transport")

    started = time.monotonic()
    with patch("httpx.Client", BlockedClient):
        result = probe_provider_endpoint(
            base_url="https://user:secret@example.test/v1?token=secret",
            timeout_seconds=0.05,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert result == ProbeOutcome(status="unreachable", detail="ProbeTimeout")
    assert not owner_closed.is_set()

    transport_release.set()
    assert owner_closed.wait(1), "probe owner did not ultimately close its client"


def test_permanently_blocked_transports_cannot_create_unbounded_probe_threads() -> None:
    transport_release = threading.Event()
    owners_closed = 0
    owners_closed_lock = threading.Lock()

    class BlockedClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            nonlocal owners_closed
            with owners_closed_lock:
                owners_closed += 1

        def stream(self, method, url, *, headers):
            transport_release.wait(5)
            raise httpx.ReadTimeout("blocked transport")

    outcomes = []
    try:
        with patch("httpx.Client", BlockedClient):
            for _ in range(5):
                outcomes.append(
                    probe_provider_endpoint(
                        base_url="https://example.test/v1",
                        timeout_seconds=0.01,
                    )
                )
    finally:
        transport_release.set()

    assert [outcome.detail for outcome in outcomes[:4]] == ["ProbeTimeout"] * 4
    assert outcomes[4] == ProbeOutcome(status="unavailable", detail="ProbeCapacity")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with owners_closed_lock:
            if owners_closed == 4:
                break
        time.sleep(0.01)
    assert owners_closed == 4
