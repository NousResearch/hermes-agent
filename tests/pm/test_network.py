"""Network retry policy keeps permanent errors immediate and attempts bounded."""

import errno
import http.client
import socket
import ssl
import urllib.error
from datetime import datetime, timezone, UTC
from email.message import Message
from email.utils import format_datetime

import pytest

from pm import network
from pm.downloader import DownloadPaused, HashError


def _http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.invalid/tool", status, "failed", headers, None)


@pytest.mark.parametrize("error,retried", [
    *[pytest.param(_http_error(code), True, id=f"http-{code}") for code in (408, 429, 500, 502, 503, 504)],
    *[pytest.param(_http_error(code), False, id=f"http-{code}") for code in (400, 401, 403, 404, 410, 501, 505)],
    pytest.param(urllib.error.URLError(TimeoutError("timeout")), True, id="connect-timeout"),
    pytest.param(TimeoutError("timeout"), True, id="read-timeout"),
    pytest.param(ConnectionResetError("reset"), True, id="connection-reset"),
    pytest.param(http.client.RemoteDisconnected("closed"), True, id="remote-close"),
    pytest.param(http.client.IncompleteRead(b"prefix", 100), True, id="short-read"),
    pytest.param(socket.gaierror(socket.EAI_AGAIN, "temporary"), True, id="temporary-dns"),
    pytest.param(socket.gaierror(socket.EAI_NONAME, "no name"), False, id="invalid-host"),
    pytest.param(ssl.SSLEOFError("eof"), True, id="tls-eof"),
    pytest.param(ssl.SSLCertVerificationError("certificate"), False, id="tls-certificate"),
    pytest.param(urllib.error.URLError(ssl.SSLCertVerificationError("certificate")), False, id="wrapped-certificate"),
    pytest.param(OSError(errno.ENOSPC, "full"), False, id="disk-full"),
    pytest.param(PermissionError("denied"), False, id="filesystem-denied"),
    pytest.param(HashError("bad pin"), False, id="hash"),
    pytest.param(DownloadPaused("pause"), False, id="pause"),
    pytest.param(KeyboardInterrupt(), False, id="interrupt"),
    pytest.param(ValueError("bad archive"), False, id="invalid-data"),
])
def test_only_transient_failures_use_a_finite_retry_budget(error, retried):
    attempts = []
    waits = []

    def fail():
        attempts.append(None)
        raise error

    with pytest.raises(type(error)) as caught:
        network.retry_network(fail, wait=waits.append)
    assert caught.value is error
    expected = network._ATTEMPTS if retried else 1
    assert len(attempts) == expected
    assert len(waits) == expected - 1
    assert all(0 < delay <= network._MAX_DELAY for delay in waits)
    assert waits == sorted(waits)


@pytest.mark.parametrize("retry_after,minimum", [
    (None, 1),
    ("7", 7),
    ("-5", 1),
    ("not a date", 1),
    pytest.param("9" * 400, network._MAX_DELAY, id="oversized-delay"),
    ("future-date", 12),
    ("expired-date", 1),
])
def test_retry_after_is_bounded_and_interruptible(monkeypatch, retry_after, minimum):
    now = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    monkeypatch.setattr(network.time, "time", lambda: now)
    if retry_after in ("future-date", "expired-date"):
        offset = 12 if retry_after == "future-date" else -12
        retry_after = format_datetime(datetime.fromtimestamp(now + offset, UTC), usegmt=True)
    error = _http_error(503, retry_after)
    attempts = []
    waits = []

    def fail():
        attempts.append(None)
        raise error

    def cancel(delay):
        waits.append(delay)
        raise DownloadPaused("cancelled during backoff")

    with pytest.raises(DownloadPaused):
        network.retry_network(fail, wait=cancel)
    assert len(attempts) == len(waits) == 1
    assert minimum <= waits[0] <= network._MAX_DELAY


@pytest.mark.parametrize("status,headers,cdn", [
    pytest.param(403, {"Server": "Vercel", "x-vercel-mitigated": "deny"}, "Vercel", id="vercel-deny"),
    pytest.param(429, {"Server": "Vercel", "x-vercel-mitigated": "challenge"}, "Vercel", id="vercel-challenge-429"),
    pytest.param(403, {"Server": "cloudflare", "cf-mitigated": "challenge", "cf-cache-status": "DYNAMIC"},
                 "Cloudflare", id="cloudflare-challenge"),
    pytest.param(403, {"Server": "cloudflare"}, None, id="cloudflare-origin-not-r2"),
    pytest.param(404, {"Server": "cloudflare"}, None, id="missing-object"),
    pytest.param(403, {"Server": "AmazonS3"}, None, id="origin-403"),
])
def test_download_failure_names_only_a_cdn_firewall_refusal(status, headers, cdn):
    from pm.downloader import DownloadTransportError

    hdrs = Message()
    for key, value in headers.items():
        hdrs[key] = value
    url = "https://mirror.example/x"
    message = str(DownloadTransportError(url, urllib.error.HTTPError(url, status, "failed", hdrs, None)))
    if cdn is None:
        assert "CDN firewall" not in message
    else:
        assert f"mirror.example's CDN firewall ({cdn}) refused this network" in message, message
