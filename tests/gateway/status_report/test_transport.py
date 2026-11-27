"""One report on the wire: credential in the header, a fresh connection, and the reply classified."""

from __future__ import annotations

import http.client
import http.server
import threading
import urllib.error

import pytest

from gateway.status_report.target import ReportTarget
from gateway.status_report.transport import post_report
from tests.gateway.status_report.support import INTERVAL_HEADER, REPORT_URL, RESPONSES, Receiver, Reply, new_credential

BODY = b'{"v":1}'


def test_the_credential_travels_in_the_authorization_header_not_the_url():
    credential, receiver = new_credential(), Receiver()

    outcome = post_report(ReportTarget(f"{REPORT_URL}?zone=eu", credential), BODY, timeout=7.0, opener=receiver)

    (seen,) = receiver.requests
    assert outcome.kind == "stored" and outcome.status == 204
    assert seen.url == f"{REPORT_URL}?zone=eu" and credential not in seen.url
    assert seen.headers["Authorization"] == f"Bearer {credential}"
    assert seen.headers["Content-type"] == "application/json"
    assert seen.headers["User-agent"] == "hermes-gateway-status-report/1"
    assert (seen.method, seen.raw, seen.timeout) == ("POST", BODY, 7.0)


@pytest.mark.parametrize("case", RESPONSES["cases"], ids=lambda case: case["name"])
def test_each_reply_in_the_contract_fixture_is_classified(case):
    receiver = Receiver()
    receiver.reply_with(Reply(case["status"], case["headers"], case.get("body")))

    outcome = post_report(ReportTarget(REPORT_URL, new_credential()), BODY, timeout=7.0, opener=receiver)

    assert (outcome.kind, outcome.status) == (case["outcome"], case["status"])
    expected_wait = case.get("wait_s") if case["then"] == "wait" else None
    if expected_wait is None:
        assert outcome.retry_after_s is None
    else:  # the cap is the sender's, so the transport reports what the receiver asked for
        assert outcome.retry_after_s == float(case["headers"]["Retry-After"])
    # So are the interval's bounds. Only a stored reply carries one: the header on any other is dropped.
    told = float(case["headers"][INTERVAL_HEADER]) if case["then"] == "told_interval" else None
    assert outcome.report_interval_s == told


def test_a_retry_after_too_long_to_read_as_an_integer_never_raises():
    """``int()`` refuses a digit string past the interpreter's limit, and a header may be that long."""
    receiver = Receiver()
    receiver.reply_with(Reply(429, {"Retry-After": "9" * 5000}))

    outcome = post_report(ReportTarget(REPORT_URL, new_credential()), BODY, timeout=7.0, opener=receiver)

    assert (outcome.kind, outcome.status) == ("throttled", 429)
    assert outcome.retry_after_s > RESPONSES["retry_after_max_s"]


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
        TimeoutError("timed out"),
        http.client.IncompleteRead(b""),
        ValueError("Invalid header value"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_network_error_is_a_failed_report_and_never_raises(error):
    receiver = Receiver()
    receiver.reply_with(Reply(error=error))

    outcome = post_report(ReportTarget(REPORT_URL, new_credential()), BODY, timeout=7.0, opener=receiver)

    assert (outcome.kind, outcome.status) == ("failed", None)


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    """Answers every request with ``302`` to ``/moved``; a client that follows it shows up as a GET."""

    protocol_version = "HTTP/1.1"  # keep-alive is on offer: one connection per report is the client's choice

    def setup(self):
        super().setup()
        self.server.connections.append(self.client_address)

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.server.requests.append((self.command, self.path, self.headers.get("Authorization"),
                                     self.rfile.read(length)))
        self.send_response(302)
        self.send_header("Location", "/moved")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_GET = _answer

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_receiver(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    server.connections, server.requests = [], []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_every_report_opens_its_own_connection_and_a_redirect_is_not_followed(loopback_receiver):
    """Through the real opener: a pooled socket would be dead after a machine freeze, and a followed
    redirect would hand the Authorization header to another address."""
    credential = new_credential()
    target = ReportTarget(f"http://127.0.0.1:{loopback_receiver.server_address[1]}/report", credential)

    outcomes = [post_report(target, BODY, timeout=10.0) for _ in range(3)]

    assert [(outcome.kind, outcome.status) for outcome in outcomes] == [("refused", 302)] * 3
    assert loopback_receiver.requests == [("POST", "/report", f"Bearer {credential}", BODY)] * 3
    assert len(set(loopback_receiver.connections)) == 3
