"""End-to-end test: the real sender against a real HTTP server.

Everything else stubs the transport. This exercises the actual code path —
urllib, gzip, headers, socket — against a live server on loopback, so a
transport-level mistake that a fake would hide fails here instead.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_sender import SharedMetricsSender

INSTALL_ID = "12a73e97-4de9-4766-830d-9ca1192c0420"
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class Ingest(BaseHTTPRequestHandler):
    """A stand-in for the ingest service that records what it receives."""

    received: list = []
    script: list = []

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(raw)
        else:
            body = raw
        type(self).received.append(
            {
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body.decode("utf-8")),
                # Keep the RAW request bytes: comparing only the parsed body
                # would not notice a non-deterministic transport encoding.
                "raw": raw,
                "raw_len": len(raw),
                "decoded_len": len(body),
            }
        )
        status, payload, extra = (
            type(self).script.pop(0) if type(self).script else (202, {}, {})
        )
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


@pytest.fixture
def server():
    Ingest.received = []
    Ingest.script = []
    httpd = HTTPServer(("127.0.0.1", 0), Ingest)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def store(tmp_path):
    return SharedMetricsStore(
        database_path=tmp_path / "metrics.sqlite3",
        outbox_directory=tmp_path / "outbox",
    )


def _endpoint(server):
    host, port = server.server_address
    return f"http://{host}:{port}/v1/telemetry"


def _add(store, package_id, day="2026-08-26", metrics=1):
    payload = {
        "schema_version": "hermes.shared_metrics.v2",
        "package_id": package_id,
        "install_id": INSTALL_ID,
        "generated_at": f"{day}T01:00:00Z",
        "period_start": f"{day}T00:00:00Z",
        "period_end": f"{day}T23:59:59Z",
        "resource": {
            "hermes_version": "0.20.5",
            "os_family": "macos",
            "architecture": "arm64",
            "install_method": "git",
        },
        "metrics": [
            {
                "name": f"hermes.metric.{i}",
                "type": "counter",
                "dimensions": {"outcome": "ok"},
                "value": i,
            }
            for i in range(metrics)
        ],
    }
    with store._connection() as connection:
        connection.execute(
            """
            INSERT INTO package_outbox(
                package_id, period_start, period_end, payload_json,
                created_at, exported_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                package_id,
                f"{day}T00:00:00Z",
                f"{day}T23:59:59Z",
                json.dumps(payload),
                f"{day}T01:00:00Z",
                f"{day}T01:00:01Z",
            ),
        )
    return payload


def _sender(store, server):
    return SharedMetricsSender(
        store, _endpoint(server), sleep=lambda _s: None, now=lambda: NOW
    )


class TestRealTransport:
    def test_a_package_is_delivered_and_marked_sent(self, store, server):
        _add(store, "pkg-1")
        outcome = _sender(store, server).send_pending()

        assert outcome.sent == 1
        assert len(Ingest.received) == 1
        assert Ingest.received[0]["body"]["package_id"] == "pkg-1"

        with store._connection() as connection:
            state = connection.execute(
                "SELECT send_state FROM package_outbox WHERE package_id = 'pkg-1'"
            ).fetchone()[0]
        assert state == "sent"

    def test_the_install_id_never_crosses_the_wire(self, store, server):
        _add(store, "pkg-1", metrics=40)
        _sender(store, server).send_pending()
        body = json.dumps(Ingest.received[0]["body"])
        assert INSTALL_ID not in body
        assert len(Ingest.received[0]["body"]["install_id"]) == 64

    def test_content_type_is_json(self, store, server):
        _add(store, "pkg-1")
        _sender(store, server).send_pending()
        assert Ingest.received[0]["headers"]["content-type"] == "application/json"

    def test_a_realistic_package_is_gzipped_over_the_wire(self, store, server):
        # ~40 metrics matches the real outbox's larger packages.
        _add(store, "pkg-1", metrics=120)
        _sender(store, server).send_pending()
        record = Ingest.received[0]
        assert record["headers"].get("content-encoding") == "gzip"
        assert record["raw_len"] < record["decoded_len"]

    def test_the_server_can_parse_what_we_send(self, store, server):
        """Proves the bytes are valid JSON after transport and decompression."""
        original = _add(store, "pkg-1", metrics=120)
        _sender(store, server).send_pending()
        received = Ingest.received[0]["body"]
        assert received["metrics"] == original["metrics"]
        assert received["resource"] == original["resource"]

    def test_400_is_permanent(self, store, server):
        _add(store, "pkg-1")
        Ingest.script = [(400, {"error": "invalid_envelope"}, {})]
        outcome = _sender(store, server).send_pending()
        assert outcome.rejected == 1
        assert len(Ingest.received) == 1

    def test_429_is_honoured(self, store, server):
        _add(store, "pkg-1")
        Ingest.script = [(429, {"error": "rate_limited"}, {"Retry-After": "90"})]
        outcome = _sender(store, server).send_pending()
        assert outcome.deferred == 1
        with store._connection() as connection:
            retry_at = connection.execute(
                "SELECT next_attempt_at FROM package_outbox WHERE package_id = 'pkg-1'"
            ).fetchone()[0]
        assert retry_at == "2026-08-26T12:01:30Z"

    def test_5xx_retries_then_succeeds(self, store, server):
        _add(store, "pkg-1")
        Ingest.script = [
            (503, {"error": "storage_unavailable"}, {}),
            (202, {"package_id": "pkg-1"}, {}),
        ]
        outcome = _sender(store, server).send_pending()
        assert outcome.sent == 1
        assert len(Ingest.received) == 2

    def test_a_retry_sends_identical_bytes(self, store, server):
        _add(store, "pkg-1", metrics=5)
        Ingest.script = [(503, {}, {}), (202, {}, {})]
        _sender(store, server).send_pending()
        first, second = Ingest.received
        assert first["body"] == second["body"]
        assert first["raw"] == second["raw"], (
            "the raw request bytes must match, not just the parsed body"
        )

    def test_a_gzipped_retry_is_byte_identical_on_the_wire(self, store, server):
        """gzip embeds an mtime by default, which would break this."""
        _add(store, "pkg-1", metrics=200)
        Ingest.script = [(503, {}, {}), (202, {}, {})]
        _sender(store, server).send_pending()
        first, second = Ingest.received
        assert first["headers"].get("content-encoding") == "gzip"
        assert first["raw"] == second["raw"]

    def test_several_packages_in_one_pass(self, store, server):
        for i in range(5):
            _add(store, f"pkg-{i}")
        outcome = _sender(store, server).send_pending()
        assert outcome.sent == 5
        assert len(Ingest.received) == 5

    def test_the_outbox_directory_is_untouched(self, store, server, tmp_path):
        _add(store, "pkg-1")
        marker = store.outbox_directory / "pkg-1.json"
        marker.write_text('{"kept": true}')
        _sender(store, server).send_pending()
        assert marker.exists()
        assert json.loads(marker.read_text()) == {"kept": True}

    def test_a_dead_server_defers_without_raising(self, store, server):
        _add(store, "pkg-1")
        host, port = server.server_address
        server.shutdown()
        server.server_close()
        sender = SharedMetricsSender(
            store,
            f"http://{host}:{port}/v1/telemetry",
            sleep=lambda _s: None,
            now=lambda: NOW,
        )
        outcome = sender.send_pending()
        assert outcome.deferred == 1
        with store._connection() as connection:
            state, error = connection.execute(
                "SELECT send_state, last_error FROM package_outbox"
                " WHERE package_id = 'pkg-1'"
            ).fetchone()
        assert state == "pending"
        assert error
