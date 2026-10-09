"""Audit: actual boto3 HTTP/EventStream parsing before Hermes normalization."""

import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.bedrock_adapter import normalize_converse_stream_events


def _frame(kind, payload):
    headers = b""
    for key, value in {
        ":message-type": "event",
        ":event-type": kind,
        ":content-type": "application/json",
    }.items():
        k, v = key.encode(), value.encode()
        headers += bytes([len(k)]) + k + b"\x07" + struct.pack(">H", len(v)) + v
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


@pytest.mark.parametrize("terminal", [False, True])
def test_bedrock_wire_requires_message_stop(terminal):
    boto3 = pytest.importorskip("boto3")
    from botocore import UNSIGNED
    from botocore.config import Config

    events = [
        ("messageStart", {"role": "assistant"}),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "partial answer"}},
        ),
        ("contentBlockStop", {"contentBlockIndex": 0}),
    ]
    if terminal:
        events.append(("messageStop", {"stopReason": "end_turn"}))
    body = b"".join(_frame(*event) for event in events)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.amazon.eventstream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        config=Config(
            signature_version=UNSIGNED,
            proxies={},
            retries={"max_attempts": 0},
            read_timeout=5,
        ),
    )
    try:
        response = client.converse_stream(
            modelId="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
        )
        if terminal:
            result = normalize_converse_stream_events(response)
            assert result.choices[0].message.content == "partial answer"
            assert result.choices[0].finish_reason == "stop"
        else:
            from agent.errors import EmptyStreamError

            with pytest.raises(
                EmptyStreamError, match="(?i)(incomplete|messageStop|truncat)"
            ):
                normalize_converse_stream_events(response)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_explicit_interrupt_keeps_partial_response_contract():
    from agent.bedrock_adapter import stream_converse_with_callbacks
    from agent.chat_completion_helpers import _finalize_bedrock_relay_events

    interrupted = False

    def on_text(text):
        nonlocal interrupted
        interrupted = True

    events = [
        {"contentBlockDelta": {"delta": {"text": "partial"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
    ]
    response = stream_converse_with_callbacks(
        {"stream": events},
        on_text_delta=on_text,
        on_interrupt_check=lambda: interrupted,
    )
    assert response.choices[0].message.content == "partial"
    # The relay finalizer has no interrupt: no messageStop -> nothing to record.
    assert _finalize_bedrock_relay_events(events) is None
