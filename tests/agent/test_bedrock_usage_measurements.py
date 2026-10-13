"""Converse usage remains unmeasured until both primary counts are reported."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import struct
import threading
from types import SimpleNamespace
from unittest.mock import patch
import zlib

import pytest

from agent.bedrock_adapter import (
    _get_bedrock_runtime_client,
    normalize_converse_response,
    normalize_converse_stream_events,
)
from agent.native_usage import NativeUsageError
from agent.transports.bedrock import BedrockTransport
from agent.transports.types import Usage


def _frame(kind, payload):
    headers = b""
    for key, value in {
        ":message-type": "event",
        ":event-type": kind,
        ":content-type": "application/json",
    }.items():
        key_bytes, value_bytes = key.encode(), value.encode()
        headers += (
            bytes([len(key_bytes)])
            + key_bytes
            + b"\x07"
            + struct.pack(">H", len(value_bytes))
            + value_bytes
        )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


@contextmanager
def _response(usage, streamed, status_code=200):
    boto3 = pytest.importorskip("boto3")
    from botocore import UNSIGNED
    from botocore.config import Config

    payload = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Answer."}]}},
        "stopReason": "end_turn",
        "usage": usage,
    }
    body = json.dumps(payload).encode()
    if streamed:
        body = b"".join(
            _frame(kind, value)
            for kind, value in (
                ("messageStart", {"role": "assistant"}),
                (
                    "contentBlockDelta",
                    {"contentBlockIndex": 0, "delta": {"text": "Answer."}},
                ),
                ("contentBlockStop", {"contentBlockIndex": 0}),
                ("messageStop", {"stopReason": "end_turn"}),
                ("metadata", {"usage": usage}),
            )
        )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status_code)
            self.send_header(
                "Content-Type",
                "application/vnd.amazon.eventstream"
                if streamed
                else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        aws_access_key_id="fixture",
        aws_secret_access_key="fixture",
        config=Config(
            signature_version=UNSIGNED, proxies={}, retries={"max_attempts": 0}
        ),
    )
    with patch("agent.bedrock_adapter._cached_client", return_value=client):
        client = _get_bedrock_runtime_client("us-east-1")
    try:
        method = client.converse_stream if streamed else client.converse
        raw = method(
            modelId="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
        )
        yield (
            normalize_converse_stream_events(raw)
            if streamed
            else normalize_converse_response(raw)
        )
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "usage,expected",
    [
        (None, None),
        ({}, None),
        ({"outputTokens": 0}, None),
        ({"inputTokens": 0}, None),
        ({"inputTokens": None, "outputTokens": 0}, None),
        ({"inputTokens": 0, "outputTokens": None}, None),
        ({"inputTokens": 0, "outputTokens": 0}, (0, 0, 0, 0, 0)),
        ({"inputTokens": 0, "outputTokens": 0, "totalTokens": None}, (0, 0, 0, 0, 0)),
        ({"inputTokens": 2, "outputTokens": 1, "totalTokens": 9}, (2, 1, 9, 0, 0)),
        (
            {
                "inputTokens": 2,
                "outputTokens": 1,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 3,
            },
            (9, 1, 10, 4, 3),
        ),
        (
            {
                "inputTokens": 2,
                "outputTokens": 1,
                "totalTokens": 19,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 3,
            },
            (9, 1, 19, 4, 3),
        ),
        (
            {
                "inputTokens": 2,
                "outputTokens": 1,
                "cacheReadInputTokens": None,
                "cacheWriteInputTokens": None,
            },
            (2, 1, 3, 0, 0),
        ),
    ],
)
def test_bedrock_sdk_preserves_unknown_zero_and_reported_measurements(
    streamed, usage, expected
):
    with _response(usage, streamed) as result:
        wanted = (
            None
            if expected is None
            else SimpleNamespace(
                **dict(
                    zip(
                        (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                        ),
                        expected,
                    )
                ),
            )
        )
        assert result.usage == wanted
        normalized = BedrockTransport().normalize_response(result)
        assert (normalized.content, normalized.finish_reason, normalized.usage) == (
            "Answer.",
            "stop",
            None
            if expected is None
            else Usage(
                prompt_tokens=expected[0],
                completion_tokens=expected[1],
                total_tokens=expected[2],
            ),
        )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
    ],
)
@pytest.mark.parametrize("value", [True, "2", 2.0, -1])
def test_bedrock_sdk_rejects_invalid_counts_before_arithmetic(streamed, field, value):
    usage = {"inputTokens": 2, "outputTokens": 1, field: value}
    with pytest.raises(NativeUsageError) as error, _response(usage, streamed):
        pass
    assert (error.value.code, error.value.field) == ("invalid_usage", f"usage.{field}")


@pytest.mark.parametrize("usage", [False, 0, "", []])
def test_bedrock_normalizer_rejects_invalid_usage_containers(usage):
    with pytest.raises(NativeUsageError) as error:
        normalize_converse_response({"usage": usage})
    assert (error.value.code, error.value.field) == ("invalid_usage", "usage")


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "field", ["totalTokens", "cacheReadInputTokens", "cacheWriteInputTokens"]
)
def test_bedrock_unknown_primary_does_not_hide_invalid_optional_count(streamed, field):
    with pytest.raises(NativeUsageError) as error, _response({field: True}, streamed):
        pass
    assert (error.value.code, error.value.field) == ("invalid_usage", f"usage.{field}")


def test_bedrock_usage_validation_preserves_provider_errors():
    pytest.importorskip("boto3")
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError), _response({"inputTokens": True}, False, 400):
        pass
