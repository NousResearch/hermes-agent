"""Tests for the LSP protocol framing layer.

The framer is small but load-bearing — Content-Length parsing is the
single most common reason for hand-rolled LSP clients to silently
deadlock.  These tests exercise:

- exact wire format of outgoing messages (encode_message)
- partial-read tolerance + EOF handling (read_message)
- envelope helpers (request, response, notification, error)
- message classification
"""
from __future__ import annotations

import asyncio
import json
import pytest

from agent.lsp.protocol import (
    ERROR_CONTENT_MODIFIED,
    ERROR_METHOD_NOT_FOUND,
    LSPProtocolError,
    LSPRequestError,
    classify_message,
    encode_message,
    make_error_response,
    make_notification,
    make_request,
    make_response,
    read_message,
)


# ---------------------------------------------------------------------------
# encode_message
# ---------------------------------------------------------------------------


def test_encode_message_uses_compact_separators_and_utf8():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "x", "params": {"k": "ä"}}
    out = encode_message(msg)
    # Header is plain ASCII Content-Length CRLF CRLF
    header_end = out.index(b"\r\n\r\n") + 4
    header = out[:header_end].decode("ascii")
    body = out[header_end:]
    assert "Content-Length:" in header
    declared = int(header.split("Content-Length:")[1].split("\r\n")[0].strip())
    # Declared length must equal actual body bytes.
    assert declared == len(body)
    # Body parses as JSON and round-trips.
    parsed = json.loads(body.decode("utf-8"))
    assert parsed == msg
    # Body uses compact separators (no spaces between kv).
    assert b'"id":1' in body


def test_encode_message_handles_unicode_in_strings():
    msg = {"jsonrpc": "2.0", "method": "log", "params": {"text": "🚀 ünıcödé"}}
    out = encode_message(msg)
    header_end = out.index(b"\r\n\r\n") + 4
    declared = int(out[: out.index(b"\r\n")].split(b": ")[1])
    assert declared == len(out[header_end:])
    assert json.loads(out[header_end:].decode("utf-8")) == msg


# ---------------------------------------------------------------------------
# read_message
# ---------------------------------------------------------------------------


async def _stream_from_bytes(data: bytes) -> asyncio.StreamReader:
    """Build an asyncio.StreamReader pre-populated with ``data``."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_read_message_round_trip():
    msg = {"jsonrpc": "2.0", "method": "ping"}
    reader = await _stream_from_bytes(encode_message(msg))
    parsed = await read_message(reader)
    assert parsed == msg


@pytest.mark.asyncio
async def test_read_message_clean_eof_returns_none():
    reader = await _stream_from_bytes(b"")
    assert await read_message(reader) is None


@pytest.mark.asyncio
async def test_read_message_truncated_body_raises():
    msg = encode_message({"jsonrpc": "2.0", "method": "x"})
    truncated = msg[: -3]  # cut the body
    reader = await _stream_from_bytes(truncated)
    with pytest.raises(LSPProtocolError):
        await read_message(reader)


@pytest.mark.asyncio
async def test_read_message_missing_content_length_raises():
    bad = b"X-Other: 5\r\n\r\n12345"
    reader = await _stream_from_bytes(bad)
    with pytest.raises(LSPProtocolError):
        await read_message(reader)


@pytest.mark.asyncio
async def test_read_message_two_messages_back_to_back():
    a = encode_message({"jsonrpc": "2.0", "method": "a"})
    b = encode_message({"jsonrpc": "2.0", "method": "b"})
    reader = await _stream_from_bytes(a + b)
    assert (await read_message(reader))["method"] == "a"
    assert (await read_message(reader))["method"] == "b"


@pytest.mark.asyncio
async def test_read_message_rejects_runaway_header():
    """A pathological server that streams headers without ever emitting
    the CRLF-CRLF terminator must not loop forever — the 8 KiB cap kicks
    in and surfaces a protocol error."""
    flood = (b"X-Junk: " + b"A" * 200 + b"\r\n") * 60   # ~12 KiB worth
    reader = await _stream_from_bytes(flood)
    with pytest.raises(LSPProtocolError) as exc:
        await read_message(reader)
    assert "8 KiB" in str(exc.value)


# ---------------------------------------------------------------------------
# envelope helpers
# ---------------------------------------------------------------------------


def test_make_request_includes_id_and_method():
    msg = make_request(7, "ping", {"v": 1})
    assert msg == {"jsonrpc": "2.0", "id": 7, "method": "ping", "params": {"v": 1}}


def test_make_request_omits_params_when_none():
    msg = make_request(7, "ping", None)
    assert "params" not in msg


def test_make_notification_omits_id():
    msg = make_notification("log", {"line": "hi"})
    assert "id" not in msg
    assert msg["method"] == "log"


def test_make_response_carries_result():
    msg = make_response(7, {"ok": True})
    assert msg["id"] == 7 and msg["result"] == {"ok": True}


def test_make_error_response_shape():
    msg = make_error_response(7, ERROR_CONTENT_MODIFIED, "stale", {"hint": "retry"})
    assert msg["error"]["code"] == ERROR_CONTENT_MODIFIED
    assert msg["error"]["message"] == "stale"
    assert msg["error"]["data"] == {"hint": "retry"}


# ---------------------------------------------------------------------------
# classify_message
# ---------------------------------------------------------------------------


def test_classify_message_request():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "x"}
    assert classify_message(msg) == ("request", 1)


def test_classify_message_response():
    msg = {"jsonrpc": "2.0", "id": 1, "result": None}
    assert classify_message(msg) == ("response", 1)


def test_classify_message_notification():
    msg = {"jsonrpc": "2.0", "method": "log"}
    assert classify_message(msg) == ("notification", "log")


def test_classify_message_invalid():
    assert classify_message({"id": 1})[0] == "invalid"
    assert classify_message({"jsonrpc": "1.0", "method": "x"})[0] == "invalid"


# ---------------------------------------------------------------------------
# LSPRequestError
# ---------------------------------------------------------------------------


def test_lsp_request_error_carries_code_and_data():
    e = LSPRequestError(ERROR_METHOD_NOT_FOUND, "no", {"x": 1})
    assert e.code == ERROR_METHOD_NOT_FOUND
    assert e.message == "no"
    assert e.data == {"x": 1}
