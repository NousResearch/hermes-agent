"""Byte-contract tests for the shared ``_sse_frame`` SSE encoder.

``_sse_frame`` is the single source of truth for SSE frame serialization
across ``_write_sse_chat_completion``, ``_write_sse_responses._write_event``,
and the ``/v1/runs`` event stream. These tests assert the *invariant* that
``_sse_frame`` reproduces the exact on-the-wire bytes the inline encoders
used to emit — not a snapshot of a frozen value. If a writer's bytes ever
diverge, a real client breaks, so we pin the relationship, not a literal.
"""

import json

from gateway.platforms.api_server import _sse_frame


def _inline_frame(data, *, event=None):
    """Reproduce the historical inline SSE encoder (pre-dedup)."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n".encode()


def test_sse_frame_matches_inline_encoder_no_event():
    for data in (
        {"id": "c1", "choices": [{"delta": {"role": "assistant"}}]},
        {"event": "ping", "sequence_number": 1},
        {"text": "plain ascii"},
    ):
        assert _sse_frame(data) == _inline_frame(data)


def test_sse_frame_matches_inline_encoder_with_event():
    for event, data in (
        ("hermes.tool.progress", {"name": "x", "status": "running"}),
        ("response.created", {"id": "r1", "status": "in_progress"}),
    ):
        assert _sse_frame(data, event=event) == _inline_frame(data, event=event)


def test_sse_frame_event_line_shape():
    out = _sse_frame({"a": 1}, event="my.event")
    assert out.startswith(b"event: my.event\n")
    assert b"data: " in out
    assert out.endswith(b"\n\n")


def test_sse_frame_default_ensure_ascii_matches_bare_json():
    payload = {"text": "café — Münchner 🏔"}
    # Default must equal a bare json.dumps (the original writers used no
    # ensure_ascii override), so existing byte streams are unchanged.
    assert _sse_frame(payload) == _inline_frame(payload)
    assert _sse_frame(payload) == f"data: {json.dumps(payload)}\n\n".encode()


def test_sse_frame_ensure_ascii_false_preserves_raw_bytes():
    payload = {"text": "café — Münchner 🏔"}
    raw = _sse_frame(payload, ensure_ascii=False)
    assert "café" in raw.decode("utf-8")
    assert raw != _sse_frame(payload)  # different bytes from the default


def test_sse_frame_ensure_ascii_false_reproduces_session_event_stream():
    """The session event stream (api_server.py:~2236) historically used
    ``json.dumps(payload, ensure_ascii=False)`` + ``.encode('utf-8')`` — the
    one genuinely unicode-distinct SSE writer. _sse_frame(event=name,
    ensure_ascii=False) must reproduce its exact bytes, raw non-ASCII included.
    """

    def old_session(name, payload):
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {name}\ndata: {data}\n\n".encode("utf-8")

    for name, payload in (
        ("session.update", {"text": "café — Münchner 🏔", "id": 1}),
        ("thread.message.delta", {"content": "héllo wörld ✓", "seq": 3}),
    ):
        assert _sse_frame(payload, event=name, ensure_ascii=False) == old_session(name, payload)


def test_sse_frame_ensure_ascii_false_survives_lone_surrogate():
    """``ensure_ascii=False`` leaves a lone UTF-16 surrogate raw in the
    serialized JSON, and UTF-8 cannot encode one.  Encoding it must not raise:
    a raise here propagates into the caller's SSE writer loop, which stops
    consuming and closes a response that already sent HTTP 200, so the client
    hangs with no terminal event.
    """
    payload = {"text": "truncated emoji \ud83d pasted from a clipboard"}

    raw = _sse_frame(payload, event="run.started", ensure_ascii=False)

    assert isinstance(raw, bytes)
    assert raw.startswith(b"event: run.started\n")
    assert raw.endswith(b"\n\n")
    decoded = raw.decode("utf-8")
    body = json.loads(decoded.split("data: ", 1)[1].strip())
    # Repaired, not dropped: the surrogate becomes U+FFFD and the surrounding
    # text is delivered intact.
    assert "�" in body["text"]
    assert "\ud83d" not in body["text"]
    assert body["text"].startswith("truncated emoji ")
    assert body["text"].endswith(" pasted from a clipboard")


def test_sse_frame_ensure_ascii_false_repairs_surrogate_in_nested_payload():
    """The session event stream nests user text under ``user_message`` — the
    repair happens on the serialized frame, so nesting depth is irrelevant.
    """
    payload = {
        "user_message": {"role": "user", "content": "hi \udccc there"},
        "session_id": "s1",
        "run_id": "r1",
        "seq": 1,
    }

    raw = _sse_frame(payload, event="run.started", ensure_ascii=False)

    body = json.loads(raw.decode("utf-8").split("data: ", 1)[1].strip())
    assert body["user_message"]["content"] == "hi � there"
    assert body["session_id"] == "s1"
    assert body["seq"] == 1


def test_sse_frame_default_escapes_lone_surrogate_without_repair():
    """``ensure_ascii=True`` (the default, used by every other writer) escapes
    the surrogate as ``\\udXXX`` text, so it never reaches the UTF-8 encoder.
    That path must keep passing the code point through untouched.
    """
    payload = {"text": "hi \ud83d there"}

    out = _sse_frame(payload)

    assert out == _inline_frame(payload)
    assert b"\\ud83d" in out
    # Escaped, so a JSON parser still reconstructs the original surrogate.
    assert json.loads(out.decode().split("data: ", 1)[1].strip()) == payload


def test_sse_frame_byte_contract_unchanged_for_encodable_payloads():
    """The repair is reachable only from the encoder failing, so every payload
    that already encoded must produce byte-identical output — in both modes.
    """
    for data in (
        {"id": "c1", "choices": [{"delta": {"role": "assistant"}}]},
        {"text": "café — Münchner 🏔"},
        {"content": "héllo wörld ✓", "seq": 3},
        {"nested": {"deep": ["plain ascii", 1, None, True]}},
    ):
        for event in (None, "session.update"):
            prefix = f"event: {event}\n" if event else ""
            for ensure_ascii in (True, False):
                assert _sse_frame(data, event=event, ensure_ascii=ensure_ascii) == (
                    f"{prefix}data: {json.dumps(data, ensure_ascii=ensure_ascii)}\n\n".encode()
                )


def test_sse_frame_typed_object_roundtrip():
    obj = {"id": "x", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
    out = _sse_frame(obj)
    line = out.decode().split("data: ", 1)[1].strip()
    assert json.loads(line) == obj
