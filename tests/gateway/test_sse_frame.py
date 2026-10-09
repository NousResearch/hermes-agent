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




def test_sse_frame_matches_inline_encoder_with_event():
    for event, data in (
        ("hermes.tool.progress", {"name": "x", "status": "running"}),
        ("response.created", {"id": "r1", "status": "in_progress"}),
    ):
        assert _sse_frame(data, event=event) == _inline_frame(data, event=event)




def test_sse_frame_default_ensure_ascii_matches_bare_json():
    payload = {"text": "café — Münchner 🏔"}
    # Default must equal a bare json.dumps (the original writers used no
    # ensure_ascii override), so existing byte streams are unchanged.
    assert _sse_frame(payload) == _inline_frame(payload)
    assert _sse_frame(payload) == f"data: {json.dumps(payload)}\n\n".encode()




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


def test_sse_frame_typed_object_roundtrip():
    obj = {"id": "x", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
    out = _sse_frame(obj)
    line = out.decode().split("data: ", 1)[1].strip()
    assert json.loads(line) == obj
