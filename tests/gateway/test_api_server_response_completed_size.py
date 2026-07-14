"""Verify SSE `data:` lines stay under CPython's effective ~128 KB MAXLINE.

Regression for #18021 / review on #21550:
- Terminal `response.completed` must be budgeted AFTER the assistant message
  is appended (not only tool items).
- Live intermediate SSE events (function_call / function_call_output) must
  also be bounded — trimming only the terminal event is not enough.
"""

from __future__ import annotations

import json

from gateway.platforms.api_server import (
    _RESPONSE_COMPLETED_SAFE_BYTES,
    _SSE_DATA_SAFE_BYTES,
    _SSE_LINE_HARD_BYTES,
    _enforce_sse_event_budget,
    _sse_payload_byte_len,
    _trim_response_completed_items,
    _trim_response_completed_payload,
)


def _huge_function_call(name: str, arg_size: int = 50_000) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(
            {
                "content": "x" * arg_size,
                "query": "y" * arg_size,
                "extra": "z" * arg_size,
            }
        ),
    }


def _huge_function_output(call_id: str, text_size: int = 50_000) -> dict:
    return {
        "id": f"out_{call_id}",
        "type": "function_call_output",
        "call_id": call_id,
        "output": [
            {"type": "input_text", "text": "a" * text_size},
            {"type": "input_text", "text": "b" * text_size},
            {"type": "input_text", "text": "c" * text_size},
        ],
    }


def test_safe_threshold_is_below_maxline():
    import http.client

    maxline_ceiling = getattr(http.client, "_MAXLINE", 65536) * 2
    assert _SSE_DATA_SAFE_BYTES + 20_000 <= maxline_ceiling
    assert _RESPONSE_COMPLETED_SAFE_BYTES == _SSE_DATA_SAFE_BYTES
    assert _SSE_LINE_HARD_BYTES < maxline_ceiling


def test_huge_payload_trimmed_below_safe_limit():
    items = [
        _huge_function_call("read_file"),
        _huge_function_output("read_file"),
        _huge_function_call("web_extract"),
        _huge_function_output("web_extract"),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "summary"}],
        },
    ]
    raw_size = len(json.dumps(items).encode("utf-8"))
    assert raw_size > 200_000, f"test setup too small ({raw_size} bytes)"

    _trim_response_completed_payload(items)
    trimmed_size = len(json.dumps(items).encode("utf-8"))
    assert trimmed_size <= _SSE_DATA_SAFE_BYTES, (
        f"after trimming, payload is still {trimmed_size} bytes "
        f"(limit {_SSE_DATA_SAFE_BYTES})"
    )


def test_soft_trim_replaces_strings_above_500_chars():
    items = [
        {
            "type": "function_call",
            "arguments": json.dumps(
                {
                    "small": "ok",
                    "big": "x" * 600,
                    "extra": "y" * 1000,
                }
            ),
        }
    ]
    _trim_response_completed_items(items, soft=True)
    args = json.loads(items[0]["arguments"])
    assert args["small"] == "ok"
    assert "truncated for SSE size cap" in args["big"]
    assert "truncated for SSE size cap" in args["extra"]


def test_soft_trim_does_not_drop_non_first_output_entries():
    items = [
        {
            "type": "function_call_output",
            "output": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
                {"type": "input_text", "text": "third"},
            ],
        }
    ]
    _trim_response_completed_items(items, soft=True)
    assert len(items[0]["output"]) == 3


def test_hard_trim_collapses_to_single_output_entry():
    items = [
        {
            "type": "function_call_output",
            "output": [
                {"type": "input_text", "text": "first" * 200},
                {"type": "input_text", "text": "second" * 200},
                {"type": "input_text", "text": "third" * 200},
            ],
        }
    ]
    _trim_response_completed_items(items, soft=False)
    assert len(items[0]["output"]) == 1


def test_message_preserved_when_under_budget():
    items = [
        {
            "type": "function_call",
            "arguments": json.dumps({"q": "tiny"}),
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello world"}],
        },
    ]
    _trim_response_completed_payload(items)
    msg = items[-1]
    assert msg["type"] == "message"
    assert msg["content"][0]["text"] == "hello world"


def test_oversized_final_message_included_in_envelope_budget():
    """Budget applies AFTER the final assistant message is in the envelope."""
    final_answer = "Z" * 150_000
    items = [
        _huge_function_call("read_file", arg_size=10_000),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_answer}],
        },
    ]
    event = {
        "type": "response.completed",
        "response": {
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "output": items,
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        },
    }
    bounded = _enforce_sse_event_budget("response.completed", event)
    nbytes = _sse_payload_byte_len("response.completed", bounded)
    assert nbytes <= _SSE_LINE_HARD_BYTES, f"SSE frame still {nbytes} bytes"
    assert nbytes <= _SSE_DATA_SAFE_BYTES + 5_000


def test_writer_bounds_live_function_call_item_event():
    item = {
        "id": "fc_abc",
        "type": "function_call",
        "status": "in_progress",
        "name": "execute_code",
        "call_id": "call_1",
        "arguments": json.dumps({"code": "print(" + repr("x" * 120_000) + ")"}),
    }
    event = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": item,
    }
    raw = _sse_payload_byte_len("response.output_item.added", event)
    assert raw > _SSE_DATA_SAFE_BYTES

    bounded = _enforce_sse_event_budget("response.output_item.added", event)
    nbytes = _sse_payload_byte_len("response.output_item.added", bounded)
    assert nbytes <= _SSE_LINE_HARD_BYTES, f"live SSE frame still {nbytes} bytes"
    assert "x" * 1000 in event["item"]["arguments"]


def test_writer_bounds_live_function_call_output_event():
    event = {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "id": "fco_abc",
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "R" * 200_000}],
            "status": "completed",
        },
    }
    bounded = _enforce_sse_event_budget("response.output_item.added", event)
    nbytes = _sse_payload_byte_len("response.output_item.added", bounded)
    assert nbytes <= _SSE_LINE_HARD_BYTES


def test_malformed_function_call_arguments_do_not_crash():
    items = [
        {
            "type": "function_call",
            "arguments": "not valid json {{{",
        }
    ]
    _trim_response_completed_items(items, soft=True)
    assert isinstance(items[0]["arguments"], str)
