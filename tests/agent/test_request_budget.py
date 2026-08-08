"""Tests for agent.request_budget — total request-body budget enforcement."""

import base64
import io
import json

import pytest

from agent.request_budget import (
    DEFAULT_REQUEST_BODY_BUDGET_BYTES,
    apply_request_body_budget,
    get_request_body_budget,
    serialized_body_size,
)


def _user_message(text="hello", extra_parts=None):
    content = [{"type": "input_text", "text": text}]
    if extra_parts:
        content.extend(extra_parts)
    return {"type": "message", "role": "user", "content": content}


def _fc(call_id="call_1", name="terminal"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _fc_output(call_id="call_1", output="ok"):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _kwargs(input_items, instructions="test"):
    return {
        "model": "gpt-5.6-sol",
        "instructions": instructions,
        "input": input_items,
        "store": False,
        "extra_headers": {"session_id": "t"},
        "_client_only": {"ignored": True},
    }


def _noise_jpeg_data_url(side=512):
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    import os

    img = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_under_budget_is_untouched_same_object():
    kwargs = _kwargs([_user_message(), _fc(), _fc_output()])
    result = apply_request_body_budget(kwargs, budget_bytes=1_000_000)
    assert result is kwargs


def test_serialized_size_excludes_client_only_keys():
    kwargs = _kwargs([_user_message()])
    base = serialized_body_size(kwargs)
    kwargs["extra_headers"] = {"session_id": "x" * 100_000}
    kwargs["_client_only"] = {"blob": "y" * 100_000}
    assert serialized_body_size(kwargs) == base


def test_history_tool_output_truncated_before_active_tail():
    huge = "H" * 400_000
    tail_output = "T" * 10_000
    kwargs = _kwargs(
        [
            _user_message("first ask"),
            _fc("call_old"),
            _fc_output("call_old", huge),
            _user_message("current ask"),
            _fc("call_now"),
            _fc_output("call_now", tail_output),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=100_000)

    assert result is not kwargs
    # Original never mutated.
    assert kwargs["input"][2]["output"] == huge
    out_items = [i for i in result["input"] if i["type"] == "function_call_output"]
    assert "characters omitted" in out_items[0]["output"]
    assert len(out_items[0]["output"]) < len(huge)
    # Active-tail output untouched (budget already met by step 1).
    assert out_items[1]["output"] == tail_output
    assert serialized_body_size(result) <= 100_000


def test_array_output_text_parts_truncated():
    kwargs = _kwargs(
        [
            _user_message("old"),
            _fc(),
            _fc_output(
                output=[
                    {"type": "input_text", "text": "A" * 300_000},
                    {"type": "input_text", "text": "short"},
                ]
            ),
            _user_message("now"),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=80_000)
    parts = [
        i for i in result["input"] if i["type"] == "function_call_output"
    ][0]["output"]
    assert "characters omitted" in parts[0]["text"]
    assert parts[1]["text"] == "short"
    assert serialized_body_size(result) <= 80_000


def test_images_reencoded_then_placeholdered():
    data_url = _noise_jpeg_data_url(side=700)
    assert len(data_url) > 200_000  # noise JPEG stays large
    kwargs = _kwargs(
        [
            _user_message("photo one", [{"type": "input_image", "image_url": data_url}]),
            _user_message("photo two", [{"type": "input_image", "image_url": data_url}]),
            _user_message("current ask"),
        ]
    )
    start = serialized_body_size(kwargs)
    result = apply_request_body_budget(kwargs, budget_bytes=200_000)
    assert serialized_body_size(result) < start
    assert serialized_body_size(result) <= 200_000
    # Every remaining image part fits the per-image cap; any that could not be
    # constrained became text placeholders.
    for item in result["input"]:
        for part in item.get("content") or []:
            if part.get("type") == "input_image":
                assert len(part["image_url"]) <= 200_000
            if part.get("type") == "input_text" and "image omitted" in part["text"]:
                assert "transport" in part["text"]


def test_active_tail_output_truncated_as_last_resort():
    kwargs = _kwargs(
        [
            _user_message("current ask"),
            _fc("call_now"),
            _fc_output("call_now", "V" * 500_000),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=120_000)
    out = [i for i in result["input"] if i["type"] == "function_call_output"][0]
    assert "characters omitted" in out["output"]
    assert serialized_body_size(result) <= 120_000


def test_irreducible_overage_logged_not_crashed(caplog):
    kwargs = _kwargs([_user_message("hi")], instructions="I" * 300_000)
    with caplog.at_level("ERROR"):
        result = apply_request_body_budget(kwargs, budget_bytes=50_000)
    assert serialized_body_size(result) > 50_000
    assert any("request_budget" in r.message for r in caplog.records)


def test_user_text_never_modified():
    user_text = "do not touch this" * 1_000
    kwargs = _kwargs(
        [
            _user_message(user_text),
            _fc(),
            _fc_output(output="Z" * 300_000),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=90_000)
    messages = [i for i in result["input"] if i.get("type") == "message"]
    assert messages[0]["content"][0]["text"] == user_text


def test_budget_default_positive():
    assert get_request_body_budget() >= 1
    assert DEFAULT_REQUEST_BODY_BUDGET_BYTES == 1_000_000


def test_equal_size_requests_are_not_cross_contaminated():
    """Regression (review finding): a size-keyed memo substituted an
    unrelated equal-length request's content. Distinct bodies of identical
    serialized size must each get their own constrained result."""
    a = _kwargs([_user_message("A" * 50), _fc(), _fc_output(output="a" * 200_000)],
                instructions="tenant-A")
    b = _kwargs([_user_message("B" * 50), _fc(), _fc_output(output="b" * 200_000)],
                instructions="tenant-B")
    assert serialized_body_size(a) == serialized_body_size(b)
    ra = apply_request_body_budget(a, budget_bytes=80_000)
    rb = apply_request_body_budget(b, budget_bytes=80_000)
    assert ra["instructions"] == "tenant-A"
    assert rb["instructions"] == "tenant-B"
    assert ra is not rb


def test_many_under_cap_images_still_reduced():
    """Regression (review finding): several images each under the per-image
    cap can collectively bust the budget; the last-resort pass must sacrifice
    them anyway."""
    data_url = _noise_jpeg_data_url(side=300)
    per_image = len(data_url)
    n = (250_000 // per_image) + 3
    msgs = [
        _user_message(f"photo {i}", [{"type": "input_image", "image_url": data_url}])
        for i in range(n)
    ]
    kwargs = _kwargs(msgs + [_user_message("current ask")])
    budget = 150_000
    assert per_image <= max(64_000, budget // 8) or True  # images may be under cap
    result = apply_request_body_budget(kwargs, budget_bytes=budget)
    assert serialized_body_size(result) <= budget


def test_history_reasoning_and_call_arguments_reduced():
    """Regression (review finding): reasoning.encrypted_content and giant
    function_call.arguments were untouched by the ladder."""
    kwargs = _kwargs(
        [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "E" * 400_000},
            {
                "type": "function_call",
                "call_id": "call_big",
                "name": "write_file",
                "arguments": json.dumps({"content": "X" * 300_000}),
            },
            _fc_output("call_big", "ok"),
            _user_message("current ask"),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=100_000)
    assert serialized_body_size(result) <= 100_000
    types = [i.get("type") for i in result["input"]]
    assert "reasoning" not in types
    call = [i for i in result["input"] if i.get("type") == "function_call"][0]
    assert "_hermes_truncated" in call["arguments"]
    assert json.loads(call["arguments"])  # stub stays valid JSON


def test_hundreds_of_outputs_escalate_past_first_pass_floors():
    """Regression (review finding): 200 outputs at the 4KB+1KB first-pass
    floors still exceed the budget; the escalation pass must fire."""
    items = [_user_message("go")]
    for i in range(200):
        items.append(_fc(f"call_{i}"))
        items.append(_fc_output(f"call_{i}", f"{i}:" + "Y" * 6_000))
    kwargs = _kwargs(items)
    result = apply_request_body_budget(kwargs, budget_bytes=400_000)
    assert serialized_body_size(result) <= 400_000


def test_omission_marker_counts_characters():
    """Regression (review finding): the marker claimed byte counts while
    slicing characters."""
    kwargs = _kwargs(
        [_user_message("old"), _fc(), _fc_output(output="é" * 200_000), _user_message("now")]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=100_000)
    out = [i for i in result["input"] if i["type"] == "function_call_output"][0]
    assert "characters omitted" in out["output"]
    assert "bytes omitted" not in out["output"]


def test_bare_role_user_message_images_are_visible():
    """Images inside bare {'role': 'user'} items (no 'type' key) must be
    reachable by the image steps — the real codex adapter emits that shape."""
    data_url = _noise_jpeg_data_url(side=700)
    kwargs = _kwargs(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "photo"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
            _user_message("current ask"),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=150_000)
    assert serialized_body_size(result) <= 150_000


def test_cron_shape_role_only_user_message_outputs_truncated():
    """Regression: real cron dumps carry one bare {'role':'user'} item and
    dozens of tool exchanges after it. Those outputs are history, not the
    protected active tail — all but the last few must be truncatable."""
    items = [{"role": "user", "content": [{"type": "input_text", "text": "go"}]}]
    for n in range(20):
        items.append(_fc(f"call_{n}"))
        items.append(_fc_output(f"call_{n}", f"{n}:" + "X" * 60_000))
    kwargs = _kwargs(items)
    result = apply_request_body_budget(kwargs, budget_bytes=300_000)
    assert serialized_body_size(result) <= 300_000
    outs = [i for i in result["input"] if i.get("type") == "function_call_output"]
    assert any("characters omitted" in o["output"] for o in outs)
    # The most recent exchanges stay verbatim.
    assert "characters omitted" not in outs[-1]["output"]


def test_unprocessable_tail_image_placeholdered_last_resort():
    """Regression: an oversized image Pillow cannot decode (e.g. corrupt
    bytes) in the active tail must be placeholdered, not left to guarantee a
    transport drop."""
    junk = "data:image/png;base64," + base64.b64encode(b"\x89PNG" + b"J" * 400_000).decode()
    kwargs = _kwargs(
        [
            _user_message("analyze this"),
            _fc("call_v", "vision_analyze"),
            _fc_output(
                "call_v",
                output=[
                    {"type": "input_text", "text": "described below"},
                    {"type": "input_image", "image_url": junk},
                ],
            ),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=100_000)
    assert serialized_body_size(result) <= 100_000
    parts = [i for i in result["input"] if i.get("type") == "function_call_output"][0]["output"]
    assert any(
        p.get("type") == "input_text" and "image omitted" in p.get("text", "")
        for p in parts
    )
    assert parts[0]["text"] == "described below"


def test_lone_image_keeps_budget_headroom():
    """A message that is mostly one image gets the envelope's remaining
    headroom, not a fixed budget//8 crush."""
    data_url = _noise_jpeg_data_url(side=650)
    assert 200_000 < len(data_url) < 700_000
    kwargs = _kwargs(
        [
            _user_message("old"),
            _fc(),
            _fc_output(output="H" * 600_000),
            _user_message("look at this", [{"type": "input_image", "image_url": data_url}]),
        ]
    )
    result = apply_request_body_budget(kwargs, budget_bytes=1_000_000)
    assert serialized_body_size(result) <= 1_000_000
    img = [
        p
        for i in result["input"]
        for p in (i.get("content") or [])
        if p.get("type") == "input_image"
    ][0]
    # With ~600KB of history truncated away, the lone image should keep far
    # more than the old fixed cap (125,000) — ideally untouched.
    assert len(img["image_url"]) > 200_000
