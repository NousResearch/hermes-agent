"""The outbound request copy budgets every data-URL image part before it is sent (#135497).

``try_shrink_image_parts_in_messages`` only runs from the image_too_large recovery branch,
which needs the provider to REJECT the request. A gateway that answers an over-window
request with HTTP 200 and the error text as assistant content never reaches that branch, so
one oversized capture stays in history and is re-sent on every turn until the window wedges.
"""
from __future__ import annotations

import base64
import io
import random

import pytest

from agent.turn_api_request import _shrink_oversized_request_images

PIL = pytest.importorskip("PIL.Image")


def _noise_png_b64(size=(1800, 1400)) -> str:
    rnd = random.Random(7)
    img = PIL.new("RGB", size)
    img.putdata([
        (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
        for _ in range(size[0] * size[1])
    ])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _flat_png_b64(size=(40, 40)) -> str:
    buf = io.BytesIO()
    PIL.new("RGB", size, (200, 40, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _native_part(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}


def _messages(part: dict) -> list:
    return [{"role": "user", "content": [{"type": "text", "text": "look"}, part]}]


def _long_edge(data_url: str) -> int:
    payload = data_url.partition(",")[2]
    with PIL.open(io.BytesIO(base64.b64decode(payload))) as img:
        return max(img.size)


def test_oversized_native_part_is_reencoded_smaller():
    part = _native_part(_noise_png_b64())
    before = len(part["source"]["data"])
    messages = _messages(part)

    assert _shrink_oversized_request_images(messages) is True

    rewritten = messages[0]["content"][1]
    after = len(rewritten["source"]["data"])
    assert after < before
    data_url = f"data:{rewritten['source']['media_type']};base64,{rewritten['source']['data']}"
    assert _long_edge(data_url) <= 1568


def test_canonical_history_part_keeps_full_resolution_bytes():
    """Only the per-attempt copy is budgeted; the stored history part must not change."""
    part = _native_part(_noise_png_b64())
    original_data = part["source"]["data"]
    messages = _messages(part)

    _shrink_oversized_request_images(messages)

    assert part["source"]["data"] == original_data
    assert messages[0]["content"][1]["source"]["data"] != original_data


def test_http_url_image_is_left_alone():
    part = {"type": "image_url", "image_url": {"url": "https://example.invalid/shot.png"}}
    assert _shrink_oversized_request_images(_messages(part)) is False
    assert part["image_url"]["url"] == "https://example.invalid/shot.png"


def test_small_image_is_not_reencoded():
    part = _native_part(_flat_png_b64())
    original_data = part["source"]["data"]

    assert _shrink_oversized_request_images(_messages(part)) is False
    assert part["source"]["data"] == original_data


def test_non_image_content_is_ignored():
    messages = [{"role": "user", "content": [{"type": "text", "text": "no pictures here"}]}]
    assert _shrink_oversized_request_images(messages) is False


def test_text_only_messages_are_a_no_op():
    assert _shrink_oversized_request_images([{"role": "user", "content": "hi"}]) is False
    assert _shrink_oversized_request_images([]) is False
