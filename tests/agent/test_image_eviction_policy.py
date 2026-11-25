"""Arithmetic contract of the shared send-path image-eviction policy (#113517).

Both outbound passes (``context_compressor.evict_stale_outbound_tool_images`` and
``anthropic_message_convert._evict_old_screenshots``) translate their message shapes into
``(carrier blocks newest-first, reserved)`` and call this one function; the shape tests in
``test_outbound_stale_vision.py`` / ``test_computer_use.py`` cover the translation, this file
covers the numbers once.
"""

from __future__ import annotations

import base64

import pytest

from agent.anthropic_message_convert import _evict_old_screenshots
from agent.context_compressor import evict_stale_outbound_tool_images
from agent.image_eviction_policy import (
    IMAGE_EVICTION_BATCH,
    OUTBOUND_IMAGE_BUDGET_BYTES,
    OUTBOUND_IMAGE_FLOOR,
    OUTBOUND_IMAGE_LIMIT,
    outbound_image_retire_count,
)
import itertools

MB = 1_000_000

# Anthropic's >20-images stricter per-side cap in px. Kept as a literal, not an import, so the
# two invariants below still run (and fail) against the pre-change policy module.
MANY_IMAGE_CAP_PX = 2000


def _png_data_url(width: int, height: int) -> str:
    """A data URL whose PNG IHDR declares ``width`` x ``height``.

    Only the 24 bytes a header-only reader inspects are needed, which is exactly the contract
    the policy relies on: classifying a request must not decode whole payloads.
    """
    ihdr = "IHDR".encode("ascii") + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    raw = b"\x89PNG\r\n\x1a\n" + len(ihdr).to_bytes(4, "big") + ihdr
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _screenshot_tool_messages(n: int, url: str) -> list[dict]:
    """``n`` OpenAI-shaped tool results, one image block each."""
    messages: list[dict] = []
    for i in range(n):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": [
                    {"type": "text", "text": f"shot {i}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        )
    return messages


def _wire_tool_results(n: int, url: str) -> list[dict]:
    """``n`` Anthropic wire user-turns, each carrying one image ``tool_result``."""
    data = url.split(",", 1)[1]
    result: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "start"}]}]
    for i in range(n):
        result.append({"role": "assistant", "content": [{"type": "text", "text": f"shot {i}"}]})
        result.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{i}",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": data},
                            }
                        ],
                    }
                ],
            }
        )
    return result


def _wire_image_blocks(result: list[dict]) -> int:
    return sum(
        1
        for msg in result
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        for inner in (block.get("content") if block.get("type") == "tool_result" else [block])
        if isinstance(inner, dict) and inner.get("type") == "image"
    )


@pytest.mark.parametrize(
    ("blocks", "reserved", "sizes", "reserved_bytes", "expected"),
    [
        # at the limit: append-only, nothing rewritten
        ([1] * OUTBOUND_IMAGE_LIMIT, 0, None, 0, 0),
        # one over: exactly one batch
        ([1] * (OUTBOUND_IMAGE_LIMIT + 1), 0, None, 0, IMAGE_EVICTION_BATCH),
        # still within the first window after a batch: same retire count (frontier holds)
        ([1] * (OUTBOUND_IMAGE_LIMIT + IMAGE_EVICTION_BATCH), 0, None, 0, IMAGE_EVICTION_BATCH),
        # second window: two batches, never a fixed single batch
        ([1] * (OUTBOUND_IMAGE_LIMIT + IMAGE_EVICTION_BATCH + 1), 0, None, 0, 2 * IMAGE_EVICTION_BATCH),
        # one carrier breaching alone: fixable, floor yields
        ([OUTBOUND_IMAGE_LIMIT + 5], 0, None, 0, 1),
        # four 10-block carriers: retire until it fits, not everything
        ([10, 10, 10, 10], 0, None, 0, 2),
        # uploads alone breach: unfixable, shelter the floor, retire only whole batches
        ([1] * 5, OUTBOUND_IMAGE_LIMIT + 1, None, 0, 0),
        ([1] * (OUTBOUND_IMAGE_FLOOR + IMAGE_EVICTION_BATCH), OUTBOUND_IMAGE_LIMIT + 1, None, 0, IMAGE_EVICTION_BATCH),
        # uploads alone breach with fewer carriers than the floor: nothing to gain
        ([1, 1], OUTBOUND_IMAGE_LIMIT + 5, None, 0, 0),
        # byte pressure from uploads: floor never shelters a 413
        ([1, 1, 1], 5, [3 * MB] * 3, 25 * MB, 3),
        # bytes bind before blocks: 2 MB frames, 13 of them
        ([1] * 13, 0, [2 * MB] * 13, 0, IMAGE_EVICTION_BATCH),
        # empty
        ([], 0, None, 0, 0),
    ],
)
def test_retire_count(blocks, reserved, sizes, reserved_bytes, expected):
    assert (
        outbound_image_retire_count(
            blocks, reserved, carrier_bytes_newest_first=sizes, reserved_bytes=reserved_bytes
        )
        == expected
    )


@pytest.mark.parametrize("n", [OUTBOUND_IMAGE_LIMIT + 1, 30])
def test_pre_shrunk_screenshots_do_not_trip_the_many_image_trigger(n):
    """Invariant 1 (#133999): a request whose images are all within the many-image per-side cap
    crosses no Anthropic constraint at image #21, so no prefix may be rewritten.

    Hermes embeds its own tool images at 1568 px (vision_analyze / browser) and 1456 px
    (computer_use), i.e. comfortably under the 2000 px cap the >20 rule would tighten. Retiring a
    batch here buys nothing and costs a full prompt-cache rewrite per batch -- plus the cache
    invalidation of every later thinking block on the preserved-thinking models. Red on base:
    8 rewrites at n=21, 16 at n=30.
    """
    shrunk = _png_data_url(1568, 882)
    assert evict_stale_outbound_tool_images(_screenshot_tool_messages(n, shrunk)) == 0

    # The wire pass reaches aux/MoA requests without the compressor's pass, so it must hold
    # the same invariant alone.
    wire = _wire_tool_results(n, shrunk)
    _evict_old_screenshots(wire)
    assert _wire_image_blocks(wire) == n


_UNREADABLE = "data:image/png;base64," + base64.b64encode(b"not an image header at all").decode("ascii")
_REMOTE = "https://example.com/shot.png"


@pytest.mark.parametrize("odd", ["oversized", "unreadable", "remote"])
def test_one_oversized_image_keeps_the_strict_trigger(odd):
    """Invariant 2 (#133999): the stricter cap is per request, so a single image over the cap --
    here a native-size user upload -- must keep the strict 20-block trigger for the whole request,
    byte-identical to the pre-#133999 behaviour. The ceiling relaxes only on dimensions the pass
    can read: an unreadable payload or a remote URL fails closed the same way."""
    shrunk = _png_data_url(1568, 882)
    url = {"oversized": _png_data_url(MANY_IMAGE_CAP_PX + 1, 8), "unreadable": _UNREADABLE, "remote": _REMOTE}[odd]

    upload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
    outbound = [upload, *_screenshot_tool_messages(OUTBOUND_IMAGE_LIMIT, shrunk)]
    assert evict_stale_outbound_tool_images(outbound) == IMAGE_EVICTION_BATCH
    # The upload itself is reserved and never rewritten.
    assert upload["content"][1]["image_url"]["url"] == url

    # Same request on the wire shape: 20 pre-shrunk carriers plus the odd reserved upload.
    wire = _wire_tool_results(OUTBOUND_IMAGE_LIMIT, shrunk)
    source = ({"type": "url", "url": url} if odd == "remote"
              else {"type": "base64", "media_type": "image/png", "data": url.split(",", 1)[1]})
    wire[0]["content"].append({"type": "image", "source": source})
    _evict_old_screenshots(wire)
    assert _wire_image_blocks(wire) == OUTBOUND_IMAGE_LIMIT - IMAGE_EVICTION_BATCH + 1


def test_wire_pass_enforces_the_byte_budget_below_the_block_ceiling():
    """The aux/MoA Anthropic path reaches the wire pass alone, and that pass used to have no
    request-size guard: 10 image blocks is half the block ceiling, but 30 MB of base64 is over
    Anthropic's 32 MB Messages limit once text and JSON framing are added. Byte pressure must
    retire carriers even while the block count is comfortably fine."""
    big = "Q" * (3 * MB)  # ~2.2 MB decoded per image, unreadable header -> strict trigger
    result: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": big}}
            ],
        }
    ]
    for i in range(9):
        result.append({"role": "assistant", "content": [{"type": "text", "text": "s"}]})
        result.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{i}",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": big}}
                        ],
                    }
                ],
            }
        )
    blocks_before = _wire_image_blocks(result)
    assert 10 * len(big) > OUTBOUND_IMAGE_BUDGET_BYTES and blocks_before < OUTBOUND_IMAGE_LIMIT
    _evict_old_screenshots(result)
    assert _wire_image_blocks(result) < blocks_before, (
        "the wire pass ignored byte pressure while under the block ceiling"
    )


def test_quantum_shrinks_to_the_fit_window_for_heavy_carriers():
    """Three-block carriers: a batch of eight is wider than the window, so the retire count
    must still be a step function (hold for window - floor turns), never ``total - floor``."""
    window = OUTBOUND_IMAGE_LIMIT // 3
    counts = [outbound_image_retire_count([3] * n, 0) for n in range(window + 1, window + 10)]
    assert all(c > 0 for c in counts)
    assert all(3 * (n - c) <= OUTBOUND_IMAGE_LIMIT for n, c in zip(range(window + 1, window + 10), counts))
    moves = sum(a != b for a, b in itertools.pairwise(counts))
    assert moves <= len(counts) // (window - OUTBOUND_IMAGE_FLOOR)


def test_unfixable_upload_breach_still_retires_in_batches():
    """#133999: when reserved uploads alone exceed the block ceiling, retiring tool images
    cannot make the request fit, so the retire count must still be a step function of the
    carrier count. Retiring to exactly ``total - floor`` would rewrite the cached prefix on
    every new tool image."""
    uploads = 51
    span = range(1, 31)
    counts = [outbound_image_retire_count([1] * n, uploads) for n in span]
    kept = [n - c for n, c in zip(span, counts)]
    assert all(k >= min(n, OUTBOUND_IMAGE_FLOOR) for n, k in zip(span, kept))
    assert all(k < OUTBOUND_IMAGE_FLOOR + IMAGE_EVICTION_BATCH for k in kept)
    moves = sum(a != b for a, b in itertools.pairwise(counts))
    assert moves <= -(-(len(span) - OUTBOUND_IMAGE_FLOOR) // IMAGE_EVICTION_BATCH)


def test_unfixable_upload_breach_under_byte_pressure_fits_and_still_steps():
    """The hold never keeps a request over the byte budget, and byte pressure narrows the
    step instead of collapsing to the floor (which would move the frontier every image)."""
    span = range(1, 31)
    counts = [
        outbound_image_retire_count(
            [1] * n, OUTBOUND_IMAGE_LIMIT + 1, carrier_bytes_newest_first=[3 * MB] * n, reserved_bytes=6 * MB
        )
        for n in span
    ]
    kept = [n - c for n, c in zip(span, counts)]
    assert all(6 * MB + 3 * MB * k <= 24 * MB for k in kept)
    assert all(k >= min(n, OUTBOUND_IMAGE_FLOOR) for n, k in zip(span, kept))
    # bytes allow 6 carriers, floor is 3 -> a 4-carrier quantum, so at most one move per 4 images
    moves = sum(a != b for a, b in itertools.pairwise(counts))
    assert moves <= -(-(len(span) - OUTBOUND_IMAGE_FLOOR) // 4)


def _mixed_tool_history(sizes: list[int], uploads: int) -> list[dict]:
    """Uploads (unreadable headers -> strict ceiling) plus one screenshot tool round per size."""
    rows: list[dict] = [
        {"role": "user", "content": [{"type": "text", "text": "look"}]
         + [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "U" * 285_000}}] * uploads}
    ]
    for i, size in enumerate(sizes):
        rows.append({"role": "assistant", "content": "shot", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]})
        rows.append({"role": "tool", "tool_call_id": f"c{i}", "content": [
            {"type": "text", "text": f"shot {i}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "Q" * size}}]})
    return rows


def test_a_small_new_screenshot_never_restores_retired_ones():
    """#133999 review: appending a small screenshot after large ones widened the byte-derived
    quantum, so the retire count fell (8 -> 5) and three retired screenshots came back inside
    already-sent history. Retirement must only grow as the conversation does, on both passes."""
    from agent.anthropic_adapter import build_anthropic_kwargs

    def retired(sizes: list[int]) -> tuple[set[str], set[str]]:
        rows = _mixed_tool_history(sizes, uploads=OUTBOUND_IMAGE_LIMIT + 1)
        outbound = [dict(m) for m in rows]
        evict_stale_outbound_tool_images(outbound)
        kept = {m["tool_call_id"] for m in outbound if m["role"] == "tool"
                and any(p.get("type") == "image_url" for p in m["content"])}
        wire = build_anthropic_kwargs(model="claude-sonnet-4-5", messages=rows, tools=None,
                                      max_tokens=None, reasoning_config=None)["messages"]
        wire_kept = {b["tool_use_id"] for m in wire if isinstance(m["content"], list) for b in m["content"]
                     if b.get("type") == "tool_result" and isinstance(b.get("content"), list)
                     and any(x.get("type") == "image" for x in b["content"])}
        ids = {f"c{i}" for i in range(len(sizes))}
        return ids - kept, ids - wire_kept

    previous = (set(), set())
    for extra in range(4):
        now = retired([2_900_000] * 11 + [100_000] * extra)
        assert now[0] >= previous[0] and now[1] >= previous[1], (extra, previous, now)
        previous = now
    assert len(previous[0]) >= IMAGE_EVICTION_BATCH


def test_retirement_is_latched_when_an_upload_leaves_the_request(tmp_path, monkeypatch):
    """#133999 x #136216 review: once the replay window drops an oversized upload, the request
    relaxes to the 100-image ceiling and a from-scratch decision restored all eight retired
    screenshots into already-sent history. Driven through the agent's request builder."""
    from agent.chat_completion_helpers import _iteration_summary_api_messages
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(api_key="k", base_url="https://api.anthropic.com", provider="anthropic",
                    model="claude-sonnet-4-5", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "SYS"
    shrunk, oversized = _png_data_url(1568, 882), _png_data_url(MANY_IMAGE_CAP_PX + 400, 8)
    tools = [m for i in range(OUTBOUND_IMAGE_LIMIT) for m in (
        {"role": "assistant", "content": "shot", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": [
            {"type": "text", "text": f"shot {i}"}, {"type": "image_url", "image_url": {"url": shrunk}}]})]

    def kept(upload_url: str | None) -> list[str]:
        upload = [{"type": "text", "text": "photo"}]
        if upload_url:
            upload.append({"type": "image_url", "image_url": {"url": upload_url}})
        history = [{"role": "user", "content": upload}, *tools, {"role": "user", "content": "next"}]
        return [m["tool_call_id"] for m in _iteration_summary_api_messages(agent, history)
                if m.get("role") == "tool" and any(p.get("type") == "image_url" for p in m["content"])]

    before = kept(oversized)
    assert len(before) == OUTBOUND_IMAGE_LIMIT - IMAGE_EVICTION_BATCH
    assert kept(None) == before
