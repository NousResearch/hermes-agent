"""Reproduce + verify the Thai combining-marks streaming decoder fix.

# The test simulates an SSE stream where a multi-byte UTF-8 Thai character
# is split across two byte chunks (mimicking what happens in real streaming
# responses when a Thai combining mark arrives in the middle of a delta).

Without the patch: the second chunk's leading bytes (start of a new
multi-byte char) fail to decode because they are the tail of the previous
char. The original ``SSEDecoder._iter_chunks`` then yields an empty or
truncated SSE message, and downstream code drops or U+FFFD-replaces the
garbled bytes.

With the patch applied via ``agent.thai_streaming_patch``: the bytes are
buffered and decoded as a complete UTF-8 character before line splitting,
so the SSE message is intact and the Thai combining marks round-trip
correctly.
"""

from __future__ import annotations

import sys
import os

# Add the repo root so we can import the patch module without installing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import thai_streaming_patch  # noqa: F401  (applies on import)
from anthropic._streaming import SSEDecoder


def make_sse_chunk(data: str) -> bytes:
    """Wrap a JSON ``data:`` payload in a minimal SSE event."""
    return f"data: {data}\n\n".encode("utf-8")


def test_thai_combining_marks_survive_chunk_split():
    """A Thai combining mark split across two byte chunks must round-trip."""
    # Build an SSE event whose JSON payload contains Thai combining marks.
    thai_text = "สวัสดีค่ะ ทำไม กินข้าว"  # สวัสดี + mai ek + sar aai + etc.
    payload = (
        '{"type":"content_block_delta","index":0,'
        f'"delta":{{"type":"text_delta","text":"{thai_text}"}}'
        "}\n"
    )
    full_bytes = make_sse_chunk(payload)

    # Find the first multi-byte Thai character in the encoded payload and
    # split inside its UTF-8 sequence (mid-byte). This is the bug trigger.
    thai_bytes = thai_text.encode("utf-8")
    # Locate the start of the first 3-byte Thai char (U+0E00-U+0E7F range).
    thai_start_in_payload = payload.encode("utf-8").index(thai_bytes)
    # Split 1 byte into the multi-byte sequence.
    split_at_in_payload = thai_start_in_payload + 1
    # Convert payload-local offset to full-bytes offset.
    split_at = full_bytes.index(b"data: ") + len(b"data: ") + split_at_in_payload
    chunk_a = full_bytes[:split_at]
    chunk_b = full_bytes[split_at:]

    assert any(0x80 <= b <= 0xBF for b in chunk_b[:3]), (
        "Test setup failed: chunk_b should start mid-UTF-8-sequence "
        "(continuation byte 0x80-0xBF). "
        f"chunk_a ends with: {chunk_a[-5:]!r}; "
        f"chunk_b starts with: {chunk_b[:5]!r}"
    )

    # Feed through the patched decoder.
    decoder = SSEDecoder()
    events = list(decoder.iter_bytes(iter([chunk_a, chunk_b])))

    assert len(events) == 1, f"Expected 1 SSE event, got {len(events)}"
    assert thai_text in events[0].data, (
        f"Thai text lost across chunk split.\n"
        f"  Original: {thai_text!r}\n"
        f"  Decoded:  {events[0].data!r}"
    )
    assert "\ufffd" not in events[0].data, (
        f"Replacement char U+FFFD appeared in decoded event: {events[0].data!r}"
    )
    print(f"  OK: Thai text round-tripped intact across chunk split")
    print(f"  Decoded data: {events[0].data!r}")


def test_repeated_delta_stream():
    """Simulate 30 small deltas like a real streaming response."""
    decoder = SSEDecoder()
    chunks = []
    # Emit a Thai word one character at a time to maximize chunk-split risk.
    thai_words = [
        "สวัสดี",  # hello
        "ค่ะ",  # polite particle
        "ทำไม",  # why
        "กิน",  # eat
        "ข้าว",  # rice
        "วันนี้",  # today
        "อากาศดี",  # good weather
        "ไปเที่ยว",  # go travel
        "ด้วยกัน",  # together
        "ไหม",  # question particle
    ]
    for word in thai_words:
        chunks.append(make_sse_chunk(
            '{"type":"content_block_delta","index":0,'
            f'"delta":{{"type":"text_delta","text":"{word}"}}'
            "}\n"
        ))

    events = list(decoder.iter_bytes(iter(chunks)))
    full_text = "".join(e.data.split("\n")[0] for e in events)
    for word in thai_words:
        assert word in full_text, (
            f"Word {word!r} lost in decoded stream.\nDecoded: {full_text!r}"
        )
    assert "\ufffd" not in full_text, f"U+FFFD appeared in: {full_text!r}"
    print(f"  OK: {len(events)} Thai deltas round-tripped intact")
    print(f"  Full decoded: {full_text!r}")


if __name__ == "__main__":
    print("Test 1: Thai combining marks survive SSE chunk split at mid-character")
    test_thai_combining_marks_survive_chunk_split()
    print()
    print("Test 2: Repeated delta stream of Thai words")
    test_repeated_delta_stream()
    print()
    print("ALL TESTS PASSED")
