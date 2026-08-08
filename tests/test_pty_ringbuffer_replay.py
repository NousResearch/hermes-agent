"""Real regression tests for the RingBuffer scrollback mechanism.

The 16 MB buffer cap is configured by PTY_BUFFER_CAP_BYTES in pty_session.py
(or web_server.py). These tests verify the buffer overflow behaviour: that when
more data than the cap is appended, the oldest bytes are dropped from the front
and the snapshot matches exactly the retained tail.

This is Priority 2 from PR #64664: the buffer mechanism the scrollback replay
depends on must have asserting tests.
"""

from __future__ import annotations

from hermes_cli.pty_session import RingBuffer


# ---------------------------------------------------------------------------
# Exact overflow trimming
# ---------------------------------------------------------------------------

def test_ringbuffer_exact_capacity_is_not_truncated():
    """Why: data that fits exactly must not trigger truncation.
    What: appends exactly capacity bytes; asserts snapshot == data, truncated=False.
    Test: snapshot length equals capacity, truncated is False.
    """
    rb = RingBuffer(8)
    rb.append(b"12345678")

    assert rb.snapshot() == b"12345678"
    assert rb.truncated is False


def test_ringbuffer_one_byte_over_capacity_trims_front():
    """Why: a single overflow byte must drop exactly one byte from the front.
    What: appends cap+1 bytes in one call.
    Test: snapshot == last cap bytes; truncated is True.
    """
    rb = RingBuffer(4)
    rb.append(b"ABCDE")   # 5 bytes into 4-byte cap

    assert rb.snapshot() == b"BCDE", f"Got: {rb.snapshot()!r}"
    assert rb.truncated is True


def test_ringbuffer_double_overflow_retains_newest_tail():
    """Why: multiple overflowing appends must always keep the newest bytes.
    What: fills buffer twice over capacity in separate appends.
    Test: snapshot matches the tail of the concatenated input.
    """
    rb = RingBuffer(4)
    rb.append(b"AAAA")   # exactly fills: AAAA
    rb.append(b"BBBB")   # overflow by 4: BBBB

    assert rb.snapshot() == b"BBBB", f"Got: {rb.snapshot()!r}"
    assert rb.truncated is True


def test_ringbuffer_incremental_appends_trim_from_front():
    """Why: real PTY output arrives in many small chunks; the ring buffer must
    handle each incrementally, always keeping the tail.
    What: appends single bytes until cap is reached, then one more.
    Test: snapshot == last cap bytes of full sequence.
    """
    rb = RingBuffer(3)
    for byte in b"ABCDE":
        rb.append(bytes([byte]))

    # Full sequence: A B C D E; last 3 = D E ... wait, let's be precise.
    # After each:
    #   A → buf="A"    (1<=3)
    #   B → buf="AB"   (2<=3)
    #   C → buf="ABC"  (3==3)
    #   D → buf="BCD"  (trim A)
    #   E → buf="CDE"  (trim B)
    assert rb.snapshot() == b"CDE", f"Got: {rb.snapshot()!r}"
    assert rb.truncated is True


def test_ringbuffer_snapshot_returns_bytes_not_bytearray():
    """Why: callers pass snapshot() directly to WebSocket.send_bytes; it must
    be bytes, not bytearray (which some transports reject).
    What: appends data; asserts snapshot type is bytes.
    Test: isinstance(rb.snapshot(), bytes) is True.
    """
    rb = RingBuffer(10)
    rb.append(b"hello")

    result = rb.snapshot()
    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"


def test_ringbuffer_zero_capacity_discards_all():
    """Why: edge case — a 0-byte buffer should silently drop everything.
    What: creates RingBuffer(0); appends data; asserts snapshot is empty.
    Test: snapshot == b""; truncated is True.
    """
    rb = RingBuffer(0)
    rb.append(b"something")

    assert rb.snapshot() == b"", f"Got: {rb.snapshot()!r}"
    assert rb.truncated is True


def test_ringbuffer_large_single_chunk_over_cap():
    """Why: a single large write (e.g. TUI initial render) must trim to cap.
    What: appends 1000 bytes to a 100-byte buffer.
    Test: snapshot length == 100; snapshot == last 100 bytes of input.
    """
    data = bytes(range(256)) * 4  # 1024 bytes
    cap = 100
    rb = RingBuffer(cap)
    rb.append(data)

    expected = data[-cap:]
    assert len(rb.snapshot()) == cap
    assert rb.snapshot() == expected, (
        f"Expected last {cap} bytes; got {rb.snapshot()!r}"
    )
    assert rb.truncated is True


def test_ringbuffer_fresh_buffer_snapshot_is_empty():
    """Why: a fresh RingBuffer must replay nothing (no stale data).
    What: creates buffer; calls snapshot before any appends.
    Test: snapshot == b""; truncated is False.
    """
    rb = RingBuffer(1024)

    assert rb.snapshot() == b""
    assert rb.truncated is False


# ---------------------------------------------------------------------------
# RED-GREEN seam: verify that snapshot content drives reconnect replay
#
# The replay contract: when a new WebSocket attaches to an existing PtySession
# the buffer snapshot is sent as the first message. These tests verify the
# snapshot content (and therefore the replay) is exactly the retained tail.
# ---------------------------------------------------------------------------

def test_ringbuffer_replay_content_matches_retained_tail():
    """Why: core contract — what the client sees on reconnect must be exactly
    the tail bytes that fit in the buffer. This is the seam PR #64664 protects.
    What: fills buffer beyond cap; records what snapshot() returns; asserts
          it equals the tail of the original stream.
    Test: snapshot == expected_tail.
    """
    cap = 8
    rb = RingBuffer(cap)

    # Simulate two TUI render frames
    frame1 = b"FRAME_ONE"     # 9 bytes
    frame2 = b"FRAME_TWO"     # 9 bytes
    rb.append(frame1)
    rb.append(frame2)

    # Combined: FRAME_ONEFRAME_TWO (18 bytes); last 8 = "AME_TWO" + ... let's calculate
    combined = frame1 + frame2  # 18 bytes
    expected_tail = combined[-cap:]

    assert rb.snapshot() == expected_tail, (
        f"Expected {expected_tail!r}, got {rb.snapshot()!r}"
    )
