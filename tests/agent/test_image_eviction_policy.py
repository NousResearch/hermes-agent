"""Arithmetic contract of the shared send-path image-eviction policy (#113517).

Both outbound passes (``context_compressor.evict_stale_outbound_tool_images`` and
``anthropic_message_convert._evict_old_screenshots``) translate their message shapes into
``(carrier blocks newest-first, reserved)`` and call this one function; the shape tests in
``test_outbound_stale_vision.py`` / ``test_computer_use.py`` cover the translation, this file
covers the numbers once.
"""

from __future__ import annotations

import pytest

from agent.image_eviction_policy import (
    IMAGE_EVICTION_BATCH,
    OUTBOUND_IMAGE_FLOOR,
    OUTBOUND_IMAGE_LIMIT,
    outbound_image_retire_count,
)

MB = 1_000_000


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
        # uploads alone breach: unfixable, keep the floor
        ([1] * 5, OUTBOUND_IMAGE_LIMIT + 1, None, 0, 5 - OUTBOUND_IMAGE_FLOOR),
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


def test_quantum_shrinks_to_the_fit_window_for_heavy_carriers():
    """Three-block carriers: a batch of eight is wider than the window, so the retire count
    must still be a step function (hold for window - floor turns), never ``total - floor``."""
    window = OUTBOUND_IMAGE_LIMIT // 3
    counts = [outbound_image_retire_count([3] * n, 0) for n in range(window + 1, window + 10)]
    assert all(c > 0 for c in counts)
    assert all(3 * (n - c) <= OUTBOUND_IMAGE_LIMIT for n, c in zip(range(window + 1, window + 10), counts))
    moves = sum(a != b for a, b in zip(counts, counts[1:]))
    assert moves <= len(counts) // (window - OUTBOUND_IMAGE_FLOOR)
