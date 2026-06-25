#!/usr/bin/env python3
"""Runnable proof for issue #48013 — image-dimension 400 session brick.

Before the fix, ``agent.conversation_compression.try_shrink_image_parts_in_messages``
silently discarded a *pixel-correct* downscale whenever the re-encoded PNG was
larger in bytes than the original (the common case for downscaled Retina
screenshots). The image was left at its original oversized dimensions, the
provider re-rejected it on retry, and the session wedged forever on the
Anthropic many-image 2000px path.

This script reproduces the exact scenario with REAL Pillow (no mocks): it
synthesizes screenshot-like PNGs at the dimensions from the issue's table —
images that are small in bytes (under the 4 MB budget) but over the 2000px
per-side cap — and runs the real recovery helper. It asserts every image is
brought under the cap and that the helper reports success.

Run directly to see a human-readable report:

    python tests/run_agent/repro_48013_image_shrink_brick.py

Or as a pytest smoke test (skipped automatically when Pillow is absent):

    scripts/run_tests.sh tests/run_agent/repro_48013_image_shrink_brick.py
"""

from __future__ import annotations

import base64
import io

import pytest


PIL = pytest.importorskip("PIL", reason="Pillow required for the real-resize proof")
from PIL import Image, ImageDraw  # noqa: E402

from agent.conversation_compression import (  # noqa: E402
    try_shrink_image_parts_in_messages,
)

# The many-image per-side cap Anthropic reported in the wild (issue #48013).
MANY_IMAGE_CAP = 2000
BYTE_BUDGET = 4 * 1024 * 1024

# Dimensions straight from the issue's per-image table. The "REJECTED" rows
# are the ones that bricked: tall/large screenshots whose downscale re-encodes
# to MORE PNG bytes than the original.
CASES = [
    (2344, 778),   # wide — shrank even before the fix
    (2374, 1144),  # wide — shrank even before the fix
    (2097, 1476),  # REJECTED before fix
    (2247, 1544),  # REJECTED before fix
    (2263, 1644),  # REJECTED before fix
]


def _make_screenshot_png(width: int, height: int) -> bytes:
    """A screenshot-like PNG: mostly flat UI regions so it compresses small.

    Flat regions keep the byte size well under the 4 MB budget, forcing the
    DIMENSION path (not the byte path) — exactly the code that bricked. The
    downscale of such an image re-encodes to a comparable-or-larger PNG, which
    is what the old byte gate wrongly rejected.
    """
    img = Image.new("RGB", (width, height), (245, 245, 247))
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 40):
        shade = 255 - (y // 40) % 6 * 4
        draw.rectangle([20, y + 5, width - 20, y + 30], fill=(shade, 250, 250))
    for x in range(0, width, 160):
        draw.rectangle([x, 0, x + 2, height], fill=(220, 220, 225))
    draw.text((40, 40), "Some UI text " * 30, fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _decode_dims(data_url: str) -> tuple[int, int]:
    payload = data_url.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as img:
        return img.size


def run_proof(verbose: bool = False) -> list[dict]:
    """Run the recovery against every case; return per-case results."""
    results: list[dict] = []
    for width, height in CASES:
        raw = _make_screenshot_png(width, height)
        url = _data_url(raw)
        # Sanity: this case must be UNDER the byte budget and OVER the pixel cap,
        # i.e. it exercises the dimension path that bricked.
        under_byte_budget = len(url) <= BYTE_BUDGET
        over_pixel_cap = max(width, height) > MANY_IMAGE_CAP

        msgs = [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}],
        }]
        changed = try_shrink_image_parts_in_messages(
            msgs, max_dimension=MANY_IMAGE_CAP,
        )
        out_url = msgs[0]["content"][0]["image_url"]["url"]
        out_dims = _decode_dims(out_url)

        result = {
            "orig": (width, height),
            "orig_bytes": len(raw),
            "under_byte_budget": under_byte_budget,
            "over_pixel_cap": over_pixel_cap,
            "changed": changed,
            "result_dims": out_dims,
            "under_cap_after": max(out_dims) <= MANY_IMAGE_CAP,
        }
        results.append(result)
        if verbose:
            status = "OK" if result["under_cap_after"] else "BRICK"
            print(
                f"  {width}x{height} ({len(raw)//1024:>3} KB)"
                f" -> changed={changed!s:>5}"
                f"  result={out_dims[0]}x{out_dims[1]}"
                f"  [{status}]"
            )
    return results


def test_issue_48013_dimension_shrink_does_not_brick():
    """Every dimension-oversized screenshot must be brought under the cap."""
    results = run_proof()
    assert results, "no cases ran"
    for r in results:
        # Precondition: we really are on the dimension path.
        assert r["under_byte_budget"], (
            f"{r['orig']} must be under the byte budget to exercise the bug"
        )
        assert r["over_pixel_cap"], f"{r['orig']} must exceed the pixel cap"
        # The fix: image lands under the cap and the helper reports success.
        assert r["under_cap_after"], (
            f"BRICK: {r['orig']} left at {r['result_dims']} "
            f"(> {MANY_IMAGE_CAP}px) — the shrink recovery discarded a "
            f"pixel-correct downscale (#48013)"
        )
        assert r["changed"] is True, (
            f"{r['orig']} shrank but helper reported no progress — caller "
            f"would surface the original error and burn the one-shot retry"
        )


def main() -> int:
    print("Issue #48013 proof — image-dimension shrink must not brick sessions")
    print(f"(many-image per-side cap = {MANY_IMAGE_CAP}px, byte budget = "
          f"{BYTE_BUDGET // (1024 * 1024)} MB)\n")
    results = run_proof(verbose=True)
    bricked = [r for r in results if not r["under_cap_after"]]
    no_progress = [r for r in results if r["under_cap_after"] and not r["changed"]]
    print()
    if bricked:
        print(f"FAIL: {len(bricked)} image(s) still over the pixel cap (BRICK).")
        return 1
    if no_progress:
        print(f"FAIL: {len(no_progress)} image(s) shrank but helper reported "
              f"no progress (would burn the retry).")
        return 1
    print(f"PASS: all {len(results)} dimension-oversized screenshots brought "
          f"under {MANY_IMAGE_CAP}px and reported as progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
