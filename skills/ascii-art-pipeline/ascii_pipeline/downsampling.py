"""Downsampling strategies for high-resolution ASCII to target grid.

Currently implements:
- majority_vote: current baseline (mode per block)
- edge_aware: Laplacian-weighted; preserves edges via median-luminance fallback
"""

from typing import List
import numpy as np
from collections import Counter

# D30 extended charset ordered by visual density (dark → light)
D30_CHARSET = ".'`^\\\",:;Il!i><~+_-?][}{1)|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
D30_LUT = {c: i / len(D30_CHARSET) for i, c in enumerate(D30_CHARSET)}


def majority_vote(high_ascii: List[str], factor: int) -> List[str]:
    """Simple block-mode majority vote (current baseline)."""
    if factor <= 0:
        raise ValueError("factor must be positive integer")
    h = len(high_ascii)
    if h == 0:
        return []
    w = len(high_ascii[0])
    if any(len(row) != w for row in high_ascii):
        raise ValueError("All rows must have equal length")
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"High-res dimensions ({w}×{h}) not exact multiples of factor={factor}. "
            f"Remainders: w%factor={w % factor}, h%factor={h % factor}."
        )

    th, tw = h // factor, w // factor
    out = []
    for y in range(th):
        row = []
        for x in range(tw):
            block = []
            for dy in range(factor):
                ly = y * factor + dy
                line = high_ascii[ly]
                for dx in range(factor):
                    lx = x * factor + dx
                    block.append(line[lx])
            row.append(Counter(block).most_common(1)[0][0])
        out.append(''.join(row))
    return out


def edge_aware_downsample(high_ascii: List[str], factor: int,
                          edge_threshold: float = 0.15) -> List[str]:
    """Block-mode downsampling with edge-aware median fallback.

    For each factor×factor block:
      - Compute local gradient magnitude from the high-res luminance map
      - If gradient > edge_threshold: pick median-luminance glyph (preserves contrast)
      - Else: pick mode glyph (preserves texture)

    Args:
        high_ascii: High-resolution ASCII grid (exact multiple of factor)
        factor: Integer downsampling factor (e.g., 12 for 576×288 → 48×24)
        edge_threshold: Gradient magnitude threshold in normalized lum space [0,1].
                        Recommended: 0.12–0.18. Lower = more edge regions.

    Returns:
        Downsampled ASCII lines.
    """
    if factor <= 0:
        raise ValueError("factor must be positive integer")
    h = len(high_ascii)
    if h == 0:
        return []
    w = len(high_ascii[0])
    if any(len(row) != w for row in high_ascii):
        raise ValueError("All rows must have equal length")
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"High-res dimensions ({w}×{h}) not exact multiples of factor={factor}. "
            f"Remainders: w%factor={w % factor}, h%factor={h % factor}."
        )

    # Convert to luminance float array for gradient computation
    lum = np.array([[D30_LUT.get(c, 0.5) for c in row] for row in high_ascii], dtype=np.float32)
    th, tw = h // factor, w // factor
    out = []

    for y in range(th):
        row_chars = []
        y0 = y * factor
        for x in range(tw):
            x0 = x * factor
            # Collect block glyphs and their luminances
            block_chars = []
            block_lums = []
            for dy in range(factor):
                ly = y0 + dy
                line = high_ascii[ly]
                lum_line = lum[ly]
                for dx in range(factor):
                    lx = x0 + dx
                    ch = line[lx]
                    block_chars.append(ch)
                    block_lums.append(lum_line[lx])

            # Estimate edge strength using center-patch gradients
            cy = y0 + factor // 2
            cx = x0 + factor // 2
            if cy + 1 < h and cx + 1 < w and cy - 1 >= 0 and cx - 1 >= 0:
                gx = abs(lum[cy, cx + 1] - lum[cy, cx - 1])
                gy = abs(lum[cy + 1, cx] - lum[cy - 1, cx])
                edge_strength = np.sqrt(gx * gx + gy * gy)
            else:
                edge_strength = 0.0

            if edge_strength > edge_threshold:
                # Median luminance preserves edge contrast without Mode's bias
                median_lum = sorted(block_lums)[len(block_lums) // 2]
                best = min(block_chars, key=lambda c: abs(D30_LUT.get(c, 0.5) - median_lum))
            else:
                # Mode is fine for smooth regions
                best = Counter(block_chars).most_common(1)[0][0]
            row_chars.append(best)
        out.append(''.join(row_chars))
    return out


def clahe_downsample(high_ascii: List[str], factor: int,
                     clip_limit: float = 2.0) -> List[str]:
    """Contrast-limited adaptive histogram equalization + area average.
    Approximate CLAHE via global percentile stretch (fast) then LANCZOS resize.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    h = len(high_ascii)
    if h == 0:
        return []
    w = len(high_ascii[0])
    if any(len(r) != w for r in high_ascii):
        raise ValueError("All rows must have equal length")
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"High-res dimensions ({w}×{h}) not exact multiples of factor={factor}.")

    # Map to luminance using D30 LUT
    lum = np.array([[D30_LUT.get(c, 0.5) for c in row] for row in high_ascii], dtype=np.float32)

    # Percentile-based contrast stretch (simplified CLAHE)
    p_low, p_high = 2, 98
    lo, hi = np.percentile(lum, (p_low, p_high))
    lum = np.clip((lum - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # PIL LANCZOS resize to target
    small_img = Image.fromarray((lum * 255).astype(np.uint8)).resize(
        (w // factor, h // factor), Image.Resampling.LANCZOS
    )
    small = np.array(small_img, dtype=np.float32) / 255.0

    # Quantize back to D30 charset (nearest neighbor in LUT)
    lut_vals = np.array(list(D30_LUT.values()))
    indices = np.abs(small[..., None] - lut_vals).argmin(axis=2)
    return [''.join(D30_CHARSET[i] for i in row) for row in indices]


def area_average_downsample(high_ascii: List[str], factor: int) -> List[str]:
    """Pure area-average (LANCZOS-style) block downsampling without edge awareness."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    h = len(high_ascii)
    if h == 0:
        return []
    w = len(high_ascii[0])
    if any(len(r) != w for r in high_ascii):
        raise ValueError("All rows must have equal length")
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Dim mismatch: {w}×{h} not divisible by {factor}")

    th, tw = h // factor, w // factor
    out = []
    for y in range(th):
        row = []
        for x in range(tw):
            block = []
            for dy in range(factor):
                for dx in range(factor):
                    block.append(D30_LUT.get(high_ascii[y*factor+dy][x*factor+dx], 0.5))
            avg = sum(block) / len(block)
            best = min(D30_CHARSET, key=lambda c: abs(D30_LUT.get(c, 0.5) - avg))
            row.append(best)
        out.append(''.join(row))
    return out
