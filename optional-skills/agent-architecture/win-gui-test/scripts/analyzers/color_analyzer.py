"""Color analysis — find dominant / accent / background colors in a region."""

import logging
from typing import Any

import numpy as np

log = logging.getLogger("win-gui-test.analyzers.color")

ColorInfo = dict[str, Any]


def dominant_colors(image: np.ndarray, n: int = 5) -> list[dict]:
    """Return the *n* most frequent colours in *image* (BGR array).

    Returns list of ``{"color": [B,G,R], "hex": "#rrggbb", "pct": float}``.
    """
    pixels = image.reshape(-1, 3)
    # quantize to reduce noise
    quantized = (pixels // 16) * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    total = counts.sum()
    top_idx = np.argsort(counts)[-n:][::-1]

    results = []
    for idx in top_idx:
        b, g, r = unique[idx].tolist()
        results.append(
            {
                "color": [int(b), int(g), int(r)],
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "pct": round(float(counts[idx]) / total * 100, 1),
            }
        )
    return results


def detect_edge_colors(image: np.ndarray, edge_width: int = 2) -> dict[str, list[int]]:
    """Read the border pixel rows/cols to guess border-color.

    Returns ``{"top": [B,G,R], "bottom": ..., "left": ..., "right": ...}``.
    """
    h, w = image.shape[:2]
    return {
        "top": image[edge_width, w // 2].tolist(),
        "bottom": image[h - 1 - edge_width, w // 2].tolist(),
        "left": image[h // 2, edge_width].tolist(),
        "right": image[h // 2, w - 1 - edge_width].tolist(),
    }
