"""Style extractor — infer CSS-like properties from element rectangles & screenshot crops."""

import logging
from typing import Any

import numpy as np

log = logging.getLogger("win-gui-test.analyzers.style")


def estimate_border_radius(crop: np.ndarray, threshold: int = 30) -> int:
    """Estimate corner radius from a cropped element image.

    Scans the top-left corner for the first non-background pixel.
    Returns approximate radius in px, or 0 if detection fails.
    """
    if crop.size == 0:
        return 0
    gray = np.mean(crop, axis=2)
    bg = gray[0, 0]
    h, w = crop.shape[:2]
    for r in range(min(h, w)):
        if abs(gray[r, r] - bg) > threshold:
            return max(0, r - 1)
    return 0


def extract_font_hint(image: np.ndarray, crop: np.ndarray) -> dict[str, Any]:
    """Return rough font characteristics from cropped text region.

    This is a heuristic — for accurate font info use OCR tools (tesseract etc).
    """
    gray = np.mean(crop, axis=2)
    # text vs background contrast
    fg_mask = gray < np.percentile(gray, 20)
    if fg_mask.sum() == 0:
        return {"note": "No text detected"}
    # estimate font size from character height
    rows_with_text = np.any(fg_mask, axis=1)
    text_rows = np.where(rows_with_text)[0]
    font_size_guess = len(text_rows) if len(text_rows) > 3 else 0
    fg_pixels = crop[fg_mask]
    fg_color = fg_pixels.mean(axis=0).tolist() if len(fg_pixels) else [0, 0, 0]
    bg_pixels = crop[~fg_mask]
    bg_color = bg_pixels.mean(axis=0).tolist() if len(bg_pixels) else [255, 255, 255]

    return {
        "estimated_font_size_px": font_size_guess,
        "text_color_bgr": [round(c) for c in fg_color],
        "background_color_bgr": [round(c) for c in bg_color],
    }
