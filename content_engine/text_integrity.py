"""Guarantee baked-in text is correct: OCR the generated image and fuzzy-match
each expected string. Free + local (Tesseract). Drives draft_media's regen loop
so a misspelled image (e.g. 'DABUG'/'MISPATCH') never ships."""

from __future__ import annotations
import re, sys
from difflib import SequenceMatcher
from typing import Iterable, Optional


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "",
                                      s.lower().replace("\n", " "))).strip()


def _present(needle: str, haystack: str, threshold: float = 0.82) -> bool:
    n = _norm(needle)
    if not n:
        return True
    if n in haystack:
        return True
    # sliding fuzzy match over comparable-length windows
    words = haystack.split()
    span = len(n.split())
    for i in range(0, max(1, len(words) - span + 1)):
        window = " ".join(words[i:i + span + 2])
        if SequenceMatcher(None, n, window).ratio() >= threshold:
            return True
    return False


def ocr_text(image_path: str) -> Optional[str]:
    try:
        import pytesseract
        from PIL import Image
        return _norm(pytesseract.image_to_string(Image.open(image_path)))
    except ImportError:
        print("[text_integrity] WARN: pytesseract not installed, OCR skipped", file=sys.stderr)
    except Exception as exc:
        print(f"[text_integrity] WARN: OCR failed: {exc}", file=sys.stderr)
    return None


def verify_text(image_path: str, expected: Iterable[str],
                threshold: float = 0.82):
    """Return (ok, missing) from locally recognised text.

    A required-text check fails closed when OCR is unavailable or cannot read
    the image. Publishing a label we cannot verify is less safe than a retry.
    """
    expected_values = list(expected)
    haystack = ocr_text(image_path)
    if haystack is None:
        print(
            f"[text_integrity] WARN: OCR unavailable for {image_path}, failing text gate",
            file=sys.stderr,
        )
        return False, expected_values
    missing = [item for item in expected_values if not _present(item, haystack, threshold)]
    return (len(missing) == 0, missing)


def has_significant_text(image_path: str, min_words: int = 2,
                         min_chars: int = 6) -> bool:
    """Return True if OCR detects significant text in the image.

    Used for SCENE/HERO types where the image is supposed to be textless. If
    the model added labels, watermarks, captions or any other text we want
    to reject and regenerate. The thresholds (min_words=2, min_chars=6) are
    loose enough to ignore single short tokens ("v1.0", "1") that the model
    might add by accident but tight enough to catch real labels.
    """
    haystack = ocr_text(image_path)
    if haystack is None:
        print(
            f"[text_integrity] WARN: OCR unavailable for {image_path}, rejecting textless gate",
            file=sys.stderr,
        )
        return True
    if not haystack:
        return False
    words = haystack.split()
    if len(words) < min_words:
        return False
    # A "significant" word is at least 3 chars and not pure digits
    sig = [w for w in words if len(w) >= 3 and not w.isdigit()]
    return len(sig) >= 1 and len(haystack) >= min_chars
