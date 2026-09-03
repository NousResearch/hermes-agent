#!/usr/bin/env python3
"""OCR + translation for the desktop's live-subtitle overlay.

The desktop app samples the subtitle band of the window the user is watching
and POSTs a changed crop to ``/api/subtitles/process``; this module answers
it. One frame in, one decision out:

- the crop's subtitle text is unchanged from ``prev_text`` → ``unchanged``
  (the desktop just refreshes its overlay hold; no translation runs);
- the crop has no subtitle text → empty result (the desktop clears the line);
- new text → translate it and return the text plus the union pixel box of the
  original line, which the desktop covers and paints over.

No model conversation is involved: OCR is local (RapidOCR, an optional
dependency), translation is one auxiliary-LLM call per NEW line with a rolling
per-stream context window — subtitle lines are short and context-poor, and
target languages with grammatical gender (the motivating user watches in
Portuguese) translate wrongly without the preceding lines. Repeated lines hit
an LRU cache instead of the model.
"""

import base64
import io
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# One OCR engine per process — model load costs ~1s, every call after is fast.
_ocr_engine = None
_ocr_engine_error: Optional[str] = None
_ocr_lock = threading.Lock()

OCR_INSTALL_HINT = (
    "Live subtitles need the local OCR engine. Install it into the Hermes "
    "environment with: pip install 'hermes-agent[subtitles]' (or: pip install "
    "rapidocr onnxruntime) and restart the backend."
)

# Translation memory. The LRU spares the model from re-translating recurring
# lines (names, catchphrases); the per-stream context deque gives each NEW
# line the preceding dialogue so gender/formality agreement survives
# line-by-line translation.
_CACHE_MAX = 512
_translation_cache: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
_CONTEXT_LINES = 6
_MAX_STREAMS = 8
_stream_contexts: Dict[str, Dict[str, Any]] = {}
_state_lock = threading.Lock()

# Lines that are player chrome, not dialogue: pure timestamps/counters
# ("1:23:45", "02 / 10") flicker through the band on hover.
_NON_DIALOGUE_RE = re.compile(r"^[\d\s:./\-]+$")

_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "pt-br": "Brazilian Portuguese",
    "zh": "Chinese",
}


def _language_name(language: str) -> str:
    return _LANGUAGE_NAMES.get(language.strip().lower(), language.strip())


def _settings() -> Dict[str, Any]:
    """The ``subtitles:`` config section, defaults applied."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("subtitles") or {}
    except Exception:
        section = {}
    return {
        "max_chars_per_line": int(section.get("max_chars_per_line") or 42),
        "min_ocr_confidence": float(section.get("min_ocr_confidence") or 0.5),
    }


def _get_ocr_engine():
    """RapidOCR singleton; a missing optional dep fails with the install hint."""
    global _ocr_engine, _ocr_engine_error
    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        if _ocr_engine_error is not None:
            raise RuntimeError(_ocr_engine_error)
        try:
            from rapidocr import RapidOCR

            _ocr_engine = RapidOCR()
        except ImportError:
            _ocr_engine_error = OCR_INSTALL_HINT
            raise RuntimeError(_ocr_engine_error) from None
        except Exception as exc:
            _ocr_engine_error = f"The subtitle OCR engine failed to start: {exc}"
            raise RuntimeError(_ocr_engine_error) from exc
        return _ocr_engine


def normalize_subtitle_text(text: str) -> str:
    """Comparison form: OCR jitter must not read as a new line."""
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def wrap_subtitle(text: str, max_chars: int = 42) -> str:
    """Greedy word wrap into display lines. Never drops words — an over-long
    unbreakable token just makes an over-long line."""
    words = (text or "").split()
    if not words:
        return ""
    lines: List[str] = []
    line = words[0]
    for word in words[1:]:
        if len(line) + 1 + len(word) <= max_chars:
            line += " " + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)


def _union_box(boxes: List[Any]) -> Optional[Dict[str, int]]:
    """Union of RapidOCR quadrilaterals → {x, y, width, height} in crop pixels."""
    xs: List[float] = []
    ys: List[float] = []
    for quad in boxes:
        for point in quad:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return None
    x0, y0 = min(xs), min(ys)
    width, height = max(xs) - x0, max(ys) - y0
    if width <= 0 or height <= 0:
        return None
    return {"x": round(x0), "y": round(y0), "width": round(width), "height": round(height)}


def _decode_image(image_bytes: bytes):
    from PIL import Image
    import numpy as np

    with Image.open(io.BytesIO(image_bytes)) as img:
        return np.asarray(img.convert("RGB"))


def _read_subtitle_lines(image_bytes: bytes) -> Tuple[List[str], List[Any]]:
    """OCR the crop → (dialogue lines top-to-bottom, their boxes)."""
    settings = _settings()
    engine = _get_ocr_engine()
    result = engine(_decode_image(image_bytes))

    # RapidOCR returns numpy arrays (ambiguous under `or`) — compare to None.
    raw_texts = getattr(result, "txts", None)
    raw_boxes = getattr(result, "boxes", None)
    raw_scores = getattr(result, "scores", None)
    texts = list(raw_texts) if raw_texts is not None else []
    boxes = list(raw_boxes) if raw_boxes is not None else []
    scores = list(raw_scores) if raw_scores is not None else []

    rows = []
    for index, text in enumerate(texts):
        cleaned = (text or "").strip()
        score = float(scores[index]) if index < len(scores) else 1.0
        if not cleaned or score < settings["min_ocr_confidence"]:
            continue
        if _NON_DIALOGUE_RE.match(cleaned):
            continue
        box = boxes[index] if index < len(boxes) else None
        top = min(float(p[1]) for p in box) if box is not None else 0.0
        left = min(float(p[0]) for p in box) if box is not None else 0.0
        rows.append((top, left, cleaned, box))

    rows.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in rows], [row[3] for row in rows if row[3] is not None]


def _stream_context(stream_id: str) -> deque:
    """The stream's rolling (source, translation) history, LRU-evicted."""
    key = stream_id or "default"
    now = time.time()
    with _state_lock:
        entry = _stream_contexts.get(key)
        if entry is None:
            if len(_stream_contexts) >= _MAX_STREAMS:
                oldest = min(_stream_contexts, key=lambda k: _stream_contexts[k]["used"])
                _stream_contexts.pop(oldest, None)
            entry = {"lines": deque(maxlen=_CONTEXT_LINES), "used": now}
            _stream_contexts[key] = entry
        entry["used"] = now
        return entry["lines"]


def _cached_translation(key: Tuple[str, str]) -> Optional[str]:
    with _state_lock:
        value = _translation_cache.get(key)
        if value is not None:
            _translation_cache.move_to_end(key)
        return value


def _remember_translation(key: Tuple[str, str], value: str) -> None:
    with _state_lock:
        _translation_cache[key] = value
        _translation_cache.move_to_end(key)
        while len(_translation_cache) > _CACHE_MAX:
            _translation_cache.popitem(last=False)


def _translate(source: str, language: str, context: deque) -> str:
    """One auxiliary-LLM call: the new line, with recent dialogue for agreement."""
    from agent.auxiliary_client import call_llm

    target = _language_name(language)
    system = (
        f"You translate live movie subtitles into {target}. Translate the final "
        "line only; earlier lines are context from the same scene. Keep the "
        "register and tone of the dialogue, and keep it as concise as a "
        "subtitle. Output ONLY the translated line — no quotes, no notes."
    )
    parts = []
    if context:
        recent = "\n".join(f"{src} -> {dst}" for src, dst in context)
        parts.append(f"Recent lines (already translated):\n{recent}\n")
    parts.append(f"Translate: {source}")

    response = call_llm(
        task="subtitles",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()
    # Some models quote the answer despite instructions; unwrap one layer.
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    if not text:
        raise RuntimeError("the translation model returned an empty line")
    return text


def process_frame(
    image_bytes: bytes,
    language: str,
    prev_text: str = "",
    stream_id: str = "",
) -> Dict[str, Any]:
    """One subtitle-band crop → what the desktop should paint. See module doc."""
    lines, boxes = _read_subtitle_lines(image_bytes)
    source_text = "\n".join(lines)

    if normalize_subtitle_text(source_text) == normalize_subtitle_text(prev_text):
        return {"ok": True, "unchanged": True}

    if not source_text:
        return {"ok": True, "text": "", "source_text": "", "box": None}

    box = _union_box(boxes)
    if box is None:
        return {"ok": True, "text": "", "source_text": "", "box": None}

    settings = _settings()
    context = _stream_context(stream_id)
    cache_key = (normalize_subtitle_text(source_text), language.strip().lower())
    translated = _cached_translation(cache_key)
    if translated is None:
        translated = _translate(source_text.replace("\n", " "), language, context)
        _remember_translation(cache_key, translated)
    context.append((source_text.replace("\n", " "), translated))

    return {
        "ok": True,
        "text": wrap_subtitle(translated, settings["max_chars_per_line"]),
        "source_text": source_text,
        "box": box,
    }


def decode_image_data_url(data_url: str, max_bytes: int) -> bytes:
    """The endpoint's payload check: a bounded base64 PNG data URL."""
    prefix = "data:image/png;base64,"
    if not (data_url or "").startswith(prefix):
        raise ValueError("image_data_url must be a base64 PNG data URL")
    encoded = data_url[len(prefix):]
    if len(encoded) > max_bytes * 4 // 3 + 4:
        raise ValueError("subtitle frame is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("image_data_url is not valid base64") from exc
    if len(raw) > max_bytes:
        raise ValueError("subtitle frame is too large")
    return raw
