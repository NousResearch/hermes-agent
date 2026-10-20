"""MEDIA-tag scanning helpers, extracted from ``gateway/platforms/base.py``.

``base.py`` imports every name here back, so ``gateway.platforms.base.<name>`` keeps resolving for
the adapters, the streaming consumer and the tests. The MEDIA regexes, ``MEDIA_DELIVERY_EXTS``,
``validate_media_delivery_path`` and the mask helpers still live in ``base.py``: ``_base()``
resolves that module lazily, because ``base.py`` imports this one at import time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

# Optional per-tag caption: ``MEDIA:<path> | <caption>``. The caption runs to the end of the line and
# rides on the attachment bubble (Telegram photo/document caption) so an image and the question it
# belongs to are delivered together. Without a ``|`` the tag behaves exactly as before (empty
# caption), so existing ``MEDIA:<path>`` output is unchanged.
_MEDIA_CAPTION_SUFFIX_RE = re.compile(r'[ \t]*\|[ \t]*(?P<caption>[^\n]*)')


def _base():
    """``gateway.platforms.base`` resolved at call time (an import-time import would be circular)."""
    from gateway.platforms import base as base_module
    return base_module


def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _path_lacks_deliverable_extension(path: str) -> bool:
    """True when ``path`` has no extension or one outside MEDIA_DELIVERY_EXTS — such paths
    take the validated delivery pass so nonexistent / denylisted ones stay visible.

    ``path`` — either the basename has no extension at all (Caddyfile, Makefile, …) or the extension is not
    in MEDIA_DELIVERY_EXTS (.py, .log, .weirdext, …). Such paths route through the validated delivery pass
    (``validate_media_delivery_path``) instead of the unconditional one, so every file type is deliverable
    (#36060) while nonexistent / denylisted paths stay visible in the text.
    """
    return Path(path).suffix.lower() not in _base().MEDIA_DELIVERY_EXTS


def _match_extensionless_path(scan_text: str, match: re.Match) -> Optional[tuple[str, int]]:
    """Extensionless MEDIA tag match -> validated on-disk ``(safe_path, end_offset)`` or None: the
    captured path first, then extended across single spaces (max 8 tokens, never past a newline
    or the next ``MEDIA:``).

    When that fails validation, the candidate is progressively extended forward across single spaces
    (validation-gated, bounded at 8 tokens, never past a newline or a subsequent ``MEDIA:`` keyword) so
    unknown-extension paths containing spaces deliver (#24032). Returns ``(safe_path, end_offset)`` where
    ``end_offset`` is the index in ``scan_text`` just past the matched path, or ``None`` when nothing
    validates.
    """
    validate = _base().validate_media_delivery_path
    path = _normalize_media_tag_path(match.group("path"))
    if not path:
        return None
    safe = validate(path)
    if safe:
        return safe, match.end("path")
    start = match.start("path")
    segment = scan_text[start:].split("\n", 1)[0]
    nxt = segment.find("MEDIA:", 1)
    if nxt != -1:
        segment = segment[:nxt]
    pos = match.end("path") - start
    for _ in range(8):
        token = re.match(r"[ \t]*[^ \t]+", segment[pos:])
        if not token:
            break
        tok_end = pos + token.end()
        safe = validate(_normalize_media_tag_path(segment[:tok_end]))
        if safe:
            return safe, start + tok_end
        pos = tok_end
    return None


def _extensionless_media_matches(masked: str):
    """Yield ``(match, safe_path, end_offset)`` for every extension-less / unknown-extension
    MEDIA tag in ``masked`` that ``validate_media_delivery_path`` accepts."""
    for match in _base().MEDIA_EXTENSIONLESS_TAG_RE.finditer(masked):
        path = _normalize_media_tag_path(match.group("path"))
        if path and _path_lacks_deliverable_extension(path):
            resolved = _match_extensionless_path(masked, match)
            if resolved is not None:
                yield match, resolved[0], resolved[1]


def _media_caption_suffix(text: str, tag_end: int) -> Optional[tuple[str, int]]:
    """``(caption, end_offset)`` for a ``| <caption>`` suffix starting at ``tag_end``, else None.

    ``end_offset`` covers the whole suffix so the caption text is deleted from the delivered
    body — it is delivered as the attachment caption, never as a separate chat message.
    """
    match = _MEDIA_CAPTION_SUFFIX_RE.match(text, tag_end)
    if match is None:
        return None
    return match.group("caption").strip(), match.end()


def _media_caption_key(path: str) -> Optional[str]:
    """Normalize a MEDIA-tag path to the same form ``filter_media_delivery_paths`` yields.

    Captions are keyed by the *delivered* path so a dispatch site can look one up after the
    safety filter (which resolves symlinks / rewrites docker paths) without re-parsing the tag.
    """
    try:
        safe = _base().validate_media_delivery_path(path)
    except (OSError, RuntimeError, ValueError):
        safe = None
    if safe:
        return safe
    try:
        return os.path.expanduser(path)
    except (OSError, RuntimeError, ValueError):
        return None


def _real_media_tag_spans(masked: str) -> list:
    """(start, end) spans of deliverable MEDIA tags on a masked copy: known-extension tags
    unconditionally, extension-less / unknown ones only if validate_media_delivery_path accepts.

    A ``| caption`` suffix is part of the span when present: the caption is delivered on the
    attachment bubble, so it must not survive in the body text."""
    spans: list = []
    for m in _base().MEDIA_TAG_CLEANUP_RE.finditer(masked):
        caption = _media_caption_suffix(masked, m.end())
        spans.append((m.start(), caption[1] if caption else m.end()))
    for match, _, end in _extensionless_media_matches(masked):
        caption = _media_caption_suffix(masked, end)
        spans.append((match.start(), caption[1] if caption else end))
    return spans


def extract_media_captions(content: str) -> dict:
    """``{delivered_path: caption}`` for ``MEDIA:<path> | <caption>`` tags (absent when no pipe).

    Sibling of ``BasePlatformAdapter.extract_media`` (which keeps its ``(path, is_voice)`` tuple
    contract): a caption rides on the attachment bubble, so dispatch sites that batch images or
    attach files look it up by path. Keys are normalized the same way
    ``filter_media_delivery_paths`` normalizes delivered paths, so a caption survives the safety
    filter (symlink resolution / docker path rewrite included).

    Duplicate tags for the same path are first-wins: ``setdefault`` keeps the first tag's caption
    and ignores the later ones.
    """
    if "MEDIA:" not in content or "|" not in content:
        return {}
    base_module = _base()
    masked = base_module._mask_media_scan_text(content)
    captions: dict = {}

    def _record(path: str, tag_end: int) -> None:
        suffix = _media_caption_suffix(masked, tag_end)
        if suffix is None or not suffix[0]:
            return
        key = _media_caption_key(path)
        if key:
            captions.setdefault(key, suffix[0])

    for match in base_module.MEDIA_TAG_CLEANUP_RE.finditer(masked):
        path = _normalize_media_tag_path(match.group("path"))
        if path:
            _record(path, match.end())
    for _match, safe_path, end in _extensionless_media_matches(masked):
        _record(safe_path, end)
    return captions
