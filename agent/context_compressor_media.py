"""Readable media-part helpers for ``agent/context_compressor.py``.

Kept in a sibling module so ``context_compressor.py`` stays below its line-count ratchet
without sacrificing readability or architectural docstrings.
"""
from __future__ import annotations

from typing import Any

from session_rules.turns import (
    _ALL_MEDIA_PART_TYPES,
    _AUDIO_PART_TYPES,
    _DOCUMENT_PART_TYPES,
    _IMAGE_PART_TYPES,
    _VIDEO_PART_TYPES,
)


def media_placeholder(part: Any, default_placeholder: str) -> str:
    """Return a modality-specific removal placeholder for *part*."""
    part_type = part.get("type") if isinstance(part, dict) else None
    if part_type in _VIDEO_PART_TYPES:
        return default_placeholder.replace("screenshot", "video").replace("image", "video")
    if part_type in _AUDIO_PART_TYPES:
        return default_placeholder.replace("screenshot", "audio").replace("image", "audio")
    if part_type in _DOCUMENT_PART_TYPES:
        return default_placeholder.replace("screenshot", "document").replace("image", "document")
    return default_placeholder


def multimodal_envelope_tag(content: dict[str, Any]) -> str:
    """Classify a ``_multimodal`` tool-result envelope as video, audio, document, or screenshot."""
    meta = content.get("meta") or {}
    inner_parts = content.get("content") or []
    modality_specs = (
        ("video", "native_video", _VIDEO_PART_TYPES),
        ("audio", "native_audio", _AUDIO_PART_TYPES),
        ("document", "native_document", _DOCUMENT_PART_TYPES),
    )
    for label, meta_flag, part_types in modality_specs:
        if meta.get(meta_flag) or any(isinstance(p, dict) and p.get("type") in part_types for p in inner_parts):
            return label
    return "screenshot"


def extract_media_payloads(parts: Any, *, images_only: bool = False) -> list[str]:
    """Extract raw data URL / base64 strings from *parts* for size-budget accounting."""
    if not isinstance(parts, list):
        return []
    sources: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type not in _ALL_MEDIA_PART_TYPES:
            continue
        if images_only and part_type not in _IMAGE_PART_TYPES:
            continue
        url_obj = (
            part.get("image_url")
            or part.get("video_url")
            or part.get("input_video")
            or part.get("audio_url")
            or part.get("document_url")
        )
        source_obj = part.get("source")
        input_audio_obj = part.get("input_audio")
        file_obj = part.get("file")
        data = (
            (url_obj.get("url") if isinstance(url_obj, dict) else url_obj)
            or (source_obj.get("data") if isinstance(source_obj, dict) else None)
            or (input_audio_obj.get("data") if isinstance(input_audio_obj, dict) else None)
            or (file_obj.get("file_data") if isinstance(file_obj, dict) else None)
            or part.get("file_data")
            or part.get("data")
        )
        sources.append(data if isinstance(data, str) else "")
    return sources
