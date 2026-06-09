"""Transcript normalization and deterministic chronological grouping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def normalize_segments(segments: Iterable[Mapping[str, Any]]) -> list[dict[str, float | str]]:
    """Normalize provider segments and reject unusable timestamp data."""
    normalized: list[dict[str, float | str]] = []
    for segment in segments:
        start = float(segment["start"])
        if "end" in segment:
            end = float(segment["end"])
        elif "duration" in segment:
            end = start + float(segment["duration"])
        else:
            raise ValueError("Transcript segment is missing end or duration")
        text = " ".join(str(segment.get("text", "")).split())
        if start < 0 or end < start:
            raise ValueError("Transcript segment timestamps are invalid")
        if text:
            normalized.append({"start": start, "end": end, "text": text})

    normalized.sort(key=lambda segment: float(segment["start"]))
    return normalized


def group_segments(
    segments: Iterable[Mapping[str, Any]],
    target_seconds: float = 120.0,
    max_chars: int = 3000,
) -> list[dict[str, float | str]]:
    """Group normalized segments without changing their source timestamps."""
    if target_seconds <= 0 or max_chars <= 0:
        raise ValueError("Chunk limits must be positive")

    normalized = normalize_segments(segments)
    chunks: list[dict[str, float | str]] = []
    current: list[dict[str, float | str]] = []

    for segment in normalized:
        projected_text = " ".join(str(item["text"]) for item in [*current, segment])
        elapsed = float(segment["end"]) - float(current[0]["start"]) if current else 0
        if current and (elapsed > target_seconds or len(projected_text) > max_chars):
            chunks.append(_merge(current))
            current = []
        current.append(segment)

    if current:
        chunks.append(_merge(current))
    return chunks


def _merge(segments: list[dict[str, float | str]]) -> dict[str, float | str]:
    return {
        "start": float(segments[0]["start"]),
        "end": float(segments[-1]["end"]),
        "text": " ".join(str(segment["text"]) for segment in segments),
    }
