"""Formatting helpers for Rasputin recall blocks.

The provider is context-only in V1, so the formatter's job is simply to turn
raw search hits into a compact injected block with provenance. Canonical memory
stays in Hermes built-ins and JSONL; Rasputin output is advisory context.

TODO:
- tighten formatting against the final Rasputin hit schema
- add richer grouping for daily-log windows if they become noisy
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping
import textwrap


def format_recall_block(results: Iterable[Mapping[str, Any]], *, limit: int = 8) -> str:
    """Render a compact recall block for prompt injection."""
    hits = [dict(hit) for hit in list(results or [])[: max(int(limit or 1), 1)]]
    if not hits:
        return ""

    facts: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []

    for hit in hits:
        bucket = _bucket_for_hit(hit)
        if bucket == "facts":
            facts.append(hit)
        elif bucket == "windows":
            windows.append(hit)
        else:
            other.append(hit)

    lines = ["# Rasputin Recall"]
    mixed = sum(bool(group) for group in (facts, windows, other)) > 1

    if facts:
        if mixed:
            lines.append("## Facts")
        lines.extend(_format_group(facts))
    if windows:
        if mixed:
            lines.append("## Windows")
        lines.extend(_format_group(windows))
    if other:
        if mixed:
            lines.append("## Other")
        lines.extend(_format_group(other))

    return "\n".join(lines)


def _format_group(hits: Iterable[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    for hit in hits:
        metadata = _metadata(hit)
        identifier = (
            str(metadata.get("canonical_id") or "").strip()
            or str(hit.get("id") or "").strip()
            or str(hit.get("source") or "").strip()
            or "rasputin-hit"
        )
        score = _score_text(hit)
        provenance = _provenance_text(hit, metadata)
        text = _truncate(_extract_text(hit), width=220)

        lines.append(f"- [score {score}] {identifier}")
        if provenance:
            lines.append(f"  {provenance}")
        if text:
            lines.append(f"  text: {text}")
        lines.append("")
    if lines:
        lines.pop()
    return lines


def _bucket_for_hit(hit: Mapping[str, Any]) -> str:
    metadata = _metadata(hit)
    kind = str(metadata.get("kind") or "").lower()
    source = str(hit.get("source") or metadata.get("source") or "").lower()
    if kind in {"typed_memory", "memory_write"}:
        return "facts"
    if kind in {"conversation_window", "session_window", "daily_log_window"}:
        return "windows"
    if "window" in kind or "window" in source:
        return "windows"
    if metadata.get("canonical_id") or metadata.get("memory_type"):
        return "facts"
    return "other"


def _metadata(hit: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = hit.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _score_text(hit: Mapping[str, Any]) -> str:
    value = hit.get("score")
    if value is None:
        value = hit.get("similarity")
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _provenance_text(hit: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    parts: List[str] = []

    source = str(hit.get("source") or metadata.get("source") or "").strip()
    if source:
        parts.append(f"source: {source}")

    fleet = str(metadata.get("fleet") or "").strip()
    if fleet:
        parts.append(f"fleet: {fleet}")

    memory_type = str(metadata.get("memory_type") or metadata.get("type") or "").strip()
    if memory_type:
        parts.append(f"type: {memory_type}")

    session_id = str(metadata.get("session_id") or "").strip()
    if session_id:
        parts.append(f"session: {session_id}")

    kind = str(metadata.get("kind") or "").strip()
    if kind and kind not in {"typed_memory", "conversation_window", "session_window", "daily_log_window", "memory_write"}:
        parts.append(f"kind: {kind}")

    return " | ".join(parts)


def _extract_text(hit: Mapping[str, Any]) -> str:
    for key in ("text", "content", "snippet", "summary"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _truncate(text: str, *, width: int) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return textwrap.shorten(text, width=width, placeholder=" …")
