"""Normalized per-session context usage for API and UI clients."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer, or ``None`` for unknown/sentinel values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def build_context_usage(
    *,
    engine: Any = None,
    compression_enabled: bool = True,
    compacted: bool = False,
    updated_at: Optional[float] = None,
    used_tokens: Any = None,
    context_window_tokens: Any = None,
    compression_threshold_tokens: Any = None,
    compression_count: Any = 0,
) -> Dict[str, Any]:
    """Build a stable context-occupancy snapshot.

    ``ContextEngine.get_status()`` is preferred so plugin engines can define
    their own readings. Attribute fallback keeps older engines compatible.
    Unknown occupancy stays ``None`` rather than being confused with an empty
    context; this also hides the built-in compressor's ``-1`` post-compaction
    sentinel.
    """
    status: Dict[str, Any] = {}
    if engine is not None:
        getter = getattr(engine, "get_status", None)
        if callable(getter):
            try:
                raw = getter()
                if isinstance(raw, dict):
                    status = raw
            except Exception:
                status = {}

        if used_tokens is None:
            used_tokens = status.get(
                "last_prompt_tokens", getattr(engine, "last_prompt_tokens", None)
            )
        if context_window_tokens is None:
            context_window_tokens = status.get(
                "context_length", getattr(engine, "context_length", None)
            )
        if compression_threshold_tokens is None:
            compression_threshold_tokens = status.get(
                "threshold_tokens", getattr(engine, "threshold_tokens", None)
            )
        if not compression_count:
            compression_count = status.get(
                "compression_count", getattr(engine, "compression_count", 0)
            )

    used = _positive_int(used_tokens)
    window = _positive_int(context_window_tokens)
    threshold = _positive_int(compression_threshold_tokens)
    try:
        count = max(0, int(compression_count or 0))
    except (TypeError, ValueError, OverflowError):
        count = 0

    usage_percent = round(used / window * 100, 2) if used and window else None
    threshold_percent = round(threshold / window * 100, 2) if threshold and window else None
    progress_percent = round(used / threshold * 100, 2) if used and threshold else None
    remaining = max(threshold - used, 0) if used and threshold else None

    return {
        "used_tokens": used,
        "context_window_tokens": window,
        "usage_percent": min(100.0, usage_percent) if usage_percent is not None else None,
        "compression_threshold_tokens": threshold,
        "compression_threshold_percent": threshold_percent,
        "compression_progress_percent": progress_percent,
        "tokens_until_compression": remaining,
        "compression_count": count,
        "compression_enabled": bool(compression_enabled),
        "compacted": bool(compacted),
        "updated_at": float(updated_at if updated_at is not None else time.time()),
    }
