"""Headroom tool output compression.

Compresses large tool outputs before they enter the conversation context.
Uses Headroom's SmartCrusher directly (no proxy, no import overhead when
disabled).  Works with any message format since it operates on the raw
tool result string before formatting.

Failure mode (prominent): this module is fail-open.  Any error — import
failure, config error, compression exception — returns None and the
original output passes through unchanged.  Compression is lossy by design
(the heuristic text compressor truncates repetitive middle sections).

Why hook inline in ``model_tools.py`` rather than via the existing
``transform_tool_result`` plugin hook: the plugin hook runs for every tool
call but is gated by ``has_hook()`` which requires plugin discovery overhead.
The inline hook has a fast-path that returns before any import when
``context.headroom.enabled`` is False (the default).  Built-in features
that need zero-overhead disabled paths hook inline (see approval,
edit-approval, observability context in the same function).

Config: ``context.headroom`` in config.yaml

    enabled: true/false            (default false)
    exclude_tools: [tool, ...]     (never compress these)
    min_chars: 2000                (skip below this length)
    min_reduction_pct: 20          (skip if compression saves less)
    max_output_chars: 8000         (hard cap on compressed output)
    keep_head_lines: 10            (head lines preserved)
    keep_tail_lines: 5             (tail lines preserved)

Toggle: ``hermes config set context.headroom.enabled true|false``
"""

from __future__ import annotations

from collections import OrderedDict

import logging
import re
import hashlib
import json
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config cache — avoids calling load_config_readonly() on every tool call
# ---------------------------------------------------------------------------
_config_cache: tuple[float, dict] | None = None
_CONFIG_TTL = 5.0  # seconds


def _config() -> dict:
    """Read headroom config with safe defaults and TTL-based caching."""
    global _config_cache
    now = time.monotonic()
    if _config_cache and now - _config_cache[0] < _CONFIG_TTL:
        return _config_cache[1]
    try:
        from hermes_cli.config import load_config_readonly
        raw = load_config_readonly()
        cfg = raw.get("context", {}).get("headroom", {}) or {}
    except Exception:
        cfg = {}
    _config_cache = (now, cfg)
    return cfg


# ---------------------------------------------------------------------------
# SmartCrusher singleton with failed-init sentinel
# ---------------------------------------------------------------------------
# Tri-state: None (untried), False (init failed — never retry), or instance
_crusher = None  # type: ignore[assignment]
_crusher_lock = threading.Lock()


def _get_crusher():
    """Lazy-init SmartCrusher.  Returns None if headroom-ai not installed."""
    global _crusher
    if _crusher is not None:  # True = instance, False = failed sentinel
        return _crusher if _crusher is not False else None
    with _crusher_lock:
        if _crusher is None:
            try:
                from headroom import SmartCrusher
                _crusher = SmartCrusher()
            except ImportError:
                _crusher = False  # sentinel: don't retry
            except Exception as e:
                logger.debug("Headroom SmartCrusher init failed: %s", e)
                _crusher = False  # sentinel: don't retry
    return _crusher if _crusher is not False else None


# ---------------------------------------------------------------------------
# Compression cache — stores originals for future retrieval
# Uses OrderedDict for LRU eviction (most recently retrieved stays)
# ---------------------------------------------------------------------------
_cache: OrderedDict[str, str] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX_DEFAULT = 5000  # fallback when config doesn't specify


def _cache_max(cfg: dict | None = None) -> int:
    """Get cache size from config, with sensible default."""
    c = cfg or _config()
    return c.get("cache_max", _CACHE_MAX_DEFAULT)


def compress_tool_result(
    tool_name: str,
    result: str,
    *,
    config: dict | None = None,
) -> Optional[str]:
    """Compress a tool result using Headroom's SmartCrusher.

    Returns the compressed version with a retrieval marker, or None if the
    original should pass through unchanged.

    Args:
        tool_name: Tool being called (for exclusion list + logging).
        result: Raw tool output string.
        config: Optional config dict (bypasses config read for testing).
                When None, reads from ``context.headroom`` in config.yaml.
    """
    cfg = config or _config()
    if not cfg.get("enabled", False):
        return None

    # Never compress headroom_retrieve output — would create recursive
    # compression and burn cache slots for already-compressed markers.
    if tool_name == "headroom_retrieve":
        return None

    # Env-var escape hatch for CI / debugging
    if __import__("os").environ.get("HERMES_COMPRESS_DISABLE") == "1":
        return None

    # Tool exclusion list
    if tool_name in cfg.get("exclude_tools", []):
        return None

    # Length gate
    if len(result) <= cfg.get("min_chars", 2000):
        return None

    min_reduction = cfg.get("min_reduction_pct", 20)

    # Try SmartCrusher first (excellent for JSON/structured data)
    # Note: SmartCrusher may be unavailable (headroom-ai not installed) —
    # heuristic fallback always runs regardless.
    crusher = _get_crusher()
    if crusher is not None:
        try:
            cr = crusher.crush(result)
            compressed = cr.compressed
            if cr.was_modified and len(result) > 0 and len(compressed) > 0:
                reduction = (1 - len(compressed) / len(result)) * 100
                if reduction >= min_reduction:
                    return _build_output(result, compressed, reduction, "smartcrusher")
        except Exception as e:
            logger.debug("Headroom SmartCrusher failed for %s: %s", tool_name, e)

    # Fallback: heuristic text compressor for code/plain text
    # SmartCrusher handles JSON; this handles the rest.
    # Always runs — doesn't depend on headroom-ai being installed.
    try:
        text_compressed = _heuristic_compress(result, cfg)
        if text_compressed is not None:
            reduction = (1 - len(text_compressed) / len(result)) * 100
            if reduction >= min_reduction:
                return _build_output(result, text_compressed, reduction, "heuristic")
    except Exception as e:
        logger.debug("Headroom text compression failed for %s: %s", tool_name, e)

    return None


def _heuristic_compress(text: str, cfg: dict) -> Optional[str]:
    """Heuristic text compressor for code/plain text that SmartCrusher skips.

    Strategy:
    - Detect line-structured content (code, logs, listings)
    - Keep first N lines (header/context)
    - Keep last N lines (trailer/context)
    - Preserve error-like lines (Traceback, Error, Exception, etc.)
    - Replace middle with a summary of unique line patterns
    """
    lines = text.split("\n")
    if len(lines) < 50:
        return None

    max_output = cfg.get("max_output_chars", 8000)
    keep_head = cfg.get("keep_head_lines", 10)
    keep_tail = cfg.get("keep_tail_lines", 5)

    # Strip blank lines (they cost tokens but add no info)
    non_blank = [l for l in lines if l.strip()]
    if len(non_blank) < 50:
        return None

    # Detect repetitive/sequential patterns (e.g. file listings, numbered output)
    patterns = _detect_repetition(non_blank)
    if patterns is not None:
        return _collapse_repetition(lines, patterns, keep_head, keep_tail, max_output)

    # General truncation: keep head + tail, summarize middle
    if len(lines) > 200:
        return _truncate_middle(lines, keep_head, keep_tail, max_output)

    return None


# Lines matching this are preserved even if they fall in the truncated middle.
# Prevents silent loss of stack traces, error messages, etc.
_ERROR_LINE_RE = re.compile(
    r"Traceback|Error|Exception|FAIL|panic|FATAL|CRITICAL|assertion failed|assert\s",
    re.IGNORECASE,
)


def _detect_repetition(lines: list[str]) -> Optional[dict]:
    """Detect if lines follow a repetitive pattern."""
    if len(lines) < 10:
        return None

    # Count lines that match a simple sequential pattern (e.g. "  N|something")
    numbered_re = re.compile(r"^\s*\d+\|")
    numbered_count = sum(1 for l in lines[:min(50, len(lines))] if numbered_re.match(l))
    if numbered_count > 0.7 * min(50, len(lines)):
        return {"type": "numbered"}

    # Count lines that are structurally similar (same prefix)
    if len(lines) < 20:
        return None
    prefixes = set()
    for l in lines[:20]:
        prefixes.add(l[:30].rstrip())
    if len(prefixes) < 5:
        return {"type": "uniform"}

    return None


def _collapse_repetition(lines: list[str], patterns: dict, keep_head: int, keep_tail: int,
                         max_output: int) -> Optional[str]:
    """Collapse repetitive content into a compact representation."""
    summary_lines = list(lines[:keep_head])
    skipped = len(lines) - keep_head - keep_tail
    summary_lines.append(f" ... {skipped} similar lines omitted ...")
    summary_lines.extend(lines[-keep_tail:])
    result = "\n".join(summary_lines)
    if len(result) < max_output and len(result) < len("\n".join(lines)):
        return result
    return None


def _truncate_middle(lines: list[str], keep_head: int, keep_tail: int,
                     max_output: int) -> Optional[str]:
    """Keep head + tail lines, replace middle with summary.

    Preserves error-like lines (Traceback, Error, Exception, etc.) even
    when they fall in the truncated middle region, plus 1 line of context
    before each error line.
    """
    if len(lines) < 200:
        return None

    middle_start = keep_head
    middle_end = len(lines) - keep_tail
    middle = lines[middle_start:middle_end]

    # Find error-like lines in the middle
    error_indices: list[int] = []
    for i, line in enumerate(middle):
        if _ERROR_LINE_RE.search(line):
            error_indices.append(i)

    # Collect unique non-error lines for the summary
    unique_patterns: set[str] = set()
    for l in middle[:500]:  # sample first 500
        if not _ERROR_LINE_RE.search(l):
            unique_patterns.add(l.strip())

    summary_lines = list(lines[:keep_head])
    skipped = len(lines) - keep_head - keep_tail
    skipped_bytes = sum(len(l) + 1 for l in middle)  # +1 for newline

    if error_indices:
        # Include error lines with context in the compressed output
        summary_lines.append(
            f" ... {skipped} lines / {skipped_bytes} bytes omitted "
            f"({len(unique_patterns)} unique patterns, "
            f"{len(error_indices)} error lines preserved) ..."
        )
        # Add error lines with 1 line of preceding context
        prev_error_end = 0
        for idx in error_indices:
            # Add context line before error
            ctx_start = max(prev_error_end, idx - 1)
            for ci in range(ctx_start, idx + 1):
                if ci < len(middle):
                    summary_lines.append(middle[ci])
            prev_error_end = idx + 1
    else:
        summary_lines.append(
            f" ... {skipped} lines / {skipped_bytes} bytes omitted "
            f"({len(unique_patterns)} unique patterns) ..."
        )

    summary_lines.extend(lines[-keep_tail:])
    result = "\n".join(summary_lines)
    if len(result) < max_output and len(result) < len("\n".join(lines)):
        return result
    return None


def _build_output(
    original: str,
    compressed: str,
    reduction: float,
    strategy: str,
) -> str:
    """Build the final compressed output with cache marker."""
    h = hashlib.md5(
        original.encode("utf-8", errors="replace"),
        usedforsecurity=False,
    ).hexdigest()[:12]

    with _cache_lock:
        # LRU eviction: move to end on access, evict from front
        max_size = _cache_max()
        if h in _cache:
            # Already cached — move to end (most recently used)
            _cache.move_to_end(h)
        else:
            if len(_cache) >= max_size:
                _cache.popitem(last=False)  # evict oldest (least recently used)
            _cache[h] = original

    # Build output with the compressed content.
    # For SmartCrusher, the compressed output is a CCR marker (no readable content),
    # so we also include a preview of the original's first lines.
    # For heuristic compression, the compressed output already IS a readable summary.
    parts = [compressed]

    # SmartCrusher produces CCR markers in some fields — add a preview only
    # when the compressed output has lost the actual content (e.g. terminal
    # tool {"output":"<<ccr:...>>"}, read_file {"content":"<<ccr:...>>"}).
    # For tool outputs where SmartCrusher already preserved representative
    # data (search_files arrays with file paths), skipping the preview avoids
    # duplicate rendering.
    if strategy == "smartcrusher":
        raw_preview = None
        try:
            comp_parsed = json.loads(compressed)
            if isinstance(comp_parsed, dict):
                # Find any string field that was replaced with a CCR marker.
                # Those are the fields where the original content was lost.
                for key, val in comp_parsed.items():
                    if isinstance(val, str) and "<<ccr:" in val:
                        orig_parsed = json.loads(original)
                        if isinstance(orig_parsed, dict):
                            raw_preview = orig_parsed.get(key, "")
                            break
            elif isinstance(comp_parsed, str):
                raw_preview = original
        except (json.JSONDecodeError, TypeError):
            pass

        if raw_preview is not None and raw_preview:
            content_lines = raw_preview.split("\n")
            preview = "\n".join(content_lines[:15])
            if len(content_lines) > 15:
                preview += f"\n... ({len(content_lines) - 15} more lines)"
            parts.append(preview)

    parts.append(f"[Full output cached: hash={h}]")

    # Build the header LAST so the savings figure reflects the actual total
    # delivered size (compressed JSON + preview + markers), not just the
    # SmartCrusher payload.
    out_parts = "\n".join(parts)
    actual_reduction = (1 - len(out_parts) / len(original)) * 100
    header = (
        f"[COMPRESSED by headroom ({strategy}): {actual_reduction:.0f}% saved, "
        f"{len(original):,}→{len(out_parts):,} chars]"
    )
    out = f"{header}\n{out_parts}"

    logger.debug(
        "Headroom: compressed %s→%s chars (%.0f%% saved, strategy=%s, hash=%s)",
        len(original), len(out_parts), actual_reduction, strategy, h,
    )
    return out


def retrieve_cached(h: str) -> Optional[str]:
    """Retrieve an original tool result from the compression cache.
    Promotes the item to most-recently-used to prevent eviction."""
    with _cache_lock:
        if h in _cache:
            _cache.move_to_end(h)  # promote to MRU
        return _cache.get(h)


def cache_stats() -> dict:
    """Return cache statistics."""
    with _cache_lock:
        return {"cached_items": len(_cache), "max_cache_size": _cache_max()}


def clear_cache():
    """Clear the compression cache."""
    with _cache_lock:
        _cache.clear()


def _reset_state():
    """Reset all module-level state.  For tests."""
    global _config_cache, _crusher
    _config_cache = None
    _crusher = None
    clear_cache()
