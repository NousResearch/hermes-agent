#!/usr/bin/env python3
"""TOON utility module for KENSEI cron pipelines.

Provides TOON encoding for structured data that gets embedded in LLM prompts.
Replaces json.dumps() calls where the output is consumed by an LLM.

Usage:
    from toon_utils import toon_encode, maybe_toon, toon_blob, size_report

    # Always encode as TOON (the typical case)
    prompt = f"Process this data:\n{toon_blob(data)}"

    # Only encode if it saves tokens (skips small/trivial data)
    prompt = f"Process this data:\n{toon_blob(data)}"  # maybe_toon is used internally
"""

import json

# Threshold in characters — below this, JSON is fine and TOON overhead isn't worth it
TOON_MIN_CHARS = 200


def _try_encode(data: object) -> str | None:
    """Try to encode data as TOON. Returns None if library unavailable."""
    try:
        from toon_format import encode
        return encode(data)
    except ImportError:
        return None
    except Exception as exc:
        import sys
        print(f"TOON encode failed: {exc}", file=sys.stderr)
        return None


def toon_encode(data: object, fallback_to_json: bool = True) -> str:
    """Encode data as TOON. Falls back to compact JSON if TOON unavailable."""
    result = _try_encode(data)
    if result is not None:
        return result
    if fallback_to_json:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(data, indent=2, ensure_ascii=False)


def maybe_toon(data: object, threshold: int = TOON_MIN_CHARS) -> str:
    """Encode as TOON only if data is large enough to justify the overhead."""
    try:
        compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        # Data isn't JSON-serializable — try TOON directly
        result = _try_encode(data)
        if result is not None:
            return result
        raise TypeError(f"data is neither JSON-serializable nor TOON-encodable: {type(data)}")

    if len(compact) < threshold:
        return compact
    result = _try_encode(data)
    if result is not None:
        return result
    return compact


def toon_blob(data: object, label: str = "toon") -> str:
    """Wrap data in a labelled codeblock for LLM prompts. Shortcut for the most common use case."""
    encoded = toon_encode(data)
    return f"```{label}\n{encoded}\n```"


def estimated_tokens(text: str) -> int:
    """Rough token estimate for English+code text (chars/4)."""
    return len(text) // 4


def size_report(data: object) -> str:
    """Return a comparison string showing TOON vs JSON size for the given data.
    Useful for debugging and monitoring actual savings.
    Falls back gracefully if data isn't JSON-serializable."""
    toon = toon_encode(data)

    try:
        compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return (
            f"TOON: {len(toon)} chars (~{estimated_tokens(toon)} tok) | "
            f"JSON compact: {len(compact)} chars (~{estimated_tokens(compact)} tok) "
            f"{100 - (len(toon) / len(compact) * 100):+.0f}% | "
            f"JSON pretty: {len(pretty)} chars (~{estimated_tokens(pretty)} tok) "
            f"{100 - (len(toon) / len(pretty) * 100):+.0f}%"
        )
    except (TypeError, ValueError):
        return (
            f"TOON: {len(toon)} chars (~{estimated_tokens(toon)} tok) | "
            f"JSON: not serializable"
        )
