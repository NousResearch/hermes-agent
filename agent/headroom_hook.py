"""
Headroom compression hook for Kensei context optimization.

Wraps headroom.compress() with try/except passthrough (G1) and
config gating (headroom.enabled in config.yaml).  Applied once per
API call in conversation_loop.py, before _build_api_kwargs.

KENSEI CUSTOM — fork patch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config helpers ──────────────────────────────────────────────────

def _get_headroom_enabled() -> bool:
    """Check config.yaml headroom.enabled (default: False)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        headroom_cfg = cfg.get("headroom")
        if isinstance(headroom_cfg, dict):
            val = headroom_cfg.get("enabled", False)
            return bool(val)
    except Exception:
        pass
    return False


def _get_headroom_config() -> Dict[str, Any]:
    """Read headroom config section from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        headroom_cfg = cfg.get("headroom")
        if isinstance(headroom_cfg, dict):
            return dict(headroom_cfg)
    except Exception:
        pass
    return {}


# ── Metrics tracking ────────────────────────────────────────────────

# Per-session accumulation (simple dict, not thread-safe — conversation
# loop is single-threaded per session so this is fine).
_session_tokens_before: int = 0
_session_tokens_after: int = 0
_session_compress_calls: int = 0
_session_compress_errors: int = 0


def get_session_metrics() -> Dict[str, int]:
    """Return accumulated compression metrics for the current session."""
    return {
        "tokens_before": _session_tokens_before,
        "tokens_after": _session_tokens_after,
        "tokens_saved": _session_tokens_before - _session_tokens_after,
        "compress_calls": _session_compress_calls,
        "compress_errors": _session_compress_errors,
    }


def reset_session_metrics() -> None:
    """Reset per-session metrics (called at session start)."""
    global _session_tokens_before, _session_tokens_after
    global _session_compress_calls, _session_compress_errors
    _session_tokens_before = 0
    _session_tokens_after = 0
    _session_compress_calls = 0
    _session_compress_errors = 0


# ── Core compression hook ───────────────────────────────────────────

def compress_messages(
    messages: List[Dict[str, Any]],
    model: str,
    model_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Apply headroom compression to messages array before API call.

    Gated by config: ``headroom.enabled`` must be True.

    Args:
        messages: API messages list (OpenAI/Anthropic format).
        model: Active model name (used for token counting).
        model_limit: Optional context window override.

    Returns:
        Compressed messages (same format). On any error, returns
        the ORIGINAL messages unchanged with a logged warning — the
        tool result is never silently dropped from context (G1).
    """
    if not _get_headroom_enabled():
        return messages

    if not messages:
        return messages

    global _session_compress_calls, _session_compress_errors
    global _session_tokens_before, _session_tokens_after

    _session_compress_calls += 1

    try:
        from headroom import compress, CompressConfig

        # Use the conversation model for token counting
        # Protect recent turns, skip user messages (coding agent pattern)
        config = CompressConfig(
            compress_user_messages=False,
            compress_system_messages=True,
            protect_recent=4,
            protect_analysis_context=True,
        )

        headroom_kwargs = _get_headroom_config()

        # Allow config.yaml overrides
        target_ratio = headroom_kwargs.get("target_ratio")
        if target_ratio is not None:
            config.target_ratio = float(target_ratio)

        min_tokens = headroom_kwargs.get("min_tokens_to_compress")
        if min_tokens is not None:
            config.min_tokens_to_compress = int(min_tokens)

        kompress_model = headroom_kwargs.get("kompress_model")
        if kompress_model:
            config.kompress_model = kompress_model

        result = compress(
            messages,
            model=model,
            model_limit=model_limit or 200000,
            config=config,
        )

        _session_tokens_before += result.tokens_before
        _session_tokens_after += result.tokens_after

        if result.tokens_saved > 0:
            logger.info(
                "headroom: saved %d tokens (%.0f%%), transforms=%s",
                result.tokens_saved,
                result.compression_ratio * 100,
                result.transforms_applied,
            )

        return result.messages

    except Exception:
        _session_compress_errors += 1
        logger.warning(
            "headroom.compress() failed — returning uncompressed messages. "
            "This is a non-fatal error: the session continues with full context.",
            exc_info=True,
        )
        return messages
