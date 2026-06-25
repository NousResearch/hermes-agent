"""Build the per-message token-breakdown footer line for bot replies.

Gated per-session by the ``/tokens`` toggle (see the gateway's
``_tokens_display`` map). The numbers are decoded from the bit-packed
``token_count`` on the turn's messages (see :mod:`hermes_token_codec`) — the
final assistant row carries output/reasoning, the prompt-tail user/tool row
carries total_input/cache_read.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _decode(role: Optional[str], token_count: Any) -> Dict[str, int]:
    from hermes_token_codec import resolve_message_tokens
    return resolve_message_tokens(role, token_count)


def build_token_line(agent_result: Dict[str, Any]) -> str:
    """Return a compact one-line token breakdown, or "" when unavailable.

    Example: ``📊 in:1520 out:234 reason:128 cache:890``.
    """
    messages: List[Dict[str, Any]] = agent_result.get("messages") or []

    out_tokens = reason_tokens = in_tokens = cache_tokens = 0

    # Final assistant row → output + reasoning (the reply we are decorating).
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            d = _decode("assistant", msg.get("token_count"))
            out_tokens, reason_tokens = d["output"], d["reasoning"]
            break

    # Nearest prompt-tail user/tool row carrying input → total_input + cache.
    # (First-turn user rows can be NULL under the best-effort write path, so
    # scan back for the most recent row that actually carries input data.)
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("user", "tool"):
            d = _decode(msg.get("role"), msg.get("token_count"))
            if d["input"] or d["cache_read"]:
                in_tokens, cache_tokens = d["input"], d["cache_read"]
                break

    # Fallback for input when no prompt row carried it (e.g. NULL first-turn
    # user row): the agent reports the prompt size as last_prompt_tokens.
    if not in_tokens:
        try:
            in_tokens = int(agent_result.get("last_prompt_tokens") or 0)
        except (TypeError, ValueError):
            in_tokens = 0

    if not (out_tokens or reason_tokens or in_tokens or cache_tokens):
        return ""

    return f"📊 in:{in_tokens} out:{out_tokens} reason:{reason_tokens} cache:{cache_tokens}"
