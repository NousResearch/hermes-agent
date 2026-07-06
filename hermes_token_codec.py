# hermes_token_codec.py
"""Bit-packed codec for messages.token_count.

Layout (64-bit, MSB→LSB):  [F:1][tag1:4][value1:27][tag2:4][value2:28]

F (bit 63) is BOTH the format flag and SQLite's signed-INTEGER sign bit:
packed rows (F=1) are stored as NEGATIVE integers; legacy rows (F=0) stay
non-negative. Read-time discriminator: token_count < 0  ->  packed.
"""
from __future__ import annotations
from typing import Optional

TAG_OUTPUT       = 0x0
TAG_REASONING    = 0x1
TAG_TOTAL_INPUT  = 0x2
TAG_CACHE_READ   = 0x3

_TAG_NAME = {
    TAG_OUTPUT:      "output_tokens",
    TAG_REASONING:   "reasoning_tokens",
    TAG_TOTAL_INPUT: "total_input_tokens",
    TAG_CACHE_READ:  "cache_read_tokens",
}

_FORMAT_FLAG = 1 << 63
_TAG_MASK    = 0xF
_V1_BITS, _V2_BITS = 27, 28
_V1_MAX = (1 << _V1_BITS) - 1
_V2_MAX = (1 << _V2_BITS) - 1
_U64     = 1 << 64
_I63     = 1 << 63

def _to_signed64(u: int) -> int:
    return u - _U64 if u >= _I63 else u

def _to_unsigned64(s: int) -> int:
    return s + _U64 if s < 0 else s

def pack_token_count(tag1: int, value1: int, tag2: int, value2: int) -> int:
    v1 = max(0, min(int(value1), _V1_MAX))
    v2 = max(0, min(int(value2), _V2_MAX))
    packed = (_FORMAT_FLAG | ((tag1 & _TAG_MASK) << 59) | (v1 << 32) | ((tag2 & _TAG_MASK) << 28) | v2)
    return _to_signed64(packed)

def unpack_token_count(token_count: Optional[int]) -> dict:
    if token_count is None:
        return {}
    if token_count >= 0:
        return {"legacy": token_count}
    u = _to_unsigned64(token_count)
    tag1   = (u >> 59) & _TAG_MASK
    value1 = (u >> 32) & _V1_MAX
    tag2   = (u >> 28) & _TAG_MASK
    value2 = u & _V2_MAX
    out: dict = {}
    out[_TAG_NAME.get(tag1, f"tag_{tag1:#x}")] = value1
    out[_TAG_NAME.get(tag2, f"tag_{tag2:#x}")] = value2
    return out

def pack_assistant_tokens(output_tokens: int, reasoning_tokens: int) -> int:
    return pack_token_count(TAG_OUTPUT, output_tokens, TAG_REASONING, reasoning_tokens)

def pack_input_tokens(total_input_tokens: int, cache_read_tokens: int) -> int:
    return pack_token_count(TAG_TOTAL_INPUT, total_input_tokens, TAG_CACHE_READ, cache_read_tokens)

# Roles eligible to carry a call's input-token attribution. The prompt-tail
# row that triggered an API call is always a user message (first turn) or a
# tool-result message (subsequent turns). An assistant row is never the prompt
# tail at response time — the reply is appended only afterwards — so an
# assistant/system/developer tail (e.g. an assistant prefill) is skipped
# rather than mis-attributed.
INPUT_TOKEN_PROMPT_TAIL_ROLES = ("user", "tool")


def attribute_input_tokens_to_prompt_tail(messages, canonical_usage) -> Optional[dict]:
    """Bit-pack a call's (total_input, cache_read) onto the prompt-tail row.

    Called right after an API response arrives, when ``messages[-1]`` is still
    the message that triggered the request — the assistant reply has not been
    appended yet. Attribution is applied ONLY when that tail is a user/tool
    row (:data:`INPUT_TOKEN_PROMPT_TAIL_ROLES`) whose ``token_count`` is not
    already bit-packed (negative); otherwise nothing is written. Mutates the
    row in place and returns it, or returns ``None`` when no eligible tail
    exists (empty list, non-dict tail, wrong role, or already-packed row).

    ``canonical_usage`` only needs ``.prompt_tokens`` and ``.cache_read_tokens``
    attributes. Kept here (not in conversation_loop) so the role/timing/
    no-clobber invariants stay verifiable without the agent import chain.
    """
    if not messages:
        return None
    tail = messages[-1]
    if not isinstance(tail, dict):
        return None
    if tail.get("role") not in INPUT_TOKEN_PROMPT_TAIL_ROLES:
        return None
    existing = tail.get("token_count")
    if existing is not None and existing < 0:
        # Already bit-packed (e.g. re-entry on a retried call) — don't clobber.
        return None
    tail["token_count"] = pack_input_tokens(
        canonical_usage.prompt_tokens,
        canonical_usage.cache_read_tokens,
    )
    return tail


def format_token_count(n: int) -> str:
    """Human-friendly token magnitude for compact footers (shared by all UIs).

      * n < 1000        -> exact integer        ("234")
      * 1000 <= n < 1e6 -> K, adaptive decimals ("1.52K", "23.5K", "123K")
      * n >= 1e6        -> M                     ("1.23M", "12.5M")

    Negative / non-numeric inputs clamp to "0".
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "0"
    if n < 1000:
        return str(max(0, n))
    for threshold, suffix in ((1_000_000, "M"), (1_000, "K")):
        if n >= threshold:
            scaled = n / threshold
            if scaled < 10:
                text = f"{scaled:.2f}"
            elif scaled < 100:
                text = f"{scaled:.1f}"
            else:
                text = f"{scaled:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(n)


def resolve_message_tokens(role: str, token_count: Optional[int]) -> dict:
    out = {"input": 0, "output": 0, "cache_read": 0, "reasoning": 0}
    d = unpack_token_count(token_count)
    if not d:
        return out
    if "legacy" in d:
        if role == "assistant":
            out["output"] = d["legacy"]
        return out
    out["output"]     = d.get("output_tokens", 0)
    out["reasoning"]  = d.get("reasoning_tokens", 0)
    out["input"]      = d.get("total_input_tokens", 0)
    out["cache_read"] = d.get("cache_read_tokens", 0)
    return out
