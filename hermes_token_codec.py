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
