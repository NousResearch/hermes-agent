"""Pure clipping logic — no I/O, no framework imports, trivially testable."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ClipResult:
    text: str
    changed: bool
    tokens_before: int
    tokens_after: int

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


def estimate_tokens(text: str, chars_per_token: float = 3.0) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / max(0.5, chars_per_token)))


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def clip(text: str, cap_tokens: int, head_ratio: float = 0.6,
         chars_per_token: float = 3.0) -> ClipResult:
    """Keep head + tail, drop the middle, leave a pointer line.

    Head/tail (not head-only) because failures and final results sit at the edges;
    a head-only cut hides the exit code and the stack trace.
    """
    before = estimate_tokens(text, chars_per_token)
    if before <= cap_tokens:
        return ClipResult(text, False, before, before)

    keep_chars = max(64, int(cap_tokens * chars_per_token))
    head_chars = max(32, int(keep_chars * head_ratio))
    tail_chars = max(32, keep_chars - head_chars)
    if len(text) <= head_chars + tail_chars:
        return ClipResult(text, False, before, before)

    omitted = len(text) - head_chars - tail_chars
    pointer = (
        f"\n\n… [{omitted:,} chars / ≈{int(omitted / max(0.5, chars_per_token)):,} tok "
        f"omitted · sha256:{fingerprint(text)}] …\n\n"
    )
    out = text[:head_chars] + pointer + text[-tail_chars:]
    return ClipResult(out, True, before, estimate_tokens(out, chars_per_token))
