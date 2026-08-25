"""Per-model stale-timeout FLOOR for known reasoning models.

Reasoning models routinely exceed the default chat-model stale detectors (stream 180s,
non-stream 90s): upstream proxies idle-kill the stream mid-think, surfacing as
``BrokenPipeError``/``RemoteProtocolError``. The stale-detector scaling applies
``max(default, floor)`` from :func:`get_reasoning_stale_timeout_floor`, so this never
overrides explicit per-model ``stale_timeout_seconds``/``request_timeout_seconds`` (that
branch never calls it), never lowers a threshold, and is ``None`` for non-allowlisted models.
"""

from __future__ import annotations

import re
from typing import Optional


# (slug, floor_seconds).  Each slug is matched as a discrete
# word-boundary component via the wrapper regex in ``_match_any``
# below.  Order is irrelevant — the first regex match wins.
_REASONING_STALE_TIMEOUT_FLOORS: tuple[tuple[str, int], ...] = (
    # NVIDIA Nemotron — reasoning models behind hosted NIM with
    # documented 60-180s upstream idle kill (NVIDIA/NemoClaw#4846:
    # 120s measured).
    ("nemotron-3-ultra", 600),
    ("nemotron-3-super", 600),
    ("nemotron-3-nano",  300),
    ("nemotron-3.5-lightning", 300),
    # DeepSeek — R1 and V4 reasoning models on hosted NIM / DeepSeek direct.
    # V4 series emits reasoning_content in a separate delta field before
    # final content, requiring the same extended stale timeout floor.
    ("deepseek-r1", 600),
    ("deepseek-reasoner", 600),
    ("deepseek-v4-flash", 600),
    ("deepseek-v4-pro", 600),
    # Qwen — QwQ reasoning + Qwen3 thinking variants.  QwQ-32B
    # preview is the stable slug; ``qwen3`` covers the family of
    # thinking-mode Qwen3 models (qwen3-235b-a22b, qwen3-32b, etc.)
    # without over-matching every Qwen3 instruct variant — the
    # right-anchor requires the slug to be at the start of the
    # remaining model name, so ``qwen3-235b-instruct`` (instruct is
    # NOT a thinking variant) would still match.  Acceptable
    # trade-off: instruct variants of qwen3 get the 180s floor
    # even though they don't reason.  The cost is a slightly longer
    # wait on a hung provider; the alternative (matching only
    # ``qwen3-.*-thinking``) breaks the moment NVIDIA or Alibaba
    # ships a slightly different naming shape.
    ("qwq-32b", 300),
    ("qwen3", 180),
    # OpenAI o-series — known multi-minute TTFB.  Each variant
    # enumerated explicitly so bare ``o1`` doesn't over-match
    # ``olmo-1`` or hypothetical future community derivatives.
    ("o1", 600),
    ("o1-mini", 600),
    ("o1-pro", 600),
    ("o1-preview", 600),
    ("o3", 600),
    ("o3-pro", 600),
    ("o3-mini", 300),
    ("o4-mini", 300),
    # Anthropic Claude 4.x thinking variants.  Anchored at
    # ``claude-opus-4`` so non-thinking Claude 3.x or future
    # non-reasoning Claude variants don't match.
    ("claude-opus-4", 240),
    ("claude-opus-5", 240),
    ("claude-sonnet-5", 180),
    ("claude-sonnet-4.5", 180),
    ("claude-sonnet-4.6", 180),
    # Anthropic Mythos-class named reasoning models (claude-fable-5, …).
    # 1M context + 128K output — heavier thinking phase than the
    # numbered Claude line, so the floor is in the deep-reasoning tier
    # alongside o1 / deepseek-r1 / nemotron-3-ultra.  Without this
    # entry the stale-stream detector kills fable-5's thinking phase
    # at the default 180s (300s with context scaling), tripping the
    # cross-turn circuit breaker after 5 consecutive stale kills.
    ("claude-fable", 600),
    # xAI Grok reasoning variants.  Explicit reasoning-only keys
    # plus one for the ``non-reasoning`` variant so users picking
    # the fast variant don't get the 300s floor.  Bare ``grok-3``,
    # ``grok-4`` etc. don't match — only the explicit reasoning /
    # non-reasoning pairs.
    ("grok-4-fast-reasoning", 300),
    ("grok-4.20-reasoning", 300),
    ("grok-4.5", 300),
    ("grok-4.6", 300),
    ("grok-4-fast-non-reasoning", 180),
    # "Ox Alpha" stealth reasoning model (stealth/ox-alpha on OpenRouter,
    # x-preview-f-free on OpenCode Zen).  Marketed as a reasoning model for
    # long-horizon coding/agentic work; 1M context — same tier as the Grok
    # reasoning variants.
    ("ox-alpha", 300),
    ("x-preview-f-free", 300),
)


# Pre-compiled once at import (immutable afterwards — safe under free-threaded Python).
# Right anchor: end-of-string or a slug separator; ``:`` because OpenRouter routing suffixes
# (``:free``, ``:nitro``) attach directly to the slug. Longest-first so ``o3-mini`` beats ``o3``.
_SORTED_REASONING_FLOORS: list[tuple[str, float, re.Pattern[str]]] = [
    (slug, floor, re.compile(r"^" + re.escape(slug) + r"(?:$|[\-._:])"))
    for slug, floor in sorted(
        ((slug, floor) for floor, slugs in _REASONING_STALE_TIMEOUT_FLOORS.items() for slug in slugs),
        key=lambda kv: -len(kv[0]),
    )
]


def get_reasoning_stale_timeout_floor(model: object) -> Optional[float]:
    """Stale-timeout floor (seconds) for a known reasoning model, else ``None``.

    The aggregator prefix (up to the last ``/``) is stripped and the slug matched
    start-anchored with an end-or-separator right anchor, so ``qwen3-235b`` matches ``qwen3``
    but ``some-other-qwen3`` and ``llama-4-70b-o1-preview`` do not.
    """
    if not model or not isinstance(model, str):
        return None
    name = model.strip().lower().rsplit("/", 1)[-1]
    for _slug, floor, pattern in _SORTED_REASONING_FLOORS:
        if pattern.search(name):
            return float(floor)
    return None
