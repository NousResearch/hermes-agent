"""
Free-model picker for the OpenRouter routing proxy.

Periodically fetches the popularity-sorted model list from
`/api/frontend/models/find?active=true&fmt=cards&order=most-popular`,
filters to:

  - free (pricing.prompt == 0 AND completion == 0)
  - within the top N most popular
  - meets size criteria: total params >= MIN_TOTAL_B AND active params >= MIN_ACTIVE_B
    (where parsable from the slug/name; unknown sizes are conservatively skipped)
  - supports tools (we'll be using it via the agent's tool-calling loop)

Returns the highest-ranked qualifier (or None). Caches for CACHE_TTL_S.

The router consults this when env ROUTER_PREFER_FREE=true. If the free model
errors or returns empty, the router falls back to the originally-requested
model so the user still gets an answer.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Optional

POPULAR_URL = (
    "https://openrouter.ai/api/frontend/models/find?active=true&fmt=cards&order=most-popular"
)
CACHE_TTL_S = 3600  # refresh hourly
TOP_N = 20         # consider this many of the most popular models
MIN_TOTAL_B = 100  # billions of total params
MIN_ACTIVE_B = 10  # billions of active params (only enforced if we can parse it)

# Known sizes for models whose names don't expose params unambiguously.
# Add to this when a useful free model appears that the regex misses.
_KNOWN_SIZES = {
    "tencent/hy3-preview": (480, 17),               # ~480B MoE, 17B active (per HF model card)
    "z-ai/glm-4.5-air": (106, 12),                  # ~106B MoE, 12B active
    "openai/gpt-oss-120b": (120, 5),                # 120B MoE, 5B active per OpenAI docs
    "moonshotai/kimi-k2.6": (1000, 32),             # estimate; only relevant if a free variant appears
}

_cache: dict = {"ts": 0.0, "model": None, "all_qualifiers": []}


def _fetch_popular() -> list[dict]:
    req = urllib.request.Request(POPULAR_URL, headers={"User-Agent": "hermes-router-free/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    return ((data.get("data") or {}).get("models") or [])


def _is_free(model: dict) -> bool:
    pricing = (model.get("endpoint") or {}).get("pricing") or {}
    try:
        return float(pricing.get("prompt") or 0) == 0 and float(pricing.get("completion") or 0) == 0
    except (TypeError, ValueError):
        return False


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bt])(?:[-_/.\s]+a(\d+(?:\.\d+)?)\s*([bt]))?", re.IGNORECASE)


def _parse_params_b(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (total_b, active_b) parsed from a model slug/name.

    `120b-a12b` → (120, 12)
    `1t` → (1000, None)
    `405b` → (405, None)
    Unknown → (None, None)
    """
    if not text:
        return None, None
    m = _PARAM_RE.search(text)
    if not m:
        return None, None
    total = float(m.group(1))
    if m.group(2).lower() == "t":
        total *= 1000
    active = None
    if m.group(3):
        active = float(m.group(3))
        if m.group(4) and m.group(4).lower() == "t":
            active *= 1000
    return total, active


def _supports_tools(model: dict) -> bool:
    sp = (model.get("endpoint") or {}).get("supported_parameters") or []
    return "tools" in sp


def _meets_criteria(model: dict) -> bool:
    slug = model.get("slug") or model.get("permaslug") or ""
    # Strip anything after a colon (e.g. ":free", ":nitro") so KNOWN_SIZES match
    base = slug.split(":")[0]
    total_b, active_b = _KNOWN_SIZES.get(base, (None, None))
    if total_b is None:
        # Try to parse from slug + short_name
        slug_total, slug_active = _parse_params_b(slug)
        name_total, name_active = _parse_params_b(model.get("short_name") or model.get("name") or "")
        total_b = slug_total or name_total
        active_b = slug_active or name_active
    if total_b is None:
        return False  # conservative: don't pick unknown sizes
    if total_b < MIN_TOTAL_B:
        return False
    if active_b is not None and active_b < MIN_ACTIVE_B:
        return False
    if not _supports_tools(model):
        return False
    return True


def get_free_model() -> Optional[str]:
    """Best free model right now, or None.

    Caches for CACHE_TTL_S so we don't hit OpenRouter's frontend on every
    proxy request.
    """
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL_S and _cache["model"] is not None:
        return _cache["model"]
    if now - _cache["ts"] < CACHE_TTL_S and _cache["model"] is None and _cache["ts"] > 0:
        # Negative cache — saw nothing last time, don't refetch yet
        return None
    try:
        models = _fetch_popular()
    except Exception:
        return _cache["model"]  # serve stale on error
    qualifiers = []
    for m in models[:TOP_N]:
        if _is_free(m) and _meets_criteria(m):
            slug = m.get("slug") or m.get("permaslug")
            qualifiers.append(slug)
    pick = qualifiers[0] if qualifiers else None
    # OpenRouter free variants need the `:free` suffix to actually route as free
    if pick and not pick.endswith(":free"):
        pick = pick + ":free"
    _cache.update(ts=now, model=pick, all_qualifiers=qualifiers)
    return pick


def get_free_qualifiers() -> list[str]:
    """All currently-qualifying free models in popularity order. Cached."""
    get_free_model()  # populate cache
    return list(_cache.get("all_qualifiers") or [])





# ----- Fast-tier (smaller free models, prioritised for low latency) ------

# Lower bound for "fast tier": active params <= this count and total <= 80B.
# Smaller models tend to be much faster TPS at the cost of capability.
FAST_MIN_TOTAL_B = 0     # no minimum (we want small)
FAST_MAX_TOTAL_B = 100   # hard cap
FAST_MAX_ACTIVE_B = 30   # if active known, cap at 30B

# Manually curated fast-tier known sizes.
_FAST_KNOWN_SIZES = {
    "meta-llama/llama-3.3-70b-instruct": (70, 70),
    "google/gemma-4-31b-it": (31, 31),
    "google/gemma-4-26b-a4b-it": (26, 4),
    "nvidia/nemotron-3-nano-30b-a3b": (30, 3),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": (30, 3),
    "nvidia/nemotron-nano-12b-v2-vl": (12, 12),
    "nvidia/nemotron-nano-9b-v2": (9, 9),
    "z-ai/glm-4.5-air": (106, 12),  # MoE — fast despite total
    "qwen/qwen3-next-80b-a3b-instruct": (80, 3),
    "openai/gpt-oss-20b": (20, 20),
}


def _meets_fast_criteria(model: dict) -> bool:
    slug = (model.get("slug") or model.get("permaslug") or "").split(":")[0]
    total_b, active_b = _FAST_KNOWN_SIZES.get(slug, (None, None))
    if total_b is None:
        slug_total, slug_active = _parse_params_b(slug)
        name_total, name_active = _parse_params_b(model.get("short_name") or model.get("name") or "")
        total_b = slug_total or name_total
        active_b = slug_active or name_active
    if total_b is None:
        return False
    if total_b > FAST_MAX_TOTAL_B:
        return False
    if active_b is not None and active_b > FAST_MAX_ACTIVE_B:
        return False
    if not _supports_tools(model):
        return False
    return True


def get_fast_free_model() -> "Optional[str]":
    """Best small / fast free model right now. Use when latency matters more
    than capability (e.g. simple chat continuations)."""
    now = time.time()
    # Reuse main cache load
    if now - _cache["ts"] >= CACHE_TTL_S:
        try:
            _fetch_popular()  # refreshes the cache via get_free_model
        except Exception:
            pass
    try:
        models = _fetch_popular()
    except Exception:
        return None
    for m in models[:TOP_N]:
        if _is_free(m) and _meets_fast_criteria(m):
            slug = m.get("slug") or m.get("permaslug")
            return slug if slug.endswith(":free") else slug + ":free"
    # Fall back to broader scan across the FULL list if top-N had no fast frees.
    # Top-N popular tends to be all big paid models; small fast frees live further down.
    for m in models:
        if _is_free(m) and _meets_fast_criteria(m):
            slug = m.get("slug") or m.get("permaslug")
            return slug if slug.endswith(":free") else slug + ":free"
    return None


if __name__ == "__main__":
    print("best free model:", get_free_model())
    print("all qualifiers:", get_free_qualifiers())
