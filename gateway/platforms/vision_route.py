"""
Vision-route picker — STUB, not wired.

When a Discord message includes an image attachment AND the bot is
addressed, the gateway could swap the model to a vision-capable one for
that single turn. Currently DeepSeek V4 Flash (default chat model) does
NOT support vision, so the bot can't see images at all.

Free vision-capable models on OpenRouter (as of 2026-05-05):
  - nvidia/nemotron-nano-12b-v2-vl:free   (12B, vision-capable, free)
  - google/gemini-2.5-flash-image         (cheap, vision input)

To enable, set HERMES_VISION_ROUTE=true in .env and ensure the gateway
patches discord.py's _handle_message to call pick_vision_model() when
message.attachments has any image/* MIME type, then pass that model
into the AIAgent for that turn.

NOT wired yet. Intentionally inert until someone reviews + enables.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

ENABLED = os.environ.get("HERMES_VISION_ROUTE", "").lower() in ("1", "true", "yes")

# Free vision-capable allowlist; primary first.
_VISION_FREE_ALLOWLIST = (
    "nvidia/nemotron-nano-12b-v2-vl:free",
)

# Paid vision fallbacks (cheap), only used if HERMES_VISION_ALLOW_PAID=true
_VISION_PAID_ALLOWLIST = (
    "google/gemini-2.5-flash-image",
    "anthropic/claude-haiku-4.5",
)

_cache = {"ts": 0.0, "primary": None}
_CACHE_TTL_S = 3600


def _model_active(slug: str) -> bool:
    """Cheap check: hit /api/v1/models/<slug>/endpoints; 200 + nonempty endpoints means usable."""
    try:
        req = urllib.request.Request(
            f"https://openrouter.ai/api/v1/models/{slug}/endpoints",
            headers={"User-Agent": "hermes-vision-stub/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read())
        eps = (data.get("data") or {}).get("endpoints") or []
        return len(eps) > 0
    except Exception:
        return False


def pick_vision_model(allow_paid: bool = False) -> Optional[str]:
    """Return the first available vision-capable model from the allowlist."""
    if not ENABLED:
        return None
    now = time.time()
    if now - _cache["ts"] < _CACHE_TTL_S and _cache["primary"] is not None:
        return _cache["primary"]
    candidates = list(_VISION_FREE_ALLOWLIST)
    if allow_paid:
        candidates += list(_VISION_PAID_ALLOWLIST)
    for slug in candidates:
        if _model_active(slug):
            _cache.update(ts=now, primary=slug)
            return slug
    _cache.update(ts=now, primary=None)
    return None


if __name__ == "__main__":
    print("ENABLED:", ENABLED)
    print("primary:", pick_vision_model())
