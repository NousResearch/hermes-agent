"""Per-image token cost learned from the provider's own usage, never from a vendor formula.

A flat per-image constant is wrong in both directions: a 1920x1080 screenshot costs ~1,100 tokens
on one provider and 4,000+ on a local mmproj model. The provider prices every image exactly on the
request that carries it, so the cost is observable: with a fresh usage anchor (real prompt count of
the previous response), the residual between the next real ``prompt_tokens`` and
``anchor + text-only delta`` is the price of the N images that delta introduced (#70328).

The learned value is kept per ``model@host`` in ``~/.hermes/cache/image_token_costs.json`` so a new
session starts calibrated, and bound per turn through a ContextVar so every estimator
(preflight trigger, tail-budget walk, gateway hygiene) prices images the same way.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TOKEN_COST = 1500
# Observations outside this band are text-estimate noise, not an image price.
_MIN_PLAUSIBLE, _MAX_PLAUSIBLE = 64, 32_768
_EMA_ALPHA = 0.5

_image_cost_var: ContextVar[Optional[int]] = ContextVar("hermes_image_token_cost", default=None)
_LEARNED: Dict[str, int] = {}
_LOADED = False


def _cache_path():
    from agent.model_metadata import _cache_file

    return _cache_file("image_token_costs.json")


def _key(model: Any, base_url: Any) -> str:
    from utils import base_url_hostname

    return f"{model or ''}@{base_url_hostname(base_url or '') or ''}"


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from agent.model_metadata import _load_json_dict

    for k, v in _load_json_dict(_cache_path()).items():
        if isinstance(v, int) and _MIN_PLAUSIBLE <= v <= _MAX_PLAUSIBLE:
            _LEARNED[k] = v


def learned_image_token_cost(model: Any, base_url: Any) -> int:
    """Learned per-image cost for ``model@host``, else the flat default."""
    _load()
    return _LEARNED.get(_key(model, base_url), DEFAULT_IMAGE_TOKEN_COST)


def current_image_token_cost() -> int:
    """Per-image cost bound for the running turn (see :func:`image_cost_context`), else the default."""
    bound = _image_cost_var.get()
    return bound if bound is not None else DEFAULT_IMAGE_TOKEN_COST


@contextlib.contextmanager
def image_cost_context(cost: Optional[int]):
    token = _image_cost_var.set(cost)
    try:
        yield
    finally:
        _image_cost_var.reset(token)


def bind_image_token_cost(agent: Any) -> None:
    """Bind the agent's learned per-image cost to the current context for the rest of the turn."""
    _image_cost_var.set(learned_image_token_cost(getattr(agent, "model", None), getattr(agent, "base_url", None)))


def count_images(messages: List[Dict[str, Any]]) -> int:
    from agent.model_metadata import _count_image_tokens

    return sum(_count_image_tokens(m, 1) for m in messages if isinstance(m, dict))


def calibrate_from_usage(agent: Any, messages: List[Dict[str, Any]], prompt_tokens: Any) -> Optional[int]:
    """Learn the per-image cost from the response that just priced ``messages``.

    Requires the PREVIOUS anchor (real count of the prior request) to still match: the residual
    ``prompt_tokens - (anchor + text-only delta)`` is then the provider's price for the images the
    delta introduced. Returns the new learned cost, or None when this response teaches nothing
    (no anchor, no new images, implausible residual)."""
    from agent.usage_anchor import anchored_context_tokens

    anchor = getattr(agent, "_usage_anchor", None)
    try:
        real = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        return None
    if real <= 0 or not isinstance(anchor, dict) or not isinstance(messages, list):
        return None
    base_count = int(anchor.get("base_count") or 0)
    delta = messages[base_count:]
    if delta and isinstance(delta[0], dict) and delta[0].get("role") == "assistant":
        delta = delta[1:]
    n_images = count_images(delta)
    if n_images <= 0:
        return None
    with image_cost_context(0):
        text_only = anchored_context_tokens(messages, anchor)
    if text_only is None:
        return None
    per_image = (real - text_only) // n_images
    if not _MIN_PLAUSIBLE <= per_image <= _MAX_PLAUSIBLE:
        return None
    key = _key(getattr(agent, "model", None), getattr(agent, "base_url", None))
    _load()
    prior = _LEARNED.get(key)
    learned = per_image if prior is None else int(prior + _EMA_ALPHA * (per_image - prior))
    _LEARNED[key] = learned
    _image_cost_var.set(learned)
    try:
        from utils import atomic_json_write

        atomic_json_write(_cache_path(), dict(_LEARNED), indent=0, separators=(",", ":"))
    except Exception:
        logger.debug("image token cost persist failed", exc_info=True)
    logger.info(
        "Image token cost calibrated from provider usage: %s images priced %s tokens each (learned %s for %s)",
        n_images, f"{per_image:,}", f"{learned:,}", key,
    )
    return learned
