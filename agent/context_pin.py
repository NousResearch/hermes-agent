"""``model.context_length`` is an explicit pin: it wins over provider metadata (#66168).

Two helpers make that visible without changing the numeric value used: a label for every
surface that renders the context window, and a one-time startup warning when the pin
disagrees with what the provider is known to advertise for the model.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_warned_pins: set = set()


def is_context_pinned(context_length, config_context_length) -> bool:
    """True when the displayed ``context_length`` is the ``model.context_length`` pin."""
    return (
        isinstance(config_context_length, int)
        and not isinstance(config_context_length, bool)
        and config_context_length > 0
        and context_length == config_context_length
    )


def context_pin_suffix(context_length, config_context_length) -> str:
    """`` (pinned)`` when the shown value comes from ``model.context_length``, else ``""``."""
    return " (pinned)" if is_context_pinned(context_length, config_context_length) else ""


def advertised_context_length(model: str, base_url: str = "") -> Optional[int]:
    """Provider-advertised window from LOCAL sources only (persistent cache learned on this
    endpoint, models.dev disk cache, hardcoded catalog). Never a network probe: users pin
    precisely when the endpoint cannot report its window, so the check must not add startup
    latency or a failing request."""
    from agent.model_metadata import (
        DEFAULT_CONTEXT_LENGTHS, _load_model_metadata_disk_cache, _longest_key_match,
        _strip_provider_prefix, get_cached_context_length,
    )
    model = _strip_provider_prefix(str(model or ""))
    if not model:
        return None
    if base_url:
        cached = get_cached_context_length(model, base_url)
        if cached:
            return int(cached)
    entry = _load_model_metadata_disk_cache().get(model) or {}
    ctx = entry.get("context_length") if isinstance(entry, dict) else None
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    return hit[1] if hit else None


def warn_once_on_pin_disagreement(model: str, base_url: str, config_context_length) -> bool:
    """Log ONE warning per (model, pin) when ``model.context_length`` disagrees with the
    advertised window. Returns True when the warning fired (the pin still wins)."""
    if not is_context_pinned(config_context_length, config_context_length):
        return False
    advertised = advertised_context_length(model, base_url)
    if not advertised or advertised == config_context_length:
        return False
    key = (str(model), int(config_context_length))
    if key in _warned_pins:
        return False
    _warned_pins.add(key)
    logger.warning(
        "model.context_length pins %s at %s tokens but the provider advertises %s; the pin wins. "
        "Remove model.context_length from config.yaml to use the advertised window.",
        model, f"{config_context_length:,}", f"{advertised:,}",
    )
    return True
