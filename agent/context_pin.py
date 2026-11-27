"""``model.context_length`` is an explicit pin: it wins over provider metadata (#66168).

Two helpers make that visible without changing the numeric value used: a label for every
surface that renders the context window, and a one-time startup warning when the pin
disagrees with what the provider is known to advertise for the model. The same label marks a
window the resolver only defaulted to (``fallback``), so it is not read as a detected limit.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_warned_pins: set = set()
_MARKER_SUFFIX = {"pinned": " (pinned)", "fallback": " (fallback)"}


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


def is_fallback_context_length(context_length, *, model: str, provider: str = "", base_url: str = "") -> bool:
    """True when ``context_length`` is a default the resolver fell back to for ``model`` on this route (Bedrock's
    static 128K, or the generic 256K), not a known or detected limit. The resolver records that itself
    (agent.context_default_record), so this agrees with every source it consults; ``provider`` is accepted for
    call-site symmetry. Fail-open: False on any error."""
    if type(context_length) is not int or context_length <= 0 or not isinstance(model, str) or not model:
        return False
    try:
        from agent.context_default_record import was_context_defaulted
        return was_context_defaulted(context_length, model, base_url)
    except Exception:
        logger.debug("context fallback check failed for %s", model, exc_info=True)
        return False


def context_window_marker(context_length, config_context_length, *, model: str, provider: str = "",
                          base_url: str = "") -> str:
    """``"pinned"`` (the ``model.context_length`` pin), ``"fallback"`` (a default, not a known limit) or ``""``."""
    if is_context_pinned(context_length, config_context_length):
        return "pinned"
    fallback = is_fallback_context_length(context_length, model=model, provider=provider, base_url=base_url)
    return "fallback" if fallback else ""


def context_window_suffix(context_length, config_context_length, **route) -> str:
    """`` (pinned)`` / `` (fallback)`` / ``""`` for a rendered window; ``route`` is model, provider, base_url."""
    return _MARKER_SUFFIX.get(context_window_marker(context_length, config_context_length, **route), "")


def agent_context_marker(agent) -> str:
    """:func:`context_window_marker` for an agent's live compressor window."""
    compressor = getattr(agent, "context_compressor", None)
    pin = getattr(compressor, "_config_context_length", getattr(agent, "_config_context_length", None))
    # The compressor's route is what resolved the window; the agent's is the fallback for doubles without one.
    route = {name: getattr(compressor, name, None) or getattr(agent, name, "") or "" for name in ("model", "provider", "base_url")}
    window = getattr(compressor, "context_length", None)  # first: reading it resolves the window
    own = getattr(compressor, "_context_default", None)  # (model, base_url, defaulted window | None) from that lookup
    if isinstance(own, tuple) and len(own) == 3 and own[:2] == (getattr(compressor, "model", None), getattr(compressor, "base_url", None)):
        if is_context_pinned(window, pin):
            return "pinned"
        return "fallback" if type(own[2]) is int and own[2] == window else ""
    return context_window_marker(window, pin, **route)


def agent_context_suffix(agent) -> str:
    """`` (pinned)`` / `` (fallback)`` / ``""`` for an agent's live compressor window."""
    return _MARKER_SUFFIX.get(agent_context_marker(agent), "")


def advertised_context_length(model: str, base_url: str = "") -> Optional[int]:
    """Provider-advertised window from LOCAL sources only (persistent cache learned on this
    endpoint, models.dev disk cache, hardcoded catalog). Never a network probe: users pin
    precisely when the endpoint cannot report its window, so the check must not add startup
    latency or a failing request. A catalog hit on a different variant of the tag (uncatalogued
    ``:`` deployment suffix) is not an advertised value — returns None so the pin warning stays
    silent instead of defending a variant the deployment is not running."""
    from agent.model_metadata import (
        DEFAULT_CONTEXT_LENGTHS, _load_model_metadata_disk_cache, _longest_key_match,
        _normalize_model_version, _strip_provider_prefix, get_cached_context_length,
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
    if not hit:
        return None
    suffix = hit[0] + ":"
    tag = model.lower()
    if suffix in tag or _normalize_model_version(suffix) in _normalize_model_version(tag):
        # The tag carries a deployment/variant segment the catalog has no entry for (Ollama
        # "glm-5.3:cloud" hitting the hosted "glm-5.3"): the hit describes a different variant
        # of the model, not this deployment, so there is nothing advertised to compare against.
        return None
    return hit[1]


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
