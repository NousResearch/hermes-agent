"""``model.context_length`` is an explicit pin: it wins over provider metadata (#66168).

Two helpers make that visible without changing the numeric value used: a label for every
surface that renders the context window, and a one-time startup warning when the pin
disagrees with what the provider is known to advertise for the model. The same label marks a
window the resolver only defaulted to (``fallback``), so it is not read as a detected limit.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_warned_pins: set = set()
# (window, model, provider, base_url) -> (monotonic ts, is_fallback). Status-bar repaints read the marker, and the
# check reads local caches; a short TTL lets a later probe or provider-confirmed write clear the label.
_FALLBACK_MEMO: dict[tuple, tuple[float, bool]] = {}
_FALLBACK_MEMO_TTL_SECONDS = 60.0
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


def _resolved_by_fallback(context_length: int, model: str, provider: str, base_url: str) -> bool:
    """True when no source knows ``model``'s window and ``context_length`` is the resolver's default. Local
    reads only (config overrides, static tables, persistent cache, models.dev/OpenRouter disk caches)."""
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT, _is_bedrock_context, get_cached_context_length
    bedrock = _is_bedrock_context(base_url, provider)
    try:
        from agent.bedrock_adapter import BEDROCK_DEFAULT_CONTEXT_LENGTH, bedrock_static_context_length
    except ImportError:  # no boto3: the resolver skipped the Bedrock step and fell through to the generic default
        bedrock, BEDROCK_DEFAULT_CONTEXT_LENGTH, bedrock_static_context_length = False, 0, None
    if context_length != (BEDROCK_DEFAULT_CONTEXT_LENGTH if bedrock else DEFAULT_FALLBACK_CONTEXT):
        return False  # every other value came from a source; the reads below run only for the default
    from agent.models_dev import _override_context_window, lookup_models_dev_context
    from hermes_cli.config import get_custom_provider_context_length
    if provider and _override_context_window(provider, model):
        return False  # model_overrides.<provider>.<model>.context_window: the operator's own number
    if base_url and get_custom_provider_context_length(model=model, base_url=base_url):
        return False
    if bedrock and bedrock_static_context_length is not None:
        confirmed = get_cached_context_length(model, base_url or "bedrock://", bedrock_confirmed=True)
        return confirmed != context_length and not bedrock_static_context_length(model)
    if advertised_context_length(model, base_url):
        return False
    return not (provider and lookup_models_dev_context(provider, model))


def is_fallback_context_length(context_length, *, model: str, provider: str = "", base_url: str = "") -> bool:
    """True when ``context_length`` is a default the resolver fell back to for ``model`` on this route (Bedrock's
    static 128K, or the generic 256K), not a known or detected limit. Fail-open: False on any error."""
    if type(context_length) is not int or context_length <= 0 or not isinstance(model, str) or not model:
        return False
    key = (context_length, model, str(provider or ""), str(base_url or ""))
    now = time.monotonic()
    hit = _FALLBACK_MEMO.get(key)
    if hit is not None and now - hit[0] < _FALLBACK_MEMO_TTL_SECONDS:
        return hit[1]
    try:
        verdict = _resolved_by_fallback(*key)
    except Exception:
        logger.debug("context fallback check failed for %s", model, exc_info=True)
        verdict = False
    if len(_FALLBACK_MEMO) > 256:
        _FALLBACK_MEMO.clear()
    _FALLBACK_MEMO[key] = (now, verdict)
    return verdict


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
    return context_window_marker(
        getattr(compressor, "context_length", None), pin, model=getattr(agent, "model", ""),
        provider=getattr(agent, "provider", "") or "", base_url=getattr(agent, "base_url", "") or "")


def agent_context_suffix(agent) -> str:
    """`` (pinned)`` / `` (fallback)`` / ``""`` for an agent's live compressor window."""
    return _MARKER_SUFFIX.get(agent_context_marker(agent), "")


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
